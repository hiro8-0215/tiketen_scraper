import os
import sys
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    RANDOM_SEED, TARGET_COLUMN,
    LLM_MODEL_ID, LLM_MAX_LENGTH, LLM_BATCH_SIZE_TRAIN,
    LLM_EPOCHS, LLM_LEARNING_RATE,
    LLM_LORA_R, LLM_LORA_ALPHA, LLM_LORA_TARGETS,
)
from data_loader import load_raw_data, clean_data, load_manual_data, merge_manual_data

OUTPUT_DIR = "./qwen_ticket_model"


def format_prompt(row):
    """チケット情報をLLMが読みやすいテキスト形式に変換"""
    tags = row.get('ticket_tags', '')
    tags_str = str(tags) if tags and str(tags) != 'nan' else 'なし'
    ttype = row.get('ticket_type', '')
    ttype_str = str(ttype) if ttype and str(ttype) != 'nan' else '不明'
    base_price = row.get('base_price', np.nan)
    bp_str = f"{int(base_price):,}円" if pd.notna(base_price) and base_price > 0 else '不明'
    fc = row.get('fc_members', np.nan)
    fc_str = f"{int(fc):,}人" if pd.notna(fc) and fc > 0 else '不明'
    cap = row.get('capacity', np.nan)
    cap_str = f"{int(cap):,}人" if pd.notna(cap) and cap > 0 else '不明'
    days = row.get('days_until_event', np.nan)
    days_str = f"{int(days)}日" if pd.notna(days) and days > -30 else '不明'
    tour_info = []
    if row.get('is_tour_first_day', 0) == 1:
        tour_info.append('ツアー初日')
    if row.get('is_tour_last_day', 0) == 1:
        tour_info.append('ツアー千秋楽')
    total_stages = row.get('total_stages', np.nan)
    if pd.notna(total_stages) and total_stages > 0:
        tour_info.append(f'全{int(total_stages)}公演')
    tour_str = '、'.join(tour_info) if tour_info else 'なし'
    seat_rule = row.get('seat_rule', np.nan)
    seat_str = str(int(seat_rule)) if pd.notna(seat_rule) else '不明'
    name_type = row.get('name_type', '')
    name_str = str(name_type) if name_type and str(name_type) != 'nan' else '不明'
    desc = row.get('raw_description', '')
    desc_str = str(desc) if desc and str(desc) != 'nan' else ''

    text = (
        f"[チケット情報]\n"
        f"アーティスト: {row.get('group_slug', '')}\n"
        f"会場: {row.get('venue', '')} (キャパ: {cap_str})\n"
        f"定価: {bp_str}\n"
        f"チケット種別: {ttype_str}\n"
        f"タグ: {tags_str}\n"
        f"枚数: {row.get('quantity', 1)}枚\n"
        f"名義タイプ: {name_str}\n"
        f"公演まで: {days_str}\n"
        f"ツアー: {tour_str}\n"
        f"FC会員数: {fc_str}\n"
        f"座席ルール: {seat_str}\n"
        f"説明: {desc_str}"
    )
    return text


def prepare_hf_dataset():
    print("データセットを準備中...")
    df = clean_data(load_raw_data())
    masters = load_manual_data()
    df = merge_manual_data(df, masters)
    if "perf_date" in df.columns and "first_observed_at" in df.columns:
        df["days_until_event"] = (df["perf_date"] - df["first_observed_at"]).dt.days
        df.loc[df["days_until_event"] < -30, "days_until_event"] = np.nan

    df["text"] = df.apply(format_prompt, axis=1)
    df["labels"] = np.log1p(df[TARGET_COLUMN])
    df = df[["text", "labels"]].dropna()

    print(f"データ件数: {len(df)}")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding=False,  # DataCollatorが動的にパディング（速度・VRAM最適化）
        truncation=True,
        max_length=LLM_MAX_LENGTH
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean())
    pred_price = np.expm1(predictions)
    true_price = np.expm1(labels)
    mae = np.abs(pred_price - true_price).mean()
    ss_res = ((labels - predictions) ** 2).sum()
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return {"rmse_log": rmse, "mae_price": mae, "r2": r2}


def main():
    if not torch.cuda.is_available():
        print("エラー: CUDA (GPU) が利用できません。")
        return

    # --- 速度最適化: TF32を有効化 (Ampere以降のGPUで演算を高速化) ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[{LLM_MODEL_ID}] の4bitモデルをロード中...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset = prepare_hf_dataset()

    print("テキストをトークナイズ中...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True,
        num_proc=4  # CPU並列でトークナイズ高速化
    ).remove_columns(["text"])
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True,
        num_proc=4
    ).remove_columns(["text"])

    # 4bit量子化設定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        LLM_MODEL_ID,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        problem_type="regression"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    # LoRA設定（精査済み）
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LLM_LORA_R,
        lora_alpha=LLM_LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=LLM_LORA_TARGETS,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LLM_LEARNING_RATE,
        per_device_train_batch_size=LLM_BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=LLM_BATCH_SIZE_TRAIN * 2,  # 推論時はバッチ大きめ
        num_train_epochs=LLM_EPOCHS,
        weight_decay=0.05,
        warmup_ratio=0.05,       # 8エポックに合わせて短縮
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae_price",
        greater_is_better=False,
        fp16=True,
        logging_steps=10,
        remove_unused_columns=False,
        # --- 速度最適化 ---
        dataloader_num_workers=4,   # CPUワーカーで次バッチを先読み
        dataloader_pin_memory=True, # GPU転送を高速化
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("学習を開始します...")
    trainer.train()

    print(f"学習完了！モデルを {OUTPUT_DIR} に保存します。")
    trainer.save_model(OUTPUT_DIR)

if __name__ == '__main__':
    main()
