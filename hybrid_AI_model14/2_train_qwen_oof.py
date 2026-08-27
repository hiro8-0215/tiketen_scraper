"""Five-fold LoRA training that emits genuinely out-of-fold Qwen predictions."""
from __future__ import annotations
import argparse
import gc
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments,
)
from config import (
    ARTIFACT_DIR, N_FOLDS, QWEN_BATCH_SIZE, QWEN_EPOCHS, QWEN_GRAD_ACCUM,
    QWEN_LR, QWEN_MAX_LENGTH, QWEN_MODEL, SEED, TARGET,
)
from data_loader import prepare_dataset


def prompt(row):
    def value(name, default="不明"):
        v = row.get(name)
        return default if pd.isna(v) or str(v) == "" else str(v)
    return (
        "次の売却済みチケットの成立価格を推定してください。\n"
        f"アーティスト:{value('group_slug')}\n公演:{value('event_id')}\n"
        f"会場:{value('venue')}\n公演日:{value('perf_date')}\n定価:{value('base_price')}\n"
        f"券種:{value('ticket_type')}\n名義:{value('name_type')}\n枚数:{value('quantity')}\n"
        f"過去成立中央値:{value('event_prior_sold_median')}\n"
        f"出品時点までの出品数:{value('event_listings_seen_before')}\n"
        f"説明:{value('raw_description', '')}\nタグ:{value('ticket_tags', '')}"
    )


def dataset_for(df):
    return Dataset.from_dict({
        "text": [prompt(row) for _, row in df.iterrows()],
        "labels": np.log1p(df[TARGET].to_numpy(float)).astype(np.float32),
    })


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForSequenceClassification.from_pretrained(
        QWEN_MODEL, num_labels=1, problem_type="regression",
        quantization_config=quant, device_map="auto",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base = prepare_model_for_kbit_training(base)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    return tokenizer, model


def tokenize(ds, tokenizer):
    return ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=QWEN_MAX_LENGTH),
                  batched=True).remove_columns(["text"])


def train_one(train_df, val_df, output_dir):
    tokenizer, model = load_base()
    train_ds = tokenize(dataset_for(train_df), tokenizer)
    val_ds = tokenize(dataset_for(val_df), tokenizer)
    args = TrainingArguments(
        output_dir=str(output_dir), seed=SEED, learning_rate=QWEN_LR,
        per_device_train_batch_size=QWEN_BATCH_SIZE, per_device_eval_batch_size=QWEN_BATCH_SIZE * 2,
        gradient_accumulation_steps=QWEN_GRAD_ACCUM, num_train_epochs=QWEN_EPOCHS,
        warmup_ratio=0.05, weight_decay=0.05, lr_scheduler_type="cosine",
        gradient_checkpointing=True, fp16=True, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=20, report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    trainer.train()
    pred = trainer.predict(val_ds).predictions.squeeze().astype(float)
    trainer.save_model(str(output_dir / "best_adapter"))
    del trainer, model, tokenizer, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    return np.atleast_1d(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, help="Run only one fold; omit to run every fold")
    parser.add_argument("--force", action="store_true", help="既に完了したfoldも再学習する")
    args = parser.parse_args()
    df = prepare_dataset()
    manifest = pd.read_csv(ARTIFACT_DIR / "folds.csv")
    if not manifest.ticket_id.equals(df.ticket_id):
        raise ValueError("folds.csv does not match current dataset; rerun make_folds.py")
    pred_path = ARTIFACT_DIR / "qwen_oof.csv"
    if pred_path.exists():
        result = pd.read_csv(pred_path)
    else:
        result = manifest.copy()
        result["qwen_pred_log"] = np.nan
    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    for fold in folds:
        val_mask = manifest.fold.eq(fold).to_numpy()
        if not args.force and result.loc[val_mask, "qwen_pred_log"].notna().all():
            print(f"fold {fold}: 完了済みのためスキップ（再学習は --force）")
            continue
        fold_dir = ARTIFACT_DIR / "qwen_folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        result.loc[val_mask, "qwen_pred_log"] = train_one(df.loc[~val_mask], df.loc[val_mask], fold_dir)
        result.to_csv(pred_path, index=False)
    print(result.groupby("fold").qwen_pred_log.count())


if __name__ == "__main__":
    main()
