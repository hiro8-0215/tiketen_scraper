import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    BitsAndBytesConfig
)
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BERT_BATCH_SIZE, BERT_PCA_DIM, DEVICE,
    LLM_MODEL_ID, LLM_MAX_LENGTH, LLM_BATCH_SIZE_INFER,
)
from data_loader import load_raw_data, clean_data, load_manual_data, merge_manual_data


def format_prompt(row):
    """1_train_llm.py と完全に同じプロンプト"""
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


PEFT_MODEL_ID = "./qwen_ticket_model"
OUTPUT_PREDS_FILE = "./llm_predictions.csv"
OUTPUT_BERT_FILE = "./bert_features.npy"


def extract_llm_predictions(df, tokenizer, model):
    """ファインチューニング済みQwenから予測値を抽出"""
    all_preds = []
    print("[LLM] 予測値を抽出中...")
    texts = df.apply(format_prompt, axis=1).tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), LLM_BATCH_SIZE_INFER)):
            batch_texts = texts[i:i+LLM_BATCH_SIZE_INFER]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=LLM_MAX_LENGTH,
                return_tensors="pt"
            ).to(model.device)

            outputs = model(**inputs)
            preds = outputs.logits.squeeze(-1).float().cpu().numpy()
            if preds.ndim == 0:
                preds = np.expand_dims(preds, axis=0)
            all_preds.extend(preds)

    return np.array(all_preds)


def extract_bert_embeddings(df):
    """BERT [CLS]ベクトル抽出 → PCA圧縮"""
    print(f"\n[BERT] {BERT_MODEL_NAME} の埋め込みを抽出中...")

    texts = df.apply(
        lambda row: str(row.get("raw_description", "")) + " " + str(row.get("ticket_tags", "")),
        axis=1
    ).tolist()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME, use_safetensors=True).to(DEVICE)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BERT_BATCH_SIZE), desc="  BERT embedding"):
        batch_texts = texts[i:i+BERT_BATCH_SIZE]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=BERT_MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(cls_embeddings)

    raw_embeddings = np.vstack(all_embeddings)
    print(f"  [BERT] 生の埋め込み: {raw_embeddings.shape}")

    # GPU メモリ解放
    del model, tokenizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # PCA圧縮 (768d → BERT_PCA_DIM)
    print(f"  [PCA] {raw_embeddings.shape[1]}d -> {BERT_PCA_DIM}d に圧縮中...")
    pca = PCA(n_components=BERT_PCA_DIM, random_state=42)
    compressed = pca.fit_transform(raw_embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  [PCA] 完了 — 累積寄与率: {explained:.1%} (出力: {compressed.shape})")

    return compressed


def main():
    if not torch.cuda.is_available():
        print("エラー: CUDA (GPU) が利用できません。")
        return

    # TF32有効化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("データセットを準備中...")
    df = clean_data(load_raw_data())
    masters = load_manual_data()
    df = merge_manual_data(df, masters)

    if "perf_date" in df.columns and "first_observed_at" in df.columns:
        df["days_until_event"] = (df["perf_date"] - df["first_observed_at"]).dt.days
        df.loc[df["days_until_event"] < -30, "days_until_event"] = np.nan

    # ==========================================
    # Part 1: Fine-tuned Qwen → 予測値
    # ==========================================
    print(f"\n[{LLM_MODEL_ID}] とLoRAアダプタをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        LLM_MODEL_ID,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        problem_type="regression"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    model.eval()

    all_preds = extract_llm_predictions(df, tokenizer, model)

    pred_df = pd.DataFrame({
        "original_index": df.index,
        "llm_pred_log": all_preds,
        "llm_pred_price": np.expm1(all_preds)
    })
    pred_df.to_csv(OUTPUT_PREDS_FILE, index=False)
    print(f"LLMの予測値を保存: {OUTPUT_PREDS_FILE}")

    # GPUメモリ解放
    del model, base_model, tokenizer
    torch.cuda.empty_cache()

    # ==========================================
    # Part 2: Pre-trained BERT → [CLS] → PCA
    # ==========================================
    bert_embeddings = extract_bert_embeddings(df)
    np.save(OUTPUT_BERT_FILE, bert_embeddings)
    print(f"BERT埋め込みを保存: {OUTPUT_BERT_FILE} (Shape: {bert_embeddings.shape})")

if __name__ == '__main__':
    main()
