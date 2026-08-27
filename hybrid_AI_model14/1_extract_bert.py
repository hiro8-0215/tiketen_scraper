"""Extract target-free raw BERT embeddings. PCA is intentionally deferred to each fold."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from config import ARTIFACT_DIR, BERT_BATCH_SIZE, BERT_MAX_LENGTH, BERT_MODEL
from data_loader import prepare_dataset


def main():
    df = prepare_dataset()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    embedding_path = ARTIFACT_DIR / "bert_raw.npy"
    rows_path = ARTIFACT_DIR / "bert_rows.json"
    if embedding_path.exists() and rows_path.exists():
        existing = np.load(embedding_path, mmap_mode="r")
        existing_rows = json.loads(rows_path.read_text(encoding="utf-8"))
        if existing.shape == (len(df), 768) and existing_rows == df["ticket_id"].tolist():
            print(f"既存の整合済みBERT埋め込みを使用します: {existing.shape}")
            return
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    texts = df["model_text"].tolist()
    outputs = []
    for start in range(0, len(texts), BERT_BATCH_SIZE):
        batch = tokenizer(texts[start:start+BERT_BATCH_SIZE], padding=True, truncation=True,
                          max_length=BERT_MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        outputs.append(pooled.float().cpu().numpy())
    np.save(embedding_path, np.vstack(outputs))
    rows_path.write_text(json.dumps(df["ticket_id"].tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()
