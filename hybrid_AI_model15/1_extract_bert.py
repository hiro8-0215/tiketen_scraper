"""Reuse cached BERT rows and encode only tickets added by a newer snapshot."""
from __future__ import annotations

import gc
import hashlib
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import (
    ARTIFACT_DIR,
    BERT_BATCH_SIZE,
    BERT_MAX_LENGTH,
    BERT_MODEL,
    QWEN_GPU_INDEX,
    QWEN_GPU_MEMORY_FRACTION,
)
from data_loader import prepare_dataset


def save_embeddings(embeddings, target):
    temporary = ARTIFACT_DIR / "bert_raw.model15.complete.tmp.npy"
    np.save(temporary, embeddings)
    os.replace(temporary, target)


def main():
    df = prepare_dataset()
    embedding_path = ARTIFACT_DIR / "bert_raw.npy"
    rows_path = ARTIFACT_DIR / "bert_rows.json"
    hashes_path = ARTIFACT_DIR / "bert_text_hashes.json"
    if not embedding_path.exists() or not rows_path.exists() or not hashes_path.exists():
        raise FileNotFoundError("Run bootstrap_artifacts.py before BERT extraction")

    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    stored_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    expected_rows = df.ticket_id.tolist()
    expected_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in df["model_text"].fillna("").astype(str)
    ]
    embeddings = np.load(embedding_path)
    if (
        rows != expected_rows
        or stored_hashes != expected_hashes
        or embeddings.shape != (len(df), 768)
    ):
        raise ValueError("BERT cache manifest does not match the latest clean dataset; run bootstrap first")

    missing_indices = np.flatnonzero(~np.isfinite(embeddings).all(axis=1))
    if not len(missing_indices):
        print(f"BERT cache complete: {embeddings.shape}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("New BERT rows require an NVIDIA CUDA GPU")

    torch.cuda.set_device(QWEN_GPU_INDEX)
    torch.cuda.set_per_process_memory_fraction(QWEN_GPU_MEMORY_FRACTION, QWEN_GPU_INDEX)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(
        f"CUDA device {QWEN_GPU_INDEX}: {torch.cuda.get_device_name(QWEN_GPU_INDEX)}; "
        f"encoding only {len(missing_indices):,}/{len(df):,} new clean rows"
    )
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL, use_safetensors=True).to("cuda").eval()

    processed = 0
    batch_size = BERT_BATCH_SIZE
    completed_batches = 0
    try:
        while processed < len(missing_indices):
            batch_indices = missing_indices[processed:processed + batch_size]
            texts = df.iloc[batch_indices]["model_text"].tolist()
            try:
                batch = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=BERT_MAX_LENGTH,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    hidden = model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            except torch.cuda.OutOfMemoryError:
                if "batch" in locals():
                    del batch
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"BERT CUDA OOM: batch-sizeを{batch_size}へ自動調整")
                continue
            embeddings[batch_indices] = pooled.float().cpu().numpy()
            processed += len(batch_indices)
            completed_batches += 1
            del batch, hidden, pooled
            print(f"BERT new rows: {processed:,}/{len(missing_indices):,}")
            if completed_batches % 10 == 0:
                save_embeddings(embeddings, embedding_path)
    finally:
        save_embeddings(embeddings, embedding_path)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"BERT cache saved: {embeddings.shape}")


if __name__ == "__main__":
    main()
