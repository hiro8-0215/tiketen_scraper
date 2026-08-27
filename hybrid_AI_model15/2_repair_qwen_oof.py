"""Repair Qwen OOF row order by re-inferring saved adapters; no training."""
from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.80",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from config import (
    ARTIFACT_DIR,
    N_FOLDS,
    QWEN_EVAL_BATCH_SIZE,
    QWEN_GPU_INDEX,
    QWEN_GPU_MEMORY_FRACTION,
    QWEN_MAX_LENGTH,
    QWEN_MODEL,
    QWEN_OOF_SCHEMA_VERSION,
)
from data_loader import prepare_dataset
from qwen_prompt import build_qwen_prompt, qwen_dataset_fingerprint
from qwen_validation import attach_qwen_diagnostics, qwen_oof_diagnostics


def configure_resources():
    if not torch.cuda.is_available():
        raise RuntimeError("Ordered Qwen repair requires an NVIDIA CUDA GPU")
    torch.cuda.set_device(QWEN_GPU_INDEX)
    torch.cuda.set_per_process_memory_fraction(QWEN_GPU_MEMORY_FRACTION, QWEN_GPU_INDEX)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"[repair] CUDA {QWEN_GPU_INDEX}: {torch.cuda.get_device_name(QWEN_GPU_INDEX)}")


def load_adapter(adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        QWEN_MODEL,
        num_labels=1,
        problem_type="regression",
        quantization_config=quantization,
        device_map={"": QWEN_GPU_INDEX},
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()
    non_cuda = [name for name, parameter in model.named_parameters() if parameter.device.type != "cuda"]
    if non_cuda:
        raise RuntimeError(f"CPU/disk offload detected during Qwen repair: {non_cuda[:5]}")
    return tokenizer, model


def predict_in_original_order(df, adapter_dir, initial_batch_size):
    """Iterate DataFrame rows sequentially; never use a length-grouped sampler."""
    tokenizer, model = load_adapter(adapter_dir)
    prompts = [build_qwen_prompt(row) for _, row in df.iterrows()]
    predictions = np.empty(len(prompts), dtype=np.float32)
    position = 0
    batch_size = initial_batch_size
    try:
        while position < len(prompts):
            batch_prompts = prompts[position:position + batch_size]
            try:
                encoded = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=QWEN_MAX_LENGTH,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**encoded).logits.reshape(-1)
                values = logits.float().cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if "encoded" in locals():
                    del encoded
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"[repair] CUDA OOM; inference batch -> {batch_size}")
                continue
            predictions[position:position + len(values)] = values
            position += len(values)
            del encoded, logits, values
            print(f"[repair] {adapter_dir.parent.name}: {position:,}/{len(prompts):,}")
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return predictions


def save_result(result, target):
    temporary = ARTIFACT_DIR / "qwen_oof.repair.tmp.csv"
    result.to_csv(temporary, index=False)
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=QWEN_EVAL_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    df = prepare_dataset()
    manifest = pd.read_csv(ARTIFACT_DIR / "folds.csv")
    if (
        not {"ticket_id", "duplicate_group", "fold"}.issubset(manifest.columns)
        or manifest["ticket_id"].tolist() != df["ticket_id"].tolist()
        or manifest["duplicate_group"].tolist() != df["duplicate_group"].tolist()
    ):
        raise ValueError("folds.csv does not match the clean Model15 dataset")

    fingerprint = qwen_dataset_fingerprint(df, manifest)
    prediction_path = ARTIFACT_DIR / "qwen_oof.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("No legacy qwen_oof.csv exists to prove adapter compatibility")
    previous = pd.read_csv(prediction_path)
    compatibility_columns = {"ticket_id", "fold", "qwen_dataset_fingerprint", "qwen_pred_log"}
    compatible = (
        compatibility_columns.issubset(previous.columns)
        and previous["ticket_id"].tolist() == df["ticket_id"].tolist()
        and previous["fold"].tolist() == manifest["fold"].tolist()
        and previous["qwen_dataset_fingerprint"].eq(fingerprint).all()
    )
    if not compatible:
        raise ValueError(
            "Saved adapters cannot be proven compatible with the current data/prompt/folds; "
            "run 2_train_qwen_oof.py instead of repair."
        )

    already_ordered = (
        "qwen_oof_schema_version" in previous
        and previous["qwen_oof_schema_version"].eq(QWEN_OOF_SCHEMA_VERSION).all()
    )
    if already_ordered and not args.force:
        try:
            diagnostics = qwen_oof_diagnostics(df, manifest, previous)
            print(f"Qwen OOF is already repaired: {diagnostics}")
            return
        except ValueError as error:
            print(f"[repair] existing ordered artifact failed validation: {error}")

    if already_ordered:
        result = previous.copy()
    else:
        result = manifest[["ticket_id", "fold"]].copy()
        result["qwen_dataset_fingerprint"] = fingerprint
        result["qwen_oof_schema_version"] = QWEN_OOF_SCHEMA_VERSION
        result["qwen_training_source"] = "recovered_existing_adapters_fp16_ordered_inference"
        result["qwen_pred_log"] = np.nan

    selected = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    configure_resources()
    for fold in selected:
        mask = manifest["fold"].eq(fold).to_numpy()
        if not args.force and result.loc[mask, "qwen_pred_log"].notna().all():
            print(f"fold {fold}: repaired predictions already exist; skipping")
            continue
        adapter_dir = ARTIFACT_DIR / "qwen_folds" / f"fold_{fold}" / "best_adapter"
        required_adapter_files = [adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"]
        if not all(path.exists() for path in required_adapter_files):
            raise FileNotFoundError(f"fold {fold} best_adapter is incomplete: {adapter_dir}")
        result.loc[mask, "qwen_pred_log"] = predict_in_original_order(
            df.loc[mask], adapter_dir, args.batch_size
        )
        save_result(result, prediction_path)

    if result["qwen_pred_log"].notna().all():
        diagnostics = attach_qwen_diagnostics(result, df, manifest)
        save_result(result, prediction_path)
        print(f"Qwen OOF repair complete: {diagnostics}")
    else:
        print(result.groupby("fold")["qwen_pred_log"].count())


if __name__ == "__main__":
    main()
