"""Fast preflight checks only; does not load model weights or start training."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil

import torch

from config import (
    ARTIFACT_DIR,
    LGBM_OPTUNA_TRIALS,
    N_FOLDS,
    QWEN_BATCH_SIZE,
    QWEN_EPOCHS,
    QWEN_GPU_INDEX,
    QWEN_GPU_MEMORY_FRACTION,
    QWEN_GRAD_ACCUM,
    QWEN_MODEL,
    ROOT,
)
from data_loader import clean_model13_population, latest_data_dir, load_snapshot

REQUIRED_COLUMNS = {
    "ticket_id", "event_id", "status", "price", "raw_description",
    "first_observed_at", "sold_at", "ticket_tags",
}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--repair", action="store_true",
        help="既存Qwen adapterの順序修復だけを行う軽量経路を確認",
    )
    mode.add_argument(
        "--meta-only", action="store_true",
        help="検証済みQwen/BERTを使うLightGBM再選択だけを確認（GPU不要）",
    )
    args = parser.parse_args()
    data_dir = latest_data_dir()
    snapshot = load_snapshot(data_dir)
    missing_columns = sorted(REQUIRED_COLUMNS - set(snapshot.columns))
    if missing_columns:
        raise ValueError(f"Latest snapshot lacks required columns: {missing_columns}")
    clean = clean_model13_population(snapshot, verbose=False)
    duplicate_groups = (
        clean["raw_description"].astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    )
    if len(clean) < N_FOLDS or duplicate_groups.nunique() < N_FOLDS:
        raise ValueError("Not enough clean rows/groups for five folds")
    if clean["ticket_id"].duplicated().any():
        raise ValueError("ticket_id remains duplicated after cleansing")

    legacy_semantics = ROOT / "hybrid_AI_model13" / "train_data" / "llm_extracted_features.json"
    if not legacy_semantics.exists():
        raise FileNotFoundError(f"Legacy semantic seed is missing: {legacy_semantics}")
    properties = None
    dedicated_gib = float("nan")
    free_gpu_gib = float("nan")
    if not args.meta_only:
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA GPU is unavailable")
        if QWEN_GPU_INDEX >= torch.cuda.device_count():
            raise RuntimeError(
                f"Configured CUDA index {QWEN_GPU_INDEX} is outside {torch.cuda.device_count()} devices"
            )
        properties = torch.cuda.get_device_properties(QWEN_GPU_INDEX)
        dedicated_gib = properties.total_memory / 2**30
        if dedicated_gib < 10:
            raise RuntimeError(f"Qwen 7B QLoRA requires about 10GiB dedicated VRAM; found {dedicated_gib:.1f}")
        free_gpu_bytes, _ = torch.cuda.mem_get_info(QWEN_GPU_INDEX)
        free_gpu_gib = free_gpu_bytes / 2**30
        if free_gpu_gib < 8:
            raise RuntimeError(
                f"Only {free_gpu_gib:.1f}GiB dedicated VRAM is currently free; close other GPU jobs first"
            )

    try:
        import psutil
        available_ram_gib = psutil.virtual_memory().available / 2**30
        physical_cpus = psutil.cpu_count(logical=False) or 0
        logical_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 0
    except ImportError:
        available_ram_gib = float("nan")
        physical_cpus = 0
        logical_cpus = os.cpu_count() or 0
    if available_ram_gib == available_ram_gib and available_ram_gib < 16:
        raise RuntimeError(f"Only {available_ram_gib:.1f}GiB system RAM is free")
    free_disk_gib = shutil.disk_usage(ARTIFACT_DIR.parent).free / 2**30
    huggingface_home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    qwen_cache = huggingface_home / "hub" / f"models--{QWEN_MODEL.replace('/', '--')}"
    minimum_free_gib = 1 if args.meta_only else 2 if args.repair else 5 if qwen_cache.exists() else 20
    if free_disk_gib < minimum_free_gib:
        raise RuntimeError(
            f"Free disk is {free_disk_gib:.1f}GiB; at least {minimum_free_gib}GiB is required "
            f"(Qwen cache present={qwen_cache.exists()})"
        )
    if args.repair:
        missing_adapters = []
        for fold in range(N_FOLDS):
            adapter = ARTIFACT_DIR / "qwen_folds" / f"fold_{fold}" / "best_adapter"
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                if not (adapter / name).exists():
                    missing_adapters.append(str(adapter / name))
        if missing_adapters:
            raise FileNotFoundError(
                "Repair mode requires all saved best adapters; missing: "
                + ", ".join(missing_adapters)
            )
        if not (ARTIFACT_DIR / "qwen_oof.csv").exists():
            raise FileNotFoundError("Repair mode requires the legacy qwen_oof.csv manifest")
    if args.meta_only:
        required_artifacts = [
            "folds.csv", "qwen_oof.csv", "bert_raw.npy", "bert_rows.json",
            "bert_text_hashes.json",
        ]
        missing = [
            str(ARTIFACT_DIR / name)
            for name in required_artifacts
            if not (ARTIFACT_DIR / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Meta-only mode requires validated cached artifacts; missing: "
                + ", ".join(missing)
            )

    versions = {
        package: importlib.metadata.version(package)
        for package in ["torch", "transformers", "lightgbm", "optuna", "scikit-learn"]
    }
    selected_mode = "meta-only" if args.meta_only else "repair" if args.repair else "full"
    print(f"Model15 preflight OK (mode={selected_mode})")
    print(f"latest_data={data_dir.name}, snapshot_rows={len(snapshot):,}, clean_rows={len(clean):,}")
    print(f"events={clean['event_id'].nunique():,}, descriptions={clean['raw_description'].nunique():,}")
    if args.meta_only:
        print("GPU=not used (validated Qwen OOF and BERT cache are reused)")
    else:
        print(
            f"GPU={properties.name}, dedicated={dedicated_gib:.1f}GiB, "
            f"free_now={free_gpu_gib:.1f}GiB, limit={QWEN_GPU_MEMORY_FRACTION:.0%}"
        )
    print(
        f"CPU={physical_cpus} physical/{logical_cpus} logical, "
        f"available_RAM={available_ram_gib:.1f}GiB"
    )
    if args.meta_only:
        print("Meta selection: CPU LightGBM only; no Qwen training/inference and no BERT extraction")
    elif args.repair:
        print(f"Qwen repair: adapters={N_FOLDS}, ordered FP16 inference only; no Qwen training")
    else:
        print(
            f"Qwen: folds={N_FOLDS}, epochs={QWEN_EPOCHS}, physical_batch={QWEN_BATCH_SIZE}, "
            f"grad_accum={QWEN_GRAD_ACCUM}, effective_batch={QWEN_BATCH_SIZE * QWEN_GRAD_ACCUM}"
        )
    print(f"LightGBM Optuna trials={LGBM_OPTUNA_TRIALS}, free_disk={free_disk_gib:.1f}GiB")
    print("versions=" + ", ".join(f"{name}={version}" for name, version in versions.items()))


if __name__ == "__main__":
    main()
