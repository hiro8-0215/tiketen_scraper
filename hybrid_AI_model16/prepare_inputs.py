"""Prepare only the reusable inputs required by Model16.

This entry point does not train Model15 or Model16.  Model16 historically
stores its target-free semantic and BERT caches in the Model15 artifact
directory, so the implementation scripts are reused while presenting one
Model16-owned workflow to the user.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MODEL16_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL16_DIR.parent
MODEL15_DIR = PROJECT_ROOT / "hybrid_AI_model15"
SOURCE_ARTIFACT_DIR = MODEL15_DIR / "artifacts"


def _run_model15(script: str, *args: str) -> None:
    command = [sys.executable, str(MODEL15_DIR / script), *args]
    print("\n[Model16入力準備]", " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=MODEL15_DIR, check=True)
    print(
        f"[Model16入力準備] 完了: {script} "
        f"({(time.time() - started) / 60:.1f}分)",
        flush=True,
    )


def input_status() -> dict:
    # Import Model16's loader, which defines the exact current population.
    sys.path.insert(0, str(MODEL16_DIR))
    from data_loader import prepare_dataset

    frame = prepare_dataset()
    ids = frame.ticket_id.astype(str).tolist()
    descriptions = set(frame.raw_description.fillna("").astype(str))
    expected_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in frame.model_text.fillna("").astype(str)
    ]

    semantic_path = SOURCE_ARTIFACT_DIR / "semantic_features.json"
    semantics = (
        json.loads(semantic_path.read_text(encoding="utf-8"))
        if semantic_path.exists() else {}
    )
    accepted_semantic_sources = {"qwen15", "qwen15_parse_fallback"}
    semantic_remaining = sum(
        description not in semantics
        or semantics[description].get("semantic_source")
        not in accepted_semantic_sources
        or semantics[description].get("semantic_schema_version")
        != "model15_semantic_v1"
        for description in descriptions
    )

    rows_path = SOURCE_ARTIFACT_DIR / "bert_rows.json"
    hashes_path = SOURCE_ARTIFACT_DIR / "bert_text_hashes.json"
    bert_path = SOURCE_ARTIFACT_DIR / "bert_raw.npy"
    bert_reusable = 0
    if rows_path.exists() and hashes_path.exists() and bert_path.exists():
        stored_rows = json.loads(rows_path.read_text(encoding="utf-8"))
        stored_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        bert = np.load(bert_path, mmap_mode="r")
        if (
            bert.ndim == 2 and bert.shape[1] == 768
            and len(stored_rows) == len(stored_hashes) == len(bert)
            and len(set(stored_rows)) == len(stored_rows)
        ):
            positions = {ticket_id: index for index, ticket_id in enumerate(stored_rows)}
            for index, ticket_id in enumerate(ids):
                position = positions.get(ticket_id)
                if (
                    position is not None
                    and stored_hashes[position] == expected_hashes[index]
                    and np.isfinite(bert[position]).all()
                ):
                    bert_reusable += 1
        del bert

    folds_path = SOURCE_ARTIFACT_DIR / "folds.csv"
    folds_ready = False
    if folds_path.exists():
        import pandas as pd

        folds = pd.read_csv(folds_path)
        folds_ready = (
            "ticket_id" in folds
            and folds.ticket_id.astype(str).tolist() == ids
            and "fold" in folds
            and sorted(folds.fold.unique().tolist()) == [0, 1, 2, 3, 4]
        )

    try:
        import torch

        cuda_ready = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_ready else None
        if cuda_ready:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            vram_free_gib = round(free_bytes / 2**30, 2)
            vram_total_gib = round(total_bytes / 2**30, 2)
        else:
            vram_free_gib = vram_total_gib = 0.0
    except ImportError:
        cuda_ready = False
        cuda_device = None
        vram_free_gib = vram_total_gib = 0.0

    report = {
        "model": "Model16 input preparation",
        "training_executed": False,
        "rows": len(frame),
        "unique_descriptions": len(descriptions),
        "semantic_remaining": int(semantic_remaining),
        "semantic_parse_fallback": int(sum(
            description in semantics
            and semantics[description].get("semantic_source")
            == "qwen15_parse_fallback"
            for description in descriptions
        )),
        "bert_reusable": int(bert_reusable),
        "bert_remaining": int(len(frame) - bert_reusable),
        "folds_ready": bool(folds_ready),
        "inputs_ready": bool(
            semantic_remaining == 0
            and bert_reusable == len(frame)
            and folds_ready
        ),
        "cuda_ready": cuda_ready,
        "cuda_device": cuda_device,
        "vram_free_gib": vram_free_gib,
        "vram_total_gib": vram_total_gib,
        "free_disk_gib": round(shutil.disk_usage(MODEL16_DIR).free / 2**30, 2),
    }
    report["next"] = (
        "Model16事前チェックを実行できます。"
        if report["inputs_ready"]
        else "[10 価格][準備] Model16 意味・BERT・fold差分更新（学習なし）を実行してください。"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only", action="store_true",
        help="差分件数と実行環境だけを確認し、生成や学習を行わない",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    before = input_status()
    print(json.dumps(before, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return
    if before["inputs_ready"]:
        subprocess.run(
            [sys.executable, str(MODEL16_DIR / "preflight.py")],
            cwd=MODEL16_DIR,
            check=True,
        )
        print("Model16入力は最新です。再生成せず、厳密事前チェックだけ完了しました。")
        return
    if not before["cuda_ready"] and (
        before["semantic_remaining"] or before["bert_remaining"]
    ):
        raise RuntimeError("Model16の差分意味/BERT生成にはNVIDIA CUDA GPUが必要です")
    if before["free_disk_gib"] < 5:
        raise RuntimeError(
            f"空き容量は{before['free_disk_gib']:.1f}GiBです。最低5GiB必要です"
        )

    # No Model15 fitting is performed.  These scripts only preserve verified
    # cache rows, extract missing target-free features, and assign folds.
    _run_model15("bootstrap_artifacts.py", "--historical")
    _run_model15(
        "1_extract_semantic_json.py",
        "--historical",
        "--refresh-legacy",
        "--batch-size", str(args.batch_size),
    )
    _run_model15("1_extract_bert.py", "--historical")
    _run_model15("make_folds.py", "--historical")

    after = input_status()
    print(json.dumps(after, ensure_ascii=False, indent=2), flush=True)
    if not after["inputs_ready"]:
        raise RuntimeError("Model16入力準備が完了していません。上の差分件数を確認してください")

    subprocess.run(
        [sys.executable, str(MODEL16_DIR / "preflight.py")],
        cwd=MODEL16_DIR,
        check=True,
    )
    print("Model16入力準備と厳密事前チェックが完了しました。学習は未実行です。")


if __name__ == "__main__":
    main()
