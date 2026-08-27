"""Reuse target-free Model 14 fold/BERT artifacts after strict row validation."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from config import ARTIFACT_DIR, ROOT
from data_loader import prepare_dataset


def link_or_copy(source: Path, target: Path):
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    source = ROOT / "hybrid_AI_model14" / "artifacts"
    df = prepare_dataset()
    source_folds = pd.read_csv(source / "folds.csv")
    source_rows = json.loads((source / "bert_rows.json").read_text(encoding="utf-8"))
    bert = np.load(source / "bert_raw.npy", mmap_mode="r")
    ids = df.ticket_id.tolist()
    if source_folds.ticket_id.tolist() != ids or source_rows != ids or bert.shape != (len(df), 768):
        raise ValueError("Model 14 artifacts do not match current sold rows")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["folds.csv", "bert_rows.json", "bert_raw.npy"]:
        link_or_copy(source / name, ARTIFACT_DIR / name)
    print(f"検証済みfold/BERT成果物を再利用しました: {len(df)} rows")


if __name__ == "__main__":
    main()

