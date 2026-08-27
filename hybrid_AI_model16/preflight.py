"""Read-only Model16 preflight; never starts model fitting."""
import importlib.metadata
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

from config import (
    ARTIFACT_DIR,
    BERT_PCA_DIMS,
    FINAL_CATBOOST_TRIALS,
    FINAL_LGBM_TRIALS,
    N_INNER_FOLDS,
    N_OUTER_FOLDS,
    OUTER_CATBOOST_TRIALS,
    OUTER_LGBM_TRIALS,
    SOURCE_ARTIFACT_DIR,
    TARGET,
)
from data_loader import catboost_frame, latest_data_dir, model_feature_columns, prepare_dataset
from modeling import make_tabular

_SNAPSHOT_AUDIT_DIR = Path(__file__).resolve().parents[1] / "decision_support_models" / "pipeline_runner"
sys.path.insert(0, str(_SNAPSHOT_AUDIT_DIR))
from snapshot_audit import validate_snapshot


def main():
    validate_snapshot(latest_data_dir())
    required = [
        "folds.csv", "bert_raw.npy", "bert_rows.json", "bert_text_hashes.json",
        "semantic_features.json",
    ]
    missing = [str(SOURCE_ARTIFACT_DIR / name) for name in required if not (SOURCE_ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Model16 source artifacts are missing: " + ", ".join(missing))
    df = prepare_dataset()
    numeric, categorical = model_feature_columns(df)
    folds = pd.read_csv(SOURCE_ARTIFACT_DIR / "folds.csv")
    rows = json.loads((SOURCE_ARTIFACT_DIR / "bert_rows.json").read_text(encoding="utf-8"))
    text_hashes = json.loads(
        (SOURCE_ARTIFACT_DIR / "bert_text_hashes.json").read_text(encoding="utf-8")
    )
    bert = np.load(SOURCE_ARTIFACT_DIR / "bert_raw.npy", mmap_mode="r")
    expected_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in df.model_text.fillna("").astype(str)
    ]
    if (
        folds.ticket_id.tolist() != df.ticket_id.tolist()
        or folds.duplicate_group.tolist() != df.duplicate_group.tolist()
        or rows != df.ticket_id.tolist()
        or text_hashes != expected_hashes
    ):
        raise ValueError("Model15 folds/BERT rows do not align with current Model16 data")
    if bert.shape != (len(df), 768) or not np.isfinite(bert).all():
        raise ValueError(f"Invalid BERT cache shape/content: {bert.shape}")
    if sorted(folds.fold.unique()) != list(range(N_OUTER_FOLDS)):
        raise ValueError("Outer fold manifest does not contain exactly five folds")
    if (
        pd.DataFrame({"group": folds.duplicate_group, "fold": folds.fold})
        .groupby("group").fold.nunique().max()
        > 1
    ):
        raise ValueError("Duplicate descriptions cross outer folds")
    target = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Target prices must be finite and positive")
    numeric_values = df[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isinf(numeric_values).any():
        raise ValueError("Numeric features contain infinity")
    if len(numeric + categorical) != len(set(numeric + categorical)):
        raise ValueError("Duplicate model feature names found")
    preprocessor = make_tabular(numeric, categorical)
    matrix = preprocessor.fit_transform(df)
    matrix_values = matrix.data if hasattr(matrix, "nnz") else np.asarray(matrix)
    if not np.isfinite(matrix_values).all():
        raise ValueError("LightGBM input matrix contains NaN or infinity")
    cat_frame = catboost_frame(df, numeric, categorical)
    cat_pool = Pool(cat_frame, cat_features=categorical)
    try:
        import psutil
        ram = psutil.virtual_memory().available / 2**30
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except ImportError:
        ram, physical, logical = float("nan"), 0, os.cpu_count() or 0
    disk = shutil.disk_usage(ARTIFACT_DIR.parent).free / 2**30
    if ram == ram and ram < 16:
        raise RuntimeError(f"Only {ram:.1f}GiB RAM is free; 16GiB is required")
    if disk < 2:
        raise RuntimeError(f"Only {disk:.1f}GiB disk is free; 2GiB is required")
    packages = {
        name: importlib.metadata.version(name)
        for name in ["lightgbm", "catboost", "optuna", "scikit-learn", "scipy"]
    }
    outer_tree_fits = N_OUTER_FOLDS * N_INNER_FOLDS * (
        2 * OUTER_LGBM_TRIALS + OUTER_CATBOOST_TRIALS
    )
    final_tree_fits = N_OUTER_FOLDS * (
        2 * FINAL_LGBM_TRIALS + FINAL_CATBOOST_TRIALS
    )
    best_trial_refits = (
        N_OUTER_FOLDS * N_INNER_FOLDS * 3
        + N_OUTER_FOLDS * 3
        + N_OUTER_FOLDS * 3
        + 3
    )
    print("Model16 preflight OK (training has not started)")
    print(
        f"rows={len(df):,}, numeric={len(numeric)}, categorical={len(categorical)}, "
        f"LightGBM_after_OHE={matrix.shape[1]}, CatBoost_columns={cat_pool.num_col()}"
    )
    print(f"CPU={physical} physical/{logical} logical, free_RAM={ram:.1f}GiB, free_disk={disk:.1f}GiB")
    print("GPU=not used; LightGBM/CatBoost/Ridge run on CPU and Qwen is excluded")
    print(f"nested CV={N_OUTER_FOLDS} outer x {N_INNER_FOLDS} inner, estimated tree fits={outer_tree_fits + final_tree_fits + best_trial_refits:,}")
    print(f"BERT PCA dimensions={BERT_PCA_DIMS}; one global weight vector, price-band routing=False")
    print("versions=" + ", ".join(f"{key}={value}" for key, value in packages.items()))


if __name__ == "__main__":
    main()
