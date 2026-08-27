"""Integrity checks for ordered, genuinely out-of-fold Qwen predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import QWEN_OOF_SCHEMA_VERSION, TARGET
from qwen_prompt import qwen_dataset_fingerprint


MAX_FOLD_LOG_MSE = 0.75
MAX_OVERALL_MAE_YEN = 15_000.0


def qwen_oof_diagnostics(
    df: pd.DataFrame, manifest: pd.DataFrame, qwen: pd.DataFrame
) -> dict:
    """Measure alignment and quality without using diagnostics as model inputs."""
    expected_fingerprint = qwen_dataset_fingerprint(df, manifest)
    required = {
        "ticket_id", "fold", "qwen_dataset_fingerprint",
        "qwen_oof_schema_version", "qwen_pred_log",
    }
    missing = sorted(required - set(qwen.columns))
    if missing:
        raise ValueError(
            "Qwen OOF predates the ordered-output repair; missing columns: "
            f"{missing}. Run 2_repair_qwen_oof.py."
        )
    if qwen["ticket_id"].tolist() != df["ticket_id"].tolist():
        raise ValueError("Qwen OOF ticket order does not match the Model15 dataset")
    if qwen["fold"].tolist() != manifest["fold"].tolist():
        raise ValueError("Qwen OOF fold order does not match folds.csv")
    if not qwen["qwen_dataset_fingerprint"].eq(expected_fingerprint).all():
        raise ValueError("Qwen OOF fingerprint is stale for the current inputs/labels/folds")
    if not qwen["qwen_oof_schema_version"].eq(QWEN_OOF_SCHEMA_VERSION).all():
        raise ValueError("Qwen OOF ordering schema is stale; run ordered inference")

    pred_log = pd.to_numeric(qwen["qwen_pred_log"], errors="coerce").to_numpy(float)
    if not np.isfinite(pred_log).all():
        raise ValueError("Qwen OOF contains missing or non-finite predictions")
    true_log = np.log1p(df[TARGET].to_numpy(float))
    pred_yen = np.maximum(0.0, np.expm1(pred_log))
    true_yen = df[TARGET].to_numpy(float)

    fold_metrics = {}
    for fold in sorted(manifest["fold"].unique()):
        mask = manifest["fold"].eq(fold).to_numpy()
        log_mse = float(np.mean((pred_log[mask] - true_log[mask]) ** 2))
        mae_yen = float(np.mean(np.abs(pred_yen[mask] - true_yen[mask])))
        fold_metrics[int(fold)] = {
            "rows": int(mask.sum()),
            "log_mse": log_mse,
            "mae_yen": mae_yen,
        }

    diagnostics = {
        "overall_mae_yen": float(np.mean(np.abs(pred_yen - true_yen))),
        "unique_predictions": int(pd.Series(pred_log).nunique()),
        "folds": fold_metrics,
    }
    bad_folds = {
        fold: values["log_mse"]
        for fold, values in fold_metrics.items()
        if values["log_mse"] > MAX_FOLD_LOG_MSE
    }
    if bad_folds or diagnostics["overall_mae_yen"] > MAX_OVERALL_MAE_YEN:
        raise ValueError(
            "Qwen OOF failed the alignment guard. This usually means predictions "
            "were saved in length-grouped rather than ticket order. "
            f"fold_log_mse={bad_folds}, overall_mae_yen="
            f"{diagnostics['overall_mae_yen']:.1f}"
        )
    return diagnostics


def attach_qwen_diagnostics(
    result: pd.DataFrame, df: pd.DataFrame, manifest: pd.DataFrame
) -> dict:
    """Attach fold-level audit values after predictions are in original row order."""
    pred_log = result["qwen_pred_log"].to_numpy(float)
    true_log = np.log1p(df[TARGET].to_numpy(float))
    pred_yen = np.maximum(0.0, np.expm1(pred_log))
    true_yen = df[TARGET].to_numpy(float)
    result["qwen_fold_log_mse"] = np.nan
    result["qwen_fold_mae_yen"] = np.nan
    for fold in sorted(manifest["fold"].unique()):
        mask = manifest["fold"].eq(fold).to_numpy()
        result.loc[mask, "qwen_fold_log_mse"] = float(
            np.mean((pred_log[mask] - true_log[mask]) ** 2)
        )
        result.loc[mask, "qwen_fold_mae_yen"] = float(
            np.mean(np.abs(pred_yen[mask] - true_yen[mask]))
        )
    return qwen_oof_diagnostics(df, manifest, result)
