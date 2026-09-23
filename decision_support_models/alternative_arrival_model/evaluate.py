"""Recompute OOF probability metrics and calibration plots."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACT_DIR
from modeling import metrics


def _evaluation_horizons(oof: pd.DataFrame) -> tuple[int, ...]:
    """Return only horizons that actually produced chronological OOF rows."""
    horizons = tuple(sorted(pd.to_numeric(
        oof.get("horizon_days", pd.Series(dtype=float)), errors="coerce"
    ).dropna().astype(int).unique().tolist()))
    if not horizons:
        raise ValueError("Alternative OOF contains no evaluable horizon rows")
    report_path = ARTIFACT_DIR / "training_report.json"
    if report_path.exists():
        training_report = json.loads(report_path.read_text(encoding="utf-8"))
        declared = tuple(sorted(
            int(value) for value in training_report.get("supported_horizons", horizons)
        ))
        if declared != horizons:
            raise ValueError(
                f"Alternative OOF horizons {horizons} do not match trained "
                f"horizons {declared}"
            )
    return horizons


def evaluate():
    oof = pd.read_csv(ARTIFACT_DIR / "oof_predictions.csv", parse_dates=["landmark_at"])
    horizons = _evaluation_horizons(oof)
    report = {}
    figure, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    for axis, horizon in zip(axes[0], horizons):
        part = oof[oof.horizon_days.eq(horizon)]
        report[str(horizon)] = metrics(part.true_alternative.to_numpy(int), part.p_alternative.to_numpy(float))
        predicted, observed = [], []
        for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = part.p_alternative.between(lower, upper, inclusive="left")
            if mask.any():
                predicted.append(part.loc[mask, "p_alternative"].mean())
                observed.append(part.loc[mask, "true_alternative"].mean())
        axis.plot(predicted, observed, marker="o")
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set(title=f"{horizon} day", xlabel="predicted", ylabel="observed", xlim=(0, 1), ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "calibration.png", dpi=160)
    plt.close(figure)
    (ARTIFACT_DIR / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
