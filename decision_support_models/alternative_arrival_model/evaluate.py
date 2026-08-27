"""Recompute OOF probability metrics and calibration plots."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, HORIZONS_DAYS
from modeling import metrics


def evaluate():
    oof = pd.read_csv(ARTIFACT_DIR / "oof_predictions.csv", parse_dates=["landmark_at"])
    report = {}
    figure, axes = plt.subplots(1, len(HORIZONS_DAYS), figsize=(5 * len(HORIZONS_DAYS), 4), squeeze=False)
    for axis, horizon in zip(axes[0], HORIZONS_DAYS):
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
