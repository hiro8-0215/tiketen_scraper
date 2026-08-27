"""Recompute OOF metrics and save calibration diagrams."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, HORIZONS_DAYS, STATUS_CLASSES
from modeling import probability_metrics


def evaluate() -> dict:
    oof = pd.read_csv(ARTIFACT_DIR / "oof_predictions.csv", parse_dates=["landmark_at"])
    report = {}
    figure, axes = plt.subplots(1, len(HORIZONS_DAYS), figsize=(5 * len(HORIZONS_DAYS), 4), squeeze=False)
    for axis, horizon in zip(axes[0], HORIZONS_DAYS):
        part = oof[oof["horizon_days"].eq(horizon)]
        probabilities = part[["p_active", "p_sold", "p_deleted"]].to_numpy(float)
        truth = part["true_state"].to_numpy(int)
        report[str(horizon)] = probability_metrics(truth, probabilities)
        for class_index, name in enumerate(STATUS_CLASSES):
            observed, predicted = [], []
            for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
                mask = (probabilities[:, class_index] >= lower) & (probabilities[:, class_index] < upper)
                if mask.any():
                    predicted.append(probabilities[mask, class_index].mean())
                    observed.append((truth[mask] == class_index).mean())
            axis.plot(predicted, observed, marker="o", label=name)
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set(title=f"{horizon} day", xlabel="predicted probability", ylabel="observed rate", xlim=(0, 1), ylim=(0, 1))
        axis.legend()
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "calibration.png", dpi=160)
    plt.close(figure)
    (ARTIFACT_DIR / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
