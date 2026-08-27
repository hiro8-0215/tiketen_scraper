"""Select policy coefficients on chronological training OOF and hold out the tail."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALTERNATIVE_OOF, ARTIFACT_DIR, DEMAND_OOF, GRID, PIPELINE_VERSION, PROFILES
from data_loader import load_oof
from decision import apply_policy, summarize


def _candidates():
    names = list(GRID)
    for values in itertools.product(*(GRID[name] for name in names)):
        yield dict(zip(names, values))


def _mean_regret(frame, policy, profile):
    discount = frame.discount_ratio.to_numpy(float)
    disappearance = frame.disappearance_probability.to_numpy(float)
    alternative = frame.p_alternative.to_numpy(float)
    score = (
        discount
        + policy["disappear_weight"] * disappearance
        - policy["alternative_weight"] * alternative
    )
    buy = score >= policy["buy_threshold"]
    wait = (~buy) & (alternative >= policy["wait_threshold"])
    savings = (
        pd.to_numeric(frame.potential_savings, errors="coerce")
        .fillna(0).clip(lower=0).to_numpy(float)
    )
    sold_miss = frame.true_state.eq(1).to_numpy(float) * profile["sold_miss_penalty"]
    deleted_miss = (
        frame.true_state.eq(2).to_numpy(float)
        * profile["sold_miss_penalty"]
        * profile["deleted_multiplier"]
    )
    wait_regret = (sold_miss + deleted_miss) * (
        1 - frame.true_alternative.clip(0, 1).to_numpy(float)
    )
    values = np.where(
        buy, savings, np.where(wait, wait_regret, (savings + wait_regret) / 2)
    )
    return float(np.mean(values))


def _atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def train(demand_path: Path = DEMAND_OOF, alternative_path: Path = ALTERNATIVE_OOF):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_oof(demand_path, alternative_path)
    dates = frame.landmark_at.sort_values().drop_duplicates()
    if len(dates) < 5:
        raise ValueError("At least five distinct landmark timestamps are needed")
    boundary = dates.iloc[max(1, int(len(dates) * 0.70) - 1)]
    training, validation = frame[frame.landmark_at.le(boundary)], frame[frame.landmark_at.gt(boundary)]
    if training.empty or validation.empty:
        raise ValueError("Chronological holdout is empty")
    selected, report = {}, {"pipeline_version": PIPELINE_VERSION, "holdout_start": str(validation.landmark_at.min()), "profiles": {}}
    for name, profile in PROFILES.items():
        selected[name], report["profiles"][name] = {}, {}
        for horizon in sorted(frame.horizon_days.unique()):
            train_part = training[training.horizon_days.eq(horizon)]
            valid_part = validation[validation.horizon_days.eq(horizon)]
            if train_part.empty or valid_part.empty:
                raise ValueError(f"Empty train/holdout partition for {horizon}d")
            best_policy, best_loss = None, float("inf")
            for candidate in _candidates():
                loss = _mean_regret(train_part, candidate, profile)
                if loss < best_loss:
                    best_loss, best_policy = loss, candidate
            selected[name][str(int(horizon))] = best_policy
            report["profiles"][name][str(int(horizon))] = {
                "policy": best_policy,
                "training": summarize(apply_policy(train_part, best_policy), profile),
                "chronological_holdout": summarize(apply_policy(valid_part, best_policy), profile),
            }
    payload = {"pipeline_version": PIPELINE_VERSION, "holdout_start": str(validation.landmark_at.min()), "profiles": selected}
    _atomic_json(payload, ARTIFACT_DIR / "policy.json")
    _atomic_json(report, ARTIFACT_DIR / "training_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-oof", type=Path, default=DEMAND_OOF)
    parser.add_argument("--alternative-oof", type=Path, default=ALTERNATIVE_OOF)
    args = parser.parse_args()
    print(json.dumps(train(args.demand_oof, args.alternative_oof), ensure_ascii=False, indent=2))
