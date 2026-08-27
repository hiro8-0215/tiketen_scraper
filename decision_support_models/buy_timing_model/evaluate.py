"""Re-evaluate the frozen policy on its untouched chronological tail."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from config import ALTERNATIVE_OOF, ARTIFACT_DIR, DEMAND_OOF, PROFILES
from data_loader import load_oof
from decision import apply_policy, summarize


def evaluate(demand_path=DEMAND_OOF, alternative_path=ALTERNATIVE_OOF):
    payload = json.loads((ARTIFACT_DIR / "policy.json").read_text(encoding="utf-8"))
    frame = load_oof(demand_path, alternative_path)
    holdout = frame[frame.landmark_at.ge(pd.Timestamp(payload["holdout_start"]))]
    result = {}
    for name, by_horizon in payload["profiles"].items():
        result[name] = {}
        for horizon, policy in by_horizon.items():
            part = holdout[holdout.horizon_days.eq(int(horizon))]
            result[name][horizon] = summarize(apply_policy(part, policy), PROFILES[name])
    path = ARTIFACT_DIR / "evaluation.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand-oof", type=Path, default=DEMAND_OOF)
    parser.add_argument("--alternative-oof", type=Path, default=ALTERNATIVE_OOF)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.demand_oof, args.alternative_oof), ensure_ascii=False, indent=2))
