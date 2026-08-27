"""Combine the two standalone prediction CSV files into buy/wait decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ARTIFACT_DIR


def predict(demand_csv: Path, alternative_csv: Path, profile="balanced"):
    policies = json.loads((ARTIFACT_DIR / "policy.json").read_text(encoding="utf-8"))["profiles"]
    if profile not in policies:
        raise ValueError(f"Unknown profile {profile}; choose from {sorted(policies)}")
    demand, alternative = pd.read_csv(demand_csv), pd.read_csv(alternative_csv)
    keys = [name for name in ("ticket_id", "event_id", "price", "as_of") if name in demand and name in alternative]
    base_columns = list(dict.fromkeys(keys + [name for name in ("fair_price", "market_prior_sold_median", "market_price_median") if name in demand]))
    result_parts = []
    for horizon in (1, 3, 7):
        required_demand = [f"p_sold_{horizon}d", f"p_deleted_{horizon}d"]
        alt_name = f"p_alternative_{horizon}d"
        if not set(required_demand).issubset(demand) or alt_name not in alternative:
            continue
        left = demand[base_columns + required_demand].copy()
        right_keys = [name for name in keys if name in alternative]
        right = alternative[right_keys + [alt_name]].copy()
        frame = left.merge(right, on=right_keys, how="inner", validate="one_to_one")
        reference = pd.Series(np.nan, index=frame.index)
        for column in ("fair_price", "market_prior_sold_median", "market_price_median", "price"):
            if column in frame:
                reference = reference.fillna(pd.to_numeric(frame[column], errors="coerce").where(lambda item: item.gt(0)))
        frame["reference_price"] = reference
        frame["discount_ratio"] = ((reference - frame.price) / reference).clip(-2, 1)
        frame["disappearance_probability"] = (frame[required_demand[0]] + frame[required_demand[1]]).clip(0, 1)
        frame["p_alternative"] = frame[alt_name]
        if str(horizon) not in policies[profile]:
            raise ValueError(f"Policy has no {horizon}d coefficients")
        policy = policies[profile][str(horizon)]
        frame["decision_score"] = frame.discount_ratio + policy["disappear_weight"] * frame.disappearance_probability - policy["alternative_weight"] * frame.p_alternative
        frame["action"] = "uncertain"
        frame.loc[frame.p_alternative.ge(policy["wait_threshold"]), "action"] = "wait"
        frame.loc[frame.decision_score.ge(policy["buy_threshold"]), "action"] = "buy_now"
        frame["horizon_days"] = horizon
        result_parts.append(frame)
    if not result_parts:
        raise ValueError("Prediction files have no common supported horizon")
    return pd.concat(result_parts, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demand", required=True, type=Path)
    parser.add_argument("--alternative", required=True, type=Path)
    parser.add_argument("--profile", choices=("safety", "balanced", "savings"), default="balanced")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "latest_decisions.csv")
    args = parser.parse_args()
    result = predict(args.demand, args.alternative, args.profile)
    result.to_csv(args.output, index=False)
    print(f"saved {len(result):,} rows to {args.output}")
