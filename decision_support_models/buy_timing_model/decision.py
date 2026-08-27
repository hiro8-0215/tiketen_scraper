"""Scoring, actions, and historically testable regret definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_policy(frame: pd.DataFrame, policy: dict) -> pd.DataFrame:
    result = frame.copy()
    result["decision_score"] = (
        result.discount_ratio
        + policy["disappear_weight"] * result.disappearance_probability
        - policy["alternative_weight"] * result.p_alternative
    )
    result["action"] = "uncertain"
    result.loc[result.p_alternative.ge(policy["wait_threshold"]), "action"] = "wait"
    result.loc[result.decision_score.ge(policy["buy_threshold"]), "action"] = "buy_now"
    return result


def regret(frame: pd.DataFrame, profile: dict) -> np.ndarray:
    savings = pd.to_numeric(frame.potential_savings, errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    buy_regret = savings
    sold_miss = frame.true_state.eq(1).to_numpy(float) * profile["sold_miss_penalty"]
    deleted_miss = frame.true_state.eq(2).to_numpy(float) * profile["sold_miss_penalty"] * profile["deleted_multiplier"]
    no_replacement = 1 - frame.true_alternative.clip(0, 1).to_numpy(float)
    wait_regret = (sold_miss + deleted_miss) * no_replacement
    actions = frame.action.to_numpy()
    # 'uncertain' represents human review; score it conservatively as the
    # average of immediate purchase and waiting opportunity losses.
    return np.where(actions == "buy_now", buy_regret, np.where(actions == "wait", wait_regret, (buy_regret + wait_regret) / 2))


def summarize(frame: pd.DataFrame, profile: dict) -> dict:
    values = regret(frame, profile)
    return {
        "rows": int(len(frame)),
        "mean_regret_yen": float(np.mean(values)),
        "median_regret_yen": float(np.median(values)),
        "actions": {str(key): int(value) for key, value in frame.action.value_counts().items()},
    }
