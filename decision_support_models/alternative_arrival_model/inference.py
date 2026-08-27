"""Predict cheaper-comparable-listing arrival probabilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import ARTIFACT_DIR, HORIZONS_DAYS
from data_loader import load_tickets
from features import add_market_features
from modeling import calibrate, enforce_monotonic_wide, predict_positive_probability
from timeline import observation_cutoff, prepare_end_times


def predict(data_dir: Path | None = None, as_of=None):
    payload = joblib.load(ARTIFACT_DIR / "alternative_arrival.joblib")
    tickets = load_tickets(data_dir)
    prepared = prepare_end_times(tickets)
    as_of = pd.Timestamp(as_of or observation_cutoff(tickets))
    active = prepared[
        prepared.first_observed_at.le(as_of)
        & (prepared.outcome_at.isna() | prepared.outcome_at.gt(as_of))
        & (prepared.performance_at.isna() | prepared.performance_at.gt(as_of))
    ].copy()
    if active.empty:
        raise ValueError(f"No active listings at {as_of}")
    active["landmark_at"] = as_of
    active["days_since_listing"] = (as_of - active.first_observed_at).dt.total_seconds() / 86400
    active["days_until_event"] = (active.performance_at - as_of).dt.total_seconds() / 86400
    frame = add_market_features(active, prepared)
    result = frame[["ticket_id", "event_id", "price"]].copy()
    result["as_of"] = as_of
    for horizon in HORIZONS_DAYS:
        selected = payload["selected_features"][str(horizon)]
        features = selected["numeric"] + selected["categorical"]
        raw = predict_positive_probability(
            payload["models"][str(horizon)], frame[features]
        )
        result[f"p_alternative_{horizon}d"] = calibrate(payload["calibrators"][str(horizon)], raw)
    return enforce_monotonic_wide(result, HORIZONS_DAYS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "latest_predictions.csv")
    args = parser.parse_args()
    value = predict(args.data_dir, args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(args.output, index=False)
    print(f"saved {len(value):,} rows to {args.output}")
