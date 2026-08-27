"""Predict competing state probabilities for listings active at an as-of time."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from config import ARTIFACT_DIR, HORIZONS_DAYS
from data_loader import load_tickets
from features import add_market_features
from modeling import aligned_probabilities, apply_temperature, enforce_monotonic_wide
from timeline import add_end_times


def predict(data_dir: Path | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    payload = joblib.load(ARTIFACT_DIR / "demand_state.joblib")
    tickets = load_tickets(data_dir)
    tickets, cutoff = add_end_times(tickets)
    as_of = pd.Timestamp(as_of) if as_of is not None else cutoff
    active = tickets[
        tickets["first_observed_at"].le(as_of)
        & (tickets["outcome_at"].isna() | tickets["outcome_at"].gt(as_of))
        & (tickets["performance_at"].isna() | tickets["performance_at"].gt(as_of))
    ].copy()
    if active.empty:
        raise ValueError(f"No active listings at {as_of}")
    active["landmark_at"] = as_of
    active["landmark_date"] = as_of.floor("D")
    active["days_since_listing"] = (as_of - active["first_observed_at"]).dt.total_seconds() / 86400
    active["days_until_event"] = (active["performance_at"] - as_of).dt.total_seconds() / 86400
    frame = add_market_features(active, tickets)
    output_columns = [column for column in (
        "ticket_id", "event_id", "price", "fair_price",
        "market_price_median", "market_prior_sold_median",
    ) if column in frame]
    output = frame[output_columns].copy()
    output["as_of"] = as_of
    for horizon in HORIZONS_DAYS:
        selected = payload["selected_features"][str(horizon)]
        features = selected["numeric"] + selected["categorical"]
        raw = aligned_probabilities(payload["models"][str(horizon)], frame[features])
        calibrated = apply_temperature(raw, payload["temperatures"][str(horizon)])
        output[[f"p_active_{horizon}d", f"p_sold_{horizon}d", f"p_deleted_{horizon}d"]] = calibrated
    return enforce_monotonic_wide(output, HORIZONS_DAYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "latest_predictions.csv")
    args = parser.parse_args()
    result = predict(args.data_dir, pd.Timestamp(args.as_of) if args.as_of else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
