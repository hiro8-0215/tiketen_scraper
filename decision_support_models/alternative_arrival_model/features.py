"""As-of supply features; all future-alternative fields are forbidden."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEMANTIC_CATEGORICAL_FEATURES, SEMANTIC_FEATURES


CATEGORICAL = ["group_slug", "event_id", "venue", "ticket_type", "name_type", "quantity", "delivery_method", "perf_day_of_week", "perf_month"] + SEMANTIC_CATEGORICAL_FEATURES
FORBIDDEN = {
    "status", "sold_at", "last_observed_at", "outcome_at", "first_observed_at",
    "performance_at", "landmark_at", "perf_date", "perf_time", "raw_description",
    "ticket_tags", "seller_name", "order_num", "created_at_unix", "details_fetched",
    "ticket_id", "duplicate_group", "artist_id", "semantic_text_hash",
}


def add_market_features(landmarks: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    result, tickets = landmarks.copy(), tickets.copy()
    if "outcome_at" not in tickets:
        tickets["outcome_at"] = pd.NaT
        tickets.loc[tickets.status.eq("sold"), "outcome_at"] = tickets.loc[tickets.status.eq("sold"), "sold_at"]
        tickets.loc[tickets.status.eq("deleted"), "outcome_at"] = tickets.loc[tickets.status.eq("deleted"), "last_observed_at"]
    source_groups = {
        event_id: source
        for event_id, source in tickets.groupby("event_id", dropna=False, observed=True)
        if pd.notna(event_id)
    }
    empty_source = tickets.iloc[0:0]
    states = []
    for event_number, (event_id, event_rows) in enumerate(
        result.groupby("event_id", dropna=False, observed=True), start=1
    ):
        source = source_groups.get(event_id, empty_source)
        moments = np.sort(
            event_rows.landmark_at.unique().astype("datetime64[ns]")
        )
        first_raw = source.first_observed_at.to_numpy(dtype="datetime64[ns]")
        outcome_raw = source.outcome_at.to_numpy(dtype="datetime64[ns]")
        prices_raw = pd.to_numeric(source.price, errors="coerce").to_numpy(float)
        first = np.sort(first_raw[~np.isnat(first_raw)])
        sold_raw = source.sold_at.to_numpy(dtype="datetime64[ns]")
        sold = np.sort(sold_raw[~np.isnat(sold_raw)])
        deleted_raw = source.loc[
            source.status.eq("deleted"), "last_observed_at"
        ].to_numpy(dtype="datetime64[ns]")
        deleted = np.sort(deleted_raw[~np.isnat(deleted_raw)])

        active_count = np.zeros(len(moments), dtype=np.int32)
        price_min = np.full(len(moments), np.nan)
        price_median = np.full(len(moments), np.nan)
        price_q25 = np.full(len(moments), np.nan)
        price_q75 = np.full(len(moments), np.nan)
        for moment_index, moment in enumerate(moments):
            active = (
                ~np.isnat(first_raw)
                & (first_raw <= moment)
                & (np.isnat(outcome_raw) | (outcome_raw > moment))
            )
            active_count[moment_index] = int(active.sum())
            prices = prices_raw[active]
            prices = prices[~np.isnan(prices)]
            if len(prices):
                price_min[moment_index] = float(np.min(prices))
                price_median[moment_index] = float(np.median(prices))
                price_q25[moment_index] = float(np.quantile(prices, .25))
                price_q75[moment_index] = float(np.quantile(prices, .75))

        event_market = pd.DataFrame({
            "event_id": event_id,
            "landmark_at": pd.to_datetime(moments),
            "market_active_count": active_count,
            "market_price_min": price_min,
            "market_price_median": price_median,
            "market_price_q25": price_q25,
            "market_price_q75": price_q75,
        })
        for days in (1, 3, 7):
            start = moments - np.timedelta64(days, "D")
            event_market[f"market_new_{days}d"] = (
                np.searchsorted(first, moments, side="left")
                - np.searchsorted(first, start, side="left")
            ).astype(np.int32)
            event_market[f"market_sold_{days}d"] = (
                np.searchsorted(sold, moments, side="left")
                - np.searchsorted(sold, start, side="left")
            ).astype(np.int32)
            event_market[f"market_deleted_{days}d"] = (
                np.searchsorted(deleted, moments, side="left")
                - np.searchsorted(deleted, start, side="left")
            ).astype(np.int32)
        states.append(event_market)
        if event_number % 100 == 0:
            print(f"[alternative] market features: {event_number:,} events", flush=True)
    market = pd.concat(states, ignore_index=True).drop_duplicates(
        ["event_id", "landmark_at"]
    )
    result = result.merge(market, on=["event_id", "landmark_at"], how="left", validate="many_to_one")
    result["price_to_market_median"] = result.price / result.market_price_median.replace(0, np.nan)
    result["market_price_spread"] = result.market_price_q75 - result.market_price_q25
    return result


def feature_columns(frame: pd.DataFrame):
    targets = {name for name in frame if name.startswith(("alternative_", "future_best_price_", "potential_savings_"))}
    categorical = [name for name in CATEGORICAL if name in frame and frame[name].nunique(dropna=False) > 1]
    excluded = FORBIDDEN | targets | set(categorical)
    numeric = [name for name in frame if name not in excluded and pd.api.types.is_numeric_dtype(frame[name]) and frame[name].nunique(dropna=False) > 1]
    if (set(numeric) | set(categorical)) & (FORBIDDEN | targets):
        raise AssertionError("Future alternative information selected as feature")
    return numeric, categorical


def feature_profiles(frame: pd.DataFrame):
    numeric, categorical = feature_columns(frame)
    return {
        "tabular": (
            [column for column in numeric if column not in SEMANTIC_FEATURES],
            [column for column in categorical if column not in SEMANTIC_FEATURES],
        ),
        "semantic": (numeric, categorical),
    }
