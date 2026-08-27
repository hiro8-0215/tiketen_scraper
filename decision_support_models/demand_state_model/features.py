"""Strictly-as-of market features and model feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEMANTIC_CATEGORICAL_FEATURES, SEMANTIC_FEATURES


TRAILING_DAYS = (1, 3, 7)
CATEGORICAL_CANDIDATES = [
    "group_slug", "event_id", "venue", "ticket_type", "name_type",
    "delivery_method", "perf_day_of_week", "perf_month", "perf_hour",
] + SEMANTIC_CATEGORICAL_FEATURES
FORBIDDEN_FEATURES = {
    "status", "sold_at", "last_observed_at", "outcome_at", "known_until",
    "landmark_at", "landmark_date", "first_observed_at", "performance_at",
    "perf_date", "perf_time", "raw_description", "ticket_tags", "seller_name",
    "order_num", "created_at_unix", "details_fetched", "ticket_id",
    "duplicate_group", "artist_id", "semantic_text_hash",
}


def _count_between(sorted_times, start, end):
    if len(sorted_times) == 0:
        return 0
    return int(
        np.searchsorted(sorted_times, np.datetime64(end), side="left")
        - np.searchsorted(sorted_times, np.datetime64(start), side="left")
    )


def add_market_features(landmarks: pd.DataFrame, tickets: pd.DataFrame) -> pd.DataFrame:
    result = landmarks.copy()
    state_parts = []
    cheaper = np.zeros(len(result), dtype=np.int32)
    tickets = tickets.copy()
    if "outcome_at" not in tickets:
        tickets["outcome_at"] = pd.NaT
        tickets.loc[tickets["status"].eq("sold"), "outcome_at"] = tickets.loc[
            tickets["status"].eq("sold"), "sold_at"
        ]
        tickets.loc[tickets["status"].eq("deleted"), "outcome_at"] = tickets.loc[
            tickets["status"].eq("deleted"), "last_observed_at"
        ]
    tickets["end_at"] = tickets["outcome_at"]
    source_groups = {
        event_id: source
        for event_id, source in tickets.groupby("event_id", dropna=False, observed=True)
        if pd.notna(event_id)
    }
    empty_source = tickets.iloc[0:0]
    for event_number, (event_id, row_positions) in enumerate(
        result.groupby("event_id", dropna=False, observed=True).indices.items(), start=1
    ):
        positions = np.asarray(row_positions, dtype=np.int64)
        event_rows = result.iloc[positions]
        moments, inverse = np.unique(
            event_rows["landmark_at"].to_numpy(dtype="datetime64[ns]"),
            return_inverse=True,
        )
        source = source_groups.get(event_id, empty_source)
        first_raw = source["first_observed_at"].to_numpy(dtype="datetime64[ns]")
        end_raw = source["end_at"].to_numpy(dtype="datetime64[ns]")
        source_prices = pd.to_numeric(source["price"], errors="coerce").to_numpy(float)
        first = np.sort(first_raw[~np.isnat(first_raw)])
        sold_raw = source["sold_at"].to_numpy(dtype="datetime64[ns]")
        sold = np.sort(sold_raw[~np.isnat(sold_raw)])
        deleted_raw = source.loc[
            source["status"].eq("deleted"), "last_observed_at"
        ].to_numpy(dtype="datetime64[ns]")
        deleted = np.sort(deleted_raw[~np.isnat(deleted_raw)])

        prior_mask = ~np.isnat(sold_raw) & ~np.isnan(source_prices)
        prior_order = np.argsort(sold_raw[prior_mask], kind="stable")
        prior_times = sold_raw[prior_mask][prior_order]
        prior_prices = source_prices[prior_mask][prior_order]

        active_count = np.zeros(len(moments), dtype=np.int32)
        price_min = np.full(len(moments), np.nan)
        price_median = np.full(len(moments), np.nan)
        price_q25 = np.full(len(moments), np.nan)
        price_q75 = np.full(len(moments), np.nan)
        prior_count = np.searchsorted(prior_times, moments, side="left").astype(np.int32)
        prior_median = np.full(len(moments), np.nan)

        # Rows for one event are processed as NumPy arrays. The active interval
        # remains exactly first_observed_at <= t < outcome_at.
        row_order = np.argsort(inverse, kind="stable")
        row_counts = np.bincount(inverse, minlength=len(moments))
        row_boundaries = np.r_[0, np.cumsum(row_counts)]
        event_row_price = pd.to_numeric(event_rows["price"], errors="coerce").to_numpy(float)
        for moment_index, moment in enumerate(moments):
            active_mask = (
                ~np.isnat(first_raw)
                & (first_raw <= moment)
                & (np.isnat(end_raw) | (end_raw > moment))
            )
            active_count[moment_index] = int(active_mask.sum())
            prices = source_prices[active_mask]
            prices = prices[~np.isnan(prices)]
            sorted_prices = np.sort(prices)
            if len(prices):
                price_min[moment_index] = float(np.min(prices))
                price_median[moment_index] = float(np.median(prices))
                price_q25[moment_index] = float(np.quantile(prices, .25))
                price_q75[moment_index] = float(np.quantile(prices, .75))
            count = int(prior_count[moment_index])
            if count:
                prior_median[moment_index] = float(np.median(prior_prices[:count]))

            local = row_order[row_boundaries[moment_index]:row_boundaries[moment_index + 1]]
            row_price = event_row_price[local]
            valid_price = np.isfinite(row_price)
            values = np.zeros(len(local), dtype=np.int32)
            values[valid_price] = np.searchsorted(
                sorted_prices, row_price[valid_price], side="left"
            ).astype(np.int32)
            cheaper[positions[local]] = values

        event_market = pd.DataFrame({
            "event_id": event_id,
            "landmark_at": pd.to_datetime(moments),
            "market_active_count": active_count,
            "market_price_min": price_min,
            "market_price_median": price_median,
            "market_price_q25": price_q25,
            "market_price_q75": price_q75,
            "market_prior_sold_count": prior_count,
            "market_prior_sold_median": prior_median,
        })
        for days in TRAILING_DAYS:
            begin = moments - np.timedelta64(int(days), "D")
            event_market[f"market_new_{days}d"] = (
                np.searchsorted(first, moments, side="left")
                - np.searchsorted(first, begin, side="left")
            ).astype(np.int32)
            event_market[f"market_sold_{days}d"] = (
                np.searchsorted(sold, moments, side="left")
                - np.searchsorted(sold, begin, side="left")
            ).astype(np.int32)
            event_market[f"market_deleted_{days}d"] = (
                np.searchsorted(deleted, moments, side="left")
                - np.searchsorted(deleted, begin, side="left")
            ).astype(np.int32)
        state_parts.append(event_market)
        if event_number % 100 == 0:
            print(f"[demand] market features: {event_number:,} events", flush=True)

    market = pd.concat(state_parts, ignore_index=True).drop_duplicates(
        ["event_id", "landmark_at"]
    )
    result = result.merge(market, on=["event_id", "landmark_at"], how="left", validate="many_to_one")
    result["price_to_market_median"] = result["price"] / result["market_price_median"].replace(0, np.nan)
    result["price_to_prior_sold_median"] = result["price"] / result["market_prior_sold_median"].replace(0, np.nan)
    if "fair_price" in result:
        result["price_to_fair"] = result["price"] / result["fair_price"].replace(0, np.nan)
        result["fair_discount_pct"] = (result["fair_price"] - result["price"]) / result["fair_price"].replace(0, np.nan)
    result["market_cheaper_active_count"] = cheaper
    for days in TRAILING_DAYS:
        result[f"market_sell_through_{days}d"] = result[f"market_sold_{days}d"] / np.maximum(
            result["market_active_count"] + result[f"market_sold_{days}d"], 1
        )
        result[f"market_disappearance_{days}d"] = (
            result[f"market_sold_{days}d"] + result[f"market_deleted_{days}d"]
        ) / np.maximum(
            result["market_active_count"] + result[f"market_sold_{days}d"] + result[f"market_deleted_{days}d"], 1
        )
    return result


def feature_columns(frame: pd.DataFrame):
    labels = {column for column in frame if column.startswith("state_")}
    categorical = [
        column for column in CATEGORICAL_CANDIDATES
        if column in frame and frame[column].nunique(dropna=False) > 1
    ]
    excluded = FORBIDDEN_FEATURES | labels | set(categorical)
    numeric = [
        column for column in frame
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].nunique(dropna=False) > 1
    ]
    leaked = (set(numeric) | set(categorical)) & (FORBIDDEN_FEATURES | labels)
    if leaked:
        raise AssertionError(f"Forbidden demand features: {sorted(leaked)}")
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
