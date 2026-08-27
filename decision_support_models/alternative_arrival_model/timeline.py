"""As-of landmarks and future cheaper-comparable-listing labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    HORIZONS_DAYS, MIN_SAVINGS_PCT, MIN_SAVINGS_YEN,
    SEMANTIC_COMPARABLE_FIELDS,
)


COMPARABLE = ("event_id", "quantity", "ticket_type", "name_type")
STATIC_KEEP = {
    "ticket_id", "event_id", "group_slug", "venue", "ticket_type", "name_type",
    "delivery_method", "quantity", "price", "seller_rating", "capacity",
    "base_price", "seat_rule", "total_stages", "fc_members", "first_observed_at",
    "performance_at", "outcome_at", "duplicate_group", "perf_day_of_week",
    "perf_month", "is_weekend", "perf_day_sin", "perf_day_cos",
    "description_length", "text_has_fc", "text_has_seat",
    "text_has_identity_check", "text_has_urgent",
    "semantic_text_hash", "semantic_seat_level", "semantic_row_position",
    "semantic_winning_route", "semantic_name_status", "semantic_identity_check",
    "semantic_distribution_type", "semantic_visibility", "semantic_is_fc_early",
    "semantic_is_random",
}


def observation_cutoff(tickets: pd.DataFrame) -> pd.Timestamp:
    return pd.concat([tickets["first_observed_at"], tickets["last_observed_at"], tickets["sold_at"]]).dropna().max()


def prepare_end_times(tickets: pd.DataFrame) -> pd.DataFrame:
    result = tickets.copy()
    prior_excluded = int(tickets.attrs.get("excluded_temporal_anomalies", 0))
    prior_ids = list(tickets.attrs.get("excluded_temporal_anomaly_ticket_ids", []))
    result["outcome_at"] = pd.NaT
    result.loc[result["status"].eq("sold"), "outcome_at"] = result.loc[result["status"].eq("sold"), "sold_at"]
    result.loc[result["status"].eq("deleted"), "outcome_at"] = result.loc[result["status"].eq("deleted"), "last_observed_at"]
    invalid = (
        result["outcome_at"].notna()
        & result["outcome_at"].lt(result["first_observed_at"])
    ) | (
        result["performance_at"].notna()
        & result["performance_at"].lt(result["first_observed_at"])
    )
    current_ids = result.loc[invalid, "ticket_id"].astype(str).tolist()
    if current_ids:
        print(
            f"[alternative] excluded {len(current_ids):,} temporal anomalies "
            "(known end < first_observed_at)",
            flush=True,
        )
        result = result.loc[~invalid].copy()
    result.attrs.update(tickets.attrs)
    result.attrs["excluded_temporal_anomalies"] = prior_excluded + len(current_ids)
    result.attrs["excluded_temporal_anomaly_ticket_ids"] = list(
        dict.fromkeys(prior_ids + current_ids)
    )
    for column in COMPARABLE:
        result[column] = result[column].fillna("__missing__").astype(str)
    for column in SEMANTIC_COMPARABLE_FIELDS:
        if column not in result:
            result[column] = "不明"
        result[column] = result[column].fillna("不明").astype(str)
    return result


def _times(start, end):
    if pd.isna(start) or pd.isna(end) or end <= start:
        return []
    return pd.date_range(start=start, end=end, freq="1D", inclusive="left")


def build_landmarks(tickets: pd.DataFrame, horizons=HORIZONS_DAYS, cutoff=None) -> pd.DataFrame:
    prepared = prepare_end_times(tickets)
    cutoff = pd.Timestamp(cutoff or observation_cutoff(prepared))
    # Raw descriptions are already represented by derived flags and must not be
    # duplicated into every daily row.
    static = [column for column in prepared if column in STATIC_KEEP]
    comparable_groups = {}
    for key, value in prepared.groupby(list(COMPARABLE), dropna=False, observed=True):
        ordered = value.sort_values("first_observed_at")
        comparable_groups[key] = (
            ordered["first_observed_at"].to_numpy(dtype="datetime64[ns]"),
            pd.to_numeric(ordered["price"], errors="coerce").to_numpy(float),
            {
                column: ordered[column].fillna("不明").astype(str).to_numpy()
                for column in SEMANTIC_COMPARABLE_FIELDS
            },
        )

    start = prepared["first_observed_at"].to_numpy(dtype="datetime64[ns]")
    outcome = prepared["outcome_at"].to_numpy(dtype="datetime64[ns]")
    performance = prepared["performance_at"].to_numpy(dtype="datetime64[ns]")
    cutoff_value = np.datetime64(cutoff, "ns")
    end = np.full(len(prepared), cutoff_value, dtype="datetime64[ns]")
    has_performance = ~np.isnat(performance)
    end[has_performance] = np.minimum(performance[has_performance], cutoff_value)
    has_outcome = ~np.isnat(outcome)
    end[has_outcome] = outcome[has_outcome]
    outcome_and_performance = has_outcome & has_performance
    end[outcome_and_performance] = np.minimum(
        end[outcome_and_performance], performance[outcome_and_performance]
    )
    valid = ~np.isnat(start) & ~np.isnat(end) & (end > start)
    day_ns = int(pd.Timedelta(days=1).value)
    counts = np.zeros(len(prepared), dtype=np.int64)
    duration_ns = (end[valid] - start[valid]).astype("timedelta64[ns]").astype(np.int64)
    counts[valid] = (duration_ns + day_ns - 1) // day_ns
    total_rows = int(counts.sum())
    if total_rows == 0:
        raise ValueError("No alternative-arrival landmarks")

    repeated_ticket = np.repeat(np.arange(len(prepared), dtype=np.int64), counts)
    block_start = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = np.arange(total_rows, dtype=np.int64) - block_start
    landmark_values = start[repeated_ticket] + (
        offsets * day_ns
    ).astype("timedelta64[ns]")
    result = prepared.iloc[repeated_ticket][static].reset_index(drop=True).copy()
    result["landmark_at"] = pd.to_datetime(landmark_values)
    result["days_since_listing"] = offsets.astype(float)
    result["days_until_event"] = (
        result["performance_at"] - result["landmark_at"]
    ).dt.total_seconds() / 86400

    target_arrays = {}
    for horizon in horizons:
        target_arrays[f"alternative_{horizon}d"] = np.full(total_rows, -1, dtype=np.int8)
        target_arrays[f"future_best_price_{horizon}d"] = np.full(total_rows, np.nan)
        target_arrays[f"alternative_first_at_{horizon}d"] = np.full(
            total_rows, np.datetime64("NaT"), dtype="datetime64[ns]"
        )
        target_arrays[f"potential_savings_{horizon}d"] = np.zeros(total_rows)

    boundaries = np.r_[0, np.cumsum(counts)]
    for ticket_number, row in enumerate(prepared.itertuples(index=False), start=1):
        left_row, right_row = int(boundaries[ticket_number - 1]), int(boundaries[ticket_number])
        if left_row == right_row:
            continue
        values = row._asdict()
        key = tuple(values[column] for column in COMPARABLE)
        candidate_times, candidate_prices, candidate_semantics = comparable_groups[key]
        price = float(values["price"])
        threshold = price - max(MIN_SAVINGS_YEN, price * MIN_SAVINGS_PCT)
        comparable = np.isfinite(candidate_prices) & (candidate_prices <= threshold)
        for column in SEMANTIC_COMPARABLE_FIELDS:
            current = str(values.get(column, "不明"))
            if current != "不明":
                candidates = candidate_semantics[column]
                comparable &= (candidates == "不明") | (candidates == current)
        qualifying_times = candidate_times[comparable]
        qualifying_prices = candidate_prices[comparable]
        moments = landmark_values[left_row:right_row]
        performance_at = values.get("performance_at")

        for horizon in horizons:
            deadline = moments + np.timedelta64(int(horizon), "D")
            fully_observed = deadline <= cutoff_value
            if pd.notna(performance_at):
                fully_observed &= deadline <= np.datetime64(performance_at, "ns")
            left = np.searchsorted(qualifying_times, moments, side="right")
            right = np.searchsorted(qualifying_times, deadline, side="right")
            has_alternative = left < right
            best_price = np.full(len(moments), np.nan)
            first_time = np.full(
                len(moments), np.datetime64("NaT"), dtype="datetime64[ns]"
            )
            for local in np.flatnonzero(has_alternative):
                best_price[local] = float(
                    qualifying_prices[left[local]:right[local]].min()
                )
                first_time[local] = qualifying_times[left[local]]
            target_arrays[f"alternative_{horizon}d"][left_row:right_row] = np.where(
                fully_observed, has_alternative.astype(np.int8), -1
            )
            target_arrays[f"future_best_price_{horizon}d"][left_row:right_row] = best_price
            target_arrays[f"alternative_first_at_{horizon}d"][left_row:right_row] = first_time
            savings = np.zeros(len(moments))
            savings[has_alternative] = np.maximum(
                0.0, price - best_price[has_alternative]
            )
            target_arrays[f"potential_savings_{horizon}d"][left_row:right_row] = savings
        if ticket_number % 10_000 == 0:
            print(
                f"[alternative] labels: {ticket_number:,}/{len(prepared):,} tickets",
                flush=True,
            )

    for column, values in target_arrays.items():
        result[column] = values
    print(
        f"[alternative] landmark table: {len(prepared):,} tickets -> "
        f"{total_rows:,} rows (vectorized)",
        flush=True,
    )
    result = result.sort_values(["landmark_at", "ticket_id"]).reset_index(drop=True)
    result.attrs["excluded_temporal_anomalies"] = int(
        prepared.attrs.get("excluded_temporal_anomalies", 0)
    )
    result.attrs["excluded_temporal_anomaly_ticket_ids"] = list(
        prepared.attrs.get("excluded_temporal_anomaly_ticket_ids", [])
    )
    for column in (
        "ticket_id", "event_id", "group_slug", "venue", "ticket_type", "name_type",
        "quantity", "delivery_method", "duplicate_group", "semantic_seat_level",
        "semantic_row_position", "semantic_winning_route", "semantic_name_status",
        "semantic_identity_check", "semantic_distribution_type", "semantic_visibility",
    ):
        if column in result:
            result[column] = result[column].astype("category")
    return result
