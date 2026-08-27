"""Leakage-safe daily landmark and competing-risk label construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZONS_DAYS, LANDMARK_STEP_DAYS


STATIC_KEEP = {
    "ticket_id", "event_id", "group_slug", "venue", "ticket_type", "name_type",
    "delivery_method", "quantity", "price", "seller_rating", "capacity",
    "base_price", "seat_rule", "total_stages", "fc_members", "fair_price",
    "first_observed_at", "performance_at", "outcome_at", "duplicate_group",
    "perf_day_of_week", "perf_month", "perf_hour", "is_weekend",
    "perf_day_sin", "perf_day_cos", "perf_hour_sin", "perf_hour_cos",
    "description_length", "text_has_doukou", "text_has_random", "text_has_no_swap",
    "text_has_fc", "text_has_identity_check", "text_has_seat", "text_has_urgent",
    "semantic_text_hash", "semantic_seat_level", "semantic_row_position",
    "semantic_winning_route", "semantic_name_status", "semantic_identity_check",
    "semantic_distribution_type", "semantic_visibility", "semantic_is_fc_early",
    "semantic_is_random",
}


def observation_cutoff(tickets: pd.DataFrame) -> pd.Timestamp:
    values = pd.concat(
        [tickets["first_observed_at"], tickets["last_observed_at"], tickets["sold_at"]],
        ignore_index=True,
    ).dropna()
    if values.empty:
        raise ValueError("Cannot determine observation cutoff")
    return values.max()


def add_end_times(tickets: pd.DataFrame, cutoff: pd.Timestamp | None = None):
    result = tickets.copy()
    prior_excluded = int(tickets.attrs.get("excluded_temporal_anomalies", 0))
    prior_ids = list(tickets.attrs.get("excluded_temporal_anomaly_ticket_ids", []))
    cutoff = cutoff or observation_cutoff(result)
    sold = result["status"].eq("sold")
    deleted = result["status"].eq("deleted")
    result["outcome_at"] = pd.NaT
    result.loc[sold, "outcome_at"] = result.loc[sold, "sold_at"]
    result.loc[deleted, "outcome_at"] = result.loc[deleted, "last_observed_at"]
    result["known_until"] = cutoff
    result.loc[sold | deleted, "known_until"] = result.loc[sold | deleted, "outcome_at"]
    if "performance_at" in result:
        valid_performance = result["performance_at"].notna()
        result.loc[valid_performance, "known_until"] = result.loc[
            valid_performance, ["known_until", "performance_at"]
        ].min(axis=1)
    invalid = result["known_until"] < result["first_observed_at"]
    current_ids = result.loc[invalid, "ticket_id"].astype(str).tolist()
    if current_ids:
        # These rows cannot contribute a leakage-safe landmark: the recorded
        # outcome predates the first time the listing was observed. Do not
        # fabricate a sale time or a positive duration; quarantine only these
        # rows and retain an explicit audit count in every training report.
        print(
            f"[demand] excluded {len(current_ids):,} temporal anomalies "
            "(known_until < first_observed_at)",
            flush=True,
        )
        result = result.loc[~invalid].copy()
    result.attrs.update(tickets.attrs)
    result.attrs["excluded_temporal_anomalies"] = prior_excluded + len(current_ids)
    result.attrs["excluded_temporal_anomaly_ticket_ids"] = list(
        dict.fromkeys(prior_ids + current_ids)
    )
    return result, cutoff


def _landmarks(start: pd.Timestamp, end: pd.Timestamp):
    if pd.isna(start) or pd.isna(end) or end <= start:
        return []
    count = int(np.ceil((end - start).total_seconds() / (86400 * LANDMARK_STEP_DAYS)))
    return [start + pd.Timedelta(days=LANDMARK_STEP_DAYS * step) for step in range(count)]


def build_landmarks(
    tickets: pd.DataFrame,
    horizons=HORIZONS_DAYS,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    prepared, cutoff = add_end_times(tickets, cutoff)
    # Do not replicate raw descriptions and other unused object columns across
    # millions of daily landmark rows. Derived text flags retain their signal.
    static_columns = [column for column in prepared.columns if column in STATIC_KEEP]
    start = prepared["first_observed_at"].to_numpy(dtype="datetime64[ns]")
    end = prepared["known_until"].to_numpy(dtype="datetime64[ns]")
    valid = ~np.isnat(start) & ~np.isnat(end) & (end > start)
    step_ns = int(pd.Timedelta(days=LANDMARK_STEP_DAYS).value)
    counts = np.zeros(len(prepared), dtype=np.int64)
    duration_ns = (end[valid] - start[valid]).astype("timedelta64[ns]").astype(np.int64)
    counts[valid] = (duration_ns + step_ns - 1) // step_ns
    total_rows = int(counts.sum())
    if total_rows == 0:
        raise ValueError("No landmark rows could be constructed")

    # Expanding the static ticket table once is value-equivalent to the old
    # ticket x day dictionary loop, but avoids millions of Python objects.
    repeated_ticket = np.repeat(np.arange(len(prepared), dtype=np.int64), counts)
    block_start = np.repeat(np.cumsum(counts) - counts, counts)
    offsets = np.arange(total_rows, dtype=np.int64) - block_start
    result = prepared.iloc[repeated_ticket][static_columns].reset_index(drop=True).copy()
    landmark_at = start[repeated_ticket] + (
        offsets * step_ns
    ).astype("timedelta64[ns]")
    result["landmark_at"] = pd.to_datetime(landmark_at)
    result["days_since_listing"] = offsets.astype(float) * LANDMARK_STEP_DAYS
    if "performance_at" in result:
        result["days_until_event"] = (
            result["performance_at"] - result["landmark_at"]
        ).dt.total_seconds() / 86400
    else:
        result["days_until_event"] = np.nan

    outcome = prepared["outcome_at"].to_numpy(dtype="datetime64[ns]")[repeated_ticket]
    known_until = end[repeated_ticket]
    sold = prepared["status"].eq("sold").to_numpy()[repeated_ticket]
    for horizon in horizons:
        deadline = landmark_at + np.timedelta64(int(horizon), "D")
        outcome_observed = ~np.isnat(outcome) & (outcome <= deadline)
        label = np.full(total_rows, -1, dtype=np.int8)
        label[outcome_observed] = np.where(sold[outcome_observed], 1, 2)
        label[~outcome_observed & (known_until >= deadline)] = 0
        result[f"state_{horizon}d"] = label

    print(
        f"[demand] landmark table: {len(prepared):,} tickets -> "
        f"{total_rows:,} rows (vectorized)",
        flush=True,
    )
    result["landmark_date"] = result["landmark_at"].dt.floor("D")
    for column in (
        "ticket_id", "event_id", "group_slug", "venue", "ticket_type", "name_type",
        "delivery_method", "duplicate_group", "semantic_seat_level",
        "semantic_row_position", "semantic_winning_route", "semantic_name_status",
        "semantic_identity_check", "semantic_distribution_type", "semantic_visibility",
    ):
        if column in result:
            result[column] = result[column].astype("category")
    result = result.sort_values(["landmark_at", "ticket_id"]).reset_index(drop=True)
    result.attrs["observation_cutoff"] = str(cutoff)
    result.attrs["excluded_temporal_anomalies"] = int(
        prepared.attrs.get("excluded_temporal_anomalies", 0)
    )
    result.attrs["excluded_temporal_anomaly_ticket_ids"] = list(
        prepared.attrs.get("excluded_temporal_anomaly_ticket_ids", [])
    )
    return result
