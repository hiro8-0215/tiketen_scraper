"""Audit a ticket snapshot before any extraction or model training."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


ALLOWED_STATUS = {"listing", "sold", "deleted"}
MAX_LABEL_LAG_DAYS = 2
MIN_LATEST_DELETION_SPIKE = 50
MAX_LATEST_DELETION_SPIKE_FRACTION = 0.05
MIN_SALE_TIME_SPIKE = 50
MAX_SALE_TIME_SPIKE_FRACTION = 0.05


def _snapshot_key(path: Path) -> tuple[int, ...]:
    values = path.name.removeprefix("data_").split("_")
    if len(values) not in {2, 3} or not all(value.isdigit() for value in values):
        return (-1, -1, -1)
    numbers = tuple(map(int, values))
    return (0, *numbers) if len(numbers) == 2 else numbers


def latest_snapshot(data_root: Path) -> Path:
    candidates = [
        path for path in data_root.glob("data_*")
        if path.is_dir() and _snapshot_key(path)[0] >= 0
        and any(path.glob("*_master.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No ticket snapshot under {data_root}")
    return max(candidates, key=_snapshot_key)


def _parse_datetime(value: str) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text)
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    except ValueError:
        return None


def _snapshot_label_date(snapshot: Path, observation_year: int) -> date | None:
    parts = snapshot.name.removeprefix("data_").split("_")
    try:
        if len(parts) == 2:
            return date(observation_year, int(parts[0]), int(parts[1]))
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    return None


def audit_snapshot(snapshot: Path) -> dict:
    files = sorted(snapshot.glob("*_master.csv"))
    if not files:
        raise FileNotFoundError(f"No *_master.csv files in {snapshot}")

    status_counts: Counter[str] = Counter()
    status_by_ticket: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    status_time_counts: Counter[tuple[str, datetime]] = Counter()
    unknown_statuses: Counter[str] = Counter()
    rows = 0
    missing_ticket_ids = 0
    invalid_last_observed = 0
    maximum_observed: datetime | None = None
    records: list[dict] = []
    sold_time_counts: Counter[datetime] = Counter()

    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"ticket_id", "status", "last_observed_at"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Snapshot file missing {sorted(required)}: {path}"
                )
            for row in reader:
                rows += 1
                ticket_id = str(row.get("ticket_id", "")).strip()
                status = str(row.get("status", "")).strip().lower()
                observed = _parse_datetime(row.get("last_observed_at", ""))
                if not ticket_id:
                    missing_ticket_ids += 1
                else:
                    occurrences[ticket_id] += 1
                    status_by_ticket[ticket_id].add(status)
                status_counts[status] += 1
                if status not in ALLOWED_STATUS:
                    unknown_statuses[status] += 1
                if observed is None:
                    invalid_last_observed += 1
                else:
                    maximum_observed = (
                        observed
                        if maximum_observed is None
                        else max(maximum_observed, observed)
                    )
                    status_time_counts[(status, observed)] += 1
                sold_at = _parse_datetime(row.get("sold_at", ""))
                if status == "sold" and sold_at is not None:
                    sold_time_counts[sold_at] += 1
                event_id = str(row.get("event_id", "")).strip()
                created_at = str(row.get("created_at_unix", "")).strip()
                logical_id = (
                    f"created:{event_id}|{created_at}"
                    if event_id and created_at
                    else f"ticket:{ticket_id}"
                )
                records.append({
                    "logical_id": logical_id, "ticket_id": ticket_id,
                    "status": status, "observed": observed,
                })

    status_priority = {"deleted": 0, "listing": 1, "sold": 2}
    canonical: dict[str, dict] = {}
    logical_ticket_ids: dict[str, set[str]] = defaultdict(set)
    logical_statuses: dict[str, set[str]] = defaultdict(set)
    for record in records:
        logical_ticket_ids[record["logical_id"]].add(record["ticket_id"])
        logical_statuses[record["logical_id"]].add(record["status"])
        rank = (
            record["observed"] or datetime.min,
            status_priority.get(record["status"], -1),
        )
        previous = canonical.get(record["logical_id"])
        if previous is None or rank > previous["rank"]:
            canonical[record["logical_id"]] = {**record, "rank": rank}

    raw_status_counts = status_counts.copy()
    raw_status_time_counts = status_time_counts.copy()
    status_counts = Counter(record["status"] for record in canonical.values())
    status_time_counts = Counter(
        (record["status"], record["observed"])
        for record in canonical.values() if record["observed"] is not None
    )
    rotated_logical_ids = {
        key for key, values in logical_ticket_ids.items() if len(values) > 1
    }
    conflicting_logical_ids = {
        key for key in rotated_logical_ids if len(logical_statuses[key]) > 1
    }

    duplicate_ids = {key for key, count in occurrences.items() if count > 1}
    conflicting_ids = {
        key for key in duplicate_ids if len(status_by_ticket[key]) > 1
    }
    label_date = (
        _snapshot_label_date(snapshot, maximum_observed.year)
        if maximum_observed is not None
        else None
    )
    lag_days = (
        (label_date - maximum_observed.date()).days
        if label_date is not None and maximum_observed is not None
        else None
    )
    latest_deleted = (
        status_time_counts.get(("deleted", maximum_observed), 0)
        if maximum_observed is not None
        else 0
    )
    deletion_spikes = [
        (count, observed)
        for (status, observed), count in status_time_counts.items()
        if status == "deleted"
    ]
    largest_deleted, largest_deleted_at = (
        max(deletion_spikes) if deletion_spikes else (0, None)
    )
    raw_deletion_spikes = [
        (count, observed)
        for (status, observed), count in raw_status_time_counts.items()
        if status == "deleted"
    ]
    raw_largest_deleted, raw_largest_deleted_at = (
        max(raw_deletion_spikes) if raw_deletion_spikes else (0, None)
    )
    largest_sold_at, largest_sold_timestamp = (
        max((count, timestamp) for timestamp, count in sold_time_counts.items())
        if sold_time_counts else (0, None)
    )
    ignored_copies = sorted(
        path.name for path in snapshot.glob("*_master(*).csv")
    )
    warnings = []
    if duplicate_ids:
        warnings.append(
            f"{len(duplicate_ids):,} ticket IDs occur in multiple master files"
        )
    if conflicting_ids:
        warnings.append(
            f"{len(conflicting_ids):,} duplicated ticket IDs have conflicting statuses"
        )
    if ignored_copies:
        warnings.append(
            f"ignored master-like copies: {ignored_copies}"
        )
    if rotated_logical_ids:
        warnings.append(
            f"{len(rotated_logical_ids):,} logical listings use multiple ticket IDs; "
            "canonicalized by event_id + created_at_unix"
        )

    return {
        "snapshot": str(snapshot.resolve()),
        "files": len(files),
        "rows": rows,
        "canonical_rows": len(canonical),
        "unique_tickets": len(occurrences),
        "status_counts": dict(status_counts),
        "raw_status_counts": dict(raw_status_counts),
        "maximum_last_observed_at": str(maximum_observed),
        "snapshot_label_date": str(label_date),
        "snapshot_label_lag_days": lag_days,
        "latest_timestamp_deleted_rows": latest_deleted,
        "largest_timestamp_deleted_rows": largest_deleted,
        "largest_timestamp_deleted_at": str(largest_deleted_at),
        "raw_largest_timestamp_deleted_rows": raw_largest_deleted,
        "raw_largest_timestamp_deleted_at": str(raw_largest_deleted_at),
        "largest_sold_at_rows": largest_sold_at,
        "largest_sold_at": str(largest_sold_timestamp),
        "rotated_logical_listing_ids": len(rotated_logical_ids),
        "conflicting_logical_status_ids": len(conflicting_logical_ids),
        "duplicate_ticket_ids": len(duplicate_ids),
        "conflicting_duplicate_status_ids": len(conflicting_ids),
        "missing_ticket_ids": missing_ticket_ids,
        "invalid_last_observed_at": invalid_last_observed,
        "unknown_statuses": dict(unknown_statuses),
        "ignored_master_copies": ignored_copies,
        "warnings": warnings,
    }


def validate_snapshot(snapshot: Path, allow_historical: bool = False) -> dict:
    report = audit_snapshot(snapshot)
    structural_errors = []
    historical_errors = []
    if report["missing_ticket_ids"]:
        structural_errors.append("ticket_id is missing")
    if report["invalid_last_observed_at"]:
        structural_errors.append("last_observed_at is invalid")
    if report["unknown_statuses"]:
        structural_errors.append(
            f"unknown statuses: {report['unknown_statuses']}"
        )
    lag = report["snapshot_label_lag_days"]
    if lag is not None and lag > MAX_LABEL_LAG_DAYS:
        historical_errors.append(
            f"folder date is {lag} days newer than its latest observation"
        )
    if report["status_counts"].get("listing", 0) == 0:
        historical_errors.append("snapshot contains zero listing rows")
    spike = report["largest_timestamp_deleted_rows"]
    if (
        spike >= MIN_LATEST_DELETION_SPIKE
        and spike / max(report["unique_tickets"], 1)
        > MAX_LATEST_DELETION_SPIKE_FRACTION
    ):
        historical_errors.append(
            f"{spike:,} rows were deleted at one timestamp "
            f"({report['largest_timestamp_deleted_at']})"
        )
    sold_spike = report["largest_sold_at_rows"]
    if (
        sold_spike >= MIN_SALE_TIME_SPIKE
        and sold_spike / max(report["canonical_rows"], 1)
        > MAX_SALE_TIME_SPIKE_FRACTION
    ):
        historical_errors.append(
            f"{sold_spike:,} sold rows share one sold_at timestamp "
            f"({report['largest_sold_at']}); bootstrap sale times are not valid "
            "demand labels"
        )
    errors = structural_errors + ([] if allow_historical else historical_errors)
    report["historical_override_used"] = bool(
        allow_historical and historical_errors
    )
    report["errors"] = errors
    report["historical_issues"] = historical_errors
    report["ok"] = not errors
    if errors:
        raise RuntimeError(
            "Snapshot quality gate failed: " + "; ".join(errors)
        )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--allow-historical", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        validate_snapshot(args.snapshot, args.allow_historical),
        ensure_ascii=False,
        indent=2,
    ))
