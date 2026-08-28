"""Load common ticket data without importing code from another model folder."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ALLOWED_STATUS,
    DATA_ROOT,
    FAIR_PRICE_CACHE,
    MANUAL_DIR,
    REQUIRE_SEMANTIC_FEATURES,
    REQUIRED_COLUMNS,
    SEMANTIC_FEATURES,
    SEMANTIC_FEATURES_FILE,
    SEMANTIC_MANIFEST_FILE,
    SEMANTIC_MAX_PARSE_ERROR_RATE,
    SEMANTIC_SCHEMA_VERSION,
)


def snapshot_key(path: Path):
    parts = path.name.removeprefix("data_").split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return (0, int(parts[0]), int(parts[1]))
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return tuple(map(int, parts))
    return (-1, -1, -1)


def latest_data_dir() -> Path:
    candidates = [
        path for path in DATA_ROOT.glob("data_*")
        if path.is_dir() and snapshot_key(path)[0] >= 0 and any(path.glob("*_master.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No ticket snapshot found under {DATA_ROOT}")
    return max(candidates, key=snapshot_key)


def _load_master(name: str) -> pd.DataFrame:
    path = MANUAL_DIR / f"master_{name}.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def merge_manual_masters(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    venue = _load_master("venue")
    if not venue.empty and {"venue", "capacity"}.issubset(venue):
        result = result.merge(
            venue[["venue", "capacity"]].drop_duplicates("venue"),
            on="venue", how="left", validate="many_to_one",
        )
    tour = _load_master("tour")
    tour_columns = [
        column for column in [
            "event_id", "venue", "base_price", "lottery_date", "seat_rule",
            "first_day", "last_day", "total_stages", "artist_id",
        ] if column in tour
    ]
    if not tour.empty and {"event_id", "venue"}.issubset(tour_columns):
        result = result.merge(
            tour[tour_columns].drop_duplicates(["event_id", "venue"]),
            on=["event_id", "venue"], how="left", validate="many_to_one",
        )
    artist = _load_master("artist")
    if not artist.empty and {"artist_id", "fc_members"}.issubset(artist):
        if "artist_id" in result:
            result = result.merge(
                artist[["artist_id", "fc_members"]].drop_duplicates("artist_id"),
                on="artist_id", how="left", validate="many_to_one",
            )
        else:
            result = result.merge(
                artist[["artist_id", "fc_members"]].drop_duplicates("artist_id"),
                left_on="group_slug", right_on="artist_id", how="left",
                validate="many_to_one",
            )
    return result


def _performance_at(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(frame["perf_date"], errors="coerce").dt.normalize()
    time_text = frame.get("perf_time", pd.Series("", index=frame.index)).fillna("").astype(str)
    parts = time_text.str.extract(r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?")
    hour = pd.to_numeric(parts["hour"], errors="coerce").fillna(23).clip(0, 23)
    minute = pd.to_numeric(parts["minute"], errors="coerce").fillna(59).clip(0, 59)
    return date + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")


def add_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["performance_at"] = _performance_at(result)
    result["perf_day_of_week"] = result["performance_at"].dt.dayofweek
    result["perf_month"] = result["performance_at"].dt.month
    result["perf_hour"] = result["performance_at"].dt.hour
    result["is_weekend"] = result["perf_day_of_week"].isin([5, 6]).astype(float)
    result["perf_day_sin"] = np.sin(2 * np.pi * result["perf_day_of_week"] / 7)
    result["perf_day_cos"] = np.cos(2 * np.pi * result["perf_day_of_week"] / 7)
    result["perf_hour_sin"] = np.sin(2 * np.pi * result["perf_hour"] / 24)
    result["perf_hour_cos"] = np.cos(2 * np.pi * result["perf_hour"] / 24)
    description = result.get("raw_description", pd.Series("", index=result.index)).fillna("").astype(str)
    tags = result.get("ticket_tags", pd.Series("", index=result.index)).fillna("").astype(str)
    text = description + " " + tags
    patterns = {
        "text_has_doukou": r"同行|同時入場",
        "text_has_random": r"ランダム",
        "text_has_no_swap": r"すり替え\s*(?:なし|無し|無)",
        "text_has_fc": r"FC|ファンクラブ|初期当選",
        "text_has_identity_check": r"本人確認|身分証",
        "text_has_seat": r"アリーナ|スタンド|\d+列|ゲート",
        "text_has_urgent": r"急ぎ|至急|即決|値下げ",
    }
    for column, pattern in patterns.items():
        result[column] = text.str.contains(pattern, case=False, na=False, regex=True).astype(float)
    result["description_length"] = description.str.len().clip(upper=1000).astype(float)
    normalized = description.str.replace(r"\s+", "", regex=True).str.lower()
    # Empty descriptions must not collapse thousands of unrelated tickets into
    # one validation group.  Non-empty repeated descriptions are kept together.
    normalized = normalized.where(normalized.ne(""), "ticket:" + result["ticket_id"].astype(str))
    result["duplicate_group"] = normalized.map(
        lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()
    )
    for column in [
        "price", "quantity", "seller_rating", "capacity", "base_price",
        "seat_rule", "total_stages", "fc_members",
    ]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def attach_complete_fair_price(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach Model16-like fair prices only when coverage is complete.

    Partial coverage would identify the sold training population and leak the
    target into the demand model, so it is rejected rather than filled.
    """
    result = frame.copy()
    if not FAIR_PRICE_CACHE.exists():
        result["fair_price"] = np.nan
        return result
    cache = pd.read_csv(FAIR_PRICE_CACHE)
    if not {"ticket_id", "fair_price"}.issubset(cache):
        raise ValueError("fair_price cache must contain ticket_id,fair_price")
    if cache["ticket_id"].isna().any() or cache["ticket_id"].duplicated().any():
        raise ValueError("fair_price cache contains empty or duplicate ticket_id values")
    missing = set(result.ticket_id) - set(cache.ticket_id)
    if missing:
        raise ValueError(
            f"Partial fair-price cache is forbidden: {len(missing):,} tickets missing"
        )
    result = result.merge(
        cache[["ticket_id", "fair_price"]], on="ticket_id", how="left",
        validate="one_to_one",
    )
    result["fair_price"] = pd.to_numeric(result["fair_price"], errors="coerce")
    if result["fair_price"].isna().any() or (result["fair_price"] <= 0).any():
        raise ValueError("fair_price cache contains invalid values")
    return result


def _semantic_text_hash(value) -> str:
    normalized = str(value or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def attach_complete_semantics(frame: pd.DataFrame, required=REQUIRE_SEMANTIC_FEATURES) -> pd.DataFrame:
    """Attach target-free semantics only with complete all-description coverage."""
    result = frame.copy()
    if not SEMANTIC_FEATURES_FILE.exists() or not SEMANTIC_MANIFEST_FILE.exists():
        if required:
            raise FileNotFoundError(
                "Complete semantic data is required. Run "
                "decision_support_models/semantic_data_builder/extract_semantic_json.py first."
            )
        for column in SEMANTIC_FEATURES:
            result[column] = 0 if column in {"semantic_is_fc_early", "semantic_is_random"} else "不明"
        return result
    manifest = json.loads(SEMANTIC_MANIFEST_FILE.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("schema_version") != SEMANTIC_SCHEMA_VERSION:
        raise ValueError("Semantic manifest is incomplete or has the wrong schema")
    if manifest.get("parse_errors", 0) / max(manifest.get("unique_descriptions", 1), 1) > SEMANTIC_MAX_PARSE_ERROR_RATE:
        raise ValueError("Semantic parse-error rate exceeds the quality gate")
    semantic = pd.read_csv(SEMANTIC_FEATURES_FILE, dtype={"text_hash": str})
    required_columns = {"text_hash", *SEMANTIC_FEATURES, "semantic_schema_version"}
    if not required_columns.issubset(semantic):
        raise ValueError(f"Semantic data missing columns: {sorted(required_columns - set(semantic))}")
    if semantic.text_hash.duplicated().any() or not semantic.semantic_schema_version.eq(SEMANTIC_SCHEMA_VERSION).all():
        raise ValueError("Semantic data has duplicate hashes or mixed schema versions")
    description = result.get("raw_description", pd.Series("", index=result.index)).fillna("")
    result["semantic_text_hash"] = description.map(_semantic_text_hash)
    result = result.merge(
        semantic[["text_hash"] + SEMANTIC_FEATURES],
        left_on="semantic_text_hash", right_on="text_hash", how="left", validate="many_to_one",
    ).drop(columns="text_hash")
    missing = result[SEMANTIC_FEATURES].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Partial semantic coverage is forbidden: {int(missing.sum()):,} ticket rows missing")
    for column in ("semantic_is_fc_early", "semantic_is_random"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def load_tickets(data_dir: Path | None = None) -> pd.DataFrame:
    selected = data_dir or latest_data_dir()
    frames = []
    for path in sorted(selected.glob("*_master.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["group_slug"] = path.name.removesuffix("_master.csv")
        frames.append(frame)
    if not frames:
        raise ValueError(f"No non-empty master CSV found in {selected}")
    result = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"Missing required ticket columns: {sorted(missing)}")
    for column in ["perf_date", "first_observed_at", "last_observed_at", "sold_at"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["status"] = result["status"].astype(str).str.lower()
    unknown = set(result["status"].dropna()) - ALLOWED_STATUS
    if unknown:
        raise ValueError(f"Unknown status values: {sorted(unknown)}")
    created = result.get("created_at_unix", pd.Series("", index=result.index)).fillna("").astype(str).str.strip()
    event = result["event_id"].fillna("").astype(str).str.strip()
    result["_logical_id"] = "ticket:" + result["ticket_id"].astype(str)
    stable = created.ne("") & event.ne("")
    result.loc[stable, "_logical_id"] = "created:" + event[stable] + "|" + created[stable]
    rotated = int(result.groupby("_logical_id")["ticket_id"].nunique().gt(1).sum())
    result["_status_priority"] = result["status"].map(
        {"deleted": 0, "listing": 1, "sold": 2}
    )
    result = (
        result.sort_values(
            ["_logical_id", "last_observed_at", "_status_priority", "ticket_id"],
            na_position="first",
        )
        .drop_duplicates("_logical_id", keep="last")
        .drop(columns=["_status_priority", "_logical_id"])
        .reset_index(drop=True)
    )
    if "sold_at_source" in result:
        unknown_sale_time = result["status"].eq("sold") & result["sold_at_source"].fillna("").eq("historical_unknown")
        excluded_unknown_sales = int(unknown_sale_time.sum())
        result = result.loc[~unknown_sale_time].copy()
    else:
        excluded_unknown_sales = 0
    if result["ticket_id"].isna().any() or result["first_observed_at"].isna().any():
        raise ValueError("ticket_id and first_observed_at must be present")
    if (result["status"].eq("sold") & result["sold_at"].isna()).any():
        raise ValueError("Every sold ticket must have sold_at")
    result = merge_manual_masters(result)
    result = add_static_features(result)
    invalid_price = result["price"].isna() | result["price"].le(0)
    invalid_price_ids = result.loc[invalid_price, "ticket_id"].astype(str).tolist()
    if invalid_price_ids:
        # Keep their valid state labels and all other features, but never let a
        # scraper placeholder price of zero become a deleted-status proxy.
        result.loc[invalid_price, "price"] = np.nan
        print(
            f"[demand] replaced {len(invalid_price_ids):,} non-positive prices "
            "with fold-imputed missing values",
            flush=True,
        )
    result = attach_complete_semantics(result)
    result = attach_complete_fair_price(result)
    result.attrs["snapshot_dir"] = str(selected)
    result.attrs["invalid_listing_price_rows"] = len(invalid_price_ids)
    result.attrs["invalid_listing_price_ticket_ids"] = invalid_price_ids
    result.attrs["invalid_listing_price_policy"] = "set_nan_then_fold_local_median_imputation"
    result.attrs["rotated_logical_listing_ids"] = rotated
    result.attrs["excluded_unknown_sale_time_rows"] = excluded_unknown_sales
    return result
