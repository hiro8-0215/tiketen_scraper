"""Self-contained common-data loader for this model only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_ROOT, MANUAL_DIR, REQUIRED_COLUMNS, REQUIRE_SEMANTIC_FEATURES,
    SEMANTIC_FEATURES, SEMANTIC_FEATURES_FILE, SEMANTIC_MANIFEST_FILE,
    SEMANTIC_SCHEMA_VERSION,
    SEMANTIC_MAX_PARSE_ERROR_RATE,
)


def snapshot_key(path: Path):
    values = path.name.removeprefix("data_").split("_")
    if len(values) == 2 and all(value.isdigit() for value in values):
        return (0, int(values[0]), int(values[1]))
    if len(values) == 3 and all(value.isdigit() for value in values):
        return tuple(map(int, values))
    return (-1, -1, -1)


def latest_data_dir() -> Path:
    choices = [path for path in DATA_ROOT.glob("data_*") if path.is_dir() and any(path.glob("*_master.csv"))]
    if not choices:
        raise FileNotFoundError(f"No snapshot under {DATA_ROOT}")
    return max(choices, key=snapshot_key)


def _read_manual(name: str) -> pd.DataFrame:
    path = MANUAL_DIR / f"master_{name}.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _merge_manual(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    venue = _read_manual("venue")
    if {"venue", "capacity"}.issubset(venue.columns):
        result = result.merge(venue[["venue", "capacity"]].drop_duplicates("venue"), on="venue", how="left", validate="many_to_one")
    tour = _read_manual("tour")
    columns = [name for name in ("event_id", "venue", "base_price", "seat_rule", "total_stages", "artist_id") if name in tour]
    if {"event_id", "venue"}.issubset(columns):
        result = result.merge(tour[columns].drop_duplicates(["event_id", "venue"]), on=["event_id", "venue"], how="left", validate="many_to_one")
    artist = _read_manual("artist")
    if "artist_id" in result and {"artist_id", "fc_members"}.issubset(artist.columns):
        result = result.merge(artist[["artist_id", "fc_members"]].drop_duplicates("artist_id"), on="artist_id", how="left", validate="many_to_one")
    return result


def _semantic_hash(value):
    return hashlib.sha256(str(value or "").strip().encode("utf-8")).hexdigest()


def attach_complete_semantics(frame: pd.DataFrame, required=REQUIRE_SEMANTIC_FEATURES):
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
        raise ValueError("Semantic manifest is incomplete or incompatible")
    if manifest.get("parse_errors", 0) / max(manifest.get("unique_descriptions", 1), 1) > SEMANTIC_MAX_PARSE_ERROR_RATE:
        raise ValueError("Semantic parse-error rate exceeds the quality gate")
    semantic = pd.read_csv(SEMANTIC_FEATURES_FILE, dtype={"text_hash": str})
    required_columns = {"text_hash", *SEMANTIC_FEATURES, "semantic_schema_version"}
    if not required_columns.issubset(semantic):
        raise ValueError(f"Semantic data missing columns: {sorted(required_columns - set(semantic))}")
    if semantic.text_hash.duplicated().any() or not semantic.semantic_schema_version.eq(SEMANTIC_SCHEMA_VERSION).all():
        raise ValueError("Semantic hashes/schema are invalid")
    description = result.get("raw_description", pd.Series("", index=result.index)).fillna("")
    result["semantic_text_hash"] = description.map(_semantic_hash)
    result = result.merge(
        semantic[["text_hash"] + SEMANTIC_FEATURES], left_on="semantic_text_hash",
        right_on="text_hash", how="left", validate="many_to_one",
    ).drop(columns="text_hash")
    if result[SEMANTIC_FEATURES].isna().any(axis=1).any():
        raise ValueError("Partial semantic coverage is forbidden")
    for column in ("semantic_is_fc_early", "semantic_is_random"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def load_tickets(data_dir: Path | None = None) -> pd.DataFrame:
    selected = data_dir or latest_data_dir()
    frames = []
    for path in sorted(selected.glob("*_master.csv")):
        value = pd.read_csv(path, low_memory=False)
        if not value.empty:
            value["group_slug"] = path.name.removesuffix("_master.csv")
            frames.append(value)
    if not frames:
        raise ValueError(f"No master CSV in {selected}")
    result = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(result)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for column in ("first_observed_at", "last_observed_at", "sold_at", "perf_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["status"] = result["status"].astype(str).str.lower()
    unknown = set(result["status"]) - {"listing", "sold", "deleted"}
    if unknown:
        raise ValueError(f"Unknown status: {sorted(unknown)}")
    result["_status_priority"] = result["status"].map(
        {"listing": 0, "deleted": 1, "sold": 2}
    )
    result = (
        result.sort_values(
            ["ticket_id", "last_observed_at", "_status_priority"],
            na_position="first",
        )
        .drop_duplicates("ticket_id", keep="last")
        .drop(columns="_status_priority")
        .reset_index(drop=True)
    )
    if (result["status"].eq("sold") & result["sold_at"].isna()).any():
        raise ValueError("sold ticket without sold_at")
    result = _merge_manual(result)
    date = result["perf_date"].dt.normalize()
    times = result.get("perf_time", pd.Series("", index=result.index)).fillna("").astype(str).str.extract(r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?")
    result["performance_at"] = date + pd.to_timedelta(pd.to_numeric(times.h, errors="coerce").fillna(23).clip(0, 23), unit="h") + pd.to_timedelta(pd.to_numeric(times.m, errors="coerce").fillna(59).clip(0, 59), unit="m")
    result["perf_day_of_week"] = result["performance_at"].dt.dayofweek
    result["perf_month"] = result["performance_at"].dt.month
    result["is_weekend"] = result["perf_day_of_week"].isin([5, 6]).astype(float)
    result["perf_day_sin"] = np.sin(2 * np.pi * result["perf_day_of_week"] / 7)
    result["perf_day_cos"] = np.cos(2 * np.pi * result["perf_day_of_week"] / 7)
    description = result.get("raw_description", pd.Series("", index=result.index)).fillna("").astype(str)
    normalized = description.str.replace(r"\s+", "", regex=True).str.lower()
    normalized = normalized.where(normalized.ne(""), "ticket:" + result["ticket_id"].astype(str))
    result["duplicate_group"] = normalized.map(lambda value: hashlib.sha1(value.encode()).hexdigest())
    result["description_length"] = description.str.len().clip(upper=1000).astype(float)
    for name, pattern in {
        "text_has_fc": r"FC|ファンクラブ|初期当選", "text_has_seat": r"アリーナ|スタンド|\d+列|ゲート",
        "text_has_identity_check": r"本人確認|身分証", "text_has_urgent": r"急ぎ|即決|値下げ",
    }.items():
        result[name] = description.str.contains(pattern, case=False, regex=True, na=False).astype(float)
    for column in ("price", "quantity", "seller_rating", "capacity", "base_price", "seat_rule", "total_stages", "fc_members"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    invalid_price = result["price"].isna() | result["price"].le(0)
    invalid_price_ids = result.loc[invalid_price, "ticket_id"].astype(str).tolist()
    if invalid_price_ids:
        # A cheaper-alternative label cannot be defined from a zero/unknown
        # current price, and zero-price candidates are scraper placeholders.
        result = result.loc[~invalid_price].copy()
        print(
            f"[alternative] excluded {len(invalid_price_ids):,} tickets with "
            "non-positive prices",
            flush=True,
        )
    result = attach_complete_semantics(result)
    result.attrs["snapshot_dir"] = str(selected)
    result.attrs["invalid_listing_price_rows"] = len(invalid_price_ids)
    result.attrs["invalid_listing_price_ticket_ids"] = invalid_price_ids
    result.attrs["invalid_listing_price_policy"] = "excluded_from_price_comparison"
    return result
