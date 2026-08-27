"""Sold-only dataset with rich listing-time and strictly-as-of features."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA_ROOT, EXCLUDE_GROUPS, FORBIDDEN_MODEL_COLUMNS, MANUAL_DIR, TARGET
from description_parser import parse_description


def latest_data_dir() -> Path:
    def date_key(p: Path):
        match = re.fullmatch(r"data_(\d+)_(\d+)", p.name)
        return tuple(map(int, match.groups())) if match else (0, 0)
    candidates = [p for p in DATA_ROOT.glob("data_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No data snapshots found in {DATA_ROOT}")
    return max(candidates, key=date_key)


def load_snapshot(data_dir: Path | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted((data_dir or latest_data_dir()).glob("*_master.csv")):
        slug = path.name.removesuffix("_master.csv")
        if slug in EXCLUDE_GROUPS:
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame["group_slug"] = slug
        frames.append(frame)
    if not frames:
        raise ValueError("No usable ticket data")
    df = pd.concat(frames, ignore_index=True)
    for col in ["perf_date", "first_observed_at", "last_observed_at", "sold_at"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _master(name: str) -> pd.DataFrame:
    path = MANUAL_DIR / f"master_{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def merge_masters(df: pd.DataFrame) -> pd.DataFrame:
    venue, tour, artist = _master("venue"), _master("tour"), _master("artist")
    if not venue.empty:
        cols = [c for c in ["venue", "capacity"] if c in venue]
        df = df.merge(venue.drop_duplicates("venue")[cols], on="venue", how="left")
    if not tour.empty:
        cols = [c for c in ["event_id", "venue", "base_price", "lottery_date", "seat_rule", "first_day", "last_day", "total_stages"] if c in tour]
        df = df.merge(tour.drop_duplicates(["event_id", "venue"])[cols], on=["event_id", "venue"], how="left")
    if not artist.empty:
        cols = [c for c in ["artist_id", "fc_members"] if c in artist]
        df = df.merge(artist.drop_duplicates("artist_id")[cols], left_on="group_slug", right_on="artist_id", how="left")
    return df


def add_asof_market_features(snapshot: pd.DataFrame, sold: pd.DataFrame) -> pd.DataFrame:
    """Use only records observable strictly before each listing was first seen."""
    out = sold.copy()
    out["event_listings_seen_before"] = 0
    out["event_prior_sold_count"] = 0
    out["event_prior_sold_mean"] = np.nan
    out["event_prior_sold_median"] = np.nan
    out["event_prior_sold_logmean"] = np.nan

    for event_id, idx in out.groupby("event_id", dropna=False).groups.items():
        targets = out.loc[idx, "first_observed_at"]
        event_all = snapshot[snapshot["event_id"].eq(event_id)]
        seen_times = event_all["first_observed_at"].dropna().sort_values().to_numpy(dtype="datetime64[ns]")
        history = event_all[event_all["status"].eq("sold") & event_all["sold_at"].notna()].copy()
        history[TARGET] = pd.to_numeric(history[TARGET], errors="coerce")
        history = history.dropna(subset=[TARGET]).sort_values("sold_at")
        sold_times = history["sold_at"].to_numpy(dtype="datetime64[ns]")
        sold_prices = history[TARGET].to_numpy(float)
        for row_idx, timestamp in targets.items():
            if pd.isna(timestamp):
                continue
            t64 = np.datetime64(timestamp.to_datetime64())
            out.at[row_idx, "event_listings_seen_before"] = int(np.searchsorted(seen_times, t64, side="left"))
            n = int(np.searchsorted(sold_times, t64, side="left"))
            if n:
                prior = sold_prices[:n]
                out.at[row_idx, "event_prior_sold_count"] = n
                out.at[row_idx, "event_prior_sold_mean"] = prior.mean()
                out.at[row_idx, "event_prior_sold_median"] = np.median(prior)
                out.at[row_idx, "event_prior_sold_logmean"] = np.log1p(prior).mean()
    return out


def prepare_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    snapshot = load_snapshot(data_dir)
    sold = snapshot[snapshot["status"].eq("sold")].copy()
    sold[TARGET] = pd.to_numeric(sold[TARGET], errors="coerce")
    sold = sold[sold[TARGET].gt(0)].copy()  # retain every meaningful positive sold price
    if "ticket_id" in sold:
        sold = sold.sort_values("first_observed_at").drop_duplicates("ticket_id", keep="last")
    sold = add_asof_market_features(snapshot, sold)
    sold = merge_masters(sold)
    for col in ["lottery_date", "first_day", "last_day"]:
        if col in sold:
            sold[col] = pd.to_datetime(sold[col], errors="coerce")
    for col in ["quantity", "base_price", "capacity", "fc_members", "total_stages", "seller_rating"]:
        if col in sold:
            sold[col] = pd.to_numeric(sold[col], errors="coerce")
    if {"perf_date", "first_observed_at"}.issubset(sold):
        sold["days_until_event"] = (sold["perf_date"] - sold["first_observed_at"]).dt.total_seconds() / 86400
    if {"first_observed_at", "lottery_date"}.issubset(sold):
        sold["days_since_lottery"] = (sold["first_observed_at"] - sold["lottery_date"]).dt.total_seconds() / 86400
    if "perf_date" in sold:
        sold["perf_day_of_week"] = sold["perf_date"].dt.dayofweek
        sold["perf_month"] = sold["perf_date"].dt.month
        sold["perf_hour_numeric"] = sold.get("perf_time", "").astype(str).str.extract(r"(\d{1,2})", expand=False).astype(float)
        sold["is_heijitsu"] = sold["perf_day_of_week"].isin(range(5)).astype(int)
        sold["is_weekend"] = sold["perf_day_of_week"].isin([5, 6]).astype(int)
        if "first_day" in sold:
            sold["is_tour_first_day"] = sold["perf_date"].eq(sold["first_day"]).astype(int)
        if "last_day" in sold:
            sold["is_tour_last_day"] = sold["perf_date"].eq(sold["last_day"]).astype(int)
    if {"fc_members", "capacity"}.issubset(sold):
        stages = sold.get("total_stages", pd.Series(1, index=sold.index)).fillna(1).clip(lower=1)
        sold["ticket_multiplier"] = sold["fc_members"] / (sold["capacity"] * stages).replace(0, np.nan)
    tags = sold.get("ticket_tags", pd.Series("", index=sold.index)).fillna("").astype(str)
    sold["tag_doukou"] = tags.str.contains("同行", regex=False).astype(int)
    sold["tag_jyouken_ari"] = tags.str.contains("条件あり", regex=False).astype(int)
    sold = parse_description(sold, "raw_description")
    desc = sold.get("raw_description", pd.Series("", index=sold.index)).fillna("").astype(str)
    sold["model_text"] = desc + " [タグ] " + tags
    normalized = desc.str.replace(r"\s+", "", regex=True).str.lower()
    sold["duplicate_group"] = normalized.map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())
    sold = sold.sort_values(["first_observed_at", "ticket_id"], na_position="first").reset_index(drop=True)
    return sold


def model_feature_columns(df: pd.DataFrame):
    categorical_candidates = [
        "group_slug", "event_id", "venue", "ticket_type", "name_type",
        "delivery_method", "gate_info",
    ]
    excluded = FORBIDDEN_MODEL_COLUMNS | {
        "ticket_id", "created_at_unix", "seller_name", "order_num", "raw_description",
        "details_fetched", "perf_date", "perf_time", "ticket_tags", "first_observed_at",
        "lottery_date", "first_day", "last_day", "artist_id", "artist_name",
        "model_text", "duplicate_group",
    }
    categorical = [c for c in categorical_candidates if c in df]
    numeric = [c for c in df if c not in excluded and c not in categorical and pd.api.types.is_numeric_dtype(df[c])]
    leaked = set(numeric + categorical) & FORBIDDEN_MODEL_COLUMNS
    if leaked:
        raise AssertionError(f"Forbidden model features: {sorted(leaked)}")
    return numeric, categorical

