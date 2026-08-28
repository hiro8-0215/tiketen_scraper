"""Build complete all-ticket price features from the completed Model 16.

Clean Model 16 training rows use their saved OOF predictions.  Every other row
uses the frozen production ensemble.  This prevents the demand model from seeing
an in-sample prediction for Model 16's own sold training population.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent
DECISION_DIR = MODEL_DIR.parent
PROJECT_ROOT = DECISION_DIR.parent
DATA_ROOT = PROJECT_ROOT / "tiketen_date_data"
MANUAL_DIR = PROJECT_ROOT / "手動_data"
MODEL16_DIR = PROJECT_ROOT / "hybrid_AI_model16"
MODEL16_ARTIFACT = MODEL16_DIR / "artifacts" / "model16.joblib"
MODEL16_OOF = MODEL16_DIR / "artifacts" / "oof_predictions_model16.csv"
SEMANTIC_FILE = PROJECT_ROOT / "semantic_feature_data" / "semantic_features.csv"
SEMANTIC_MANIFEST = PROJECT_ROOT / "semantic_feature_data" / "semantic_manifest.json"
OUTPUT_FILE = DECISION_DIR / "demand_state_model" / "artifacts" / "fair_price_all_tickets.csv"
REPORT_FILE = MODEL_DIR / "artifacts" / "bridge_report.json"

EXCLUDE_GROUPS = {"ambitious", "b-and-zai", "banzai", "boys-be"}
MIN_PRICE = 2_000
MAX_PRICE = 150_000
MIN_DESCRIPTION_LENGTH = 5
EVENT_PRICE_LOW_QUANTILE = 0.05
EVENT_PRICE_HIGH_QUANTILE = 0.98
NOISE_DESCRIPTION_PATTERN = (
    r"専用|代理|取り置き|ダミー|相場理解|即決額|手渡し|"
    r"別途\s*(?:定価|支払|決済|負担)|即\s*\d(?:\.\d)?|当日\s*\d(?:\.\d)?"
)
RAW_PRICE_SCALE = 10_000.0
WEIGHT_EPSILON = 1e-8
SEMANTIC_COLUMNS = [
    "semantic_seat_level", "semantic_row_position", "semantic_winning_route",
    "semantic_name_status", "semantic_identity_check",
    "semantic_distribution_type", "semantic_visibility",
    "semantic_is_fc_early", "semantic_is_random",
]


# Model 16 itself inherits this audited Model 13 parser.  Importing it here
# avoids maintaining a subtly different regex copy in the bridge.
MODEL15_DIR = PROJECT_ROOT / "hybrid_AI_model15"
if str(MODEL15_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL15_DIR))
from description_parser import parse_description  # noqa: E402


def snapshot_key(path: Path) -> tuple[int, ...]:
    parts = path.name.removeprefix("data_").split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return 0, int(parts[0]), int(parts[1])
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return tuple(map(int, parts))
    return -1, -1, -1


def latest_data_dir() -> Path:
    candidates = [
        path for path in DATA_ROOT.glob("data_*")
        if path.is_dir()
        and snapshot_key(path)[0] >= 0
        and any(path.glob("*_master.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No ticket snapshot found under {DATA_ROOT}")
    return max(candidates, key=snapshot_key)


def load_snapshot(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("*_master.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty:
            continue
        frame["group_slug"] = path.name.removesuffix("_master.csv")
        frames.append(frame)
    if not frames:
        raise ValueError(f"No non-empty master CSV found in {data_dir}")
    result = pd.concat(frames, ignore_index=True)
    required = {
        "ticket_id", "event_id", "status", "price", "raw_description",
        "first_observed_at", "last_observed_at", "sold_at", "perf_date",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"Snapshot is missing columns: {sorted(missing)}")
    for column in ["perf_date", "first_observed_at", "last_observed_at", "sold_at"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["status"] = result["status"].astype(str).str.lower()
    unknown = set(result["status"]) - {"listing", "sold", "deleted"}
    if unknown:
        raise ValueError(f"Unknown status: {sorted(unknown)}")
    created = result.get("created_at_unix", pd.Series("", index=result.index)).fillna("").astype(str).str.strip()
    event = result["event_id"].fillna("").astype(str).str.strip()
    result["_logical_id"] = "ticket:" + result["ticket_id"].astype(str)
    stable = created.ne("") & event.ne("")
    result.loc[stable, "_logical_id"] = "created:" + event[stable] + "|" + created[stable]
    result["_status_priority"] = result["status"].map({"deleted": 0, "listing": 1, "sold": 2})
    result = (
        result.sort_values(
            ["_logical_id", "last_observed_at", "_status_priority", "ticket_id"],
            na_position="first",
        )
        .drop_duplicates("_logical_id", keep="last")
        .drop(columns=["_status_priority", "_logical_id"])
        .reset_index(drop=True)
    )
    if result.ticket_id.isna().any() or result.first_observed_at.isna().any():
        raise ValueError("ticket_id and first_observed_at are required")
    return result


def clean_model16_population(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[
        snapshot.status.eq("sold") & ~snapshot.group_slug.isin(EXCLUDE_GROUPS)
    ].copy()
    description = frame.raw_description.astype("string")
    frame = frame[
        description.notna() & description.str.strip().str.len().ge(MIN_DESCRIPTION_LENGTH)
    ].copy()
    frame = frame[
        ~frame.raw_description.astype("string").str.contains(
            NOISE_DESCRIPTION_PATTERN, na=False, regex=True
        )
    ].copy()
    frame["price"] = pd.to_numeric(frame.price, errors="coerce")
    frame = frame[frame.price.between(MIN_PRICE, MAX_PRICE, inclusive="both")].copy()
    low = frame.groupby("event_id").price.transform(
        lambda values: values.quantile(EVENT_PRICE_LOW_QUANTILE)
    )
    high = frame.groupby("event_id").price.transform(
        lambda values: values.quantile(EVENT_PRICE_HIGH_QUANTILE)
    )
    frame = frame[frame.price.ge(low) & frame.price.le(high)].copy()
    return (
        frame.sort_values("first_observed_at")
        .drop_duplicates("ticket_id", keep="last")
        .reset_index(drop=True)
    )


def merge_masters(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    venue_path = MANUAL_DIR / "master_venue.csv"
    if venue_path.exists():
        venue = pd.read_csv(venue_path, low_memory=False)
        if {"venue", "capacity"}.issubset(venue):
            result = result.merge(
                venue[["venue", "capacity"]].drop_duplicates("venue"),
                on="venue", how="left", validate="many_to_one",
            )
    tour_path = MANUAL_DIR / "master_tour.csv"
    if tour_path.exists():
        tour = pd.read_csv(tour_path, low_memory=False)
        columns = [
            column for column in [
                "event_id", "venue", "base_price", "lottery_date", "seat_rule",
                "first_day", "last_day", "total_stages",
            ] if column in tour
        ]
        if {"event_id", "venue"}.issubset(columns):
            result = result.merge(
                tour[columns].drop_duplicates(["event_id", "venue"]),
                on=["event_id", "venue"], how="left", validate="many_to_one",
            )
    artist_path = MANUAL_DIR / "master_artist.csv"
    if artist_path.exists():
        artist = pd.read_csv(artist_path, low_memory=False)
        if {"artist_id", "fc_members"}.issubset(artist):
            result = result.merge(
                artist[["artist_id", "fc_members"]].drop_duplicates("artist_id"),
                left_on="group_slug", right_on="artist_id", how="left",
                validate="many_to_one",
            )
    return result


def add_asof_market_features(
    snapshot: pd.DataFrame, targets: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Add market history strictly before each ticket's first observation."""
    result = targets.copy()
    result["event_listings_seen_before"] = 0
    result["event_prior_sold_count"] = 0
    for column in (
        "event_prior_sold_mean", "event_prior_sold_median", "event_prior_sold_logmean"
    ):
        result[column] = np.nan

    for event_id, indices in result.groupby("event_id", dropna=False).groups.items():
        if pd.isna(event_id):
            continue
        event_all = snapshot[snapshot.event_id.eq(event_id)]
        seen_times = (
            event_all.first_observed_at.dropna().sort_values().to_numpy(dtype="datetime64[ns]")
        )
        event_history = history[
            history.event_id.eq(event_id) & history.sold_at.notna()
        ].copy()
        event_history = event_history.sort_values("sold_at")
        sold_times = event_history.sold_at.to_numpy(dtype="datetime64[ns]")
        sold_prices = pd.to_numeric(event_history.price, errors="coerce").to_numpy(float)
        target_times = result.loc[indices, "first_observed_at"].to_numpy(dtype="datetime64[ns]")
        result.loc[indices, "event_listings_seen_before"] = np.searchsorted(
            seen_times, target_times, side="left"
        )
        prior_counts = np.searchsorted(sold_times, target_times, side="left")
        result.loc[indices, "event_prior_sold_count"] = prior_counts
        if len(sold_prices):
            cumulative = np.cumsum(sold_prices)
            cumulative_log = np.cumsum(np.log1p(sold_prices))
            for count in np.unique(prior_counts[prior_counts > 0]):
                selected = np.asarray(indices)[prior_counts == count]
                result.loc[selected, "event_prior_sold_mean"] = cumulative[count - 1] / count
                result.loc[selected, "event_prior_sold_median"] = np.median(sold_prices[:count])
                result.loc[selected, "event_prior_sold_logmean"] = cumulative_log[count - 1] / count
    return result


def merge_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    if not SEMANTIC_FILE.exists() or not SEMANTIC_MANIFEST.exists():
        raise FileNotFoundError("Complete all-ticket semantic data is required")
    manifest = json.loads(SEMANTIC_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("schema_version") != "target_free_semantic_v1":
        raise ValueError("Semantic manifest is incomplete or incompatible")
    semantic = pd.read_csv(SEMANTIC_FILE, dtype={"text_hash": str})
    required = {"text_hash", *SEMANTIC_COLUMNS, "semantic_source"}
    if not required.issubset(semantic):
        raise ValueError(f"Semantic data is missing columns: {sorted(required - set(semantic))}")
    if semantic.text_hash.duplicated().any():
        raise ValueError("Semantic data contains duplicate text_hash values")
    result = frame.copy()
    description = result.raw_description.fillna("").astype(str).str.strip()
    result["semantic_text_hash"] = description.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    result = result.merge(
        semantic[["text_hash", *SEMANTIC_COLUMNS, "semantic_source"]],
        left_on="semantic_text_hash", right_on="text_hash", how="left",
        validate="many_to_one",
    ).drop(columns="text_hash")
    missing = result[SEMANTIC_COLUMNS].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Semantic coverage is incomplete for {int(missing.sum()):,} tickets")
    parse_error = result.semantic_source.eq("parse_error")
    # Match categories seen by the frozen Model 16 while making cache provenance
    # uniform across sold/deleted/listing rows.
    result["semantic_source"] = np.where(parse_error, "parse_error", "qwen15")
    result["semantic_confidence"] = np.where(parse_error, 0.0, 1.0)
    result["semantic_available"] = (~parse_error).astype(float)
    for column in ("semantic_is_fc_early", "semantic_is_random"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def _delivery_channel(text: str) -> str:
    if "同行" in text or "同時入場" in text:
        return "同行・同時入場"
    if "ログイン" in text:
        return "ログイン情報"
    if "QR" in text.upper():
        return "QR共有"
    if "郵送" in text:
        return "郵送"
    if "手渡" in text:
        return "手渡し"
    if "ランダム" in text:
        return "ランダム配布"
    if "発券" in text or "コンビニ" in text:
        return "発券"
    return "その他"


def _delivery_timing(text: str) -> str:
    if "入金後" in text or "即時" in text or "すぐ" in text:
        return "入金後"
    if "当日" in text:
        return "当日"
    if "前日" in text or "1日前" in text:
        return "前日"
    match = re.search(r"(\d+)日(?:前|まで)", text)
    if match:
        return "2-3日前" if int(match.group(1)) <= 3 else "4日以上前"
    return "時期不明"


def _safe_ratio(numerator, denominator):
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return top / bottom


def add_model16_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["lottery_date", "first_day", "last_day"]:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    for column in [
        "quantity", "base_price", "capacity", "fc_members", "total_stages", "seller_rating"
    ]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["days_until_event"] = (
        result.perf_date - result.first_observed_at
    ).dt.total_seconds() / 86400
    if "lottery_date" in result:
        result["days_since_lottery"] = (
            result.first_observed_at - result.lottery_date
        ).dt.total_seconds() / 86400
    result["perf_day_of_week"] = result.perf_date.dt.dayofweek
    result["perf_month"] = result.perf_date.dt.month
    time_text = result.get("perf_time", pd.Series("", index=result.index)).fillna("").astype(str)
    result["perf_hour_numeric"] = pd.to_numeric(
        time_text.str.extract(r"(\d{1,2})", expand=False), errors="coerce"
    )
    result["is_heijitsu"] = result.perf_day_of_week.isin(range(5)).astype(int)
    result["is_weekend"] = result.perf_day_of_week.isin([5, 6]).astype(int)
    result["is_tour_first_day"] = (
        result.perf_date.eq(result.first_day).astype(int) if "first_day" in result else 0
    )
    result["is_tour_last_day"] = (
        result.perf_date.eq(result.last_day).astype(int) if "last_day" in result else 0
    )
    stages = result.get("total_stages", pd.Series(1, index=result.index)).fillna(1).clip(lower=1)
    result["ticket_multiplier"] = result.get("fc_members", np.nan) / (
        result.get("capacity", np.nan) * stages
    ).replace(0, np.nan)
    tags = result.get("ticket_tags", pd.Series("", index=result.index)).fillna("").astype(str)
    result["tag_doukou"] = tags.str.contains("同行", regex=False).astype(int)
    result["tag_jyouken_ari"] = tags.str.contains("条件あり", regex=False).astype(int)
    result = parse_description(result, "raw_description")
    result["bante_x_baseprice"] = result.bante.fillna(0) * result.base_price.fillna(0)
    result["surikae_x_bante"] = result.surikae_nashi * result.bante.fillna(0)

    delivery = result.get("delivery_method", pd.Series("", index=result.index)).fillna("").astype(str)
    result["delivery_channel"] = delivery.map(_delivery_channel)
    result["delivery_timing"] = delivery.map(_delivery_timing)
    result["delivery_text_length"] = delivery.str.len().clip(upper=500).astype(float)
    result["delivery_digit_count"] = delivery.str.count(r"\d").astype(float)
    result["delivery_has_refund"] = delivery.str.contains("返金", regex=False).astype(float)

    code_sources = [
        "seat_rule", "perf_day_of_week", "perf_month", "perf_hour_numeric",
        "random_type", "doukou_type", "tousen_type", "seat_type",
    ]
    for source in code_sources:
        values = pd.to_numeric(result.get(source, np.nan), errors="coerce")
        result[f"{source}_category"] = values.map(
            lambda value: "__missing__" if pd.isna(value) else str(int(value))
        )
    day, month, hour = result.perf_day_of_week, result.perf_month, result.perf_hour_numeric
    result["perf_day_sin"] = np.sin(2 * np.pi * day / 7)
    result["perf_day_cos"] = np.cos(2 * np.pi * day / 7)
    result["perf_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["perf_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    result["perf_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["perf_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    prior_count = pd.to_numeric(result.event_prior_sold_count, errors="coerce").fillna(0)
    listing_count = pd.to_numeric(result.event_listings_seen_before, errors="coerce").fillna(0)
    result["event_prior_available"] = prior_count.gt(0).astype(float)
    result["log_event_prior_sold_count"] = np.log1p(prior_count.clip(lower=0))
    result["log_event_listings_seen_before"] = np.log1p(listing_count.clip(lower=0))
    result["prior_median_to_base_price"] = _safe_ratio(result.event_prior_sold_median, result.base_price)
    result["prior_mean_to_base_price"] = _safe_ratio(result.event_prior_sold_mean, result.base_price)
    result["prior_mean_to_median"] = _safe_ratio(result.event_prior_sold_mean, result.event_prior_sold_median)
    for source, target in [
        ("capacity", "log_capacity"), ("fc_members", "log_fc_members"),
        ("total_stages", "log_total_stages"),
    ]:
        result[target] = np.log1p(pd.to_numeric(result[source], errors="coerce").clip(lower=0))
    return result


def prepare_all_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    history = clean_model16_population(snapshot)
    result = add_asof_market_features(snapshot, snapshot, history)
    result = merge_masters(result)
    result = add_model16_features(result)
    result = merge_semantics(result)
    return result


def _catboost_frame(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    result = frame[numeric + categorical].copy()
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    for column in categorical:
        result[column] = result[column].fillna("__missing__").astype(str)
    return result


def predict_model16(payload: dict, frame: pd.DataFrame, batch_size: int = 20_000) -> np.ndarray:
    numeric = list(payload["numeric"])
    categorical = list(payload["categorical"])
    for column in numeric:
        if column not in frame:
            frame[column] = np.nan
    for column in categorical:
        if column not in frame:
            frame[column] = "__missing__"
    weights = np.asarray(payload["weights"], float)
    if len(weights) != len(payload["expert_order"]) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Invalid Model 16 ensemble weights")
    result = np.zeros(len(frame), float)
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        chunk = frame.iloc[start:stop]
        blended = np.zeros(len(chunk), float)
        for name, weight in zip(payload["expert_order"], weights):
            if weight < WEIGHT_EPSILON:
                continue
            expert = payload["experts"][name]
            if name in {"lgbm_log_mae", "lgbm_raw_mape"}:
                matrix = expert["preprocessor"].transform(chunk)
                native = expert["model"].booster_.predict(matrix)
                prediction = (
                    np.expm1(native) if name == "lgbm_log_mae" else native * RAW_PRICE_SCALE
                )
            elif name == "catboost_raw_mae":
                prediction = expert["model"].predict(
                    _catboost_frame(chunk, numeric, categorical)
                ) * RAW_PRICE_SCALE
            else:
                raise ValueError(f"Positive-weight unsupported Model 16 expert: {name}")
            blended += weight * np.maximum(0, prediction)
        result[start:stop] = blended
        print(f"Model 16 inference: {stop:,}/{len(frame):,}", flush=True)
    if not np.isfinite(result).all():
        raise ValueError("Model 16 predictions contain NaN or infinity")
    return np.maximum(result, 1.0)


def build(data_dir: Path | None = None) -> dict:
    selected = data_dir or latest_data_dir()
    if not MODEL16_ARTIFACT.exists() or not MODEL16_OOF.exists():
        raise FileNotFoundError("Completed Model 16 artifact and OOF predictions are required")
    snapshot = load_snapshot(selected)
    payload = joblib.load(MODEL16_ARTIFACT)
    if payload.get("pipeline_version") != "model16_global_nested_ensemble_v2":
        raise ValueError("Model 16 artifact has an incompatible pipeline version")
    features = prepare_all_features(snapshot)
    production = predict_model16(payload, features)

    oof = pd.read_csv(MODEL16_OOF)
    required = {"ticket_id", "true_price", "pred_global_convex_ensemble"}
    if not required.issubset(oof):
        raise ValueError(f"Model 16 OOF is missing columns: {sorted(required - set(oof))}")
    if oof.ticket_id.duplicated().any():
        raise ValueError("Model 16 OOF contains duplicate ticket IDs")
    oof_prediction = oof.set_index("ticket_id").pred_global_convex_ensemble
    identifier = snapshot.ticket_id.astype(str)
    mapped = identifier.map(oof_prediction)
    use_oof = mapped.notna()
    fair_price = np.where(use_oof, mapped, production)
    absolute_error = np.abs(oof.pred_global_convex_ensemble - oof.true_price)
    error_mae = float(absolute_error.mean())
    error_q80 = float(absolute_error.quantile(0.80))
    cache = pd.DataFrame({
        "ticket_id": identifier,
        "fair_price": np.maximum(1, fair_price),
        "fair_price_source": np.where(use_oof, "model16_oof", "model16_production"),
        "fair_price_error_mae": error_mae,
        "fair_price_error_q80": error_q80,
    })
    if cache.ticket_id.duplicated().any() or cache.fair_price.isna().any():
        raise ValueError("Generated fair-price cache is incomplete")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".csv.tmp")
    cache.to_csv(temporary, index=False)
    os.replace(temporary, OUTPUT_FILE)
    report = {
        "snapshot": str(selected),
        "rows": len(cache),
        "model16_oof_rows": int(use_oof.sum()),
        "model16_production_rows": int((~use_oof).sum()),
        "model16_oof_mae_yen": error_mae,
        "model16_oof_abs_error_q80_yen": error_q80,
        "output": str(OUTPUT_FILE),
        "training_executed": False,
        "llm_executed": False,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report_temp = REPORT_FILE.with_suffix(".json.tmp")
    report_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(report_temp, REPORT_FILE)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.data_dir), ensure_ascii=False, indent=2))
