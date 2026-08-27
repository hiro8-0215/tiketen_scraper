"""Model16 data adapter over Model15's audited sold-only population."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import FORBIDDEN_MODEL_COLUMNS, MODEL15_DIR


if str(MODEL15_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL15_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "_model15_data_loader", MODEL15_DIR / "data_loader.py"
)
_MODEL15 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODEL15)


CODE_CATEGORICAL_SOURCES = [
    "seat_rule", "perf_day_of_week", "perf_month", "perf_hour_numeric",
    "random_type", "doukou_type", "tousen_type", "seat_type",
]
ENGINEERED_CATEGORICAL = [
    "delivery_channel", "delivery_timing",
] + [f"{column}_category" for column in CODE_CATEGORICAL_SOURCES]
ENGINEERED_NUMERIC = [
    "delivery_text_length",
    "delivery_digit_count",
    "delivery_has_refund",
    "event_prior_available",
    "log_event_prior_sold_count",
    "log_event_listings_seen_before",
    "prior_median_to_base_price",
    "prior_mean_to_base_price",
    "prior_mean_to_median",
    "log_capacity",
    "log_fc_members",
    "log_total_stages",
    "perf_day_sin", "perf_day_cos",
    "perf_month_sin", "perf_month_cos",
    "perf_hour_sin", "perf_hour_cos",
]


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
        days = int(match.group(1))
        return "2-3日前" if days <= 3 else "4日以上前"
    return "時期不明"


def _safe_ratio(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return numerator / denominator


def prepare_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Load Model13-clean sold rows and add listing-time-only global features."""
    df = _MODEL15.prepare_dataset(data_dir).copy()
    delivery = df.get("delivery_method", pd.Series("", index=df.index)).fillna("").astype(str)
    df["delivery_channel"] = delivery.map(_delivery_channel)
    df["delivery_timing"] = delivery.map(_delivery_timing)
    df["delivery_text_length"] = delivery.str.len().clip(upper=500).astype(float)
    df["delivery_digit_count"] = delivery.str.count(r"\d").astype(float)
    df["delivery_has_refund"] = delivery.str.contains("返金", regex=False).astype(float)
    for source in CODE_CATEGORICAL_SOURCES:
        values = pd.to_numeric(
            df.get(source, pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        df[f"{source}_category"] = values.map(
            lambda value: "__missing__" if pd.isna(value) else str(int(value))
        )

    day = pd.to_numeric(df.get("perf_day_of_week"), errors="coerce")
    month = pd.to_numeric(df.get("perf_month"), errors="coerce")
    hour = pd.to_numeric(df.get("perf_hour_numeric"), errors="coerce")
    df["perf_day_sin"] = np.sin(2 * np.pi * day / 7)
    df["perf_day_cos"] = np.cos(2 * np.pi * day / 7)
    df["perf_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["perf_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["perf_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["perf_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    prior_count = pd.to_numeric(
        df.get("event_prior_sold_count", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    listings_seen = pd.to_numeric(
        df.get("event_listings_seen_before", pd.Series(0, index=df.index)), errors="coerce"
    ).fillna(0)
    df["event_prior_available"] = prior_count.gt(0).astype(float)
    df["log_event_prior_sold_count"] = np.log1p(prior_count.clip(lower=0))
    df["log_event_listings_seen_before"] = np.log1p(listings_seen.clip(lower=0))
    df["prior_median_to_base_price"] = _safe_ratio(
        df.get("event_prior_sold_median"), df.get("base_price")
    )
    df["prior_mean_to_base_price"] = _safe_ratio(
        df.get("event_prior_sold_mean"), df.get("base_price")
    )
    df["prior_mean_to_median"] = _safe_ratio(
        df.get("event_prior_sold_mean"), df.get("event_prior_sold_median")
    )
    for source, target in [
        ("capacity", "log_capacity"),
        ("fc_members", "log_fc_members"),
        ("total_stages", "log_total_stages"),
    ]:
        values = pd.to_numeric(
            df.get(source, pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        df[target] = np.log1p(values.clip(lower=0))
    return df


def model_feature_columns(df: pd.DataFrame):
    numeric, categorical = _MODEL15.model_feature_columns(df)
    categorical = list(dict.fromkeys(categorical + [c for c in ENGINEERED_CATEGORICAL if c in df]))
    numeric = list(dict.fromkeys(numeric + [c for c in ENGINEERED_NUMERIC if c in df]))
    # Remove globally constant inputs such as tag_jyouken_ari in this snapshot.
    # This uses feature values only, never the target.
    numeric = [column for column in numeric if df[column].nunique(dropna=False) > 1]
    categorical = [
        column for column in categorical if df[column].nunique(dropna=False) > 1
    ]
    leaked = set(numeric + categorical) & FORBIDDEN_MODEL_COLUMNS
    if leaked:
        raise AssertionError(f"Forbidden Model16 features: {sorted(leaked)}")
    return numeric, categorical


def catboost_frame(df: pd.DataFrame, numeric, categorical) -> pd.DataFrame:
    result = df[numeric + categorical].copy()
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    for column in categorical:
        result[column] = result[column].fillna("__missing__").astype(str)
    return result


latest_data_dir = _MODEL15.latest_data_dir
load_snapshot = _MODEL15.load_snapshot
clean_model13_population = _MODEL15.clean_model13_population
