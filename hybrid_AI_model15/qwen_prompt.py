"""Canonical, versioned Qwen price-regression input and cache fingerprint."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from config import TARGET

QWEN_PROMPT_VERSION = "model15_price_v2"


def value(row, name, default="不明"):
    item = row.get(name)
    return default if pd.isna(item) or str(item) == "" else str(item)


def build_qwen_prompt(row):
    """Use only information that is available at listing time."""
    return (
        "次の売却済みチケットについて、出品時点の情報だけから成立価格を推定してください。\n"
        f"アーティスト:{value(row, 'group_slug')}\n公演:{value(row, 'event_id')}\n"
        f"会場:{value(row, 'venue')}\n公演日:{value(row, 'perf_date')}\n"
        f"定価:{value(row, 'base_price')}\n券種:{value(row, 'ticket_type')}\n"
        f"名義:{value(row, 'name_type')}\n枚数:{value(row, 'quantity')}\n"
        f"出品時点より前の成立中央値:{value(row, 'event_prior_sold_median')}\n"
        f"出品時点までの出品数:{value(row, 'event_listings_seen_before')}\n"
        f"説明:{value(row, 'raw_description', '')}\n"
        f"タグ:{value(row, 'ticket_tags', '')}"
    )


def qwen_input_hashes(df):
    return [
        hashlib.sha256(
            f"{QWEN_PROMPT_VERSION}\n{build_qwen_prompt(row)}".encode("utf-8")
        ).hexdigest()
        for _, row in df.iterrows()
    ]


def qwen_dataset_fingerprint(df, manifest):
    """Invalidate every fold if any input, target, membership, or code version changes."""
    input_hashes = qwen_input_hashes(df)
    records = [
        [
            str(ticket_id),
            int(fold),
            input_hash,
            format(float(price), ".12g"),
        ]
        for ticket_id, fold, input_hash, price in zip(
            df["ticket_id"], manifest["fold"], input_hashes, df[TARGET]
        )
    ]
    encoded = json.dumps(
        {"version": QWEN_PROMPT_VERSION, "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
