"""Prompt, normalization, and stable input hashing."""
from __future__ import annotations

import hashlib
import json
import re

from config import SCHEMA_VERSION, SEMANTIC_CATEGORICAL, SEMANTIC_NUMERIC


ALLOWED = {
    "semantic_seat_level": {"アリーナ", "スタンド", "バルコニー", "立見", "不明"},
    "semantic_row_position": {"最前", "前方", "中列", "後方", "最後列", "不明"},
    "semantic_winning_route": {"FC初期", "復活", "制作開放", "一般", "不明"},
    "semantic_name_status": {"本人名義", "他人名義", "男性名義", "女性名義", "名義変更可", "不明"},
    "semantic_identity_check": {"対応可", "対応不可", "確認あり", "確認なし", "不明"},
    "semantic_distribution_type": {"単独", "同行", "ランダム", "番手選択", "QR譲渡", "不明"},
    "semantic_visibility": {"通常", "注釈", "見切れ", "機材開放", "不明"},
}

CODE_FIELDS = (
    ("semantic_seat_level", ("不明", "アリーナ", "スタンド", "バルコニー", "立見")),
    ("semantic_row_position", ("不明", "最前", "前方", "中列", "後方", "最後列")),
    ("semantic_winning_route", ("不明", "FC初期", "復活", "制作開放", "一般")),
    ("semantic_name_status", ("不明", "本人名義", "他人名義", "男性名義", "女性名義", "名義変更可")),
    ("semantic_identity_check", ("不明", "対応可", "対応不可", "確認あり", "確認なし")),
    ("semantic_distribution_type", ("不明", "単独", "同行", "ランダム", "番手選択", "QR譲渡")),
    ("semantic_visibility", ("不明", "通常", "注釈", "見切れ", "機材開放")),
    ("semantic_is_fc_early", (0, 1)),
    ("semantic_is_random", (0, 1)),
)

# A compact fixed-order classification is substantially faster and more robust
# than asking a 7B model to freely generate a long JSON object.  The cache still
# stores the same named schema after deterministic decoding.
SYSTEM_PROMPT = """チケット説明文に明記された意味だけを9個の整数へ分類してください。
価格・需要・売れる確率・将来状態は推定禁止。未記載は0。
順序とコード:
S座席=0不明,1アリーナ,2スタンド,3バルコニー,4立見
R列位置=0不明,1最前,2前方,3中列,4後方,5最後列
W当選=0不明,1FC初期,2復活,3制作開放,4一般
N名義=0不明,1本人,2他人,3男性,4女性,5変更可
I本人確認=0不明,1対応可,2対応不可,3確認あり,4確認なし
D配布=0不明,1単独,2同行,3ランダム,4番手選択,5QR譲渡
V視界=0不明,1通常,2注釈,3見切れ,4機材開放
F=FC初期なら1、それ以外0
L=ランダム配布なら1、それ以外0
回答は必ず [S,R,W,N,I,D,V,F,L] の整数配列1個だけ。説明文は禁止。"""


def normalize_text(value) -> str:
    return str(value or "").strip()


def text_hash(value) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def prompt(description: str, retry: bool = False):
    instruction = SYSTEM_PROMPT
    if retry:
        instruction += "\n形式エラーの再試行です。先頭を数字、末尾を]として9整数だけを返してください。"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": normalize_text(description)},
    ]


def _binary(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "なし"}:
            return 0
        if lowered in {"1", "true", "yes", "あり"}:
            return 1
    return int(bool(value))


def normalize(payload: dict, source: str) -> dict:
    clean = {
        key: str(payload.get(key, "不明")) if str(payload.get(key, "不明")) in allowed else "不明"
        for key, allowed in ALLOWED.items()
    }
    clean["semantic_is_fc_early"] = _binary(payload.get("semantic_is_fc_early", 0))
    clean["semantic_is_random"] = _binary(payload.get("semantic_is_random", 0))
    clean["semantic_source"] = source
    clean["semantic_schema_version"] = SCHEMA_VERSION
    return clean


def unknown(source="empty_description") -> dict:
    return normalize({}, source)


def parse_response(text: str) -> dict:
    # Prefer the compact response.  Search every bracket pair because a model
    # may occasionally add a short prefix despite assistant prefill.
    translated = str(text).translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    for match in re.finditer(r"\[\s*([0-9\s,;；|｜]+)\s*\]", translated):
        values = [int(value) for value in re.findall(r"\d+", match.group(1))]
        # Qwen 2.5 occasionally returns the seven categorical fields plus F,
        # omitting the final redundant L flag. The first seven positions stay
        # aligned (verified against the persisted failure log). F and L are
        # definitions derived from W and D, so recovering them is lossless.
        if len(values) == len(CODE_FIELDS) - 1:
            categorical_fields = CODE_FIELDS[:7]
            if any(
                value < 0 or value >= len(choices)
                for value, (_, choices) in zip(values[:7], categorical_fields)
            ) or values[7] not in {0, 1}:
                continue
            payload = {
                name: choices[value]
                for value, (name, choices) in zip(values[:7], categorical_fields)
            }
            payload["semantic_is_fc_early"] = int(
                payload["semantic_winning_route"] == "FC初期"
            )
            payload["semantic_is_random"] = int(
                payload["semantic_distribution_type"] == "ランダム"
            )
            return normalize(payload, "qwen_semantic_compact_recovered8")
        if len(values) != len(CODE_FIELDS):
            continue
        if any(value < 0 or value >= len(choices) for value, (_, choices) in zip(values, CODE_FIELDS)):
            continue
        payload = {
            name: choices[value]
            for value, (name, choices) in zip(values, CODE_FIELDS)
        }
        return normalize(payload, "qwen_semantic_compact")

    # Retain backward compatibility for already-tested JSON responses and make
    # forbidden price/status keys fail closed.
    start = text.find("{")
    if start < 0:
        raise ValueError("JSON object not found")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise ValueError("Semantic response must be an object")
    forbidden = {key for key in payload if "price" in key.lower() or key in {"status", "sold_at", "deleted_at"}}
    if forbidden:
        raise ValueError(f"Forbidden semantic keys: {sorted(forbidden)}")
    return normalize(payload, "qwen_semantic")


def validate_record(record: dict):
    for key in SEMANTIC_CATEGORICAL:
        if record.get(key) not in ALLOWED[key]:
            raise ValueError(f"Invalid {key}: {record.get(key)!r}")
    for key in SEMANTIC_NUMERIC:
        if record.get(key) not in {0, 1}:
            raise ValueError(f"Invalid {key}: {record.get(key)!r}")
