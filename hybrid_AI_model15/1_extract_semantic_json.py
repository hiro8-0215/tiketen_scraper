"""Resumable target-free Qwen extraction of explicit ticket semantics."""
from __future__ import annotations
import argparse
import gc
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.80",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    QWEN_GPU_INDEX, QWEN_GPU_MEMORY_FRACTION, QWEN_MODEL,
    SEMANTIC_FEATURES_FILE,
)
from data_loader import clean_model13_population, load_snapshot

SEMANTIC_SCHEMA_VERSION = "model15_semantic_v1"

ALLOWED = {
    "semantic_seat_level": {"アリーナ", "スタンド", "バルコニー", "立見", "不明"},
    "semantic_row_position": {"最前", "前方", "中列", "後方", "最後列", "不明"},
    "semantic_winning_route": {"FC初期", "復活", "制作開放", "一般", "不明"},
    "semantic_name_status": {"本人名義", "他人名義", "男性名義", "女性名義", "名義変更可", "不明"},
    "semantic_identity_check": {"対応可", "対応不可", "確認あり", "確認なし", "不明"},
    "semantic_distribution_type": {"単独", "同行", "ランダム", "番手選択", "QR譲渡", "不明"},
    "semantic_visibility": {"通常", "注釈", "見切れ", "機材開放", "不明"},
}


def prompt(description):
    return [
        {"role": "system", "content": """チケット説明文から、出品時点で分かる意味情報だけを抽出してください。価格を推定してはいけません。JSON以外は出力しないでください。
出力キー:
semantic_seat_level: アリーナ/スタンド/バルコニー/立見/不明
semantic_row_position: 最前/前方/中列/後方/最後列/不明
semantic_winning_route: FC初期/復活/制作開放/一般/不明
semantic_name_status: 本人名義/他人名義/男性名義/女性名義/名義変更可/不明
semantic_identity_check: 対応可/対応不可/確認あり/確認なし/不明
semantic_distribution_type: 単独/同行/ランダム/番手選択/QR譲渡/不明
semantic_visibility: 通常/注釈/見切れ/機材開放/不明
semantic_is_fc_early: 0または1
semantic_is_random: 0または1
semantic_confidence: 0.0から1.0"""},
        {"role": "user", "content": str(description)},
    ]


def normalize(row):
    clean = {}
    for key, allowed in ALLOWED.items():
        value = str(row.get(key, "不明"))
        clean[key] = value if value in allowed else "不明"
    def binary(value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"0", "false", "no", "なし"}:
                return 0
            if value in {"1", "true", "yes", "あり"}:
                return 1
        return int(bool(value))

    clean["semantic_is_fc_early"] = binary(row.get("semantic_is_fc_early", 0))
    clean["semantic_is_random"] = binary(row.get("semantic_is_random", 0))
    try:
        confidence = float(row.get("semantic_confidence", 0))
        clean["semantic_confidence"] = min(1.0, max(0.0, confidence))
    except (TypeError, ValueError):
        clean["semantic_confidence"] = 0
    clean["semantic_available"] = 1
    clean["semantic_source"] = "qwen15"
    clean["semantic_schema_version"] = SEMANTIC_SCHEMA_VERSION
    return clean


def parse_json(text):
    # Decode the first complete object instead of greedily spanning multiple
    # objects/code fences, which caused avoidable parse_error rows.
    start = text.find("{")
    if start < 0:
        raise ValueError("JSON object not found")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise ValueError("Semantic response is not a JSON object")
    return normalize(payload)


def save_semantics(payload):
    SEMANTIC_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SEMANTIC_FEATURES_FILE.with_name("semantic_features.model15.tmp.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, SEMANTIC_FEATURES_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--refresh-legacy", action="store_true", help="legacy entriesも拡張schemaで再抽出")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="GPU生成batch。CUDA OOM時は自動で半分に下げる")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    raw = clean_model13_population(load_snapshot())
    descriptions = raw["raw_description"].astype(str).drop_duplicates().tolist()
    existing = {} if args.reset or not SEMANTIC_FEATURES_FILE.exists() else json.loads(SEMANTIC_FEATURES_FILE.read_text(encoding="utf-8"))
    for cached in existing.values():
        cached.pop("price_estimate", None)
        cached.pop("semantic_price_estimate", None)
    remaining = [
        description for description in descriptions
        if description not in existing
        or (
            args.refresh_legacy
            and (
                existing[description].get("semantic_source") != "qwen15"
                or existing[description].get("semantic_schema_version") != SEMANTIC_SCHEMA_VERSION
            )
        )
    ]
    print(f"descriptions={len(descriptions):,}, remaining={len(remaining):,}")
    if not remaining:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Model15意味抽出にはCUDA対応NVIDIA GPUが必要です")
    torch.cuda.set_device(QWEN_GPU_INDEX)
    torch.cuda.set_per_process_memory_fraction(QWEN_GPU_MEMORY_FRACTION, QWEN_GPU_INDEX)
    print(f"CUDA device {QWEN_GPU_INDEX}: {torch.cuda.get_device_name(QWEN_GPU_INDEX)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        quantization_config=quant,
        device_map={"": QWEN_GPU_INDEX},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    device_map = getattr(model, "hf_device_map", {}) or {}
    if any(str(device).lower() in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"CPU/disk offload detected and disabled: {device_map}")
    model.eval()
    errors = 0
    processed = 0
    batch_size = args.batch_size
    progress = tqdm(total=len(remaining), desc="semantic JSON")
    try:
        while processed < len(remaining):
            descriptions = remaining[processed:processed + batch_size]
            rendered = [tokenizer.apply_chat_template(prompt(description), tokenize=False,
                                                       add_generation_prompt=True)
                        for description in descriptions]
            try:
                inputs = tokenizer(rendered, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to("cuda")
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=220, do_sample=False,
                                            use_cache=True, pad_token_id=tokenizer.pad_token_id)
                responses = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                                   skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                if "inputs" in locals():
                    del inputs
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                progress.write(f"CUDA OOM: batch-sizeを{batch_size}へ自動調整して再試行")
                continue

            for description, response in zip(descriptions, responses):
                try:
                    existing[description] = parse_json(response)
                except Exception:
                    errors += 1
                    existing[description] = normalize({})
                    existing[description]["semantic_source"] = "parse_error"
                    existing[description]["semantic_available"] = 0
            processed += len(descriptions)
            progress.update(len(descriptions))
            del inputs, output
            if processed % 20 < len(descriptions):
                save_semantics(existing)
    finally:
        progress.close()
        save_semantics(existing)
        print(f"saved={len(existing):,}, errors={errors:,}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
