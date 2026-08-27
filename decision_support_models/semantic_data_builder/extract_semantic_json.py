"""Resumable all-description Qwen extraction. Runs only when explicitly invoked."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.80")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList,
)

from config import (
    COMPATIBLE_LEGACY_SCHEMA, CUDA_DEVICE, DEFAULT_BATCH_SIZE,
    FAILURE_LOG_FILE, GPU_MEMORY_FRACTION, LEGACY_MODEL15_FILE, MANIFEST_FILE,
    MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, OUTPUT_DIR, OUTPUT_FILE, QWEN_MODEL,
    RETRY_OUTPUT_TOKENS, SAVE_EVERY, SCHEMA_VERSION, SEMANTIC_FEATURES,
)
from data_loader import load_descriptions
from schema import normalize, parse_response, prompt, text_hash, unknown, validate_record


class _AllSequencesClosed(StoppingCriteria):
    def __init__(self, start_length: int, closing_token_id: int):
        self.start_length = start_length
        self.closing_token_id = closing_token_id

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[:, self.start_length:]
        if generated.shape[1] == 0:
            return False
        return bool(
            torch.all(torch.any(generated.eq(self.closing_token_id), dim=1)).item()
        )


def _load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame(columns=["text_hash"] + SEMANTIC_FEATURES + ["semantic_source", "semantic_schema_version"])
    frame = pd.read_csv(OUTPUT_FILE, dtype={"text_hash": str})
    required = {"text_hash", *SEMANTIC_FEATURES, "semantic_schema_version"}
    if not required.issubset(frame):
        raise ValueError(f"Existing semantic data has incompatible columns: {sorted(required - set(frame))}")
    if not frame.semantic_schema_version.eq(SCHEMA_VERSION).all():
        raise ValueError("Existing semantic data uses another schema; use --reset")
    if frame.text_hash.duplicated().any():
        raise ValueError("Duplicate text_hash in semantic data")
    return frame


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["text_hash"] + SEMANTIC_FEATURES + ["semantic_source", "semantic_schema_version"])


def _seed_model15(descriptions: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if not LEGACY_MODEL15_FILE.exists():
        return existing
    payload = json.loads(LEGACY_MODEL15_FILE.read_text(encoding="utf-8"))
    valid_hashes = set(descriptions.text_hash)
    known = set(existing.text_hash)
    records = []
    for description, value in payload.items():
        hashed = text_hash(description)
        if hashed in known or hashed not in valid_hashes:
            continue
        if value.get("semantic_source") != "qwen15" or value.get("semantic_schema_version") != COMPATIBLE_LEGACY_SCHEMA:
            continue
        clean = normalize(value, "qwen15_reused")
        validate_record(clean)
        records.append({"text_hash": hashed, **clean})
    if records:
        existing = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    return existing.drop_duplicates("text_hash", keep="last")


def _atomic_save(frame: pd.DataFrame, manifest: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".csv.tmp")
    frame.sort_values("text_hash").to_csv(temporary, index=False)
    os.replace(temporary, OUTPUT_FILE)
    manifest_temp = MANIFEST_FILE.with_suffix(".json.tmp")
    manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_temp, MANIFEST_FILE)


def _render_compact(
    tokenizer, description: str, retry: bool, overhead: int
) -> tuple[str, bool]:
    """Fit only the description into the token budget; never cut instructions.

    Tokenizer-level truncation of a complete chat prompt removes its final
    assistant generation marker on long descriptions.  Preserve the beginning
    and end of the description instead and keep the entire chat structure.
    """
    available = max(32, MAX_INPUT_TOKENS - overhead - 24)
    tokens = tokenizer(str(description or ""), add_special_tokens=False).input_ids
    truncated = len(tokens) > available
    if not truncated:
        fitted = str(description or "")
    else:
        head = max(1, int(available * 0.70))
        tail = max(1, available - head)
        fitted = (
            tokenizer.decode(tokens[:head], skip_special_tokens=True)
            + "\n…\n"
            + tokenizer.decode(tokens[-tail:], skip_special_tokens=True)
        )
    rendered = tokenizer.apply_chat_template(
        prompt(fitted, retry=retry), tokenize=False, add_generation_prompt=True
    ) + "["
    # Re-encoding a decoded token slice can differ by a few tokens.  Shrink the
    # description conservatively until the complete prompt is within budget.
    while len(tokenizer(rendered, add_special_tokens=False).input_ids) > MAX_INPUT_TOKENS:
        available = max(16, available - 24)
        head = max(1, int(available * 0.70))
        tail = max(1, available - head)
        fitted = (
            tokenizer.decode(tokens[:head], skip_special_tokens=True)
            + "\n…\n"
            + tokenizer.decode(tokens[-tail:], skip_special_tokens=True)
        )
        rendered = tokenizer.apply_chat_template(
            prompt(fitted, retry=retry), tokenize=False, add_generation_prompt=True
        ) + "["
    return rendered, truncated


def _generate_compact(model, tokenizer, batch: list[dict], max_new_tokens: int, retry=False):
    # Prefill the opening bracket as part of the assistant response.  This
    # anchors deterministic decoding to the short integer-array grammar.
    empty = tokenizer.apply_chat_template(
        prompt("", retry=retry), tokenize=False, add_generation_prompt=True
    ) + "["
    overhead = len(tokenizer(empty, add_special_tokens=False).input_ids)
    rendered_and_flags = [
        _render_compact(
            tokenizer, item["description"], retry=retry, overhead=overhead
        )
        for item in batch
    ]
    rendered = [value[0] for value in rendered_and_flags]
    truncated_count = sum(value[1] for value in rendered_and_flags)
    inputs = tokenizer(
        rendered, return_tensors="pt", padding=True, truncation=False,
    ).to("cuda")
    if inputs.input_ids.shape[1] > MAX_INPUT_TOKENS:
        raise AssertionError("Complete compact prompt exceeds MAX_INPUT_TOKENS")
    closing_ids = tokenizer("]", add_special_tokens=False).input_ids
    stopping = (
        StoppingCriteriaList([
            _AllSequencesClosed(inputs.input_ids.shape[1], closing_ids[0])
        ])
        if len(closing_ids) == 1 else None
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    decoded = tokenizer.batch_decode(
        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    del inputs, output
    return ["[" + response for response in decoded], truncated_count


def _append_failures(records: list[dict]):
    if not records:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with FAILURE_LOG_FILE.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract(data_dir: Path | None = None, batch_size=DEFAULT_BATCH_SIZE, reset=False):
    descriptions, ticket_rows = load_descriptions(data_dir)
    existing = _load_existing() if not reset else _empty_frame()
    existing = existing[existing.text_hash.isin(descriptions.text_hash)].copy()
    if "semantic_source" in existing:
        # Retry prior parse failures on every explicit extraction run.
        existing = existing[~existing.semantic_source.eq("parse_error")].copy()
    existing = _seed_model15(descriptions, existing)
    empty_hash = text_hash("")
    if empty_hash not in set(existing.text_hash):
        existing = pd.concat([existing, pd.DataFrame([{"text_hash": empty_hash, **unknown()}])], ignore_index=True)
    remaining = descriptions[~descriptions.text_hash.isin(existing.text_hash) & descriptions.description.ne("")]
    manifest = {
        "schema_version": SCHEMA_VERSION, "snapshot_dir": descriptions.attrs.get("snapshot_dir"),
        "ticket_rows": ticket_rows, "unique_descriptions": len(descriptions),
        "complete": False, "qwen_model": QWEN_MODEL,
        "generation_format": "compact_integer_array_v1",
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    print(f"unique_descriptions={len(descriptions):,} reused={len(existing):,} remaining={len(remaining):,}")
    if remaining.empty:
        manifest.update({"complete": True, "semantic_rows": len(existing)})
        _atomic_save(existing, manifest)
        return manifest
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応NVIDIA GPUが必要です")
    torch.cuda.set_device(CUDA_DEVICE)
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, CUDA_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, quantization_config=quantization, device_map={"": CUDA_DEVICE},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    device_map = getattr(model, "hf_device_map", {}) or {}
    if any(str(device).lower() in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"CPU/disk offload is forbidden: {device_map}")
    model.eval()
    rows, current_batch = [], max(1, int(batch_size))
    items = remaining[["text_hash", "description"]].to_dict("records")
    position, errors, recovered, truncated_inputs = 0, 0, 0, 0
    started_at = time.perf_counter()
    progress = tqdm(total=len(items), desc="all-ticket semantic JSON")
    try:
        while position < len(items):
            batch = items[position:position + current_batch]
            try:
                responses, batch_truncated = _generate_compact(
                    model, tokenizer, batch, MAX_OUTPUT_TOKENS, retry=False
                )
                parsed = []
                failed_positions = []
                for index, response in enumerate(responses):
                    try:
                        parsed.append(parse_response(response))
                    except Exception:
                        parsed.append(None)
                        failed_positions.append(index)
                retry_responses = []
                if failed_positions:
                    retry_batch = [batch[index] for index in failed_positions]
                    retry_responses, _ = _generate_compact(
                        model, tokenizer, retry_batch, RETRY_OUTPUT_TOKENS, retry=True
                    )
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
                progress.write(f"CUDA OOM: batch-size={current_batch}で再試行")
                continue
            truncated_inputs += batch_truncated
            failures = []
            for retry_index, batch_index in enumerate(failed_positions):
                try:
                    parsed[batch_index] = parse_response(retry_responses[retry_index])
                    recovered += 1
                except Exception:
                    errors += 1
                    parsed[batch_index] = unknown("parse_error")
                    failures.append({
                        "text_hash": batch[batch_index]["text_hash"],
                        "first_response": responses[batch_index][:1000],
                        "retry_response": retry_responses[retry_index][:1000],
                    })
            _append_failures(failures)
            for item, semantic in zip(batch, parsed):
                rows.append({"text_hash": item["text_hash"], **semantic})
            position += len(batch)
            progress.update(len(batch))
            if position % SAVE_EVERY < len(batch):
                combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("text_hash", keep="last")
                manifest.update({
                    "semantic_rows": len(combined), "parse_errors": errors,
                    "recovered_format_errors": recovered,
                    "generated_this_run": position,
                    "current_batch_size": current_batch,
                    "truncated_descriptions": truncated_inputs,
                })
                _atomic_save(combined, manifest)
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                rate = position / elapsed
                eta_hours = (len(items) - position) / max(rate, 1e-9) / 3600
                progress.write(
                    f"checkpoint={position:,}/{len(items):,} "
                    f"rate={rate:.2f}/s eta={eta_hours:.1f}h "
                    f"parse_errors={errors} recovered={recovered} "
                    f"batch={current_batch}"
                )
        final = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("text_hash", keep="last")
        complete = set(descriptions.text_hash) == set(final.text_hash)
        manifest.update({
            "complete": complete, "semantic_rows": len(final), "parse_errors": errors,
            "recovered_format_errors": recovered, "generated_this_run": position,
            "current_batch_size": current_batch,
            "truncated_descriptions": truncated_inputs,
        })
        _atomic_save(final, manifest)
        if not complete:
            raise RuntimeError("Semantic extraction finished without complete coverage")
        return manifest
    finally:
        progress.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    print(json.dumps(extract(args.data_dir, args.batch_size, args.reset), ensure_ascii=False, indent=2))
