"""Check extraction coverage and CUDA placement without loading Qwen."""
from __future__ import annotations

import importlib.util
import json
import shutil

import pandas as pd

from config import (
    BUILDER_DIR,
    LEGACY_MODEL15_FILE, MANIFEST_FILE, MAX_PARSE_ERROR_RATE, OUTPUT_FILE,
    SCHEMA_VERSION, SEMANTIC_FEATURES,
)
from data_loader import load_descriptions
from schema import text_hash


def check():
    descriptions, ticket_rows = load_descriptions()
    report = {
        "ok": True, "snapshot": descriptions.attrs.get("snapshot_dir"),
        "ticket_rows": ticket_rows, "unique_descriptions": len(descriptions),
        "existing_model15_cache": LEGACY_MODEL15_FILE.exists(),
        "output_exists": OUTPUT_FILE.exists(), "training_or_extraction_executed": False,
    }
    reusable = 0
    if LEGACY_MODEL15_FILE.exists():
        legacy = json.loads(LEGACY_MODEL15_FILE.read_text(encoding="utf-8"))
        current_hashes = set(descriptions.text_hash)
        reusable = len({
            text_hash(description)
            for description, value in legacy.items()
            if text_hash(description) in current_hashes
            and value.get("semantic_source") == "qwen15"
            and value.get("semantic_schema_version") == "model15_semantic_v1"
        })
    report["reusable_qwen15_descriptions"] = int(reusable)
    report["estimated_remaining_extractions"] = max(0, len(descriptions) - reusable - 1)
    if OUTPUT_FILE.exists():
        semantic = pd.read_csv(OUTPUT_FILE, dtype={"text_hash": str})
        missing_columns = {"text_hash", *SEMANTIC_FEATURES, "semantic_schema_version"} - set(semantic)
        usable = (
            semantic[~semantic.get("semantic_source", pd.Series("", index=semantic.index)).eq("parse_error")]
            if not missing_columns else semantic.iloc[0:0]
        )
        covered = descriptions.text_hash.isin(usable.text_hash).sum() if not missing_columns else 0
        report.update({
            "semantic_rows": len(semantic), "usable_semantic_rows": len(usable),
            "covered_descriptions": int(covered),
            "coverage_pct": round(100 * covered / max(len(descriptions), 1), 2),
            "estimated_remaining_extractions": max(0, len(descriptions) - int(covered)),
            "schema_ok": bool(
                not missing_columns
                and semantic.semantic_schema_version.eq(SCHEMA_VERSION).all()
            ),
        })
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8")) if MANIFEST_FILE.exists() else {}
        report["manifest_complete"] = bool(manifest.get("complete"))
        parse_errors = int(manifest.get("parse_errors", 0))
        report["parse_errors"] = parse_errors
        report["parse_error_rate"] = round(parse_errors / max(len(descriptions), 1), 4)
        report["generation_format"] = manifest.get("generation_format", "legacy_free_json")
        report["recovered_format_errors"] = int(manifest.get("recovered_format_errors", 0))
        report["quality_ok"] = bool(
            report["parse_error_rate"] <= MAX_PARSE_ERROR_RATE
        )
        report["ok"] = bool(
            report["schema_ok"] and report["quality_ok"]
            and report["manifest_complete"] and covered == len(descriptions)
        )
    else:
        report.update({"semantic_rows": 0, "coverage_pct": 0.0, "schema_ok": False})
        report["ok"] = False
    report["cuda_packages_available"] = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "bitsandbytes"))
    report["free_disk_gib"] = round(shutil.disk_usage(BUILDER_DIR).free / 1024 ** 3, 2)
    if importlib.util.find_spec("torch") is not None:
        import torch
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            report["cuda_device_0"] = torch.cuda.get_device_name(0)
            report["dedicated_vram_free_gib"] = round(free_bytes / 1024 ** 3, 2)
            report["dedicated_vram_total_gib"] = round(total_bytes / 1024 ** 3, 2)
            report["gpu_ready"] = free_bytes / 1024 ** 3 >= 8
        else:
            report["gpu_ready"] = False
    report["next"] = None if report["ok"] else "Run extract_semantic_json.py explicitly"
    return report


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
