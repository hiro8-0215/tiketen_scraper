"""Recover only persisted semantic parse errors without regenerating successes.

Run this after the normal extraction process has finished. It re-parses the
saved first/retry responses with the audited compact-response recovery rule,
updates only rows whose semantic_source is ``parse_error``, and enforces the
same one-percent quality gate used by the downstream models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psutil

from config import (
    FAILURE_LOG_FILE,
    MANIFEST_FILE,
    MAX_PARSE_ERROR_RATE,
    OUTPUT_FILE,
    SCHEMA_VERSION,
    SEMANTIC_FEATURES,
)
from schema import parse_response, validate_record


ACTIVE_COMMAND_MARKERS = (
    "semantic_data_builder/extract_semantic_json.py",
    "pipeline_runner/run_all.py",
)


def _assert_pipeline_is_idle() -> None:
    """Refuse concurrent writes while extraction/the full runner is active."""
    current_pid = os.getpid()
    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if process.info["pid"] == current_pid:
                continue
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        normalized = command.replace("\\", "/").lower()
        if any(marker in normalized for marker in ACTIVE_COMMAND_MARKERS):
            active.append((process.info["pid"], command))
    if active:
        details = "\n".join(f"  PID {pid}: {command}" for pid, command in active)
        raise RuntimeError(
            "LLM extraction/pipeline is still running. Wait for it to stop before "
            f"repairing parse errors.\n{details}"
        )


def _load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Semantic manifest is missing: {MANIFEST_FILE}")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Semantic manifest uses an incompatible schema")
    if not manifest.get("complete"):
        raise RuntimeError(
            "Semantic extraction is incomplete. Let the current extraction finish; "
            "this repair launcher must not be run concurrently."
        )
    return manifest


def _load_features() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Semantic feature file is missing: {OUTPUT_FILE}")
    frame = pd.read_csv(OUTPUT_FILE, dtype={"text_hash": str})
    required = {
        "text_hash", *SEMANTIC_FEATURES, "semantic_source", "semantic_schema_version"
    }
    if not required.issubset(frame):
        raise ValueError(f"Semantic feature columns are missing: {sorted(required - set(frame))}")
    if frame.text_hash.duplicated().any():
        raise ValueError("Semantic feature data contains duplicate text hashes")
    if not frame.semantic_schema_version.eq(SCHEMA_VERSION).all():
        raise ValueError("Semantic feature data contains mixed schema versions")
    return frame


def _latest_failures(path: Path = FAILURE_LOG_FILE) -> tuple[dict[str, dict], int]:
    if not path.exists():
        raise FileNotFoundError(f"Parse-failure log is missing: {path}")
    latest: dict[str, dict] = {}
    malformed = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                identifier = str(record.get("text_hash", "")).strip()
                if not identifier:
                    raise ValueError("missing text_hash")
                latest[identifier] = record
            except (json.JSONDecodeError, ValueError, TypeError):
                malformed += 1
    return latest, malformed


def recover_frame(
    frame: pd.DataFrame, failures: dict[str, dict]
) -> tuple[pd.DataFrame, dict]:
    """Return a copy with only parse-error rows replaced by valid log output."""
    result = frame.copy().set_index("text_hash", drop=False)
    targets = result.index[result.semantic_source.eq("parse_error")].tolist()
    recovered = 0
    recovered_from_retry = 0
    recovered_from_first = 0
    missing_log = 0

    for identifier in targets:
        failure = failures.get(identifier)
        if failure is None:
            missing_log += 1
            continue
        semantic = None
        selected = None
        # Prefer the explicit format-retry response; fall back to the first
        # response when the retry contains an out-of-range or repeated value.
        for source_name in ("retry_response", "first_response"):
            try:
                semantic = parse_response(str(failure.get(source_name, "")))
                validate_record(semantic)
                selected = source_name
                break
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        if semantic is None:
            continue
        for column, value in semantic.items():
            if column in result.columns:
                result.at[identifier, column] = value
        recovered += 1
        recovered_from_retry += int(selected == "retry_response")
        recovered_from_first += int(selected == "first_response")

    result = result.reset_index(drop=True)
    unresolved = int(result.semantic_source.eq("parse_error").sum())
    return result, {
        "target_parse_errors": len(targets),
        "recovered": recovered,
        "recovered_from_retry": recovered_from_retry,
        "recovered_from_first": recovered_from_first,
        "missing_failure_log": missing_log,
        "unresolved": unresolved,
    }


def _atomic_save(frame: pd.DataFrame, manifest: dict) -> None:
    feature_temp = OUTPUT_FILE.with_suffix(".csv.tmp")
    frame.sort_values("text_hash").to_csv(feature_temp, index=False)
    os.replace(feature_temp, OUTPUT_FILE)
    manifest_temp = MANIFEST_FILE.with_suffix(".json.tmp")
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(manifest_temp, MANIFEST_FILE)


def repair() -> dict:
    _assert_pipeline_is_idle()
    manifest = _load_manifest()
    frame = _load_features()
    failures, malformed_lines = _latest_failures()
    repaired, report = recover_frame(frame, failures)

    denominator = max(int(manifest.get("unique_descriptions", len(repaired))), 1)
    error_rate = report["unresolved"] / denominator
    quality_gate_passed = error_rate <= MAX_PARSE_ERROR_RATE
    report.update({
        "unique_descriptions": denominator,
        "parse_error_rate": error_rate,
        "max_parse_error_rate": MAX_PARSE_ERROR_RATE,
        "quality_gate_passed": quality_gate_passed,
        "malformed_failure_log_lines": malformed_lines,
    })
    manifest.update({
        "semantic_rows": len(repaired),
        "parse_errors": report["unresolved"],
        "parse_error_rate": error_rate,
        "quality_gate_passed": quality_gate_passed,
        "log_recovery": report,
    })
    _atomic_save(repaired, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not quality_gate_passed:
        raise RuntimeError(
            "Parse-error repair was saved, but the remaining rate still exceeds "
            f"{MAX_PARSE_ERROR_RATE:.1%}. Run the normal semantic extractor once "
            "more to regenerate only the unresolved rows."
        )
    print(
        "Semantic quality gate passed. Next run the full decision pipeline; "
        "successful semantic rows will be reused.",
        flush=True,
    )
    return report


if __name__ == "__main__":
    repair()
