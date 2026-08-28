"""Run the decision-support pipeline from one resumable entry point.

This file orchestrates existing model folders.  It deliberately does not import
their Python modules, so each model remains self-contained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from snapshot_audit import validate_snapshot


RUNNER_DIR = Path(__file__).resolve().parent
DECISION_DIR = RUNNER_DIR.parent
PROJECT_ROOT = DECISION_DIR.parent
STATE_DIR = RUNNER_DIR / "artifacts"
STATE_FILE = STATE_DIR / "pipeline_state.json"
LOG_DIR = RUNNER_DIR / "logs"

MODEL16_ARTIFACT = PROJECT_ROOT / "hybrid_AI_model16" / "artifacts" / "model16.joblib"
FAIR_PRICE_CACHE = DECISION_DIR / "demand_state_model" / "artifacts" / "fair_price_all_tickets.csv"
SEMANTIC_FILE = PROJECT_ROOT / "semantic_feature_data" / "semantic_features.csv"
SEMANTIC_MANIFEST = PROJECT_ROOT / "semantic_feature_data" / "semantic_manifest.json"


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    directory: Path
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


def snapshot_key(path: Path) -> tuple[int, ...]:
    parts = path.name.removeprefix("data_").split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return 0, int(parts[0]), int(parts[1])
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return tuple(map(int, parts))
    return -1, -1, -1


def latest_snapshot() -> Path:
    root = PROJECT_ROOT / "tiketen_date_data"
    candidates = [
        path for path in root.glob("data_*")
        if path.is_dir()
        and snapshot_key(path)[0] >= 0
        and any(path.glob("*_master.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No ticket snapshot found under {root}")
    return max(candidates, key=snapshot_key)


def pipeline_fingerprint(snapshot: Path) -> str:
    """Fingerprint inputs and code so stale completed stages are not reused."""
    files = list(snapshot.glob("*_master.csv"))
    files += list((PROJECT_ROOT / "手動_data").glob("*.csv"))
    files += list(DECISION_DIR.glob("**/*.py"))
    files += [MODEL16_ARTIFACT]
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in files if item.exists()}):
        stat = path.stat()
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path
        digest.update(str(relative).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def ticket_ids(snapshot: Path) -> set[str]:
    canonical: dict[str, tuple[tuple, str]] = {}
    priority = {"deleted": 0, "listing": 1, "sold": 2}
    for path in snapshot.glob("*_master.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "ticket_id" not in (reader.fieldnames or []):
                raise ValueError(f"ticket_id is missing: {path}")
            for row in reader:
                identifier = str(row.get("ticket_id", "")).strip()
                if not identifier:
                    continue
                event = str(row.get("event_id", "")).strip()
                created = str(row.get("created_at_unix", "")).strip()
                logical = f"created:{event}|{created}" if event and created else f"ticket:{identifier}"
                try:
                    observed = datetime.fromisoformat(str(row.get("last_observed_at", "")).strip())
                except ValueError:
                    observed = datetime.min
                rank = (observed, priority.get(str(row.get("status", "")).lower(), -1), identifier)
                if logical not in canonical or rank > canonical[logical][0]:
                    canonical[logical] = (rank, identifier)
    return {value[1] for value in canonical.values()}


def validate_model16(
    snapshot: Path, allow_price_fallback: bool, cache_will_be_built: bool
) -> None:
    if not MODEL16_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Model 16 artifact is required but missing: {MODEL16_ARTIFACT}"
        )
    if cache_will_be_built:
        state = "will be generated" if not FAIR_PRICE_CACHE.exists() else "will be validated and refreshed if needed"
        print(f"Model 16 guard: fair-price cache {state} after semantic extraction.", flush=True)
        return
    if not FAIR_PRICE_CACHE.exists():
        message = (
            "Model 16 fair-price cache is missing. Running now would silently use "
            "market/listing-price fallback instead of Model 16: "
            f"{FAIR_PRICE_CACHE}"
        )
        if not allow_price_fallback:
            raise RuntimeError(message + "\nUse --allow-price-fallback only for an explicit baseline run.")
        print(f"WARNING: {message}", flush=True)
        return

    expected = ticket_ids(snapshot)
    cached: set[str] = set()
    with FAIR_PRICE_CACHE.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ticket_id", "fair_price"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Fair-price cache must contain {sorted(required)}")
        for row in reader:
            identifier = str(row.get("ticket_id", "")).strip()
            if not identifier:
                raise ValueError("Fair-price cache contains an empty ticket_id")
            try:
                price = float(row.get("fair_price", ""))
            except ValueError as error:
                raise ValueError(f"Invalid fair_price for {identifier}") from error
            if not 0 < price < float("inf"):
                raise ValueError(f"Invalid fair_price for {identifier}: {price}")
            if identifier in cached:
                raise ValueError(f"Duplicate ticket_id in fair-price cache: {identifier}")
            cached.add(identifier)
    missing = expected - cached
    if missing:
        raise ValueError(
            f"Partial Model 16 fair-price cache is forbidden: {len(missing):,} tickets missing"
        )
    print(
        f"Model 16 guard: artifact OK, fair-price coverage={len(expected):,}/{len(expected):,}",
        flush=True,
    )


def stages(batch_size: int) -> list[Stage]:
    python = sys.executable
    return [
        Stage(
            "semantic",
            "全ticket LLM意味特徴（再開対応）",
            DECISION_DIR / "semantic_data_builder",
            (python, "extract_semantic_json.py", "--batch-size", str(batch_size)),
            (SEMANTIC_FILE, SEMANTIC_MANIFEST),
        ),
        Stage(
            "model16_prices",
            "Model 16全ticket価格キャッシュ生成",
            DECISION_DIR / "model16_price_bridge",
            (python, "build_fair_price_cache.py"),
            (
                FAIR_PRICE_CACHE,
                DECISION_DIR / "model16_price_bridge" / "artifacts" / "bridge_report.json",
            ),
        ),
        Stage(
            "demand",
            "需要状態モデルの学習・評価",
            DECISION_DIR / "demand_state_model",
            (python, "run_training.py"),
            (
                DECISION_DIR / "demand_state_model" / "artifacts" / "demand_state.joblib",
                DECISION_DIR / "demand_state_model" / "artifacts" / "oof_predictions.csv",
                DECISION_DIR / "demand_state_model" / "artifacts" / "training_report.json",
            ),
        ),
        Stage(
            "alternative",
            "安価な代替出品モデルの学習・評価",
            DECISION_DIR / "alternative_arrival_model",
            (python, "run_training.py"),
            (
                DECISION_DIR / "alternative_arrival_model" / "artifacts" / "alternative_arrival.joblib",
                DECISION_DIR / "alternative_arrival_model" / "artifacts" / "oof_predictions.csv",
                DECISION_DIR / "alternative_arrival_model" / "artifacts" / "training_report.json",
            ),
        ),
        Stage(
            "buy_inputs",
            "買い時モデルのOOF入力準備",
            DECISION_DIR / "buy_timing_model",
            (python, "prepare_inputs.py"),
            (
                DECISION_DIR / "buy_timing_model" / "inputs" / "demand_oof.csv",
                DECISION_DIR / "buy_timing_model" / "inputs" / "alternative_oof.csv",
            ),
        ),
        Stage(
            "buy",
            "買い時方針の学習・評価",
            DECISION_DIR / "buy_timing_model",
            (python, "run_training.py"),
            (
                DECISION_DIR / "buy_timing_model" / "artifacts" / "policy.json",
                DECISION_DIR / "buy_timing_model" / "artifacts" / "training_report.json",
            ),
        ),
    ]


def load_state(fingerprint: str, force: bool) -> dict:
    if force or not STATE_FILE.exists():
        return {"fingerprint": fingerprint, "completed": []}
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"fingerprint": fingerprint, "completed": []}
    if state.get("fingerprint") != fingerprint:
        print("Input/code fingerprint changed; starting a new pipeline state.", flush=True)
        return {"fingerprint": fingerprint, "completed": []}
    return state


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_FILE)


def stage_outputs_valid(stage: Stage) -> bool:
    if not all(path.exists() for path in stage.outputs):
        return False
    if stage.name == "semantic":
        try:
            manifest = json.loads(SEMANTIC_MANIFEST.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return bool(
            manifest.get("complete")
            and manifest.get("schema_version") == "target_free_semantic_v1"
        )
    if stage.name == "model16_prices":
        report_path = DECISION_DIR / "model16_price_bridge" / "artifacts" / "bridge_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            cached_ids: set[str] = set()
            prices_ok = True
            with FAIR_PRICE_CACHE.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields_ok = {"ticket_id", "fair_price"}.issubset(reader.fieldnames or [])
                rows = 0
                for row in reader:
                    rows += 1
                    identifier = str(row.get("ticket_id", "")).strip()
                    if not identifier or identifier in cached_ids:
                        return False
                    cached_ids.add(identifier)
                    try:
                        price = float(row.get("fair_price", ""))
                    except ValueError:
                        prices_ok = False
                        break
                    prices_ok = prices_ok and 0 < price < float("inf")
            snapshot = latest_snapshot()
            expected_ids = ticket_ids(snapshot)
        except (json.JSONDecodeError, OSError, FileNotFoundError, ValueError):
            return False
        return bool(
            fields_ok and prices_ok and cached_ids == expected_ids
            and rows == int(report.get("rows", -1)) and rows > 0
            and Path(report.get("snapshot", "")).resolve() == snapshot.resolve()
        )
    if stage.name in {"demand", "alternative"}:
        model_dir = DECISION_DIR / (
            "demand_state_model" if stage.name == "demand"
            else "alternative_arrival_model"
        )
        report_path = model_dir / "artifacts" / "training_report.json"
        oof_path = model_dir / "artifacts" / "oof_predictions.csv"
        expected_version = (
            "demand_state_semantic_selection_v4_logical_identity"
            if stage.name == "demand"
            else "alternative_arrival_semantic_selection_v4_logical_identity"
        )
        required_oof = (
            {"ticket_id", "landmark_at", "horizon_days", "fold"}
            | ({"true_state", "p_active", "p_sold", "p_deleted"}
               if stage.name == "demand"
               else {"true_alternative", "p_alternative"})
        )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            with oof_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields_ok = required_oof.issubset(reader.fieldnames or [])
                has_data = next(reader, None) is not None
        except (json.JSONDecodeError, OSError):
            return False
        return bool(
            report.get("pipeline_version") == expected_version
            and set(report.get("metrics_by_horizon", {})) == {"1", "3", "7"}
            and fields_ok and has_data
            and Path(report.get("snapshot_dir", "")).resolve() == latest_snapshot().resolve()
        )
    return True


def run_stage(stage: Stage, log_stream) -> None:
    banner = f"\n===== {stage.name}: {stage.label} ====="
    print(banner, flush=True)
    log_stream.write(banner + "\n")
    log_stream.flush()
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        stage.command,
        cwd=stage.directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, stage.command)
    if not stage_outputs_valid(stage):
        raise RuntimeError(
            f"Stage exited successfully but its outputs are missing or invalid: {stage.outputs}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model 16-based decision-support pipeline runner"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Qwen extraction batch size")
    parser.add_argument("--force", action="store_true", help="rerun completed stages")
    parser.add_argument("--from-stage", choices=[stage.name for stage in stages(8)])
    parser.add_argument("--dry-run", action="store_true", help="show checks and commands only")
    parser.add_argument(
        "--allow-price-fallback",
        action="store_true",
        help="explicitly allow a baseline run without the complete Model 16 price cache",
    )
    parser.add_argument(
        "--allow-historical-snapshot",
        action="store_true",
        help=(
            "explicitly allow a stale/no-listing snapshot for historical model "
            "evaluation; never use this for current recommendations"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    snapshot = latest_snapshot()
    snapshot_report = validate_snapshot(
        snapshot, allow_historical=args.allow_historical_snapshot
    )
    selected_stages = stages(args.batch_size)
    if args.from_stage:
        start = [stage.name for stage in selected_stages].index(args.from_stage)
        selected_stages = selected_stages[start:]

    print(f"Python: {sys.executable}")
    print(f"Snapshot: {snapshot}")
    print(
        "Snapshot guard: "
        f"rows={snapshot_report['rows']:,}, "
        f"listing={snapshot_report['status_counts'].get('listing', 0):,}, "
        f"latest={snapshot_report['maximum_last_observed_at']}",
        flush=True,
    )
    if snapshot_report["historical_override_used"]:
        print(
            "WARNING: historical snapshot override is active; outputs must not "
            "be used for current demand or buy-timing recommendations.",
            flush=True,
        )
    print(f"Model 16: {MODEL16_ARTIFACT}")
    validate_model16(
        snapshot,
        args.allow_price_fallback,
        cache_will_be_built=any(stage.name == "model16_prices" for stage in selected_stages),
    )
    for stage in selected_stages:
        print(f"  {stage.name:12s} cwd={stage.directory.name} command={' '.join(stage.command[1:])}")
    if args.dry_run:
        print("Dry-run complete. No extraction or training was executed.")
        return

    fingerprint = pipeline_fingerprint(snapshot)
    state = load_state(fingerprint, args.force)
    completed = set(state.get("completed", []))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
        for stage in selected_stages:
            output_ready = stage_outputs_valid(stage)
            if not args.force and stage.name in completed and output_ready:
                message = f"SKIP {stage.name}: completed for the same input/code fingerprint"
                print(message, flush=True)
                log_stream.write(message + "\n")
                continue
            run_stage(stage, log_stream)
            completed.add(stage.name)
            state.update({
                "fingerprint": fingerprint,
                "snapshot": str(snapshot),
                "completed": [item.name for item in stages(args.batch_size) if item.name in completed],
                "last_completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            save_state(state)
    elapsed = time.perf_counter() - started
    print(f"\nAll selected stages completed in {elapsed / 3600:.2f} hours.")
    print(f"Log: {log_path}")
    print(f"State: {STATE_FILE}")


if __name__ == "__main__":
    main()
