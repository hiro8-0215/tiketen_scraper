"""Safely train every Model16 expert from a genuinely fresh state.

The new run lives outside the production artifact directory.  ``start``
removes no data: an earlier unfinished fresh run is archived, then a clean
workspace is created.  ``resume`` keeps checkpoints from that clean run.
Only a fully trained, plotted, and frozen result is promoted to ``artifacts``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


MODEL_DIR = Path(__file__).resolve().parent
OFFICIAL_DIR = MODEL_DIR / "artifacts"
FRESH_DIR = MODEL_DIR / "fresh_training"
HISTORY_DIR = MODEL_DIR / "artifacts_history"
EXPECTED_OUTPUTS = {
    "evaluation_model16.json",
    "oof_predictions_model16.csv",
    "model16.joblib",
    "model16_optuna.db",
    "candidate_comparison.csv",
    "actual_vs_predicted_model16.png",
    "evaluation_dashboard_model16.png",
    "incremental_baseline.json",
    "incremental_baseline_population.csv",
}


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _archive_directory(source: Path, label: str) -> Path | None:
    if not source.exists():
        return None
    if not any(source.iterdir()):
        source.rmdir()
        return None
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    destination = HISTORY_DIR / f"{label}_{_stamp()}"
    shutil.move(str(source), str(destination))
    return destination


def _run(script: str, *arguments: str, fresh: bool = False) -> None:
    environment = os.environ.copy()
    if fresh:
        environment["MODEL16_ARTIFACT_DIR"] = str(FRESH_DIR)
    command = [sys.executable, str(MODEL_DIR / script), *arguments]
    print("\n実行:", " ".join(command), flush=True)
    subprocess.run(command, cwd=MODEL_DIR, env=environment, check=True)


def _validate_fresh_result() -> dict:
    missing = sorted(name for name in EXPECTED_OUTPUTS if not (FRESH_DIR / name).exists())
    if missing:
        raise RuntimeError("新規Model16成果物が不足しています: " + ", ".join(missing))
    report = json.loads((FRESH_DIR / "evaluation_model16.json").read_text(encoding="utf-8"))
    baseline = json.loads((FRESH_DIR / "incremental_baseline.json").read_text(encoding="utf-8"))
    if (
        report.get("dataset_fingerprint") != baseline.get("dataset_fingerprint")
        or report.get("rows") != baseline.get("training_rows")
        or report.get("price_band_routing") is not False
    ):
        raise RuntimeError("新規Model16のモデル・評価・追加評価基準が一致しません")
    return report


def _initialize_workspace(mode: str) -> None:
    if mode == "start":
        archived = _archive_directory(FRESH_DIR, "incomplete")
        if archived:
            print(f"以前の未完了新規学習を退避しました: {archived}", flush=True)
        FRESH_DIR.mkdir(parents=True, exist_ok=False)
    elif not FRESH_DIR.exists():
        # "Resume" is intentionally forgiving: on a first run (or after a
        # completed run was promoted) there is nothing to resume, so create a
        # clean workspace and proceed from zero instead of making the user pick
        # another launch entry.
        FRESH_DIR.mkdir(parents=True, exist_ok=False)
        print(
            "再開対象がないため、Model16を0から新規学習します。",
            flush=True,
        )


def _promote() -> Path | None:
    report = _validate_fresh_result()
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in OFFICIAL_DIR.iterdir() if path.name != "README.md"]
    backup = None
    if existing:
        backup = HISTORY_DIR / f"production_{_stamp()}"
        backup.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.move(str(path), str(backup / path.name))
    for path in list(FRESH_DIR.iterdir()):
        shutil.move(str(path), str(OFFICIAL_DIR / path.name))
    FRESH_DIR.rmdir()
    print(
        f"\nModel16正式成果物を更新しました: rows={report['rows']:,}, "
        f"primary={report['primary_candidate']}",
        flush=True,
    )
    if backup:
        print(f"旧Model16退避先: {backup}", flush=True)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("start", "resume"),
        help="start=完全新規、resume=この専用実行のcheckpointから再開",
    )
    args = parser.parse_args()

    # Target-free semantic/BERT caches may be reused.  They are input features,
    # not fitted price models.  Missing rows are generated before fresh fitting.
    _run("prepare_inputs.py")

    _initialize_workspace(args.mode)

    _run("preflight.py", fresh=True)
    print(
        "\n4 expertを新規学習します: LightGBM log-MAE / LightGBM MAPE / "
        "CatBoost MAE / BERT Ridge",
        flush=True,
    )
    _run("train_model16.py", fresh=True)
    _run("incremental_evaluation.py", "--freeze", fresh=True)
    _run("view_results.py", fresh=True)
    _promote()


if __name__ == "__main__":
    main()
