"""Evaluate a frozen Model16 on sold tickets added after its training cutoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, DATA_ROOT, PIPELINE_VERSION, TARGET
from data_loader import latest_data_dir, load_snapshot, prepare_dataset
from inference import predict_payload
from modeling import metrics


BASELINE_JSON = ARTIFACT_DIR / "incremental_baseline.json"
BASELINE_ROWS = ARTIFACT_DIR / "incremental_baseline_population.csv"
INCREMENTAL_DIR = ARTIFACT_DIR / "incremental_evaluation"
LATEST_REPORT = INCREMENTAL_DIR / "latest_evaluation.json"
LATEST_PREDICTIONS = INCREMENTAL_DIR / "latest_predictions.csv"
HISTORY = INCREMENTAL_DIR / "evaluation_history.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _logical_keys(df: pd.DataFrame) -> pd.Series:
    ticket = df["ticket_id"].astype("string").fillna("").str.strip()
    event = df.get(
        "event_id", pd.Series("", index=df.index)
    ).astype("string").fillna("").str.strip()
    created = df.get(
        "created_at_unix", pd.Series("", index=df.index)
    ).astype("string").fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    keys = "ticket:" + ticket
    stable = event.ne("") & created.ne("")
    keys.loc[stable] = "created:" + event.loc[stable] + "|" + created.loc[stable]
    return keys


def _collect_oof_population(
    oof: pd.DataFrame, model_path: Path, report: dict
) -> tuple[pd.DataFrame, pd.Timestamp, str]:
    """Recover the exact trained population even after newer snapshots arrive."""
    wanted = set(oof["ticket_id"].astype(str))
    pieces = []
    source_name = report.get("training_snapshot")
    source_cutoff = pd.to_datetime(report.get("training_cutoff"), errors="coerce")
    if source_name:
        def snapshot_key(path: Path) -> tuple[int, int, int]:
            numbers = tuple(map(int, path.name.removeprefix("data_").split("_")))
            return (0, *numbers) if len(numbers) == 2 else numbers

        source_key = snapshot_key(DATA_ROOT / source_name)
        directories = [
            directory for directory in DATA_ROOT.glob("data_*")
            if snapshot_key(directory) <= source_key
        ]
    else:
        # Legacy reports did not store their source. Ignore folders created after
        # the immutable model artifact was written.
        model_mtime = model_path.stat().st_mtime
        directories = [
            directory for directory in DATA_ROOT.glob("data_*")
            if directory.stat().st_mtime <= model_mtime + 1.0
        ]
    snapshot_latest = (pd.NaT, "unknown")
    for directory in directories:
        snapshot = load_snapshot(directory)
        observed_all = pd.to_datetime(snapshot.get("last_observed_at"), errors="coerce").max()
        if pd.notna(observed_all) and (pd.isna(snapshot_latest[0]) or observed_all > snapshot_latest[0]):
            snapshot_latest = (observed_all, directory.name)
        match = snapshot.loc[snapshot["ticket_id"].astype(str).isin(wanted)].copy()
        if match.empty:
            continue
        match["_snapshot"] = directory.name
        pieces.append(match)
    if not pieces:
        raise RuntimeError("OOFのticket IDを履歴snapshotから復元できません")
    raw = pd.concat(pieces, ignore_index=True)
    raw["ticket_id"] = raw["ticket_id"].astype(str)
    missing = wanted - set(raw["ticket_id"])
    if missing:
        raise RuntimeError(f"OOFの学習行を履歴snapshotから{len(missing)}件復元できません")
    raw["_observed"] = pd.to_datetime(raw.get("last_observed_at"), errors="coerce")
    raw = raw.sort_values(["ticket_id", "_observed"], na_position="first")
    population = raw.drop_duplicates("ticket_id", keep="last").copy()
    population["logical_ticket_id"] = _logical_keys(population)
    if len(population) != len(oof) or population["logical_ticket_id"].duplicated().any():
        raise RuntimeError("OOF学習母集団の論理ticket IDを一意に復元できません")
    cutoff = source_cutoff if pd.notna(source_cutoff) else snapshot_latest[0]
    if pd.isna(cutoff):
        cutoff = raw["_observed"].max()
    if pd.isna(cutoff):
        raise ValueError("有効な観測時刻がなく、Model16学習基準日時を固定できません")
    columns = ["ticket_id", "logical_ticket_id", "first_observed_at"]
    return population[columns], cutoff, source_name or snapshot_latest[1]


def _load_model_state():
    report_path = ARTIFACT_DIR / "evaluation_model16.json"
    oof_path = ARTIFACT_DIR / "oof_predictions_model16.csv"
    model_path = ARTIFACT_DIR / "model16.joblib"
    missing = [str(path) for path in [report_path, oof_path, model_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Model16学習済み成果物が不足しています: " + ", ".join(missing))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    oof = pd.read_csv(oof_path)
    payload = joblib.load(model_path)
    fingerprints = {
        report.get("dataset_fingerprint"), payload.get("dataset_fingerprint")
    }
    if (
        report.get("pipeline_version") != PIPELINE_VERSION
        or payload.get("pipeline_version") != PIPELINE_VERSION
        or len(fingerprints) != 1
        or None in fingerprints
        or len(oof) != report.get("rows")
        or oof.ticket_id.nunique() != len(oof)
    ):
        raise ValueError("Model16のモデル・OOF・評価JSONが同じ学習結果ではありません")
    return report, oof, payload, model_path


def freeze_baseline() -> dict:
    report, oof, payload, model_path = _load_model_state()
    population, cutoff, snapshot_name = _collect_oof_population(oof, model_path, report)
    rows_tmp = BASELINE_ROWS.with_suffix(".tmp.csv")
    population.to_csv(rows_tmp, index=False, encoding="utf-8-sig")
    os.replace(rows_tmp, BASELINE_ROWS)
    manifest = {
        "model": "Model16",
        "pipeline_version": PIPELINE_VERSION,
        "dataset_fingerprint": report["dataset_fingerprint"],
        "training_rows": int(len(population)),
        "training_snapshot": snapshot_name,
        "training_cutoff": cutoff.isoformat(),
        "model_sha256": _sha256(model_path),
        "oof_sha256": _sha256(ARTIFACT_DIR / "oof_predictions_model16.csv"),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text(BASELINE_JSON, json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("現在のModel16を追加データ外部評価の基準として固定しました。学習は未実行です。")
    return manifest


def _safe_metrics(frame: pd.DataFrame) -> dict | None:
    if frame.empty:
        return None
    result = metrics(frame[TARGET].to_numpy(float), frame["prediction"].to_numpy(float))
    if not np.isfinite(result["r2"]):
        result["r2"] = None
    return result


def _semantic_coverage(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    if "semantic_available" not in frame:
        return 0.0
    available = pd.to_numeric(frame["semantic_available"], errors="coerce").fillna(0)
    return float(available.gt(0).mean() * 100)


def _price_bands(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    bins = [2_000, 10_000, 20_000, 30_000, 50_000, 80_000, 150_001]
    labels = ["2–9千円", "10–19千円", "20–29千円", "30–49千円", "50–79千円", "80–150千円"]
    band = pd.cut(frame[TARGET], bins=bins, labels=labels, right=False)
    rows = []
    for label, group in frame.groupby(band, observed=True):
        value = _safe_metrics(group)
        rows.append({"price_band": str(label), **value})
    return rows


def evaluate() -> dict:
    if not BASELINE_JSON.exists() or not BASELINE_ROWS.exists():
        raise FileNotFoundError(
            "追加評価基準がありません。[10 価格][追加評価][基準固定]を先に一度実行してください"
        )
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline_rows = pd.read_csv(BASELINE_ROWS)
    report, oof, payload, model_path = _load_model_state()
    if (
        baseline.get("dataset_fingerprint") != report.get("dataset_fingerprint")
        or baseline.get("model_sha256") != _sha256(model_path)
        or baseline.get("oof_sha256") != _sha256(ARTIFACT_DIR / "oof_predictions_model16.csv")
        or len(baseline_rows) != baseline.get("training_rows")
    ):
        raise RuntimeError("固定した追加評価基準と現在のModel16が一致しません。基準を固定し直してください")
    if payload.get("requires_bert"):
        raise RuntimeError("このModel16はBERT重みが正のため、追加行BERT生成なしでは評価できません")

    df = prepare_dataset().copy()
    df["logical_ticket_id"] = _logical_keys(df)
    baseline_keys = set(baseline_rows.logical_ticket_id.astype(str))
    added = df.loc[~df.logical_ticket_id.isin(baseline_keys)].copy()
    cutoff = pd.Timestamp(baseline["training_cutoff"])
    observed = pd.to_datetime(added.get("first_observed_at"), errors="coerce")
    # Normalize timezone representation without changing the instant.
    if cutoff.tzinfo is not None:
        observed = pd.to_datetime(observed, utc=True)
        cutoff = cutoff.tz_convert("UTC")
    added["strict_future"] = observed.gt(cutoff)

    if not added.empty:
        added["prediction"] = predict_payload(payload, added)
        added["absolute_error"] = (added["prediction"] - added[TARGET]).abs()
        added["absolute_percentage_error_pct"] = (
            added["absolute_error"] / added[TARGET].clip(lower=1) * 100
        )
    else:
        for column in ["prediction", "absolute_error", "absolute_percentage_error_pct"]:
            added[column] = pd.Series(dtype=float)
    strict = added.loc[added.strict_future].copy()
    semantic_coverage = _semantic_coverage(strict)
    newest_dir = latest_data_dir().name
    evaluation_key = hashlib.sha256(
        (baseline["dataset_fingerprint"] + "|" + strict.sort_values("logical_ticket_id").to_csv(index=False)).encode("utf-8")
    ).hexdigest()
    result = {
        "model": "Model16 frozen incremental evaluation",
        "pipeline_version": PIPELINE_VERSION,
        "dataset_fingerprint": baseline["dataset_fingerprint"],
        "training_snapshot": baseline["training_snapshot"],
        "training_cutoff": baseline["training_cutoff"],
        "latest_snapshot": newest_dir,
        "evaluation_key": evaluation_key,
        "training_rows": int(baseline["training_rows"]),
        "newly_sold_clean_rows": int(len(added)),
        "strict_future_rows": int(len(strict)),
        "previously_observed_then_sold_rows": int((~added.strict_future).sum()),
        "semantic_coverage_pct": semantic_coverage,
        "strict_future_metrics": _safe_metrics(strict),
        "all_newly_sold_metrics": _safe_metrics(added),
        "strict_future_price_bands": _price_bands(strict),
        "warning": (
            "正式判断にはstrict_future_rowsを500件以上集めることを推奨します"
            if len(strict) < 500 else None
        ),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_executed": False,
    }
    INCREMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "ticket_id", "logical_ticket_id", "first_observed_at", "event_id",
        TARGET, "prediction", "absolute_error", "absolute_percentage_error_pct",
        "strict_future", "semantic_available", "semantic_source",
    ]
    available = [column for column in columns if column in added]
    pred_tmp = LATEST_PREDICTIONS.with_suffix(".tmp.csv")
    added[available].to_csv(pred_tmp, index=False, encoding="utf-8-sig")
    os.replace(pred_tmp, LATEST_PREDICTIONS)
    _atomic_text(LATEST_REPORT, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

    history_row = {
        "evaluation_key": evaluation_key,
        "evaluated_at_utc": result["evaluated_at_utc"],
        "latest_snapshot": newest_dir,
        "strict_future_rows": len(strict),
        "semantic_coverage_pct": semantic_coverage,
    }
    if result["strict_future_metrics"]:
        history_row.update(result["strict_future_metrics"])
    history = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame()
    if history.empty or evaluation_key not in set(history.get("evaluation_key", pd.Series(dtype=str)).astype(str)):
        history = pd.concat([history, pd.DataFrame([history_row])], ignore_index=True)
        history_tmp = HISTORY.with_suffix(".tmp.csv")
        history.to_csv(history_tmp, index=False, encoding="utf-8-sig")
        os.replace(history_tmp, HISTORY)

    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if strict.empty:
        print("学習基準後に初観測された新規soldはまだありません。再学習は不要です。")
    else:
        value = result["strict_future_metrics"]
        print(
            f"追加外部評価: n={len(strict):,}, MAE={value['mae_yen']:,.1f}円, "
            f"MAPE={value['mape_pct']:.2f}%, ±20%以内={value['within_20_pct']:.2f}%"
        )
    print("Model16の再学習は実行していません。")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="現在の学習結果を追加評価基準として固定")
    args = parser.parse_args()
    freeze_baseline() if args.freeze else evaluate()


if __name__ == "__main__":
    main()
