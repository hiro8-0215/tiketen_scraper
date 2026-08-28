"""Train standalone cheaper-alternative arrival models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    ARTIFACT_DIR, HORIZONS_DAYS, PIPELINE_VERSION, SEED,
    SEMANTIC_MIN_LOGLOSS_IMPROVEMENT, MANUAL_DIR, SEMANTIC_FEATURES_FILE,
    SEMANTIC_MANIFEST_FILE, N_TEMPORAL_FOLDS, LGBM_PARAMS,
)
from data_loader import load_tickets
from features import add_market_features, feature_profiles
from modeling import (
    calibrate, enforce_monotonic_horizons, fit_calibrator, make_pipeline,
    metrics, temporal_group_splits, predict_positive_probability,
)
from timeline import build_landmarks, observation_cutoff, prepare_end_times


TRAINING_FRAME_CACHE = ARTIFACT_DIR / "training_frame_cache.joblib"
TRAINING_FRAME_META = ARTIFACT_DIR / "training_frame_cache.json"
OOF_CHECKPOINT_DIR = ARTIFACT_DIR / "oof_checkpoints"
TEMPORAL_SPLIT_POLICY = "adaptive_warmup_min_leaf_v1"


def _replace_joblib(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _fit_with_heartbeat(model, features, target, label: str):
    finished = threading.Event()
    started = time.perf_counter()

    def report_progress():
        while not finished.wait(60):
            print(
                f"[{label}] fitting... elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    heartbeat = threading.Thread(target=report_progress, daemon=True)
    heartbeat.start()
    try:
        return model.fit(features, target)
    finally:
        finished.set()
        heartbeat.join(timeout=1)


def _hash_file(digest, path: Path) -> None:
    digest.update(str(path.resolve()).encode("utf-8"))
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)


def _training_frame_fingerprint(tickets: pd.DataFrame) -> str:
    snapshot = Path(str(tickets.attrs.get("snapshot_dir", "")))
    model_dir = Path(__file__).resolve().parent
    paths = list(snapshot.glob("*_master.csv")) if snapshot.is_dir() else []
    paths += list(MANUAL_DIR.glob("*.csv"))
    paths += [SEMANTIC_FEATURES_FILE, SEMANTIC_MANIFEST_FILE]
    paths += [
        model_dir / "config.py", model_dir / "data_loader.py",
        model_dir / "timeline.py", model_dir / "features.py",
    ]
    digest = hashlib.sha256()
    digest.update(PIPELINE_VERSION.encode("utf-8"))
    digest.update(repr(tuple(HORIZONS_DAYS)).encode("ascii"))
    for path in sorted({item.resolve() for item in paths if item.exists()}):
        _hash_file(digest, path)
    return digest.hexdigest()


def _load_or_build_training_frame(
    tickets: pd.DataFrame, prepared: pd.DataFrame, cutoff: pd.Timestamp
):
    fingerprint = _training_frame_fingerprint(tickets)
    required = {
        "market_active_count",
        *[f"alternative_{horizon}d" for horizon in HORIZONS_DAYS],
    }
    if TRAINING_FRAME_CACHE.exists() and TRAINING_FRAME_META.exists():
        try:
            metadata = json.loads(TRAINING_FRAME_META.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                frame = joblib.load(TRAINING_FRAME_CACHE)
                if (
                    required.issubset(frame.columns)
                    and len(frame) == int(metadata.get("rows", -1))
                ):
                    print(
                        f"[alternative] reused verified training-frame cache: {len(frame):,} rows",
                        flush=True,
                    )
                    return frame, fingerprint, True
                print("[alternative] training-frame cache schema mismatch; rebuilding", flush=True)
        except (OSError, EOFError, ValueError, TypeError, KeyError, pickle.UnpicklingError) as error:
            print(f"[alternative] invalid training-frame cache; rebuilding: {error}", flush=True)

    started = time.perf_counter()
    frame = add_market_features(
        build_landmarks(tickets, cutoff=cutoff), prepared
    )
    print(
        f"[alternative] training frame ready: {len(frame):,} rows in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    temporary_cache = TRAINING_FRAME_CACHE.with_suffix(".joblib.tmp")
    joblib.dump(frame, temporary_cache, compress=0)
    os.replace(temporary_cache, TRAINING_FRAME_CACHE)
    metadata = {
        "fingerprint": fingerprint,
        "rows": len(frame),
        "pipeline_version": PIPELINE_VERSION,
    }
    temporary_meta = TRAINING_FRAME_META.with_suffix(".json.tmp")
    temporary_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_meta, TRAINING_FRAME_META)
    print(
        f"[alternative] saved verified training-frame cache: {TRAINING_FRAME_CACHE}",
        flush=True,
    )
    return frame, fingerprint, False


def _split_fingerprints(splits) -> dict[int, str]:
    result = {}
    for fold, train_index, valid_index in splits:
        digest = hashlib.sha256()
        digest.update(np.asarray(train_index, dtype=np.int64).tobytes())
        digest.update(np.asarray(valid_index, dtype=np.int64).tobytes())
        result[int(fold)] = digest.hexdigest()
    return result


def _audit_all_temporal_splits(landmarks: pd.DataFrame) -> dict[str, list[dict]]:
    """Validate every full-frame split before any expensive model fit."""
    required_classes = {0, 1}
    min_class_count = max(int(LGBM_PARAMS.get("min_child_samples", 2)), 2)
    audit: dict[str, list[dict]] = {}
    for horizon in HORIZONS_DAYS:
        target = f"alternative_{horizon}d"
        columns = ["duplicate_group", "landmark_at", target]
        eligible = landmarks.loc[landmarks[target].ge(0), columns].reset_index(drop=True)
        full_counts = eligible[target].value_counts().sort_index()
        if set(full_counts.index.astype(int)) != required_classes:
            raise RuntimeError(
                f"Alternative {horizon}d requires negative/positive classes; got "
                f"{full_counts.to_dict()}"
            )
        splits = list(temporal_group_splits(
            eligible, horizon, target=target, min_class_count=min_class_count
        ))
        if len(splits) != N_TEMPORAL_FOLDS:
            raise RuntimeError(
                f"Alternative {horizon}d produced {len(splits)} temporal folds; "
                f"expected {N_TEMPORAL_FOLDS}"
            )
        fold_audit = []
        for fold, train_index, valid_index in splits:
            training = eligible.loc[train_index]
            validation = eligible.loc[valid_index]
            train_counts = training[target].value_counts().sort_index()
            valid_counts = validation[target].value_counts().sort_index()
            if (
                set(train_counts.index.astype(int)) != required_classes
                or int(train_counts.min()) < min_class_count
            ):
                raise RuntimeError(
                    f"Alternative prefit audit failed for fold {fold}/{horizon}d: "
                    f"training classes={train_counts.to_dict()}, "
                    f"minimum_per_class={min_class_count}"
                )
            train_label_end = (
                training["landmark_at"] + pd.Timedelta(days=horizon)
            ).max()
            valid_start = validation["landmark_at"].min()
            if not train_label_end < valid_start:
                raise RuntimeError(
                    f"Alternative prefit audit found temporal leakage in fold "
                    f"{fold}/{horizon}d: train_label_end={train_label_end}, "
                    f"valid_start={valid_start}"
                )
            if set(training["duplicate_group"]) & set(validation["duplicate_group"]):
                raise RuntimeError(
                    f"Alternative prefit audit found duplicate-group leakage in fold "
                    f"{fold}/{horizon}d"
                )
            row = {
                "fold": int(fold),
                "train_rows": int(len(training)),
                "valid_rows": int(len(validation)),
                "train_class_counts": {
                    str(key): int(value) for key, value in train_counts.items()
                },
                "valid_class_counts": {
                    str(key): int(value) for key, value in valid_counts.items()
                },
                "train_label_end": str(train_label_end),
                "valid_start": str(valid_start),
            }
            fold_audit.append(row)
            print(
                f"[alternative prefit {horizon}d] fold={fold} "
                f"train={len(training):,} valid={len(validation):,} "
                f"train_classes={train_counts.to_dict()}",
                flush=True,
            )
        audit[str(horizon)] = fold_audit
    print("[alternative] all full-frame temporal folds passed prefit audit", flush=True)
    return audit


def _oof_checkpoint_key(
    frame_fingerprint, split_fingerprint, horizon, profile_name, features
):
    model_dir = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    for path in (model_dir / "train.py", model_dir / "modeling.py"):
        code_digest.update(path.read_bytes())
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "frame_fingerprint": frame_fingerprint,
        "split_fingerprint": split_fingerprint,
        "horizon": int(horizon),
        "profile": profile_name,
        "features": list(features),
        "lgbm_params": LGBM_PARAMS,
        "training_code_fingerprint": code_digest.hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_oof_checkpoint(path, key, validation_index):
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        raw = np.asarray(payload["raw"], dtype=float)
        saved_index = np.asarray(payload["validation_index"], dtype=np.int64)
        if (
            payload.get("key") == key
            and np.array_equal(saved_index, np.asarray(validation_index, dtype=np.int64))
            and raw.shape == (len(validation_index),)
            and np.isfinite(raw).all()
        ):
            return raw
    except (OSError, EOFError, ValueError, TypeError, KeyError, pickle.UnpicklingError):
        return None
    return None


def _oof_profile(
    eligible, target, horizon, profile_name, numeric, categorical, splits,
    frame_fingerprint, split_fingerprints,
):
    features = numeric + categorical
    history_y, history_p, parts, fold_audit = [], [], [], []
    started = time.perf_counter()
    for fold, train_index, valid_index in splits:
        training, validation = eligible.loc[train_index], eligible.loc[valid_index]
        if training[target].nunique() < 2:
            raise RuntimeError(
                f"Alternative fold {fold}/{horizon}d has fewer than two "
                f"training classes after prefit audit: "
                f"{training[target].value_counts().to_dict()}"
            )
        checkpoint = OOF_CHECKPOINT_DIR / (
            f"alternative_{horizon}d_{profile_name}_fold{fold}.joblib"
        )
        key = _oof_checkpoint_key(
            frame_fingerprint, split_fingerprints[int(fold)], horizon,
            profile_name, features,
        )
        raw = _load_oof_checkpoint(checkpoint, key, valid_index)
        checkpoint_reused = raw is not None
        if raw is None:
            model = make_pipeline(numeric, categorical)
            _fit_with_heartbeat(
                model,
                training[features],
                training[target].astype(int),
                f"alternative {horizon}d {profile_name} fold={fold}",
            )
            raw = predict_positive_probability(model, validation[features])
            OOF_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            _replace_joblib({
                "key": key,
                "validation_index": np.asarray(valid_index, dtype=np.int64),
                "raw": raw,
            }, checkpoint)
        else:
            print(
                f"[alternative {horizon}d {profile_name}] fold={fold} "
                "verified checkpoint reused",
                flush=True,
            )
        old_calibrator = fit_calibrator(
            np.concatenate(history_y), np.concatenate(history_p)
        ) if history_y else None
        probability = calibrate(old_calibrator, raw)
        if not np.isfinite(probability).all():
            raise ValueError(
                f"Non-finite alternative probabilities in fold "
                f"{fold}/{horizon}d/{profile_name}"
            )
        columns = [name for name in (
            "ticket_id", "event_id", "landmark_at", "price",
            f"future_best_price_{horizon}d", f"potential_savings_{horizon}d",
            f"alternative_first_at_{horizon}d",
        ) if name in validation]
        part = validation[columns].copy()
        part["fold"] = fold
        part["horizon_days"] = horizon
        part["feature_profile"] = profile_name
        part["true_alternative"] = validation[target].to_numpy(int)
        part["p_alternative"] = probability
        parts.append(part)
        history_y.append(validation[target].to_numpy(int))
        history_p.append(raw)
        fold_audit.append({
            "fold": int(fold),
            "train_rows": int(len(training)),
            "valid_rows": int(len(validation)),
            "train_class_counts": {
                str(key): int(value)
                for key, value in training[target].value_counts().items()
            },
            "valid_class_counts": {
                str(key): int(value)
                for key, value in validation[target].value_counts().items()
            },
            "checkpoint_reused": checkpoint_reused,
        })
        print(
            f"[alternative {horizon}d {profile_name}] fold={fold} "
            f"train={len(training):,} valid={len(validation):,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not parts:
        raise RuntimeError(f"No binary fold for {horizon}d/{profile_name}")
    oof = pd.concat(parts, ignore_index=True).sort_values("landmark_at")
    report = metrics(oof.true_alternative.to_numpy(int), oof.p_alternative.to_numpy(float))
    report["folds"] = fold_audit
    final_calibrator = fit_calibrator(np.concatenate(history_y), np.concatenate(history_p))
    return oof, report, final_calibrator


def _select_profile(reports):
    tabular, semantic = reports["tabular"], reports["semantic"]
    improves_logloss = semantic["log_loss"] <= tabular["log_loss"] - SEMANTIC_MIN_LOGLOSS_IMPROVEMENT
    preserves_pr = semantic.get("pr_auc", 0.0) >= tabular.get("pr_auc", 0.0) - 0.005
    preserves_brier = semantic["brier"] <= tabular["brier"] + 0.0005
    return "semantic" if improves_logloss and preserves_pr and preserves_brier else "tabular"


def _chronological_profile_oof(profile_oof):
    """Select each validation fold using only earlier validation folds."""
    selected_parts, selected_by_fold = [], {}
    folds = sorted(
        set().union(*(set(frame["fold"].unique()) for frame in profile_oof.values()))
    )
    for fold in folds:
        if fold == folds[0]:
            selected = "tabular"
        else:
            history_reports = {}
            for profile_name, frame in profile_oof.items():
                history = frame[frame["fold"].lt(fold)]
                history_reports[profile_name] = metrics(
                    history.true_alternative.to_numpy(int),
                    history.p_alternative.to_numpy(float),
                )
            selected = _select_profile(history_reports)
        selected_by_fold[str(int(fold))] = selected
        selected_parts.append(
            profile_oof[selected][profile_oof[selected]["fold"].eq(fold)].copy()
        )
    if not selected_parts:
        raise RuntimeError("No chronological alternative OOF rows were selected")
    result = pd.concat(selected_parts, ignore_index=True).sort_values("landmark_at")
    keys = ["ticket_id", "landmark_at", "horizon_days"]
    if result.duplicated(keys).any():
        raise AssertionError("Duplicate chronological alternative OOF keys")
    return result, selected_by_fold


def train(data_dir: Path | None = None):
    np.random.seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = load_tickets(data_dir)
    observation_days = (tickets["last_observed_at"].max() - tickets["last_observed_at"].min()).total_seconds() / 86400
    if observation_days < max(HORIZONS_DAYS):
        raise RuntimeError(
            f"Alternative-arrival training needs at least {max(HORIZONS_DAYS)} days of clean observation; only {observation_days:.2f} days are available"
        )
    cutoff = observation_cutoff(tickets)
    prepared = prepare_end_times(tickets)
    landmarks, frame_fingerprint, cache_reused = _load_or_build_training_frame(
        tickets, prepared, cutoff
    )
    temporal_split_audit = _audit_all_temporal_splits(landmarks)
    profiles = feature_profiles(landmarks)
    models, calibrators, selected_oof_parts, ablation_oof_parts, reports = {}, {}, [], [], {}
    selected_features = {}
    for horizon in HORIZONS_DAYS:
        target = f"alternative_{horizon}d"
        eligible = landmarks[landmarks[target].ge(0)].copy().reset_index(drop=True)
        if set(eligible[target].unique()) != {0, 1}:
            raise RuntimeError(
                f"Alternative {horizon}d requires negative/positive classes; got "
                f"{eligible[target].value_counts().to_dict()}"
            )
        min_class_count = max(int(LGBM_PARAMS.get("min_child_samples", 2)), 2)
        splits = list(temporal_group_splits(
            eligible, horizon, target=target, min_class_count=min_class_count
        ))
        if len(splits) != N_TEMPORAL_FOLDS:
            raise RuntimeError(
                f"Alternative {horizon}d produced {len(splits)} temporal folds; "
                f"expected {N_TEMPORAL_FOLDS}"
            )
        split_fingerprints = _split_fingerprints(splits)
        profile_oof, profile_reports, profile_calibrators = {}, {}, {}
        for profile_name, (numeric, categorical) in profiles.items():
            oof, report, calibrator = _oof_profile(
                eligible, target, horizon, profile_name, numeric, categorical, splits,
                frame_fingerprint, split_fingerprints,
            )
            profile_oof[profile_name] = oof
            profile_reports[profile_name] = report
            profile_calibrators[profile_name] = calibrator
            ablation_oof_parts.append(oof)
        selected = _select_profile(profile_reports)
        numeric, categorical = profiles[selected]
        features = numeric + categorical
        model = make_pipeline(numeric, categorical)
        _fit_with_heartbeat(
            model,
            eligible[features],
            eligible[target].astype(int),
            f"alternative {horizon}d final {selected}",
        )
        models[str(horizon)] = model
        calibrators[str(horizon)] = profile_calibrators[selected]
        selected_features[str(horizon)] = {
            "profile": selected, "numeric": numeric, "categorical": categorical,
        }
        reports[str(horizon)] = {
            "selected_profile": selected,
            "semantic_logloss_effect": profile_reports["tabular"]["log_loss"] - profile_reports["semantic"]["log_loss"],
            "profiles": profile_reports,
        }
        chronological_oof, selected_by_fold = _chronological_profile_oof(profile_oof)
        reports[str(horizon)]["chronological_oof_profile_by_fold"] = selected_by_fold
        selected_oof_parts.append(chronological_oof)
        print(f"[alternative {horizon}d] selected_profile={selected}", flush=True)
    payload = {
        "pipeline_version": PIPELINE_VERSION, "snapshot_dir": tickets.attrs.get("snapshot_dir"),
        "observation_cutoff": str(cutoff),
        "training_frame_fingerprint": frame_fingerprint,
        "selected_features": selected_features,
        "models": models, "calibrators": calibrators,
    }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_dir": tickets.attrs.get("snapshot_dir"),
        "observation_cutoff": str(cutoff),
        "training_frame_fingerprint": frame_fingerprint,
        "training_frame_cache_reused": cache_reused,
        "temporal_split_policy": TEMPORAL_SPLIT_POLICY,
        "prefit_temporal_split_audit": temporal_split_audit,
        "invalid_listing_price_rows": int(
            tickets.attrs.get("invalid_listing_price_rows", 0)
        ),
        "invalid_listing_price_policy": tickets.attrs.get(
            "invalid_listing_price_policy"
        ),
        "excluded_temporal_anomalies": int(
            prepared.attrs.get("excluded_temporal_anomalies", 0)
        ),
        "excluded_temporal_anomaly_ticket_ids": list(
            prepared.attrs.get("excluded_temporal_anomaly_ticket_ids", [])
        ),
        "feature_count_by_horizon": {key: len(value["numeric"] + value["categorical"]) for key, value in selected_features.items()},
        "metrics_by_horizon": reports,
    }
    selected_oof = enforce_monotonic_horizons(pd.concat(selected_oof_parts, ignore_index=True))
    report["projected_metrics_by_horizon"] = {
        str(horizon): metrics(
            part.true_alternative.to_numpy(int), part.p_alternative.to_numpy(float)
        )
        for horizon in HORIZONS_DAYS
        for part in [selected_oof[selected_oof.horizon_days.eq(horizon)]]
    }
    _replace_joblib(payload, ARTIFACT_DIR / "alternative_arrival.joblib")
    temporary = ARTIFACT_DIR / "oof_predictions.csv.tmp"
    selected_oof.to_csv(temporary, index=False)
    os.replace(temporary, ARTIFACT_DIR / "oof_predictions.csv")
    temporary_ablation = ARTIFACT_DIR / "oof_ablation_predictions.csv.tmp"
    pd.concat(ablation_oof_parts, ignore_index=True).to_csv(temporary_ablation, index=False)
    os.replace(temporary_ablation, ARTIFACT_DIR / "oof_ablation_predictions.csv")
    _atomic_json(report, ARTIFACT_DIR / "training_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    print(json.dumps(train(parser.parse_args().data_dir), ensure_ascii=False, indent=2))
