"""Train the standalone demand-state models. This module is never auto-run."""
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
    SEMANTIC_MANIFEST_FILE, FAIR_PRICE_CACHE, N_TEMPORAL_FOLDS, LGBM_PARAMS,
    STATUS_CLASSES,
)
from data_loader import load_tickets
from features import add_market_features, feature_profiles
from modeling import (
    aligned_probabilities,
    apply_temperature,
    fit_temperature,
    enforce_monotonic_horizons,
    make_pipeline,
    probability_metrics,
    temporal_group_splits,
)
from timeline import add_end_times, build_landmarks


TRAINING_FRAME_CACHE = ARTIFACT_DIR / "training_frame_cache.joblib"
TRAINING_FRAME_META = ARTIFACT_DIR / "training_frame_cache.json"
OOF_CHECKPOINT_DIR = ARTIFACT_DIR / "oof_checkpoints"
TEMPORAL_SPLIT_POLICY = "adaptive_warmup_min_leaf_v1"
# These checkpoints were produced with this exact fit/preprocessing policy.
# Metric-only and artifact-writing fixes must not force 24 identical refits.
# Bump this value only when data passed to model.fit or fit parameters change.
OOF_FIT_CODE_FINGERPRINT = (
    "fb465323866c846884926a71422041153d7aa8c2a6751aecd73d16ced2dc7531"
)


def _atomic_joblib(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
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
    """Hash every input and implementation file that can change the frame."""
    snapshot = Path(str(tickets.attrs.get("snapshot_dir", "")))
    model_dir = Path(__file__).resolve().parent
    paths = list(snapshot.glob("*_master.csv")) if snapshot.is_dir() else []
    paths += list(MANUAL_DIR.glob("*.csv"))
    paths += [SEMANTIC_FEATURES_FILE, SEMANTIC_MANIFEST_FILE, FAIR_PRICE_CACHE]
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


def _load_or_build_training_frame(tickets: pd.DataFrame, cutoff: pd.Timestamp):
    fingerprint = _training_frame_fingerprint(tickets)
    required = {"market_active_count", *[f"state_{horizon}d" for horizon in HORIZONS_DAYS]}
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
                        f"[demand] reused verified training-frame cache: {len(frame):,} rows",
                        flush=True,
                    )
                    return frame, fingerprint, True
                print("[demand] training-frame cache schema mismatch; rebuilding", flush=True)
        except (OSError, EOFError, ValueError, TypeError, KeyError, pickle.UnpicklingError) as error:
            print(f"[demand] invalid training-frame cache; rebuilding: {error}", flush=True)

    started = time.perf_counter()
    frame = add_market_features(build_landmarks(tickets, cutoff=cutoff), tickets)
    print(
        f"[demand] training frame ready: {len(frame):,} rows in "
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
    print(f"[demand] saved verified training-frame cache: {TRAINING_FRAME_CACHE}", flush=True)
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
    """Validate every horizon on the full frame before the first model fit.

    Keeping this separate from the per-horizon training loop prevents a late
    failure after an earlier horizon has already consumed hours of compute.
    Only the three columns needed by the splitter are copied during the audit.
    """
    required_classes = {0, 1, 2}
    min_class_count = max(int(LGBM_PARAMS.get("min_child_samples", 2)), 2)
    audit: dict[str, list[dict]] = {}
    for horizon in HORIZONS_DAYS:
        target = f"state_{horizon}d"
        columns = ["duplicate_group", "landmark_at", target]
        eligible = landmarks.loc[landmarks[target].ge(0), columns].reset_index(drop=True)
        full_counts = eligible[target].value_counts().sort_index()
        if set(full_counts.index.astype(int)) != required_classes:
            raise RuntimeError(
                f"Demand {horizon}d requires active/sold/deleted classes; got "
                f"{full_counts.to_dict()}"
            )
        splits = list(temporal_group_splits(
            eligible, horizon, target=target, min_class_count=min_class_count
        ))
        if len(splits) != N_TEMPORAL_FOLDS:
            raise RuntimeError(
                f"Demand {horizon}d produced {len(splits)} temporal folds; "
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
                    f"Demand prefit audit failed for fold {fold}/{horizon}d: "
                    f"training classes={train_counts.to_dict()}, "
                    f"minimum_per_class={min_class_count}"
                )
            train_label_end = (
                training["landmark_at"] + pd.Timedelta(days=horizon)
            ).max()
            valid_start = validation["landmark_at"].min()
            if not train_label_end < valid_start:
                raise RuntimeError(
                    f"Demand prefit audit found temporal leakage in fold "
                    f"{fold}/{horizon}d: train_label_end={train_label_end}, "
                    f"valid_start={valid_start}"
                )
            if set(training["duplicate_group"]) & set(validation["duplicate_group"]):
                raise RuntimeError(
                    f"Demand prefit audit found duplicate-group leakage in fold "
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
                f"[demand prefit {horizon}d] fold={fold} "
                f"train={len(training):,} valid={len(validation):,} "
                f"train_classes={train_counts.to_dict()}",
                flush=True,
            )
        audit[str(horizon)] = fold_audit
    print("[demand] all full-frame temporal folds passed prefit audit", flush=True)
    return audit


def _oof_checkpoint_key(
    frame_fingerprint, split_fingerprint, horizon, profile_name, feature_names
):
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "frame_fingerprint": frame_fingerprint,
        "split_fingerprint": split_fingerprint,
        "horizon": int(horizon),
        "profile": profile_name,
        "features": list(feature_names),
        "lgbm_params": LGBM_PARAMS,
        "training_code_fingerprint": OOF_FIT_CODE_FINGERPRINT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_oof_checkpoint(path, key, validation_index, probability_columns):
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        raw = np.asarray(payload["raw"], dtype=float)
        saved_index = np.asarray(payload["validation_index"], dtype=np.int64)
        if (
            payload.get("key") == key
            and np.array_equal(saved_index, np.asarray(validation_index, dtype=np.int64))
            and raw.shape == (len(validation_index), probability_columns)
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
    feature_names = numeric + categorical
    raw_history_y, raw_history_p, fold_parts, fold_audit = [], [], [], []
    started = time.perf_counter()
    for fold, train_index, valid_index in splits:
        training, validation = eligible.loc[train_index], eligible.loc[valid_index]
        observed_classes = set(training[target].astype(int).unique())
        required_classes = set(range(len(STATUS_CLASSES)))
        if observed_classes != required_classes:
            raise RuntimeError(
                f"Demand fold {fold}/{horizon}d is missing training classes after "
                f"prefit audit: {training[target].value_counts().to_dict()}"
            )
        class_count = int(training[target].nunique())
        checkpoint = OOF_CHECKPOINT_DIR / (
            f"demand_{horizon}d_{profile_name}_fold{fold}.joblib"
        )
        key = _oof_checkpoint_key(
            frame_fingerprint, split_fingerprints[int(fold)], horizon,
            profile_name, feature_names,
        )
        raw = _load_oof_checkpoint(
            checkpoint, key, valid_index, len(STATUS_CLASSES)
        )
        checkpoint_reused = raw is not None
        if raw is None:
            model = make_pipeline(
                numeric, categorical, num_classes=class_count
            )
            _fit_with_heartbeat(
                model,
                training[feature_names],
                training[target].astype(int),
                f"demand {horizon}d {profile_name} fold={fold}",
            )
            raw = aligned_probabilities(model, validation[feature_names])
            OOF_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            _atomic_joblib({
                "key": key,
                "validation_index": np.asarray(valid_index, dtype=np.int64),
                "raw": raw,
            }, checkpoint)
        else:
            print(
                f"[demand {horizon}d {profile_name}] fold={fold} "
                "verified checkpoint reused",
                flush=True,
            )
        temperature = fit_temperature(
            np.concatenate(raw_history_y), np.vstack(raw_history_p)
        ) if raw_history_y else 1.0
        calibrated = apply_temperature(raw, temperature)
        if not np.isfinite(calibrated).all():
            raise ValueError(
                f"Non-finite demand probabilities in fold {fold}/{horizon}d/{profile_name}"
            )
        part_columns = [
            column for column in [
                "ticket_id", "event_id", "landmark_at", "price", "fair_price",
                "market_price_median", "market_prior_sold_median", "outcome_at",
            ] if column in validation
        ]
        part = validation[part_columns].copy()
        part["fold"] = fold
        part["horizon_days"] = horizon
        part["feature_profile"] = profile_name
        part["true_state"] = validation[target].to_numpy(int)
        part[["p_active", "p_sold", "p_deleted"]] = calibrated
        fold_parts.append(part)
        raw_history_y.append(validation[target].to_numpy(int))
        raw_history_p.append(raw)
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
            f"[demand {horizon}d {profile_name}] fold={fold} "
            f"train={len(training):,} valid={len(validation):,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if not fold_parts:
        raise RuntimeError(f"No temporal validation fold for {horizon}d/{profile_name}")
    oof = pd.concat(fold_parts, ignore_index=True).sort_values("landmark_at")
    report = probability_metrics(
        oof["true_state"].to_numpy(int),
        oof[["p_active", "p_sold", "p_deleted"]].to_numpy(float),
    )
    report["folds"] = fold_audit
    final_temperature = fit_temperature(np.concatenate(raw_history_y), np.vstack(raw_history_p))
    return oof, report, final_temperature


def _select_profile(profile_reports):
    tabular = profile_reports["tabular"]
    semantic = profile_reports["semantic"]
    improves_logloss = semantic["log_loss"] <= tabular["log_loss"] - SEMANTIC_MIN_LOGLOSS_IMPROVEMENT
    preserves_brier = semantic["multiclass_brier"] <= tabular["multiclass_brier"] + 0.0005
    return "semantic" if improves_logloss and preserves_brier else "tabular"


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
                history_reports[profile_name] = probability_metrics(
                    history["true_state"].to_numpy(int),
                    history[["p_active", "p_sold", "p_deleted"]].to_numpy(float),
                )
            selected = _select_profile(history_reports)
        selected_by_fold[str(int(fold))] = selected
        selected_parts.append(
            profile_oof[selected][profile_oof[selected]["fold"].eq(fold)].copy()
        )
    if not selected_parts:
        raise RuntimeError("No chronological demand OOF rows were selected")
    result = pd.concat(selected_parts, ignore_index=True).sort_values("landmark_at")
    keys = ["ticket_id", "landmark_at", "horizon_days"]
    if result.duplicated(keys).any():
        raise AssertionError("Duplicate chronological demand OOF keys")
    return result, selected_by_fold


def train(data_dir: Path | None = None) -> dict:
    np.random.seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tickets = load_tickets(data_dir)
    tickets, cutoff = add_end_times(tickets)
    landmarks, frame_fingerprint, cache_reused = _load_or_build_training_frame(
        tickets, cutoff
    )
    temporal_split_audit = _audit_all_temporal_splits(landmarks)
    profiles = feature_profiles(landmarks)
    models, temperatures, reports, selected_oof_parts, ablation_oof_parts = {}, {}, {}, [], []
    selected_features = {}

    for horizon in HORIZONS_DAYS:
        target = f"state_{horizon}d"
        eligible = landmarks.loc[landmarks[target].ge(0)].copy().reset_index(drop=True)
        if set(eligible[target].unique()) != {0, 1, 2}:
            raise RuntimeError(
                f"Demand {horizon}d requires active/sold/deleted classes; got "
                f"{eligible[target].value_counts().to_dict()}"
            )
        min_class_count = max(int(LGBM_PARAMS.get("min_child_samples", 2)), 2)
        splits = list(temporal_group_splits(
            eligible, horizon, target=target, min_class_count=min_class_count
        ))
        if len(splits) != N_TEMPORAL_FOLDS:
            raise RuntimeError(
                f"Demand {horizon}d produced {len(splits)} temporal folds; "
                f"expected {N_TEMPORAL_FOLDS}"
            )
        split_fingerprints = _split_fingerprints(splits)
        profile_oof, profile_reports, profile_temperatures = {}, {}, {}
        for profile_name, (numeric, categorical) in profiles.items():
            oof, profile_report, temperature = _oof_profile(
                eligible, target, horizon, profile_name, numeric, categorical, splits,
                frame_fingerprint, split_fingerprints,
            )
            profile_oof[profile_name] = oof
            profile_reports[profile_name] = profile_report
            profile_temperatures[profile_name] = temperature
            ablation_oof_parts.append(oof)
        selected = _select_profile(profile_reports)
        numeric, categorical = profiles[selected]
        feature_names = numeric + categorical
        final_model = make_pipeline(
            numeric, categorical, num_classes=eligible[target].nunique()
        )
        _fit_with_heartbeat(
            final_model,
            eligible[feature_names],
            eligible[target].astype(int),
            f"demand {horizon}d final {selected}",
        )
        models[str(horizon)] = final_model
        temperatures[str(horizon)] = profile_temperatures[selected]
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
        print(f"[demand {horizon}d] selected_profile={selected}", flush=True)

    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_dir": tickets.attrs.get("snapshot_dir"),
        "observation_cutoff": str(cutoff),
        "training_frame_fingerprint": frame_fingerprint,
        "excluded_temporal_anomalies": int(
            tickets.attrs.get("excluded_temporal_anomalies", 0)
        ),
        "invalid_listing_price_rows": int(
            tickets.attrs.get("invalid_listing_price_rows", 0)
        ),
        "selected_features": selected_features,
        "models": models,
        "temperatures": temperatures,
    }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "snapshot_dir": tickets.attrs.get("snapshot_dir"),
        "observation_cutoff": str(cutoff),
        "training_frame_fingerprint": frame_fingerprint,
        "training_frame_cache_reused": cache_reused,
        "temporal_split_policy": TEMPORAL_SPLIT_POLICY,
        "prefit_temporal_split_audit": temporal_split_audit,
        "excluded_temporal_anomalies": int(
            tickets.attrs.get("excluded_temporal_anomalies", 0)
        ),
        "excluded_temporal_anomaly_ticket_ids": list(
            tickets.attrs.get("excluded_temporal_anomaly_ticket_ids", [])
        ),
        "invalid_listing_price_rows": int(
            tickets.attrs.get("invalid_listing_price_rows", 0)
        ),
        "invalid_listing_price_policy": tickets.attrs.get(
            "invalid_listing_price_policy"
        ),
        "feature_count_by_horizon": {key: len(value["numeric"] + value["categorical"]) for key, value in selected_features.items()},
        "metrics_by_horizon": reports,
    }
    selected_oof = enforce_monotonic_horizons(pd.concat(selected_oof_parts, ignore_index=True))
    report["projected_metrics_by_horizon"] = {
        str(horizon): probability_metrics(
            part["true_state"].to_numpy(int),
            part[["p_active", "p_sold", "p_deleted"]].to_numpy(float),
        )
        for horizon in HORIZONS_DAYS
        for part in [selected_oof[selected_oof.horizon_days.eq(horizon)]]
    }
    _atomic_joblib(payload, ARTIFACT_DIR / "demand_state.joblib")
    _atomic_csv(selected_oof, ARTIFACT_DIR / "oof_predictions.csv")
    _atomic_csv(pd.concat(ablation_oof_parts, ignore_index=True), ARTIFACT_DIR / "oof_ablation_predictions.csv")
    _atomic_json(report, ARTIFACT_DIR / "training_report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
