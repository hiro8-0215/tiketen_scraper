"""Binary arrival model, forward validation, calibration, and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import LGBM_PARAMS, N_TEMPORAL_FOLDS


def make_pipeline(numeric, categorical):
    transforms = []
    if numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical))
    return Pipeline([
        ("prepare", ColumnTransformer(transforms, remainder="drop")),
        ("model", LGBMClassifier(**LGBM_PARAMS)),
    ])


def predict_positive_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Predict via Booster to avoid sklearn feature-name warning spam."""
    prepared = model.named_steps["prepare"].transform(frame)
    classifier = model.named_steps["model"]
    classes = classifier.classes_.astype(int)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Alternative model classes must be [0, 1], got {classes.tolist()}")
    probability = np.asarray(classifier.booster_.predict(prepared), dtype=float)
    if probability.shape != (len(frame),):
        raise ValueError(
            f"Alternative probability shape mismatch: {probability.shape}"
        )
    return probability


def temporal_group_splits(
    frame, horizon, n_splits=N_TEMPORAL_FOLDS, target=None, min_class_count=1
):
    groups = frame.groupby(
        "duplicate_group", dropna=False, observed=True
    )["landmark_at"].min().sort_values().reset_index()
    ordered_groups = groups.duplicate_group.to_numpy()
    if len(ordered_groups) < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} duplicate groups; got {len(ordered_groups)}"
        )
    # Preserve the exact original np.array_split boundaries for folds 1+;
    # only the first warm-up boundary may be adapted below.
    cohort_sizes = [
        len(cohort)
        for cohort in np.array_split(np.arange(len(ordered_groups)), n_splits + 1)
    ]
    boundaries = np.cumsum(cohort_sizes).astype(int).tolist()

    if target is not None:
        if target not in frame:
            raise KeyError(f"Temporal split target is missing: {target}")
        if min_class_count < 1:
            raise ValueError("min_class_count must be at least 1")
        required_classes = set(frame[target].dropna().astype(int).unique().tolist())

        def has_enough_classes(counts):
            by_class = {int(key): int(value) for key, value in counts.items()}
            return all(
                by_class.get(class_id, 0) >= min_class_count
                for class_id in required_classes
            )

        def training_class_counts(boundary_index):
            training_groups = set(ordered_groups[:boundary_index].tolist())
            valid_start = frame.loc[
                frame.duplicate_group.eq(ordered_groups[boundary_index]),
                "landmark_at",
            ].min()
            mask = frame.duplicate_group.isin(training_groups) & (
                frame.landmark_at + pd.Timedelta(days=horizon) < valid_start
            )
            return frame.loc[mask, target].value_counts(), valid_start

        original = boundaries[0]
        counts, _ = training_class_counts(original)
        if not has_enough_classes(counts):
            low, high = original + 1, boundaries[1] - 1
            found = None
            while low <= high:
                middle = (low + high) // 2
                middle_counts, _ = training_class_counts(middle)
                if has_enough_classes(middle_counts):
                    found = middle
                    high = middle - 1
                else:
                    low = middle + 1
            if found is None:
                raise RuntimeError(
                    f"Cannot establish all target classes before the second temporal "
                    f"boundary for {target}: required={sorted(required_classes)}, "
                    f"min_class_count={min_class_count}"
                )
            boundaries[0] = found
            counts, valid_start = training_class_counts(found)
            print(
                f"[temporal split {target}] warm-up groups {original:,}->{found:,}; "
                f"first validation={valid_start}; min_class_count={min_class_count}; "
                f"train_classes={counts.to_dict()}",
                flush=True,
            )

    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        valid_groups = set(ordered_groups[start:end].tolist())
        cohort_mask = frame.duplicate_group.isin(valid_groups)
        valid_start = frame.loc[cohort_mask, "landmark_at"].min()
        valid_end = None
        if end < len(ordered_groups):
            valid_end = frame.loc[
                frame.duplicate_group.eq(ordered_groups[end]), "landmark_at"
            ].min()
        valid_mask = cohort_mask & frame.landmark_at.ge(valid_start)
        if valid_end is not None:
            valid_mask &= frame.landmark_at.lt(valid_end)
        training_groups = set(ordered_groups[:start].tolist())
        train_mask = frame.duplicate_group.isin(training_groups) & ((frame.landmark_at + pd.Timedelta(days=horizon)) < valid_start)
        if train_mask.any() and valid_mask.any():
            yield fold, frame.index[train_mask].to_numpy(), frame.index[valid_mask].to_numpy()


def fit_calibrator(y_true, probability):
    if len(y_true) < 50 or len(np.unique(y_true)) < 2:
        return None
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    return LogisticRegression(C=1.0, solver="lbfgs").fit(logits.reshape(-1, 1), y_true)


def calibrate(calibrator, probability):
    if calibrator is None:
        return np.asarray(probability, dtype=float)
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def metrics(y_true, probability):
    report = {
        "rows": int(len(y_true)), "positive_rows": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        report["roc_auc"] = float(roc_auc_score(y_true, probability))
        report["pr_auc"] = float(average_precision_score(y_true, probability))
    return report


def enforce_monotonic_horizons(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_values(["ticket_id", "landmark_at", "horizon_days"])
    result["p_alternative"] = result.groupby(
        ["ticket_id", "landmark_at"], observed=True
    )["p_alternative"].cummax()
    return result.sort_index()


def enforce_monotonic_wide(frame: pd.DataFrame, horizons) -> pd.DataFrame:
    result = frame.copy()
    columns = [f"p_alternative_{horizon}d" for horizon in horizons]
    result[columns] = np.maximum.accumulate(result[columns].to_numpy(float), axis=1)
    return result
