"""Model, forward-chaining validation, calibration, and metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import LGBM_PARAMS, N_TEMPORAL_FOLDS, STATUS_CLASSES


def make_pipeline(
    numeric: list[str], categorical: list[str], num_classes: int = len(STATUS_CLASSES)
) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        category_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("categorical", category_pipe, categorical))
    if not transformers:
        raise ValueError("No usable features")
    parameters = dict(LGBM_PARAMS)
    parameters["num_class"] = int(num_classes)
    return Pipeline([
        ("prepare", ColumnTransformer(transformers, remainder="drop")),
        ("model", LGBMClassifier(**parameters)),
    ])


def temporal_group_splits(
    frame: pd.DataFrame,
    horizon_days: int,
    n_splits: int = N_TEMPORAL_FOLDS,
    target: str | None = None,
    min_class_count: int = 1,
):
    """Yield leakage-safe train/validation indices.

    Repeated descriptions belong to one chronological cohort. Training labels
    whose outcome window crosses the validation start are purged.
    """
    groups = (
        frame.groupby("duplicate_group", dropna=False, observed=True)["landmark_at"]
        .min().sort_values().reset_index()
    )
    ordered_groups = groups["duplicate_group"].to_numpy()
    if len(ordered_groups) < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} duplicate groups; got {len(ordered_groups)}"
        )
    # Match the original np.array_split cohort boundaries exactly.  In
    # particular, adapting the first warm-up boundary must not move any of the
    # later validation boundaries by even a few duplicate groups.
    cohort_sizes = [
        len(cohort)
        for cohort in np.array_split(np.arange(len(ordered_groups)), n_splits + 1)
    ]
    boundaries = np.cumsum(cohort_sizes).astype(int).tolist()

    # An early survival cohort can contain only the active class even though
    # later data contains all competing outcomes. Move only the first
    # validation boundary forward, never backward, until its training prefix
    # contains every class. Later 40/60/80% boundaries remain unchanged.
    if target is not None:
        if target not in frame:
            raise KeyError(f"Temporal split target is missing: {target}")
        if min_class_count < 1:
            raise ValueError("min_class_count must be at least 1")
        required_classes = set(frame[target].dropna().astype(int).unique().tolist())

        def has_enough_classes(counts: pd.Series) -> bool:
            by_class = {int(key): int(value) for key, value in counts.items()}
            return all(
                by_class.get(class_id, 0) >= min_class_count
                for class_id in required_classes
            )

        def training_class_counts(boundary_index: int):
            train_groups = set(ordered_groups[:boundary_index].tolist())
            valid_start = frame.loc[
                frame["duplicate_group"].eq(ordered_groups[boundary_index]),
                "landmark_at",
            ].min()
            mask = (
                frame["duplicate_group"].isin(train_groups)
                & (frame["landmark_at"] + pd.Timedelta(days=horizon_days) < valid_start)
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

    starts = boundaries[:-1]
    ends = boundaries[1:]
    for fold, (start, end) in enumerate(zip(starts, ends)):
        train_groups = set(ordered_groups[:start].tolist())
        valid_groups = set(ordered_groups[start:end].tolist())
        cohort_mask = frame["duplicate_group"].isin(valid_groups)
        if not cohort_mask.any():
            continue
        valid_start = frame.loc[cohort_mask, "landmark_at"].min()
        valid_end = None
        if end < len(ordered_groups):
            valid_end = frame.loc[
                frame["duplicate_group"].eq(ordered_groups[end]), "landmark_at"
            ].min()
        valid_mask = cohort_mask & frame["landmark_at"].ge(valid_start)
        if valid_end is not None:
            valid_mask &= frame["landmark_at"].lt(valid_end)
        train_mask = (
            frame["duplicate_group"].isin(train_groups)
            & (frame["landmark_at"] + pd.Timedelta(days=horizon_days) < valid_start)
        )
        if train_mask.sum() and valid_mask.sum():
            yield fold, frame.index[train_mask].to_numpy(), frame.index[valid_mask].to_numpy()


def aligned_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    prepared = model.named_steps["prepare"].transform(frame)
    classifier = model.named_steps["model"]
    raw = np.asarray(classifier.booster_.predict(prepared), dtype=float)
    classes = classifier.classes_.astype(int)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.shape != (len(frame), len(classes)):
        raise ValueError(
            f"Demand probability/class mismatch: probabilities={raw.shape}, "
            f"classes={classes.tolist()}"
        )
    aligned = np.full((len(frame), len(STATUS_CLASSES)), 1e-12, dtype=float)
    aligned[:, classes] = raw
    return aligned / aligned.sum(axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 0.05)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y_true) < 50 or len(np.unique(y_true)) < 2:
        return 1.0
    result = minimize_scalar(
        lambda value: log_loss(
            y_true,
            apply_temperature(probabilities, value),
            labels=np.arange(len(STATUS_CLASSES)),
        ),
        bounds=(0.25, 4.0),
        method="bounded",
    )
    return float(result.x) if result.success and math.isfinite(result.x) else 1.0


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(float)
    total = max(len(y_true), 1)
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            result += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    """Return finite probability rows clipped and normalized to the simplex.

    Competing-risk projection can produce values such as -2.2e-16 from
    floating-point subtraction.  Clipping only these numerical boundary
    errors and renormalizing does not change the fitted model or ranking.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(STATUS_CLASSES):
        raise ValueError(
            f"Expected probability matrix with {len(STATUS_CLASSES)} columns; "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Probability matrix contains non-finite values")
    values = np.clip(values, 0.0, 1.0)
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0.0).any():
        raise ValueError("Probability matrix contains a zero-sum row")
    return values / totals


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = normalize_probability_rows(probabilities)
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(len(STATUS_CLASSES))[y_true]
    report = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(len(STATUS_CLASSES)))),
        "multiclass_brier": float(mean_squared_error(one_hot, probabilities)),
        "ece_10bin": expected_calibration_error(y_true, probabilities),
        "class_counts": {name: int((y_true == index).sum()) for index, name in enumerate(STATUS_CLASSES)},
    }
    for index, name in enumerate(STATUS_CLASSES):
        binary = (y_true == index).astype(int)
        if len(np.unique(binary)) == 2:
            report[f"{name}_roc_auc"] = float(roc_auc_score(binary, probabilities[:, index]))
            report[f"{name}_pr_auc"] = float(average_precision_score(binary, probabilities[:, index]))
    return report


def enforce_monotonic_horizons(frame: pd.DataFrame) -> pd.DataFrame:
    """Project long-form cumulative competing-risk probabilities to monotonic paths."""
    result = frame.copy().sort_values(["ticket_id", "landmark_at", "horizon_days"])
    keys = ["ticket_id", "landmark_at"]
    result["p_sold"] = result.groupby(keys, observed=True)["p_sold"].cummax()
    result["p_deleted"] = result.groupby(keys, observed=True)["p_deleted"].cummax()
    total = result["p_sold"] + result["p_deleted"]
    scale = np.maximum(total, 1.0)
    result["p_sold"] = result["p_sold"] / scale
    result["p_deleted"] = result["p_deleted"] / scale
    result["p_active"] = np.clip(
        1.0 - result["p_sold"] - result["p_deleted"], 0.0, 1.0
    )
    normalized = normalize_probability_rows(
        result[["p_active", "p_sold", "p_deleted"]].to_numpy(float)
    )
    result[["p_active", "p_sold", "p_deleted"]] = normalized
    return result.sort_index()


def enforce_monotonic_wide(output: pd.DataFrame, horizons) -> pd.DataFrame:
    result = output.copy()
    sold_columns = [f"p_sold_{horizon}d" for horizon in horizons]
    deleted_columns = [f"p_deleted_{horizon}d" for horizon in horizons]
    sold = np.maximum.accumulate(result[sold_columns].to_numpy(float), axis=1)
    deleted = np.maximum.accumulate(result[deleted_columns].to_numpy(float), axis=1)
    scale = np.maximum(sold + deleted, 1.0)
    sold, deleted = sold / scale, deleted / scale
    active = np.clip(1.0 - sold - deleted, 0.0, 1.0)
    triples = np.stack([active, sold, deleted], axis=2)
    normalized = normalize_probability_rows(triples.reshape(-1, 3)).reshape(
        triples.shape
    )
    active, sold, deleted = (
        normalized[:, :, 0], normalized[:, :, 1], normalized[:, :, 2]
    )
    result[sold_columns] = sold
    result[deleted_columns] = deleted
    result[[f"p_active_{horizon}d" for horizon in horizons]] = active
    return result
