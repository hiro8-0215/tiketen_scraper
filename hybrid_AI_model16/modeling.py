"""Leakage-safe components shared by Model16 training and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    BLEND_MAPE_TOLERANCE_PCT_POINTS,
    CATBOOST_THREAD_COUNT,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    RAW_PRICE_SCALE,
    SEED,
    WEIGHT_EPSILON,
)


def metrics(y, prediction):
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Targets must be finite positive prices")
    if not np.isfinite(prediction).all():
        raise ValueError("Predictions contain NaN or infinity")
    prediction = np.maximum(0, prediction)
    error = np.abs(prediction - y)
    ape = error / np.maximum(y, 1)
    return {
        "count": int(len(y)),
        "mae_yen": float(error.mean()),
        "rmse_yen": float(mean_squared_error(y, prediction) ** 0.5),
        "mape_pct": float(ape.mean() * 100),
        "mdape_pct": float(np.median(ape) * 100),
        "wmape_pct": float(error.sum() / max(y.sum(), 1) * 100),
        "within_20_pct": float((ape <= 0.20).mean() * 100),
        "r2": float(r2_score(y, prediction)),
        "bias_yen": float((prediction - y).mean()),
    }


def make_tabular(numeric, categorical):
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def assign_inner_folds(df: pd.DataFrame, n_splits: int, seed: int, target: str):
    bins = pd.qcut(np.log1p(df[target]), q=10, labels=False, duplicates="drop")
    result = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(
        splitter.split(df, bins, df["duplicate_group"])
    ):
        result[validation] = fold
    if (result < 0).any():
        raise AssertionError("Inner fold assignment is incomplete")
    return result


def lgbm_trial_parameters(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 128),
        "max_depth": trial.suggest_int("max_depth", 5, 13),
        "min_child_samples": trial.suggest_int("min_child_samples", 8, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.95),
        "subsample": trial.suggest_float("subsample", 0.60, 0.98),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
    }


def catboost_trial_parameters(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.05, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 3),
    }


def lgbm_base_parameters(objective, overrides, n_estimators=MAX_BOOST_ROUNDS):
    result = {
        "objective": objective,
        "n_estimators": int(n_estimators),
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
        "deterministic": True,
        "metric": "None",
        "first_metric_only": True,
    }
    result.update(overrides)
    return result


def mape_training_weight(y):
    y = np.maximum(np.asarray(y, float), 1.0)
    weight = np.median(y) / y
    return weight / np.mean(weight)


def fit_lgbm(x_fit, y_fit_yen, x_val, y_val_yen, family, parameters):
    if family == "lgbm_log_mae":
        target_fit = np.log1p(y_fit_yen)
        target_val = np.log1p(y_val_yen)
        sample_weight = None
        objective = "regression_l1"

        def evaluate(y_true, pred):
            return "mae_yen", float(np.abs(np.expm1(pred) - np.expm1(y_true)).mean()), False

        convert = lambda pred: np.maximum(0, np.expm1(pred))
    elif family == "lgbm_raw_mape":
        target_fit = np.asarray(y_fit_yen) / RAW_PRICE_SCALE
        target_val = np.asarray(y_val_yen) / RAW_PRICE_SCALE
        sample_weight = mape_training_weight(y_fit_yen)
        objective = "regression_l1"

        def evaluate(y_true, pred):
            true_yen = y_true * RAW_PRICE_SCALE
            pred_yen = np.maximum(0, pred * RAW_PRICE_SCALE)
            return "mape_pct", float(np.mean(np.abs(pred_yen - true_yen) / np.maximum(true_yen, 1)) * 100), False

        convert = lambda pred: np.maximum(0, pred * RAW_PRICE_SCALE)
    else:
        raise ValueError(f"Unknown LightGBM family: {family}")

    model = LGBMRegressor(**lgbm_base_parameters(objective, parameters))
    model.fit(
        x_fit,
        target_fit,
        sample_weight=sample_weight,
        eval_set=[(x_val, target_val)],
        eval_metric=evaluate,
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )
    native = model.booster_.predict(x_val, num_iteration=model.best_iteration_)
    return model, convert(native)


def fit_lgbm_full(x, y_yen, family, parameters, iterations):
    if family == "lgbm_log_mae":
        target = np.log1p(y_yen)
        sample_weight = None
    elif family == "lgbm_raw_mape":
        target = np.asarray(y_yen) / RAW_PRICE_SCALE
        sample_weight = mape_training_weight(y_yen)
    else:
        raise ValueError(f"Unknown LightGBM family: {family}")
    model = LGBMRegressor(
        **lgbm_base_parameters("regression_l1", parameters, iterations)
    )
    model.fit(x, target, sample_weight=sample_weight)
    return model


def catboost_parameters(overrides, iterations=MAX_BOOST_ROUNDS):
    result = {
        "iterations": int(iterations),
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": SEED,
        "thread_count": CATBOOST_THREAD_COUNT,
        "task_type": "CPU",
        "boosting_type": "Ordered",
        "bootstrap_type": "Bayesian",
        "allow_writing_files": False,
        "verbose": False,
    }
    result.update(overrides)
    return result


def fit_catboost(fit_frame, y_fit_yen, val_frame, y_val_yen, categorical, parameters):
    train_pool = Pool(
        fit_frame,
        label=np.asarray(y_fit_yen) / RAW_PRICE_SCALE,
        cat_features=categorical,
    )
    validation_pool = Pool(
        val_frame,
        label=np.asarray(y_val_yen) / RAW_PRICE_SCALE,
        cat_features=categorical,
    )
    model = CatBoostRegressor(**catboost_parameters(parameters))
    model.fit(
        train_pool,
        eval_set=validation_pool,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    prediction = np.maximum(0, model.predict(validation_pool) * RAW_PRICE_SCALE)
    return model, prediction


def fit_catboost_full(frame, y_yen, categorical, parameters, iterations):
    pool = Pool(
        frame,
        label=np.asarray(y_yen) / RAW_PRICE_SCALE,
        cat_features=categorical,
    )
    model = CatBoostRegressor(**catboost_parameters(parameters, iterations))
    model.fit(pool, verbose=False)
    return model


def optimize_global_weights(y, prediction_matrix, tolerance_pct_points=BLEND_MAPE_TOLERANCE_PCT_POINTS):
    """Find one constant simplex weight vector; never route individual rows."""
    y = np.asarray(y, float)
    prediction_matrix = np.asarray(prediction_matrix, float)
    if prediction_matrix.ndim != 2 or prediction_matrix.shape[0] != len(y):
        raise ValueError("prediction_matrix must be rows x experts")
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Targets must be finite positive prices")
    if not np.isfinite(prediction_matrix).all():
        raise ValueError("Expert predictions contain NaN or infinity")
    expert_mae = np.mean(np.abs(prediction_matrix - y[:, None]), axis=0)
    best = int(np.argmin(expert_mae))
    base_weights = np.zeros(prediction_matrix.shape[1], dtype=float)
    base_weights[best] = 1.0
    base_mape = np.mean(
        np.abs(prediction_matrix[:, best] - y) / np.maximum(y, 1)
    )
    mape_limit = base_mape + tolerance_pct_points / 100.0

    def mae(weights):
        return float(np.mean(np.abs(prediction_matrix @ weights - y)))

    def mape_margin(weights):
        prediction = prediction_matrix @ weights
        value = np.mean(np.abs(prediction - y) / np.maximum(y, 1))
        return float(mape_limit - value)

    starts = [base_weights, np.full(prediction_matrix.shape[1], 1 / prediction_matrix.shape[1])]
    starts.extend(np.eye(prediction_matrix.shape[1]))
    best_weights, best_score = base_weights, mae(base_weights)
    for start in starts:
        result = minimize(
            mae,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * prediction_matrix.shape[1],
            constraints=[
                {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
                {"type": "ineq", "fun": mape_margin},
            ],
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if result.success and mape_margin(result.x) >= -1e-7:
            score = mae(result.x)
            if score < best_score:
                best_weights, best_score = result.x, score
    best_weights = np.maximum(0, best_weights)
    best_weights /= best_weights.sum()
    # SLSQP can leave numerical dust on an unused expert.  Removing it prevents
    # a practically zero BERT coefficient from needlessly requiring embeddings
    # during inference.
    pruned = best_weights.copy()
    pruned[pruned < WEIGHT_EPSILON] = 0.0
    pruned /= pruned.sum()
    if mape_margin(pruned) >= -1e-7:
        best_weights = pruned
    return best_weights


def blend_predictions(prediction_matrix, weights):
    prediction_matrix = np.asarray(prediction_matrix, float)
    weights = np.asarray(weights, float)
    if prediction_matrix.ndim != 2 or weights.shape != (prediction_matrix.shape[1],):
        raise ValueError("Blend inputs have incompatible dimensions")
    if not np.isfinite(prediction_matrix).all() or not np.isfinite(weights).all():
        raise ValueError("Blend inputs contain NaN or infinity")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Blend weights must be a nonnegative simplex vector")
    return np.maximum(0, prediction_matrix @ weights)
