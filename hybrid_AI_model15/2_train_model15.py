"""Model 15 leakage-safe feature/family selection and final LightGBM fit.

Qwen and BERT are treated as optional inputs instead of assumptions.  A joint
Optuna study selects the feature profile, objective family, and LightGBM
hyperparameters on the same duplicate-grouped OOF folds.  The production model
is then fitted with exactly the winning feature profile.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    ARTIFACT_DIR,
    BERT_PCA_DIM,
    LGBM_OPTUNA_TRIALS,
    N_FOLDS,
    PIPELINE_VERSION,
    QWEN_OOF_SCHEMA_VERSION,
    RAW_PRICE_SCALE,
    SEED,
    SEMANTIC_CATEGORICAL_FEATURES,
    SEMANTIC_NUMERIC_FEATURES,
    TARGET,
)
from data_loader import latest_data_dir, model_feature_columns, prepare_dataset
from qwen_prompt import qwen_dataset_fingerprint
from qwen_validation import qwen_oof_diagnostics


# Each profile is searched rather than presumed to be useful.  JSON semantics
# remain in the tabular block and are measured separately with a fair ablation.
FEATURE_PROFILES = {
    "full": {"bert": True, "qwen": True},
    "qwen_only": {"bert": False, "qwen": True},
    "bert_only": {"bert": True, "qwen": False},
    "tabular_only": {"bert": False, "qwen": False},
}
MODEL_FAMILIES = ("log_l1", "log_weighted", "raw_huber")


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.maximum(0, np.asarray(pred, float))
    error = np.abs(pred - y)
    ape = error / np.maximum(y, 1)
    return {
        "count": int(len(y)),
        "mae_yen": float(error.mean()),
        "rmse_yen": float(mean_squared_error(y, pred) ** 0.5),
        "mape_pct": float(ape.mean() * 100),
        "mdape_pct": float(np.median(ape) * 100),
        "wmape_pct": float(error.sum() / np.maximum(y.sum(), 1) * 100),
        "within_20_pct": float((ape <= 0.2).mean() * 100),
        "r2": float(r2_score(y, pred)),
        "bias_yen": float((pred - y).mean()),
    }


def make_tabular(numeric, categorical):
    return ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def base_params(objective, overrides=None):
    result = dict(
        objective=objective,
        n_estimators=3500,
        learning_rate=0.015,
        num_leaves=48,
        max_depth=10,
        min_child_samples=20,
        colsample_bytree=0.82,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
        deterministic=True,
        metric="None",
        first_metric_only=True,
    )
    if overrides:
        result.update(overrides)
    return result


def family_settings(family, y_log, y_scaled, weight):
    if family == "log_l1":
        return "regression_l1", y_log, None, "log"
    if family == "log_weighted":
        return "regression", y_log, weight, "log"
    if family == "raw_huber":
        return "huber", y_scaled, weight, "scaled_raw"
    raise ValueError(f"Unknown model family: {family}")


def native_prediction_to_yen(prediction, family):
    prediction = np.asarray(prediction, float)
    if family in {"log_l1", "log_weighted"}:
        return np.maximum(0, np.expm1(prediction))
    if family == "raw_huber":
        return np.maximum(0, prediction * RAW_PRICE_SCALE)
    raise ValueError(f"Unknown model family: {family}")


def fit_model(
    x_fit,
    y_fit,
    x_val,
    y_val,
    objective,
    weight=None,
    overrides=None,
    target_space="log",
):
    """Fit and early-stop on yen MAE, preserving sparse feature order."""
    if target_space not in {"log", "scaled_raw"}:
        raise ValueError(f"Unknown target space: {target_space}")

    def yen_mae(y_true, prediction):
        if target_space == "log":
            y_true = np.expm1(y_true)
            prediction = np.maximum(0, np.expm1(prediction))
        else:
            y_true = y_true * RAW_PRICE_SCALE
            prediction = np.maximum(0, prediction * RAW_PRICE_SCALE)
        return "mae_yen", float(np.mean(np.abs(prediction - y_true))), False

    model = LGBMRegressor(**base_params(objective, overrides))
    model.fit(
        x_fit,
        y_fit,
        sample_weight=weight,
        eval_set=[(x_val, y_val)],
        eval_metric=yen_mae,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    # Calling LGBMRegressor.predict() makes sklearn compare LightGBM's synthetic
    # feature names with a nameless CSR matrix.  Booster.predict() is equivalent
    # here and avoids the misleading repeated warning without hiding real checks.
    prediction = model.booster_.predict(
        x_val,
        num_iteration=model.best_iteration_,
    )
    return model, prediction


def assemble_profile(tabular, bert, qwen, profile):
    """Join blocks in one fixed order for a named feature profile."""
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {profile}")
    parts = [csr_matrix(tabular)]
    settings = FEATURE_PROFILES[profile]
    if settings["bert"]:
        parts.append(csr_matrix(bert))
    if settings["qwen"]:
        parts.append(csr_matrix(qwen))
    return hstack(parts, format="csr") if len(parts) > 1 else parts[0]


def trial_parameters(trial, family):
    result = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.06, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.95),
        "subsample": trial.suggest_float("subsample", 0.55, 0.95),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    if family in {"log_weighted", "raw_huber"}:
        result["price_weight_power"] = trial.suggest_float(
            "price_weight_power", 0.20, 0.80
        )
    if family == "raw_huber":
        result["alpha"] = trial.suggest_float("alpha", 0.70, 0.95)
    return result


def default_trial_parameters():
    return {
        "learning_rate": 0.015,
        "num_leaves": 48,
        "max_depth": 10,
        "min_child_samples": 20,
        "colsample_bytree": 0.82,
        "subsample": 0.85,
        "subsample_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }


def seeded_trial_parameters(family, prior_weighted):
    result = dict(
        prior_weighted if family == "log_weighted" else default_trial_parameters()
    )
    if family in {"log_weighted", "raw_huber"}:
        result["price_weight_power"] = 0.50
    if family == "raw_huber":
        result["alpha"] = 0.90
    return result


def resolve_training_parameters(parameters, family, price_ratio):
    """Separate LightGBM parameters from label-derived training weights."""
    model_parameters = dict(parameters)
    power = float(model_parameters.pop("price_weight_power", 0.50))
    if family in {"log_weighted", "raw_huber"}:
        all_weight = np.power(price_ratio, power)
    else:
        all_weight = None
    return model_parameters, all_weight, power


def previous_best_weighted_parameters():
    """Use the completed v2 best trial as a seed, never as a final answer."""
    path = ARTIFACT_DIR / "evaluation_model15.json"
    if not path.exists():
        return default_trial_parameters()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        previous = report.get("lgbm_optuna_best_params", {})
        required = set(default_trial_parameters())
        if required.issubset(previous):
            return {name: previous[name] for name in required}
    except (OSError, ValueError, TypeError):
        pass
    return default_trial_parameters()


def tune_joint_search(fold_cache, y_log, y_scaled, y_raw, dataset_fingerprint):
    """Jointly tune feature inclusion, target family, and tree parameters."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = (ARTIFACT_DIR / "model15_optuna.db").resolve().as_posix()
    study_name = f"model15_joint_profile_family_v3_{dataset_fingerprint[:16]}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    # Guarantee that every one of the 12 combinations has a valid baseline.
    # The former 50-trial winner is transferred only as an initial suggestion.
    prior_weighted = previous_best_weighted_parameters()
    existing_combinations = {
        (
            trial.params.get("feature_profile"),
            trial.params.get("model_family"),
        )
        for trial in study.trials
        if trial.state in {
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.COMPLETE,
        }
    }
    for profile in FEATURE_PROFILES:
        for family in MODEL_FAMILIES:
            if (profile, family) in existing_combinations:
                continue
            seed = seeded_trial_parameters(family, prior_weighted)
            study.enqueue_trial(
                {"feature_profile": profile, "model_family": family, **seed},
            )

    def objective(trial):
        profile = trial.suggest_categorical("feature_profile", list(FEATURE_PROFILES))
        family = trial.suggest_categorical("model_family", list(MODEL_FAMILIES))
        hyperparameters = trial_parameters(trial, family)
        oof = np.zeros(len(y_raw), dtype=float)
        for cached in fold_cache:
            fit, val = cached["fit"], cached["val"]
            model_parameters, all_weight, _ = resolve_training_parameters(
                hyperparameters, family, cached["price_ratio"]
            )
            objective_name, target, all_weight, target_space = family_settings(
                family, y_log, y_scaled, all_weight
            )
            fit_weight = None if all_weight is None else all_weight[fit]
            model, native_prediction = fit_model(
                cached["profiles"][profile]["fit"],
                target[fit],
                cached["profiles"][profile]["val"],
                target[val],
                objective_name,
                fit_weight,
                model_parameters,
                target_space,
            )
            oof[val] = native_prediction_to_yen(native_prediction, family)
            del model
        score = float(np.mean(np.abs(oof - y_raw)))
        trial.set_user_attr("mae_yen", score)
        gc.collect()
        return score

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, LGBM_OPTUNA_TRIALS - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    return study, study_name


def best_trials_by_combination(study):
    best = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        profile = trial.params.get("feature_profile")
        family = trial.params.get("model_family")
        if profile not in FEATURE_PROFILES or family not in MODEL_FAMILIES:
            continue
        key = (profile, family)
        if key not in best or trial.value < best[key].value:
            best[key] = trial
    missing = [
        (profile, family)
        for profile in FEATURE_PROFILES
        for family in MODEL_FAMILIES
        if (profile, family) not in best
    ]
    if missing:
        raise RuntimeError(f"Joint Optuna search lacks baseline combinations: {missing}")
    return best


def candidate_oof(fold_cache, y_log, y_scaled, family, profile, hyperparameters, no_semantic=False):
    result = np.zeros(len(y_log), dtype=float)
    iterations = []
    for cached in fold_cache:
        fit, val = cached["fit"], cached["val"]
        model_parameters, all_weight, _ = resolve_training_parameters(
            hyperparameters, family, cached["price_ratio"]
        )
        objective_name, target, all_weight, target_space = family_settings(
            family, y_log, y_scaled, all_weight
        )
        fit_weight = None if all_weight is None else all_weight[fit]
        matrix_key = "profiles_no_semantic" if no_semantic else "profiles"
        model, native_prediction = fit_model(
            cached[matrix_key][profile]["fit"],
            target[fit],
            cached[matrix_key][profile]["val"],
            target[val],
            objective_name,
            fit_weight,
            model_parameters,
            target_space,
        )
        result[val] = native_prediction_to_yen(native_prediction, family)
        iterations.append(int(model.best_iteration_))
        del model
    gc.collect()
    return result, iterations


def scope_metrics(df, prediction):
    result = {"clean_sold": metrics(df[TARGET], prediction)}
    central = df[TARGET].between(5_000, 80_000, inclusive="left").to_numpy()
    result["central_5k_80k"] = metrics(
        df.loc[central, TARGET], np.asarray(prediction)[central]
    )
    return result


def candidate_name(profile, family):
    return f"{family}__{profile}"


def choose_primary(candidates, predictions, y_raw):
    """Choose the lowest-MAE technically healthy predeclared candidate."""
    health = {}
    eligible = []
    target_std = max(float(np.std(y_raw)), 1.0)
    for name, prediction in predictions.items():
        family = name.split("__", 1)[0]
        std_ratio = float(np.std(prediction) / target_std)
        r2 = candidates[name]["clean_sold"]["r2"]
        is_healthy = bool(np.isfinite(prediction).all())
        if family == "raw_huber":
            is_healthy = is_healthy and std_ratio >= 0.20 and r2 > 0
        health[name] = {
            "std_ratio": std_ratio,
            "r2": r2,
            "healthy": is_healthy,
        }
        if is_healthy:
            eligible.append(name)
    if not eligible:
        raise RuntimeError("No technically healthy Model15 candidate remains")
    primary = min(eligible, key=lambda name: candidates[name]["clean_sold"]["mae_yen"])
    return primary, eligible, health


def optimized_feature_effect(candidates, include_key):
    """Positive means the named component improves the best optimized model."""
    with_component = []
    without_component = []
    for name, scopes in candidates.items():
        _, profile = name.split("__", 1)
        score = scopes["clean_sold"]["mae_yen"]
        target = with_component if FEATURE_PROFILES[profile][include_key] else without_component
        target.append(score)
    return float(min(without_component) - min(with_component))


def validate_artifacts(df, folds, qwen, bert, bert_rows, bert_hashes):
    ids = df.ticket_id.tolist()
    if (
        not {"ticket_id", "duplicate_group", "fold"}.issubset(folds.columns)
        or folds.ticket_id.tolist() != ids
        or folds.duplicate_group.tolist() != df.duplicate_group.tolist()
        or qwen.ticket_id.tolist() != ids
        or bert_rows != ids
    ):
        raise ValueError("Artifact rows do not match Model 15 dataset")
    expected_bert_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in df["model_text"].fillna("").astype(str)
    ]
    if (
        bert.shape != (len(df), 768)
        or bert_hashes != expected_bert_hashes
        or not np.isfinite(bert).all()
    ):
        raise ValueError("BERT cache is incomplete or stale; run bootstrap and BERT extraction")
    expected_qwen_fingerprint = qwen_dataset_fingerprint(df, folds)
    if (
        "qwen_dataset_fingerprint" not in qwen
        or not qwen["qwen_dataset_fingerprint"].eq(expected_qwen_fingerprint).all()
    ):
        raise ValueError("Qwen OOF is stale for the current inputs/labels/folds")
    qwen_diagnostics = qwen_oof_diagnostics(df, folds, qwen)
    return expected_qwen_fingerprint, qwen_diagnostics


def build_fold_cache(df, folds, bert, qwen, numeric, categorical, y_raw):
    semantic = set(SEMANTIC_NUMERIC_FEATURES + SEMANTIC_CATEGORICAL_FEATURES)
    numeric_no_semantic = [column for column in numeric if column not in semantic]
    categorical_no_semantic = [column for column in categorical if column not in semantic]
    cache = []
    for fold in range(N_FOLDS):
        val = folds.fold.eq(fold).to_numpy()
        fit = ~val
        # Derive weight normalization from the training side only.  Validation
        # prices must not influence even a constant used by the fitted model.
        price_ratio = y_raw / max(float(np.median(y_raw[fit])), 1.0)

        tabular = make_tabular(numeric, categorical)
        tab_fit = tabular.fit_transform(df.loc[fit])
        tab_val = tabular.transform(df.loc[val])
        tabular_no_semantic = make_tabular(
            numeric_no_semantic, categorical_no_semantic
        )
        tab0_fit = tabular_no_semantic.fit_transform(df.loc[fit])
        tab0_val = tabular_no_semantic.transform(df.loc[val])

        pca = PCA(
            n_components=BERT_PCA_DIM,
            svd_solver="randomized",
            random_state=SEED,
        )
        bert_fit = pca.fit_transform(bert[fit])
        bert_val = pca.transform(bert[val])
        qwen_fit = qwen.loc[fit, ["qwen_pred_log"]].to_numpy(float)
        qwen_val = qwen.loc[val, ["qwen_pred_log"]].to_numpy(float)

        profiles = {}
        profiles_no_semantic = {}
        for profile in FEATURE_PROFILES:
            profiles[profile] = {
                "fit": assemble_profile(tab_fit, bert_fit, qwen_fit, profile),
                "val": assemble_profile(tab_val, bert_val, qwen_val, profile),
            }
            profiles_no_semantic[profile] = {
                "fit": assemble_profile(tab0_fit, bert_fit, qwen_fit, profile),
                "val": assemble_profile(tab0_val, bert_val, qwen_val, profile),
            }
        cache.append(
            {
                "fold": fold,
                "fit": fit,
                "val": val,
                "price_ratio": price_ratio,
                "profiles": profiles,
                "profiles_no_semantic": profiles_no_semantic,
            }
        )
    return cache


def main():
    df = prepare_dataset()
    folds = pd.read_csv(ARTIFACT_DIR / "folds.csv")
    qwen = pd.read_csv(ARTIFACT_DIR / "qwen_oof.csv")
    bert = np.load(ARTIFACT_DIR / "bert_raw.npy")
    bert_rows = json.loads(
        (ARTIFACT_DIR / "bert_rows.json").read_text(encoding="utf-8")
    )
    bert_hashes = json.loads(
        (ARTIFACT_DIR / "bert_text_hashes.json").read_text(encoding="utf-8")
    )
    expected_qwen_fingerprint, qwen_diagnostics = validate_artifacts(
        df, folds, qwen, bert, bert_rows, bert_hashes
    )
    print(f"Qwen OOF integrity OK: {qwen_diagnostics}")

    numeric, categorical = model_feature_columns(df)
    feature_value_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            df[numeric + categorical], index=True, categorize=True
        ).values.tobytes()
    ).hexdigest()
    meta_fingerprint = hashlib.sha256(
        (
            PIPELINE_VERSION
            + QWEN_OOF_SCHEMA_VERSION
            + expected_qwen_fingerprint
            + feature_value_hash
            + hashlib.sha256(np.ascontiguousarray(bert).tobytes()).hexdigest()
            + hashlib.sha256(
                qwen["qwen_pred_log"].to_numpy(float).tobytes()
            ).hexdigest()
        ).encode("ascii")
    ).hexdigest()

    y_raw = df[TARGET].to_numpy(float)
    y_log = np.log1p(y_raw)
    y_scaled = y_raw / RAW_PRICE_SCALE
    fold_cache = build_fold_cache(
        df, folds, bert, qwen, numeric, categorical, y_raw
    )

    study, study_name = tune_joint_search(
        fold_cache, y_log, y_scaled, y_raw, meta_fingerprint
    )
    print(
        "Joint Optuna best: "
        f"MAE={study.best_value:,.1f} yen, trial={study.best_trial.number}, "
        f"profile={study.best_trial.params['feature_profile']}, "
        f"family={study.best_trial.params['model_family']}"
    )

    best_trials = best_trials_by_combination(study)
    candidates = {}
    predictions = {}
    iterations = {}
    candidate_tuning = {}
    for profile in FEATURE_PROFILES:
        for family in MODEL_FAMILIES:
            trial = best_trials[(profile, family)]
            hyperparameters = {
                name: value
                for name, value in trial.params.items()
                if name not in {"feature_profile", "model_family"}
            }
            name = candidate_name(profile, family)
            prediction, best_iterations = candidate_oof(
                fold_cache,
                y_log,
                y_scaled,
                family,
                profile,
                hyperparameters,
            )
            predictions[name] = prediction
            iterations[name] = best_iterations
            candidates[name] = scope_metrics(df, prediction)
            candidate_tuning[name] = {
                "trial": int(trial.number),
                "study_mae_yen": float(trial.value),
                "params": hyperparameters,
                "best_iterations": best_iterations,
            }
            print(
                f"candidate={name} MAE="
                f"{candidates[name]['clean_sold']['mae_yen']:,.1f} yen"
            )

    primary_name, eligible, candidate_health = choose_primary(
        candidates, predictions, y_raw
    )
    primary_family, primary_profile = primary_name.split("__", 1)
    primary_prediction = predictions[primary_name]
    primary_parameters = candidate_tuning[primary_name]["params"]

    no_semantic_prediction, no_semantic_iterations = candidate_oof(
        fold_cache,
        y_log,
        y_scaled,
        primary_family,
        primary_profile,
        primary_parameters,
        no_semantic=True,
    )
    no_semantic_name = f"ablation_no_llm_json__{primary_name}"
    no_semantic_metrics = scope_metrics(df, no_semantic_prediction)

    feature_profile_best = {
        profile: min(
            candidates[candidate_name(profile, family)]["clean_sold"]["mae_yen"]
            for family in MODEL_FAMILIES
        )
        for profile in FEATURE_PROFILES
    }
    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    report = {
        "model": "Model 15",
        "snapshot": latest_data_dir().name,
        "rows": len(df),
        "pipeline_version": PIPELINE_VERSION,
        "cleaning_policy": "model13_exact",
        "policy": (
            "Model13-clean sold population; duplicate-grouped 5-fold OOF; "
            "joint selection of text feature profile, target family, and LightGBM parameters"
        ),
        "semantic_coverage_pct": float(df.semantic_available.mean() * 100),
        "semantic_qwen15_pct": float(df.semantic_source.eq("qwen15").mean() * 100),
        "lgbm_optuna_trials": completed_trials,
        "lgbm_optuna_target_trials": LGBM_OPTUNA_TRIALS,
        "lgbm_optuna_study": study_name,
        "lgbm_optuna_best_trial": int(study.best_trial.number),
        "lgbm_optuna_best_mae_yen": float(study.best_value),
        "lgbm_optuna_best_params": dict(study.best_trial.params),
        "meta_dataset_fingerprint": meta_fingerprint,
        "qwen_oof_schema_version": QWEN_OOF_SCHEMA_VERSION,
        "qwen_oof_diagnostics": qwen_diagnostics,
        "qwen_training_source": (
            str(qwen["qwen_training_source"].iloc[0])
            if "qwen_training_source" in qwen
            else "fresh_ordered_training"
        ),
        "feature_profiles": FEATURE_PROFILES,
        "model_families": list(MODEL_FAMILIES),
        "feature_profile_best_mae_yen": feature_profile_best,
        "primary_candidate": primary_name,
        "primary_feature_profile": primary_profile,
        "primary_model_family": primary_family,
        "eligible_primary_candidates": eligible,
        "candidate_health": candidate_health,
        "candidate_tuning": candidate_tuning,
        "candidates": candidates,
        "llm_json_ablation_candidate": no_semantic_name,
        "llm_json_ablation_metrics": no_semantic_metrics,
        "llm_json_effect_mae_yen": float(
            no_semantic_metrics["clean_sold"]["mae_yen"]
            - candidates[primary_name]["clean_sold"]["mae_yen"]
        ),
        "qwen_effect_mae_yen": optimized_feature_effect(candidates, "qwen"),
        "bert_effect_mae_yen": optimized_feature_effect(candidates, "bert"),
        "evaluation_caveat": (
            "OOF metrics include model selection and are internal estimates. "
            "Confirm the locked profile on a later untouched snapshot."
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    oof_output = pd.DataFrame(
        {
            "ticket_id": df.ticket_id,
            "fold": folds.fold,
            "true_price": y_raw,
            "pred_primary": primary_prediction,
            **{f"pred_{name}": prediction for name, prediction in predictions.items()},
            f"pred_{no_semantic_name}": no_semantic_prediction,
        }
    )
    oof_target = ARTIFACT_DIR / "oof_predictions_model15.csv"
    oof_temporary = ARTIFACT_DIR / "oof_predictions_model15.tmp.csv"
    oof_output.to_csv(oof_temporary, index=False)

    evaluation_target = ARTIFACT_DIR / "evaluation_model15.json"
    evaluation_temporary = ARTIFACT_DIR / "evaluation_model15.tmp.json"
    evaluation_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparison_rows = []
    for name, scopes in candidates.items():
        comparison_rows.append(
            {
                "candidate": name,
                "feature_profile": name.split("__", 1)[1],
                "model_family": name.split("__", 1)[0],
                **scopes["clean_sold"],
            }
        )
    comparison_target = ARTIFACT_DIR / "candidate_comparison.csv"
    comparison_temporary = ARTIFACT_DIR / "candidate_comparison.tmp.csv"
    pd.DataFrame(comparison_rows).sort_values("mae_yen").to_csv(
        comparison_temporary,
        index=False,
        encoding="utf-8-sig",
    )

    # Fit one production model with exactly the OOF-selected profile/family.
    final_tabular_pipeline = make_tabular(numeric, categorical)
    final_tabular = final_tabular_pipeline.fit_transform(df)
    final_pca = None
    final_bert = None
    if FEATURE_PROFILES[primary_profile]["bert"]:
        final_pca = PCA(
            n_components=BERT_PCA_DIM,
            svd_solver="randomized",
            random_state=SEED,
        )
        final_bert = final_pca.fit_transform(bert)
    final_qwen = qwen[["qwen_pred_log"]].to_numpy(float)
    final_x = assemble_profile(
        final_tabular,
        final_bert,
        final_qwen,
        primary_profile,
    )
    final_price_ratio = y_raw / max(float(np.median(y_raw)), 1.0)
    final_model_overrides, final_weight, selected_weight_power = resolve_training_parameters(
        primary_parameters, primary_family, final_price_ratio
    )
    objective_name, final_target, final_sample_weight, _ = family_settings(
        primary_family, y_log, y_scaled, final_weight
    )
    final_parameters = base_params(objective_name, final_model_overrides)
    final_parameters["n_estimators"] = max(
        50, int(np.median(iterations[primary_name]))
    )
    final_model = LGBMRegressor(**final_parameters)
    final_model.fit(final_x, final_target, sample_weight=final_sample_weight)

    requires_qwen = FEATURE_PROFILES[primary_profile]["qwen"]
    model_payload = {
        "model": final_model,
        "models": {primary_name: final_model},
        "tabular": final_tabular_pipeline,
        "pca": final_pca,
        "numeric": numeric,
        "categorical": categorical,
        "primary": primary_name,
        "feature_profile": primary_profile,
        "model_family": primary_family,
        "requires_bert": FEATURE_PROFILES[primary_profile]["bert"],
        "requires_qwen": requires_qwen,
        "pipeline_version": PIPELINE_VERSION,
        "cleaning_policy": "model13_exact",
        "snapshot": latest_data_dir().name,
        "meta_dataset_fingerprint": meta_fingerprint,
        "qwen_oof_schema_version": QWEN_OOF_SCHEMA_VERSION,
        "raw_price_scale": RAW_PRICE_SCALE,
        "lgbm_optuna_best_params": primary_parameters,
        "price_weight_power": (
            selected_weight_power
            if primary_family in {"log_weighted", "raw_huber"}
            else None
        ),
        "best_iterations": iterations[primary_name],
        "semantic_ablation_iterations": no_semantic_iterations,
        "qwen_adapters": (
            [
                str(ARTIFACT_DIR / "qwen_folds" / f"fold_{fold}" / "best_adapter")
                for fold in range(N_FOLDS)
            ]
            if requires_qwen
            else []
        ),
    }
    model_target = ARTIFACT_DIR / "model15.joblib"
    model_temporary = ARTIFACT_DIR / "model15.tmp.joblib"
    joblib.dump(model_payload, model_temporary)
    # evaluation_model15.json is the commit marker and is replaced last.  A
    # failed final fit can therefore never make an old model look like v3.
    os.replace(model_temporary, model_target)
    os.replace(oof_temporary, oof_target)
    os.replace(comparison_temporary, comparison_target)
    os.replace(evaluation_temporary, evaluation_target)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
