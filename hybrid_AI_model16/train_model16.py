"""Train Model16's single global nested ensemble.

Every prediction uses one fixed convex formula.  There is no price-band split,
row-specific expert routing, or use of the validation target to select an
expert.  Outer folds measure the complete inner tuning and blending procedure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACT_DIR,
    BASE_EXPERTS,
    BERT_PCA_DIMS,
    BERT_PCA_MAX_DIM,
    BERT_RIDGE_ALPHAS,
    FINAL_CATBOOST_TRIALS,
    FINAL_LGBM_TRIALS,
    N_INNER_FOLDS,
    N_OUTER_FOLDS,
    OUTER_CATBOOST_TRIALS,
    OUTER_LGBM_TRIALS,
    PIPELINE_VERSION,
    RAW_PRICE_SCALE,
    SEED,
    SOURCE_ARTIFACT_DIR,
    TARGET,
    WEIGHT_EPSILON,
)
from data_loader import catboost_frame, model_feature_columns, prepare_dataset
from modeling import (
    assign_inner_folds,
    blend_predictions,
    catboost_trial_parameters,
    fit_catboost,
    fit_catboost_full,
    fit_lgbm,
    fit_lgbm_full,
    lgbm_trial_parameters,
    make_tabular,
    metrics,
    optimize_global_weights,
)


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def study_for(name, trials, objective):
    storage = (ARTIFACT_DIR / "model16_optuna.db").resolve().as_posix()
    study = optuna.create_study(
        direction="minimize",
        study_name=name,
        storage=f"sqlite:///{storage}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    return study


def validate_inputs(df):
    folds = pd.read_csv(SOURCE_ARTIFACT_DIR / "folds.csv")
    bert = np.load(SOURCE_ARTIFACT_DIR / "bert_raw.npy", mmap_mode="r")
    rows = json.loads((SOURCE_ARTIFACT_DIR / "bert_rows.json").read_text(encoding="utf-8"))
    hashes = json.loads(
        (SOURCE_ARTIFACT_DIR / "bert_text_hashes.json").read_text(encoding="utf-8")
    )
    ids = df.ticket_id.tolist()
    expected_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in df.model_text.fillna("").astype(str)
    ]
    if (
        folds.ticket_id.tolist() != ids
        or folds.duplicate_group.tolist() != df.duplicate_group.tolist()
        or rows != ids
        or hashes != expected_hashes
        or bert.shape != (len(df), 768)
    ):
        raise ValueError("Model15 folds/BERT cache do not align with the Model16 population")
    if sorted(folds.fold.unique()) != list(range(N_OUTER_FOLDS)):
        raise ValueError("Outer fold manifest is invalid")
    if (
        pd.DataFrame({"group": folds.duplicate_group, "fold": folds.fold})
        .groupby("group").fold.nunique().max()
        > 1
    ):
        raise ValueError("Duplicate descriptions cross outer folds")
    return folds, np.asarray(bert)


def dataset_fingerprint(df, folds, numeric, categorical, bert):
    schema_hash = hashlib.sha256(
        json.dumps(
            {"numeric": numeric, "categorical": categorical},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    feature_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            df[numeric + categorical + [TARGET]], index=True, categorize=True
        ).values.tobytes()
    ).hexdigest()
    bert_hash = hashlib.sha256(np.ascontiguousarray(bert).tobytes()).hexdigest()
    fold_hash = hashlib.sha256(
        pd.util.hash_pandas_object(folds[["ticket_id", "fold"]], index=False).values.tobytes()
    ).hexdigest()
    return hashlib.sha256(
        (PIPELINE_VERSION + schema_hash + feature_hash + bert_hash + fold_hash).encode("ascii")
    ).hexdigest()


def build_lgbm_cache(df, assignments, numeric, categorical):
    cache = []
    for fold in sorted(np.unique(assignments)):
        val = assignments == fold
        fit = ~val
        preprocessor = make_tabular(numeric, categorical)
        cache.append(
            {
                "fit": fit,
                "val": val,
                "x_fit": preprocessor.fit_transform(df.loc[fit]),
                "x_val": preprocessor.transform(df.loc[val]),
            }
        )
    return cache


def tune_lgbm(
    df, y, assignments, numeric, categorical, family, study_name, trials,
    cache=None,
):
    # The two LightGBM expert families use the same rows and feature matrix.
    # Reusing this fold-local cache avoids performing imputation/one-hot encoding
    # twice without changing either expert's training data or search space.
    if cache is None:
        cache = build_lgbm_cache(df, assignments, numeric, categorical)

    def objective(trial):
        params = lgbm_trial_parameters(trial)
        prediction = np.zeros(len(df), float)
        for part in cache:
            _, pred = fit_lgbm(
                part["x_fit"], y[part["fit"]], part["x_val"], y[part["val"]],
                family, params,
            )
            prediction[part["val"]] = pred
        result = metrics(y, prediction)
        return result["mape_pct" if family == "lgbm_raw_mape" else "mae_yen"]

    study = study_for(study_name, trials, objective)
    params = dict(study.best_params)
    prediction = np.zeros(len(df), float)
    iterations = []
    for part in cache:
        model, pred = fit_lgbm(
            part["x_fit"], y[part["fit"]], part["x_val"], y[part["val"]],
            family, params,
        )
        prediction[part["val"]] = pred
        iterations.append(int(model.best_iteration_))
    return params, prediction, iterations, float(study.best_value)


def tune_catboost(frame, y, assignments, categorical, study_name, trials):
    parts = []
    for fold in sorted(np.unique(assignments)):
        val = assignments == fold
        parts.append((~val, val))

    def objective(trial):
        params = catboost_trial_parameters(trial)
        prediction = np.zeros(len(frame), float)
        for fit, val in parts:
            _, pred = fit_catboost(
                frame.loc[fit], y[fit], frame.loc[val], y[val], categorical, params
            )
            prediction[val] = pred
        return metrics(y, prediction)["mae_yen"]

    study = study_for(study_name, trials, objective)
    params = dict(study.best_params)
    prediction = np.zeros(len(frame), float)
    iterations = []
    for fit, val in parts:
        model, pred = fit_catboost(
            frame.loc[fit], y[fit], frame.loc[val], y[val], categorical, params
        )
        prediction[val] = pred
        best = model.get_best_iteration()
        iterations.append(int(best + 1 if best is not None and best >= 0 else 1000))
    return params, prediction, iterations, float(study.best_value)


def bert_fold_components(bert, y, assignments):
    parts = []
    for fold in sorted(np.unique(assignments)):
        val = assignments == fold
        fit = ~val
        pca = PCA(
            n_components=BERT_PCA_MAX_DIM,
            svd_solver="randomized",
            random_state=SEED,
        )
        fit_pca = pca.fit_transform(bert[fit])
        val_pca = pca.transform(bert[val])
        parts.append((fit, val, fit_pca, val_pca))
    return parts


def tune_bert_ridge(bert, y, assignments):
    parts = bert_fold_components(bert, y, assignments)
    best = None
    for dimension in BERT_PCA_DIMS:
        for alpha in BERT_RIDGE_ALPHAS:
            prediction = np.zeros(len(y), float)
            for fit, val, fit_pca, val_pca in parts:
                scaler = StandardScaler()
                x_fit = scaler.fit_transform(fit_pca[:, :dimension])
                x_val = scaler.transform(val_pca[:, :dimension])
                model = Ridge(alpha=alpha)
                model.fit(x_fit, np.log1p(y[fit]))
                prediction[val] = np.maximum(0, np.expm1(model.predict(x_val)))
            score = metrics(y, prediction)["mae_yen"]
            if best is None or score < best[0]:
                best = (score, dimension, alpha, prediction.copy())
    return {
        "dimension": int(best[1]), "alpha": float(best[2])
    }, best[3], float(best[0])


def fit_bert_full(bert_fit, y_fit, bert_val, parameters):
    dimension = int(parameters["dimension"])
    pca = PCA(
        n_components=dimension, svd_solver="randomized", random_state=SEED
    )
    fit_pca = pca.fit_transform(bert_fit)
    val_pca = pca.transform(bert_val)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_pca)
    x_val = scaler.transform(val_pca)
    model = Ridge(alpha=float(parameters["alpha"]))
    model.fit(x_fit, np.log1p(y_fit))
    prediction = np.maximum(0, np.expm1(model.predict(x_val)))
    return {"pca": pca, "scaler": scaler, "model": model}, prediction


def fit_outer_experts(
    train_df, train_bert, y_train, validation_df, validation_bert,
    numeric, categorical, cat_frame_train, cat_frame_val, tuned,
):
    predictions = []
    payload = {}
    for family in ("lgbm_log_mae", "lgbm_raw_mape"):
        preprocessor = make_tabular(numeric, categorical)
        x_fit = preprocessor.fit_transform(train_df)
        x_val = preprocessor.transform(validation_df)
        iterations = max(50, int(np.median(tuned[family]["iterations"])))
        model = fit_lgbm_full(
            x_fit, y_train, family, tuned[family]["params"], iterations
        )
        native = model.booster_.predict(x_val)
        pred = (
            np.maximum(0, np.expm1(native))
            if family == "lgbm_log_mae"
            else np.maximum(0, native * RAW_PRICE_SCALE)
        )
        predictions.append(pred)
        payload[family] = {"preprocessor": preprocessor, "model": model}

    cat_iterations = max(50, int(np.median(tuned["catboost_raw_mae"]["iterations"])))
    cat_model = fit_catboost_full(
        cat_frame_train, y_train, categorical,
        tuned["catboost_raw_mae"]["params"], cat_iterations,
    )
    cat_pred = np.maximum(0, cat_model.predict(cat_frame_val) * RAW_PRICE_SCALE)
    predictions.append(cat_pred)
    payload["catboost_raw_mae"] = {"model": cat_model}

    bert_payload, bert_pred = fit_bert_full(
        train_bert, y_train, validation_bert, tuned["bert_ridge"]["params"]
    )
    predictions.append(bert_pred)
    payload["bert_ridge"] = bert_payload
    return np.column_stack(predictions), payload


def fit_production_experts(df, bert, y, numeric, categorical, cat_frame, tuned):
    payload = {}
    for family in ("lgbm_log_mae", "lgbm_raw_mape"):
        preprocessor = make_tabular(numeric, categorical)
        matrix = preprocessor.fit_transform(df)
        iterations = max(50, int(np.median(tuned[family]["iterations"])))
        model = fit_lgbm_full(
            matrix, y, family, tuned[family]["params"], iterations
        )
        payload[family] = {"preprocessor": preprocessor, "model": model}
    cat_iterations = max(
        50, int(np.median(tuned["catboost_raw_mae"]["iterations"]))
    )
    payload["catboost_raw_mae"] = {
        "model": fit_catboost_full(
            cat_frame, y, categorical,
            tuned["catboost_raw_mae"]["params"], cat_iterations,
        )
    }
    bert_parameters = tuned["bert_ridge"]["params"]
    dimension = int(bert_parameters["dimension"])
    pca = PCA(n_components=dimension, svd_solver="randomized", random_state=SEED)
    transformed = pca.fit_transform(bert)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(transformed)
    ridge = Ridge(alpha=float(bert_parameters["alpha"]))
    ridge.fit(transformed, np.log1p(y))
    payload["bert_ridge"] = {"pca": pca, "scaler": scaler, "model": ridge}
    return payload


def tune_all_experts(df, bert, y, assignments, numeric, categorical, prefix, lgbm_trials, cat_trials):
    frame = catboost_frame(df, numeric, categorical)
    lgbm_cache = build_lgbm_cache(df, assignments, numeric, categorical)
    tuned = {}
    inner_predictions = []
    for family in ("lgbm_log_mae", "lgbm_raw_mape"):
        params, prediction, iterations, best_value = tune_lgbm(
            df, y, assignments, numeric, categorical, family,
            f"{prefix}_{family}", lgbm_trials, cache=lgbm_cache,
        )
        tuned[family] = {
            "params": params, "iterations": iterations, "study_best": best_value
        }
        inner_predictions.append(prediction)
    params, prediction, iterations, best_value = tune_catboost(
        frame, y, assignments, categorical, f"{prefix}_catboost_raw_mae", cat_trials
    )
    tuned["catboost_raw_mae"] = {
        "params": params, "iterations": iterations, "study_best": best_value
    }
    inner_predictions.append(prediction)
    params, prediction, best_value = tune_bert_ridge(bert, y, assignments)
    tuned["bert_ridge"] = {"params": params, "iterations": [], "study_best": best_value}
    inner_predictions.append(prediction)
    return tuned, np.column_stack(inner_predictions), frame


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_dataset()
    folds, bert = validate_inputs(df)
    numeric, categorical = model_feature_columns(df)
    y = df[TARGET].to_numpy(float)
    fingerprint = dataset_fingerprint(df, folds, numeric, categorical, bert)
    print(
        f"Model16 rows={len(df):,}, numeric={len(numeric)}, categorical={len(categorical)}, "
        f"BERT dimensions searched={BERT_PCA_DIMS}"
    )
    print("Global experts only; price-band routing is disabled.")

    oof_base = np.zeros((len(df), len(BASE_EXPERTS)), float)
    oof_ensemble = np.zeros(len(df), float)
    outer_reports = []
    for outer_fold in range(N_OUTER_FOLDS):
        validation = folds.fold.eq(outer_fold).to_numpy()
        fit = ~validation
        checkpoint_path = ARTIFACT_DIR / f"outer_fold_{outer_fold}.npz"
        checkpoint_report_path = ARTIFACT_DIR / f"outer_fold_{outer_fold}.json"
        if checkpoint_path.exists() and checkpoint_report_path.exists():
            try:
                saved_report = json.loads(
                    checkpoint_report_path.read_text(encoding="utf-8")
                )
                with np.load(checkpoint_path, allow_pickle=False) as saved:
                    saved_indices = saved["validation_indices"].copy()
                    saved_base = saved["base_predictions"].copy()
                    saved_ensemble = saved["ensemble_prediction"].copy()
                expected_indices = np.flatnonzero(validation)
                saved_weights = np.asarray(
                    [saved_report["weights"][name] for name in BASE_EXPERTS], float
                )
                valid_checkpoint = (
                    saved_report.get("dataset_fingerprint") == fingerprint
                    and saved_report.get("outer_fold") == outer_fold
                    and np.array_equal(saved_indices, expected_indices)
                    and saved_base.shape == (len(expected_indices), len(BASE_EXPERTS))
                    and saved_ensemble.shape == (len(expected_indices),)
                    and np.isfinite(saved_base).all()
                    and np.isfinite(saved_ensemble).all()
                    and (saved_weights >= 0).all()
                    and np.isclose(saved_weights.sum(), 1.0)
                    and np.allclose(
                        saved_ensemble,
                        blend_predictions(saved_base, saved_weights),
                        rtol=1e-9,
                        atol=1e-6,
                    )
                )
            except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
                valid_checkpoint = False
            if valid_checkpoint:
                oof_base[validation] = saved_base
                oof_ensemble[validation] = saved_ensemble
                outer_reports.append(saved_report)
                print(f"outer={outer_fold} checkpoint reused")
                continue
            print(f"outer={outer_fold} checkpoint invalid/stale; recomputing")
        train_df = df.loc[fit].reset_index(drop=True)
        val_df = df.loc[validation].reset_index(drop=True)
        train_bert, val_bert = bert[fit], bert[validation]
        y_train = y[fit]
        inner = assign_inner_folds(
            train_df, N_INNER_FOLDS, SEED + outer_fold, TARGET
        )
        prefix = f"m16_{fingerprint[:12]}_outer{outer_fold}"
        tuned, inner_matrix, train_cat = tune_all_experts(
            train_df, train_bert, y_train, inner, numeric, categorical,
            prefix, OUTER_LGBM_TRIALS, OUTER_CATBOOST_TRIALS,
        )
        weights = optimize_global_weights(y_train, inner_matrix)
        val_cat = catboost_frame(val_df, numeric, categorical)
        outer_matrix, _ = fit_outer_experts(
            train_df, train_bert, y_train, val_df, val_bert,
            numeric, categorical, train_cat, val_cat, tuned,
        )
        oof_base[validation] = outer_matrix
        oof_ensemble[validation] = blend_predictions(outer_matrix, weights)
        outer_report = {
            "dataset_fingerprint": fingerprint,
            "outer_fold": outer_fold,
            "train_rows": int(fit.sum()),
            "validation_rows": int(validation.sum()),
            "weights": dict(zip(BASE_EXPERTS, map(float, weights))),
            "tuned": tuned,
            "validation_metrics": metrics(y[validation], oof_ensemble[validation]),
        }
        outer_reports.append(outer_report)
        checkpoint_temporary = ARTIFACT_DIR / f"outer_fold_{outer_fold}.tmp.npz"
        np.savez_compressed(
            checkpoint_temporary,
            validation_indices=np.flatnonzero(validation),
            base_predictions=outer_matrix,
            ensemble_prediction=oof_ensemble[validation],
        )
        os.replace(checkpoint_temporary, checkpoint_path)
        atomic_json(checkpoint_report_path, outer_report)
        print(
            f"outer={outer_fold} MAE={outer_report['validation_metrics']['mae_yen']:,.1f} "
            f"MAPE={outer_report['validation_metrics']['mape_pct']:.2f}% weights={weights}"
        )

    predictions = {
        expert: oof_base[:, index] for index, expert in enumerate(BASE_EXPERTS)
    }
    predictions["global_convex_ensemble"] = oof_ensemble
    candidates = {name: metrics(y, pred) for name, pred in predictions.items()}
    primary = min(candidates, key=lambda name: candidates[name]["mae_yen"])

    # Tune the deployable global models on all rows using only the established
    # outer folds, then learn one constant production weight vector.
    final_prefix = f"m16_{fingerprint[:12]}_final"
    final_tuned, final_oof_matrix, final_cat_frame = tune_all_experts(
        df, bert, y, folds.fold.to_numpy(), numeric, categorical,
        final_prefix, FINAL_LGBM_TRIALS, FINAL_CATBOOST_TRIALS,
    )
    final_weights = optimize_global_weights(y, final_oof_matrix)
    if primary != "global_convex_ensemble":
        final_weights = np.zeros(len(BASE_EXPERTS), float)
        final_weights[BASE_EXPERTS.index(primary)] = 1.0
    final_payload = fit_production_experts(
        df, bert, y, numeric, categorical, final_cat_frame, final_tuned
    )

    report = {
        "model": "Model16",
        "pipeline_version": PIPELINE_VERSION,
        "dataset_fingerprint": fingerprint,
        "rows": len(df),
        "cleaning_policy": "model13_exact_sold_only",
        "selection_policy": "single global predictor selected by nested grouped OOF MAE",
        "price_band_routing": False,
        "primary_metric": "mae_yen",
        "primary_candidate": primary,
        "base_experts": list(BASE_EXPERTS),
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "bert_dimensions_searched": list(BERT_PCA_DIMS),
        "qwen_used": False,
        "candidates": candidates,
        "outer_folds": outer_reports,
        "production_weights": dict(zip(BASE_EXPERTS, map(float, final_weights))),
        "final_tuned": final_tuned,
        "evaluation_caveat": (
            "Nested grouped OOF evaluates tuning and constant blending without held-out labels. "
            "Confirm the locked model on a later untouched snapshot."
        ),
    }
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "primary": primary,
        "expert_order": list(BASE_EXPERTS),
        "weights": final_weights,
        "experts": final_payload,
        "numeric": numeric,
        "categorical": categorical,
        "requires_bert": bool(
            final_weights[BASE_EXPERTS.index("bert_ridge")] >= WEIGHT_EPSILON
        ),
        "requires_qwen": False,
        "dataset_fingerprint": fingerprint,
    }

    output = pd.DataFrame(
        {
            "ticket_id": df.ticket_id,
            "fold": folds.fold,
            "true_price": y,
            "pred_primary": predictions[primary],
            **{f"pred_{name}": pred for name, pred in predictions.items()},
        }
    )
    oof_tmp = ARTIFACT_DIR / "oof_predictions_model16.tmp.csv"
    eval_tmp = ARTIFACT_DIR / "evaluation_model16.tmp.json"
    model_tmp = ARTIFACT_DIR / "model16.tmp.joblib"
    output.to_csv(oof_tmp, index=False)
    eval_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(payload, model_tmp)
    os.replace(model_tmp, ARTIFACT_DIR / "model16.joblib")
    os.replace(oof_tmp, ARTIFACT_DIR / "oof_predictions_model16.csv")
    os.replace(eval_tmp, ARTIFACT_DIR / "evaluation_model16.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
