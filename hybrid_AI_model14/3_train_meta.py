"""Fold-safe BERT/PCA + tabular + Qwen stacking, with LightGBM/CatBoost blending."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import ARTIFACT_DIR, BERT_PCA_DIM, N_FOLDS, SEED, TARGET
from data_loader import model_feature_columns, prepare_dataset

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def metrics(y_log, pred_log):
    y, pred = np.expm1(y_log), np.maximum(0, np.expm1(pred_log))
    ape = np.abs(pred - y) / np.maximum(y, 1)
    return {
        "count": len(y), "mae_yen": mean_absolute_error(y, pred),
        "rmse_yen": mean_squared_error(y, pred) ** 0.5,
        "mape_pct": ape.mean() * 100, "mdape_pct": np.median(ape) * 100,
        "within_20_pct": (ape <= .2).mean() * 100, "r2": r2_score(y, pred),
    }


def make_tabular(numeric, categorical):
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical),
    ])


def lgbm():
    return LGBMRegressor(
        objective="regression_l1", n_estimators=3000, learning_rate=.015,
        num_leaves=48, max_depth=10, min_child_samples=20,
        colsample_bytree=.8, subsample=.85, reg_alpha=.1, reg_lambda=1,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def main():
    df = prepare_dataset()
    manifest = pd.read_csv(ARTIFACT_DIR / "folds.csv")
    qwen = pd.read_csv(ARTIFACT_DIR / "qwen_oof.csv")
    bert = np.load(ARTIFACT_DIR / "bert_raw.npy")
    bert_rows = json.loads((ARTIFACT_DIR / "bert_rows.json").read_text(encoding="utf-8"))
    if list(df.ticket_id) != list(manifest.ticket_id) or list(df.ticket_id) != list(qwen.ticket_id) or list(df.ticket_id) != bert_rows:
        raise ValueError("Artifact row identities do not match current sold dataset")
    if qwen.qwen_pred_log.isna().any():
        raise ValueError("Qwen OOF is incomplete; run every fold first")
    numeric, categorical = model_feature_columns(df)
    y = np.log1p(df[TARGET].to_numpy(float))
    oof_lgb = np.zeros(len(df))
    oof_cat = np.full(len(df), np.nan)
    best_iterations = []

    for fold in range(N_FOLDS):
        val = manifest.fold.eq(fold).to_numpy()
        fit = ~val
        tab = make_tabular(numeric, categorical)
        x_fit_tab = tab.fit_transform(df.loc[fit])
        x_val_tab = tab.transform(df.loc[val])
        pca = PCA(n_components=min(BERT_PCA_DIM, fit.sum() - 1), svd_solver="randomized", random_state=SEED)
        x_fit_bert = pca.fit_transform(bert[fit])
        x_val_bert = pca.transform(bert[val])
        x_fit = hstack([x_fit_tab, csr_matrix(x_fit_bert), csr_matrix(qwen.loc[fit, ["qwen_pred_log"]].to_numpy())], format="csr")
        x_val = hstack([x_val_tab, csr_matrix(x_val_bert), csr_matrix(qwen.loc[val, ["qwen_pred_log"]].to_numpy())], format="csr")
        model = lgbm()
        model.fit(x_fit, y[fit], eval_set=[(x_val, y[val])], callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        oof_lgb[val] = model.predict(x_val, num_iteration=model.best_iteration_)
        best_iterations.append(model.best_iteration_)
        if CatBoostRegressor is not None:
            cat = CatBoostRegressor(loss_function="MAE", iterations=model.best_iteration_, depth=8,
                                    learning_rate=.03, random_seed=SEED, verbose=False)
            cat.fit(x_fit, y[fit])
            oof_cat[val] = cat.predict(x_val)
        print(f"fold={fold} rows={val.sum()} lgb_iteration={model.best_iteration_}")

    best_weight, best_mae = 1.0, float("inf")
    for weight in np.linspace(0, 1, 21) if CatBoostRegressor is not None else [1.0]:
        pred = weight * oof_lgb + (1 - weight) * oof_cat
        mae = mean_absolute_error(np.expm1(y), np.maximum(0, np.expm1(pred)))
        if mae < best_mae:
            best_weight, best_mae = float(weight), float(mae)
    blended = best_weight * oof_lgb + (1 - best_weight) * oof_cat if CatBoostRegressor is not None else oof_lgb
    report = {
        "policy": "sold-only duplicate-grouped random OOF; all preprocessing fold-local",
        "rows": len(df), "events": df.event_id.nunique(), "lgbm": metrics(y, oof_lgb),
        "blend_weight_lgbm": best_weight, "blended": metrics(y, blended),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Production stack: fit on every sold row only after OOF evaluation is frozen.
    final_tab = make_tabular(numeric, categorical)
    final_tabular = final_tab.fit_transform(df)
    final_pca = PCA(n_components=BERT_PCA_DIM, svd_solver="randomized", random_state=SEED)
    final_bert = final_pca.fit_transform(bert)
    final_x = hstack([final_tabular, csr_matrix(final_bert), csr_matrix(qwen[["qwen_pred_log"]].to_numpy())], format="csr")
    final_model = lgbm()
    final_model.set_params(n_estimators=max(50, int(np.median(best_iterations))))
    final_model.fit(final_x, y)
    joblib.dump({"model": final_model, "tabular": final_tab, "pca": final_pca,
                 "numeric": numeric, "categorical": categorical,
                 "qwen_adapters": [str(ARTIFACT_DIR / "qwen_folds" / f"fold_{f}" / "best_adapter") for f in range(N_FOLDS)]},
                ARTIFACT_DIR / "model14_meta.joblib")
    pd.DataFrame({"ticket_id": df.ticket_id, "fold": manifest.fold, "true_price": df[TARGET],
                  "pred_price": np.maximum(0, np.expm1(blended))}).to_csv(ARTIFACT_DIR / "oof_predictions.csv", index=False)
    (ARTIFACT_DIR / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
