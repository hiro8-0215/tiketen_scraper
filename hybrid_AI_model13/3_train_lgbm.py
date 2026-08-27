import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_COLUMN, RANDOM_SEED, CATEGORICAL_FEATURES, OUTPUT_DIR
from data_loader import prepare_dataset, get_tabular_feature_columns

LLM_PREDS_FILE = "./llm_predictions.csv"
BERT_FEATURES_FILE = "./bert_features.npy"

def setup_japanese_font():
    font_paths = [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/yuGothM.ttc"
    ]
    for path in font_paths:
        if os.path.exists(path):
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            return

def main():
    setup_japanese_font()
    
    print("[1] 表データの準備中...")
    df, tabular_cols, label_encoders = prepare_dataset()
    
    # listing_duration_days を明示的に除外（未来情報リーク）
    if 'listing_duration_days' in tabular_cols:
        tabular_cols.remove('listing_duration_days')
        print("  ⚠ listing_duration_days を除外しました（未来情報のため）")
    
    cat_cols = [col + "_encoded" for col in label_encoders.keys() if col + "_encoded" in df.columns]
    X_tabular = df[tabular_cols].copy()
    y = np.log1p(df[TARGET_COLUMN])
    sample_weights = np.sqrt(df[TARGET_COLUMN])  # 価格の平方根をウェイトにして高額帯を重視
    
    print(f"  表データ特徴量: {len(tabular_cols)}次元")
    
    # ==========================================
    # LLM予測値のロード
    # ==========================================
    print("\n[2] LLM予測値のロード...")
    if not os.path.exists(LLM_PREDS_FILE):
        print(f"エラー: {LLM_PREDS_FILE} が見つかりません。先に 2_extract_features.py を実行してください。")
        return
        
    llm_preds_df = pd.read_csv(LLM_PREDS_FILE)
    
    # ==========================================
    # BERT埋め込みのロード
    # ==========================================
    print("[3] BERT埋め込みのロード...")
    if not os.path.exists(BERT_FEATURES_FILE):
        print(f"エラー: {BERT_FEATURES_FILE} が見つかりません。先に 2_extract_features.py を実行してください。")
        return
    
    bert_embeddings = np.load(BERT_FEATURES_FILE)
    
    # 行数チェック
    if len(df) != len(llm_preds_df) or len(df) != len(bert_embeddings):
        print(f"エラー: データ行数が一致しません。")
        print(f"  表データ: {len(df)}件, LLM予測: {len(llm_preds_df)}件, BERT: {len(bert_embeddings)}件")
        return
    
    # ==========================================
    # LLM予測値のキャリブレーション (Isotonic Regression)
    # ==========================================
    print(f"\n[4] LLM予測値をIsotonic Regressionでキャリブレーション中...")
    llm_raw_log = llm_preds_df["llm_pred_log"].values

    # Cross-fittingでデータリークを防ぎながらキャリブレーション
    llm_calibrated = np.zeros_like(llm_raw_log)
    kf_calib = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, val_idx in kf_calib.split(llm_raw_log):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(llm_raw_log[train_idx], y.values[train_idx])
        llm_calibrated[val_idx] = iso.predict(llm_raw_log[val_idx])

    calib_r2 = r2_score(y.values, llm_calibrated)
    print(f"  キャリブレーション前 R²(log): {r2_score(y.values, llm_raw_log):.4f}")
    print(f"  キャリブレーション後 R²(log): {calib_r2:.4f}")

    # ==========================================
    # 特徴量の結合
    # ==========================================
    print(f"\n[5] 特徴量の結合...")

    # LLM予測値 (キャリブレーション済み)
    X_pred = pd.DataFrame({"llm_pred_log": llm_calibrated}, index=df.index)

    # BERT埋め込み
    bert_cols = [f"bert_{i}" for i in range(bert_embeddings.shape[1])]
    X_bert = pd.DataFrame(bert_embeddings, columns=bert_cols, index=df.index)

    # 全結合: 表データ + LLM予測 + BERT + (JSON特徴量は表データに含まれている)
    X = pd.concat([X_tabular, X_pred, X_bert], axis=1)
    
    # カテゴリカル特徴量の型設定
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    
    all_feature_cols = list(X.columns)
    cat_feature_names = [c for c in cat_cols if c in all_feature_cols]
    
    total_dim = len(tabular_cols) + 1 + bert_embeddings.shape[1]
    print(f"  表データ: {len(tabular_cols)}次元 (JSON特徴量含む)")
    print(f"  LLM予測: 1次元 (キャリブレーション済み)")
    print(f"  BERT: {bert_embeddings.shape[1]}次元")
    print(f"  合計: {total_dim}次元")

    # ==========================================
    # Optunaによるハイパーパラメータ最適化
    # ==========================================
    print(f"\n[6] OptunaによるLightGBMのハイパーパラメータ探索 (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 20, 128),
            'max_depth': trial.suggest_int("max_depth", 4, 12),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.3, 0.9),
            'bagging_fraction': trial.suggest_float("bagging_fraction", 0.5, 0.95),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 10),
            'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            'random_state': RANDOM_SEED,
            'verbose': -1
        }
        
        kf = KFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature_names, weight=sample_weights.iloc[train_idx])
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, weight=sample_weights.iloc[val_idx])
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            cv_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['boosting_type'] = 'gbdt'
    best_params['random_state'] = RANDOM_SEED
    best_params['verbose'] = -1
    
    print(f"  最良RMSE (log空間): {study.best_value:.4f}")
    
    # ==========================================
    # 最終モデルの学習と評価 (5-Fold OOF)
    # ==========================================
    print(f"\n[7] 最終モデルの学習 (5-Fold OOF)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    feature_importances = np.zeros(X.shape[1])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature_names, weight=sample_weights.iloc[train_idx])
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, weight=sample_weights.iloc[val_idx])
        
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        feature_importances += model.feature_importance(importance_type='gain')
        print(f"  Fold {fold+1}/5 完了 (best_iteration: {model.best_iteration})")
        
    feature_importances /= 5
    
    # 評価
    true_price = np.expm1(y)
    pred_price = np.expm1(oof_preds)
    
    mae = mean_absolute_error(true_price, pred_price)
    r2 = r2_score(true_price, pred_price)
    rmse = np.sqrt(mean_squared_error(true_price, pred_price))
    
    print(f"\n{'='*50}")
    print(f"Model 12 評価結果")
    print(f"{'='*50}")
    print(f"全データ: {len(X)} 件 ({total_dim}次元)")
    print(f"RMSE: {rmse:,.0f} 円")
    print(f"MAE : {mae:,.0f} 円")
    print(f"R²  : {r2:.4f}")
    
    # 特徴量重要度
    print(f"\n--- 特徴量重要度 TOP 30 ---")
    feat_imp = pd.DataFrame({"name": all_feature_cols, "importance": feature_importances})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False).head(30)
    for _, row in feat_imp.iterrows():
        print(f"  {row['name']:30s} : {row['importance']:10.1f}")
    
    # 散布図
    plt.figure(figsize=(10, 10))
    plt.scatter(true_price, pred_price, alpha=0.3, color='#1f77b4', s=15, label='予測結果')
    
    max_val = max(true_price.max(), pred_price.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='理想的な予測')
    
    plt.xlabel('実際の価格 (円)', fontsize=12)
    plt.ylabel('予測された価格 (円)', fontsize=12)
    plt.title(
        f'[Model 12] Qwen 7B + BERT(PCA48) + JSON + LightGBM\n'
        f'{len(true_price)}件 | MAE: {mae:,.0f}円 | R²: {r2:.3f}',
        fontsize=14
    )
    plt.xlim(0, 200000)
    plt.ylim(0, 200000)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model12_result.png"))
    print(f"\n[*] 散布図を {os.path.join(OUTPUT_DIR, 'model12_result.png')} に保存しました。")

if __name__ == "__main__":
    main()
