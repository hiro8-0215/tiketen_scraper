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

# 現在のディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_COLUMN, RANDOM_SEED, CATEGORICAL_FEATURES, OUTPUT_DIR
from data_loader import prepare_dataset

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
    
    print("==========================================================")
    print(" [Model 12] 誤差1,000円以内の予測結果 抽出＆グラフ化スクリプト")
    print("==========================================================")
    
    print("\n[1] 表データの準備中...")
    df, tabular_cols, label_encoders = prepare_dataset()
    
    # listing_duration_days を明示的に除外（未来情報リーク対策）
    if 'listing_duration_days' in tabular_cols:
        tabular_cols.remove('listing_duration_days')
        print("  ⚠ listing_duration_days を除外しました（未来情報のため）")
    
    cat_cols = [col + "_encoded" for col in label_encoders.keys() if col + "_encoded" in df.columns]
    X_tabular = df[tabular_cols].copy()
    y = np.log1p(df[TARGET_COLUMN])
    sample_weights = np.sqrt(df[TARGET_COLUMN])
    
    # ==========================================
    # LLM予測値とBERT特徴量のロード
    # ==========================================
    if not os.path.exists(LLM_PREDS_FILE) or not os.path.exists(BERT_FEATURES_FILE):
        print(f"エラー: {LLM_PREDS_FILE} または {BERT_FEATURES_FILE} が見つかりません。")
        return
        
    llm_preds_df = pd.read_csv(LLM_PREDS_FILE)
    bert_embeddings = np.load(BERT_FEATURES_FILE)
    
    if len(df) != len(llm_preds_df) or len(df) != len(bert_embeddings):
        print("エラー: 行数が一致しません。")
        return
    
    # LLM予測値のキャリブレーション (Isotonic Regression)
    llm_raw_log = llm_preds_df["llm_pred_log"].values
    llm_calibrated = np.zeros_like(llm_raw_log)
    kf_calib = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, val_idx in kf_calib.split(llm_raw_log):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(llm_raw_log[train_idx], y.values[train_idx])
        llm_calibrated[val_idx] = iso.predict(llm_raw_log[val_idx])
    
    X_pred = pd.DataFrame({"llm_pred_log": llm_calibrated}, index=df.index)
    bert_cols = [f"bert_{i}" for i in range(bert_embeddings.shape[1])]
    X_bert = pd.DataFrame(bert_embeddings, columns=bert_cols, index=df.index)
    
    X = pd.concat([X_tabular, X_pred, X_bert], axis=1)
    
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
            
    all_feature_cols = list(X.columns)
    cat_feature_names = [c for c in cat_cols if c in all_feature_cols]
    
    # ==========================================
    # 簡易Optuna探索＆5-Fold OOF予測の実行
    # ==========================================
    print("\n[2] ハイパーパラメータ最適化とOOF予測を実行中...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 24, 96),
            'max_depth': trial.suggest_int("max_depth", 5, 10),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.4, 0.9),
            'bagging_fraction': trial.suggest_float("bagging_fraction", 0.6, 0.95),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 5),
            'min_child_samples': trial.suggest_int("min_child_samples", 15, 60),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            'random_state': RANDOM_SEED,
            'verbose': -1
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = []
        for train_idx, val_idx in kf.split(X):
            train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=cat_feature_names, weight=sample_weights.iloc[train_idx])
            val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data, weight=sample_weights.iloc[val_idx])
            model = lgb.train(params, train_data, num_boost_round=600, valid_sets=[val_data], callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
            cv_scores.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
        return np.mean(cv_scores)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_SEED,
        'verbose': -1
    })
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=cat_feature_names, weight=sample_weights.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data, weight=sample_weights.iloc[val_idx])
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        print(f"  Fold {fold+1}/5 完了")

    true_price = np.expm1(y.values)
    pred_price = np.expm1(oof_preds)
    abs_error = np.abs(true_price - pred_price)
    
    # ==========================================
    # 誤差1,000円以内のデータを抽出
    # ==========================================
    mask_1000 = abs_error <= 1000
    n_total = len(true_price)
    n_within = np.sum(mask_1000)
    ratio_within = (n_within / n_total) * 100
    
    true_1000 = true_price[mask_1000]
    pred_1000 = pred_price[mask_1000]
    
    mae_1000 = mean_absolute_error(true_1000, pred_1000) if n_within > 0 else 0
    r2_1000 = r2_score(true_1000, pred_1000) if n_within > 1 else 0
    
    print("\n==========================================================")
    print(" [集計結果] 誤差1,000円以内の予測パフォーマンス")
    print("==========================================================")
    print(f" 全データ数                 : {n_total:,} 件")
    print(f" 誤差1,000円以内の件数      : {n_within:,} 件 ({ratio_within:.2f} %)")
    print(f" 抽出データの MAE           : {mae_1000:,.1f} 円")
    print(f" 抽出データの R² score      : {r2_1000:.4f}")
    
    # 価格帯別の内訳
    print("\n--- 価格帯別の適合件数（誤差1,000円以内）---")
    bins = [0, 20000, 50000, 100000, 200000]
    labels = ["〜20,000円", "20,001〜50,000円", "50,001〜100,000円", "100,001円〜"]
    df_eval = pd.DataFrame({"true": true_price, "pred": pred_price, "within_1000": mask_1000})
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        sub = df_eval[(df_eval["true"] > low) & (df_eval["true"] <= high)]
        c_tot = len(sub)
        c_win = sub["within_1000"].sum()
        pct = (c_win / c_tot * 100) if c_tot > 0 else 0
        print(f"  {labels[i]:16s} : {c_win:4d} / {c_tot:4d} 件 ({pct:5.1f}%)")
    
    # ==========================================
    # 散布図作成（誤差1,000円以内のみプロット）
    # ==========================================
    if n_within > 0:
        plt.figure(figsize=(9, 9))
        plt.scatter(true_1000, pred_1000, alpha=0.5, color='#1f77b4', s=25, label=f'予測結果 (誤差≤1,000円: {n_within}件)')
        
        # 表示範囲は該当データが含まれる範囲＋余裕をもたせる
        max_val = max(true_1000.max(), pred_1000.max()) * 1.05
        max_val = max(max_val, 30000) # 最低3万円は表示
        
        plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='理想的な予測 (y=x)')
        
        plt.xlabel('実際の価格 (円)', fontsize=12)
        plt.ylabel('予測された価格 (円)', fontsize=12)
        plt.title(
            f'[Model 12] 誤差1,000円以内の予測のみ抽出\n'
            f'対象: {n_within:,}件 / 全{n_total:,}件 ({ratio_within:.1f}%) | MAE: {mae_1000:,.0f}円 | R²: {r2_1000:.3f}',
            fontsize=13
        )
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        
        out_path = os.path.join(OUTPUT_DIR, "model12_within_1000yen.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n[*] 散布図を保存しました: {out_path}")
        
        # 抽出した予測結果をCSVとしても出力
        csv_path = os.path.join(OUTPUT_DIR, "within_1000yen_predictions.csv")
        df_1000 = df_eval[df_eval["within_1000"]].copy()
        df_1000["abs_error"] = np.abs(df_1000["true"] - df_1000["pred"])
        df_1000.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[*] 対象データの予測一覧をCSVに保存しました: {csv_path}")
    else:
        print("\n⚠ 誤差1,000円以内のデータが存在しませんでした。")

if __name__ == "__main__":
    main()
