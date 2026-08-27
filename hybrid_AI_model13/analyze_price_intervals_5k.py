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
    print(" [Model 12] 価格帯（5,000円ごと）別の誤差1,000円未満 適合率分析")
    print("==========================================================")
    
    print("\n[1] 表データの準備中...")
    df, tabular_cols, label_encoders = prepare_dataset()
    
    if 'listing_duration_days' in tabular_cols:
        tabular_cols.remove('listing_duration_days')
    
    cat_cols = [col + "_encoded" for col in label_encoders.keys() if col + "_encoded" in df.columns]
    X_tabular = df[tabular_cols].copy()
    y = np.log1p(df[TARGET_COLUMN])
    sample_weights = np.sqrt(df[TARGET_COLUMN])
    
    llm_preds_df = pd.read_csv(LLM_PREDS_FILE)
    bert_embeddings = np.load(BERT_FEATURES_FILE)
    
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
    
    print("\n[2] Optuna最適化と5-Fold OOF予測を実行中...")
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

    true_price = np.expm1(y.values)
    pred_price = np.expm1(oof_preds)
    abs_error = np.abs(true_price - pred_price)
    
    # 全予測の保存
    df_all_preds = pd.DataFrame({
        "true_price": true_price,
        "pred_price": pred_price,
        "abs_error": abs_error,
        "under_1000": abs_error < 1000
    })
    csv_all_path = os.path.join(OUTPUT_DIR, "all_oof_predictions_model12.csv")
    df_all_preds.to_csv(csv_all_path, index=False, encoding="utf-8-sig")
    
    # ==========================================
    # 5,000円ごと価格帯別の集計
    # ==========================================
    print("\n=========================================================================")
    print(" [5,000円価格帯別] 総数 vs 誤差1,000円未満の適合件数 ＆ 割合(%)")
    print("=========================================================================")
    print(f" {'価格帯(円)':<20} | {'総数':>6} | {'1,000円未満':>10} | {'適合率(%)':>9} | {'区内MAE(円)':>11}")
    print("-" * 73)
    
    max_p = int(np.ceil(true_price.max() / 5000.0) * 5000)
    intervals = []
    
    for low in range(0, max_p, 5000):
        high = low + 5000
        mask = (true_price > low) & (true_price <= high) if low > 0 else (true_price >= low) & (true_price <= high)
        sub = df_all_preds[mask]
        tot = len(sub)
        if tot == 0:
            continue
        win = sub["under_1000"].sum()
        pct = (win / tot) * 100
        mae_sub = sub["abs_error"].mean()
        
        label_str = f"{low:,}〜{high:,}"
        print(f" {label_str:<20} | {tot:>6,d} | {win:>10,d} | {pct:>8.1f}% | {mae_sub:>11,.0f}")
        
        intervals.append({
            "label": f"{int(high/1000)}k",
            "full_label": label_str,
            "low": low,
            "high": high,
            "total": tot,
            "under_1000": win,
            "accuracy_pct": pct,
            "mae": mae_sub
        })
        
    print("-" * 73)
    
    # DataFrameとしてまとめる
    df_summary = pd.DataFrame(intervals)
    csv_sum_path = os.path.join(OUTPUT_DIR, "accuracy_by_5k_intervals.csv")
    df_summary.to_csv(csv_sum_path, index=False, encoding="utf-8-sig")
    print(f"[*] 集計結果CSVを保存しました: {csv_sum_path}")
    
    # ==========================================
    # グラフ出力（2軸: 総数・適合数バー ＋ 適合率折れ線）
    # ==========================================
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    x_idx = np.arange(len(df_summary))
    labels = df_summary["label"].tolist()
    
    # バーチャート（左軸: 件数）
    b1 = ax1.bar(x_idx - 0.18, df_summary["total"], width=0.36, color='#BDC3C7', label='総数 (件)', alpha=0.85)
    b2 = ax1.bar(x_idx + 0.18, df_summary["under_1000"], width=0.36, color='#2980B9', label='1,000円未満の数 (件)', alpha=0.9)
    
    # ラインチャート（右軸: 適合率 %）
    line = ax2.plot(x_idx, df_summary["accuracy_pct"], color='#C0392B', marker='o', linewidth=2.5, markersize=7, label='適合率 (%)')
    
    # 軸・ラベル設定
    ax1.set_xlabel('チケット価格帯 (上限額: k=1,000円)', fontsize=12)
    ax1.set_ylabel('チケット件数 (件)', fontsize=12)
    ax2.set_ylabel('誤差1,000円未満 適合率 (%)', fontsize=12, color='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#C0392B')
    ax2.set_ylim(0, 100)
    
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    
    # 各バーに適合率数値を表示（主要な価格帯）
    for i, row in df_summary.iterrows():
        if row["total"] >= 10:  # 10件以上のところには％を表示
            ax2.annotate(f'{row["accuracy_pct"]:.1f}%',
                         (i, row["accuracy_pct"]),
                         textcoords="offset points",
                         xytext=(0, 8),
                         ha='center',
                         fontsize=9,
                         fontweight='bold',
                         color='#C0392B')
            
    plt.title('[Model 12] 5,000円価格帯別: チケット総数 vs 誤差1,000円未満の適合数 ＆ 適合率(%)', fontsize=14, pad=15)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 凡例を統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, "model12_accuracy_by_5k.png")
    plt.savefig(out_img, dpi=150)
    print(f"[*] 価格帯別パフォーマンスグラフを保存しました: {out_img}")

if __name__ == "__main__":
    main()
