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

PREDS_CSV_PATH = os.path.join(OUTPUT_DIR, "all_oof_predictions_model12.csv")
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

def get_or_create_predictions():
    """保存済みの全OOF予測があれば読み込み、なければ学習して作成する"""
    if os.path.exists(PREDS_CSV_PATH):
        print(f"[*] 保存済みの予測データを使用します: {PREDS_CSV_PATH}")
        return pd.read_csv(PREDS_CSV_PATH)
        
    print("[*] 保存済み予測データが見つからないため、5-Fold OOF予測を計算します...")
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
    cat_feature_names = [c for c in cat_cols if c in X.columns]
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 45,
        'max_depth': 8,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.85,
        'bagging_freq': 2,
        'min_child_samples': 25,
        'random_state': RANDOM_SEED,
        'verbose': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=cat_feature_names, weight=sample_weights.iloc[train_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=train_data, weight=sample_weights.iloc[val_idx])
        model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        
    df_all_preds = pd.DataFrame({
        "true_price": np.expm1(y.values),
        "pred_price": np.expm1(oof_preds),
        "abs_error": np.abs(np.expm1(y.values) - np.expm1(oof_preds)),
        "under_1000": np.abs(np.expm1(y.values) - np.expm1(oof_preds)) < 1000
    })
    df_all_preds.to_csv(PREDS_CSV_PATH, index=False, encoding="utf-8-sig")
    return df_all_preds

def main():
    setup_japanese_font()
    
    print("==========================================================")
    print(" [Model 12] 価格帯別 相対誤差（何％外れているか）分析")
    print("==========================================================")
    
    df_preds = get_or_create_predictions()
    true_p = df_preds["true_price"].values
    pred_p = df_preds["pred_price"].values
    
    # 相対誤差（％）: (予測額 - 実際額) / 実際額 * 100
    # 例: 実際 10,000円, 予測 7,000円 -> -30.0%
    #     実際 70,000円, 予測 65,000円 -> -7.1%
    signed_pct_error = ((pred_p - true_p) / true_p) * 100.0
    abs_pct_error = np.abs(signed_pct_error)
    
    df_preds["signed_pct_error"] = signed_pct_error
    df_preds["abs_pct_error"] = abs_pct_error
    
    # ==========================================
    # 1. 全体としての相対誤差（MAPE・誤差精度帯）
    # ==========================================
    mape_overall = np.mean(abs_pct_error)
    median_ape = np.median(abs_pct_error)
    bias_overall = np.mean(signed_pct_error)
    
    print(f"\n[全体サマリー] 総データ数: {len(df_preds):,} 件")
    print(f"  MAPE (平均絶対百分率誤差) : {mape_overall:.2f} %")
    print(f"  MdAPE (中央値百分率誤差)  : {median_ape:.2f} %")
    print(f"  平均バイアス (符号付き誤差) : {bias_overall:+.2f} % (プラスは高め予測、マイナスは低め予測)")
    
    print("\n--- 全体の誤差範囲（何％の誤差内に収まっているか）---")
    for thresh in [5, 10, 15, 20, 30]:
        c_win = np.sum(abs_pct_error <= thresh)
        pct = (c_win / len(df_preds)) * 100
        print(f"  誤差 ±{thresh:2d}% 以内 : {c_win:5,d} / {len(df_preds):5,d} 件 ({pct:5.1f}%)")

    # ==========================================
    # 2. 5,000円価格帯ごとの相対誤差分析
    # ==========================================
    print("\n==========================================================================================")
    print(" [5,000円価格帯別] 誤差±10%以内・±20%以内の適合率 ＆ 平均MAPE(%) ＆ 平均バイアス(%)")
    print("==========================================================================================")
    print(f" {'価格帯(円)':<18} | {'総数':>5} | {'±10%以内':>9} | {'±20%以内':>9} | {'区内MAPE':>9} | {'バイアス':>9}")
    print("-" * 88)
    
    max_p = int(np.ceil(true_p.max() / 5000.0) * 5000)
    intervals = []
    
    for low in range(0, max_p, 5000):
        high = low + 5000
        mask = (true_p > low) & (true_p <= high) if low > 0 else (true_p >= low) & (true_p <= high)
        sub = df_preds[mask]
        tot = len(sub)
        if tot == 0:
            continue
            
        c_10 = np.sum(sub["abs_pct_error"] <= 10.0)
        c_20 = np.sum(sub["abs_pct_error"] <= 20.0)
        pct_10 = (c_10 / tot) * 100
        pct_20 = (c_20 / tot) * 100
        
        mape_sub = sub["abs_pct_error"].mean()
        bias_sub = sub["signed_pct_error"].mean()
        
        label_str = f"{low:,}〜{high:,}"
        print(f" {label_str:<18} | {tot:>5,d} | {pct_10:>7.1f}% | {pct_20:>7.1f}% | {mape_sub:>8.1f}% | {bias_sub:>+8.1f}%")
        
        intervals.append({
            "label": f"{int(high/1000)}k",
            "full_label": label_str,
            "low": low,
            "high": high,
            "total": tot,
            "within_10pct_count": c_10,
            "within_20pct_count": c_20,
            "within_10pct_rate": pct_10,
            "within_20pct_rate": pct_20,
            "mape": mape_sub,
            "bias": bias_sub
        })
        
    print("-" * 88)
    df_summary = pd.DataFrame(intervals)
    csv_out_path = os.path.join(OUTPUT_DIR, "percentage_error_by_5k.csv")
    df_summary.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    print(f"[*] 価格帯別の割合誤差集計結果をCSVに保存しました: {csv_out_path}")
    
    # ==========================================
    # 3. グラフ出力（2つの観点で視覚化）
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    
    x_idx = np.arange(len(df_summary))
    labels = df_summary["label"].tolist()
    
    # --- パネル1: チケット総数バー ＋ ±10%以内・±20%以内 適合率(%) ---
    ax1_bars = ax1.bar(x_idx, df_summary["total"], width=0.45, color='#BDC3C7', label='総数 (件)', alpha=0.7)
    ax1_line = ax1.twinx()
    
    l1 = ax1_line.plot(x_idx, df_summary["within_20pct_rate"], color='#2980B9', marker='o', linewidth=2.5, label='誤差 ±20%以内 適合率 (%)')
    l2 = ax1_line.plot(x_idx, df_summary["within_10pct_rate"], color='#C0392B', marker='s', linewidth=2.5, label='誤差 ±10%以内 適合率 (%)')
    
    ax1.set_ylabel('チケット件数 (件)', fontsize=12)
    ax1_line.set_ylabel('適合率 (%)', fontsize=12)
    ax1_line.set_ylim(0, 100)
    ax1.set_title('[Model 12] 価格帯（5,000円ごと）別の相違・精度: 誤差 ±10%以内 ＆ ±20%以内の適合率(%)', fontsize=13, pad=10)
    
    # 凡例をまとめる
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1_line.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper right', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # --- パネル2: 平均絶対誤差率 (MAPE %) ＆ 平均バイアス (符号付き誤差 %) ---
    ax2.plot(x_idx, df_summary["mape"], color='#8E44AD', marker='D', linewidth=2.2, label='MAPE (平均絶対％誤差)')
    ax2.plot(x_idx, df_summary["bias"], color='#E67E22', marker='o', linewidth=2.2, linestyle='--', label='平均バイアス (プラス=高め予測, マイナス=低め予測)')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.7)
    
    ax2.set_xlabel('チケット価格帯 (上限額: k=1,000円)', fontsize=12)
    ax2.set_ylabel('誤差率 (%)', fontsize=12)
    ax2.set_title('[Model 12] 価格帯別の割合誤差特徴: MAPE (絶対的なズレ%) と バイアス (過大/過小の系統的ズレ%)', fontsize=13, pad=10)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    img_out_path = os.path.join(OUTPUT_DIR, "model12_percentage_error_analysis.png")
    plt.savefig(img_out_path, dpi=150)
    print(f"[*] 相対誤差（％）分析グラフを保存しました: {img_out_path}")

if __name__ == "__main__":
    main()
