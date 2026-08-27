import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
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

def train_eval_lgb(X, y, sample_weights, cat_cols, label="Model"):
    """指定された特徴量セットXで5-Fold OOF予測を行い、評価指標を計算する"""
    all_feature_cols = list(X.columns)
    cat_feature_names = [c for c in cat_cols if c in all_feature_cols]
    
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
        model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data], callbacks=[lgb.early_stopping(40, verbose=False)])
        oof_preds[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        
    true_price = np.expm1(y.values)
    pred_price = np.expm1(oof_preds)
    
    abs_error = np.abs(pred_price - true_price)
    pct_error = ((pred_price - true_price) / true_price) * 100.0
    abs_pct_error = np.abs(pct_error)
    
    mae = np.mean(abs_error)
    rmse = np.sqrt(mean_squared_error(true_price, pred_price))
    r2 = r2_score(true_price, pred_price)
    mape = np.mean(abs_pct_error)
    mdape = np.median(abs_pct_error)
    win_20pct = np.sum(abs_pct_error <= 20.0) / len(true_price) * 100.0
    win_1000yen = np.sum(abs_error <= 1000) / len(true_price) * 100.0
    
    print(f"  [{label}] MAE: {mae:,.1f}円 | MAPE: {mape:.2f}% | MdAPE: {mdape:.2f}% | ±20%内: {win_20pct:.1f}% | 1000円内: {win_1000yen:.1f}% | R²: {r2:.4f}")
    
    return pred_price, {
        "label": label,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "mdape": mdape,
        "win_20pct": win_20pct,
        "win_1000yen": win_1000yen,
        "abs_error": abs_error,
        "abs_pct_error": abs_pct_error
    }

def main():
    setup_japanese_font()
    
    print("==================================================================")
    print(" [Model 1〜12 歴代世代比較] 第1世代 vs 第2世代 vs 第3世代 精度比較")
    print("==================================================================")
    
    print("\n[1] データのロードおよび特徴量準備中...")
    df, tabular_cols, label_encoders = prepare_dataset()
    if 'listing_duration_days' in tabular_cols:
        tabular_cols.remove('listing_duration_days')
        
    cat_cols = [col + "_encoded" for col in label_encoders.keys() if col + "_encoded" in df.columns]
    y = np.log1p(df[TARGET_COLUMN])
    sample_weights = np.sqrt(df[TARGET_COLUMN])
    
    llm_preds_df = pd.read_csv(LLM_PREDS_FILE)
    bert_embeddings = np.load(BERT_FEATURES_FILE)
    
    # LLM推論値の Isotonic キャリブレーション
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
    
    # ==========================================
    # 世代別 特徴量の分割
    # ==========================================
    # 【第3世代(Model 12)】までのドメイン知識特徴量
    domain_cols_gen3 = [
        'bante', 'total_meigi', 'random_type', 'surikae_nashi', 'doukou_type', 'tousen_type',
        'seat_type', 'row_number', 'block_rank', 'is_front_row', 'gate_info_encoded',
        'is_valid_term', 'has_shimosanketa', 'is_honnin_taiou', 'henbo_ari', 'is_gaitou_meigi',
        'is_chakuburo_nashi', 'is_ren_error_nashi', 'has_refund_policy', 'desc_length',
        'num_conditions', 'ticket_count_offered'
    ]
    # 【第4世代(Model 13 追加)】「安い理由」および「平日公演」特徴量
    domain_cols_gen4_new = [
        'is_seisaku_kaihou', 'is_mikire_chuushaku', 'is_bara', 'is_urgent_sale',
        'is_teikaware_keyword', 'negative_keyword_count', 'is_heijitsu'
    ]
    all_domain_cols = domain_cols_gen3 + domain_cols_gen4_new
    
    # 【第1世代】初期表データのみ (Model 4〜7 等の再現: ドメイン知識・BERT・LLMなし)
    gen1_cols = [c for c in tabular_cols if c not in all_domain_cols]
    X_gen1 = df[gen1_cols].copy()
    for c in cat_cols:
        if c in X_gen1.columns:
            X_gen1[c] = X_gen1[c].astype("category")
            
    # 【第2世代】表データ ＋ BERT ＋ LLM (ドメイン知識なし、Model 8〜9等の再現)
    X_gen2 = pd.concat([X_gen1, X_pred, X_bert], axis=1)
    for c in cat_cols:
        if c in X_gen2.columns:
            X_gen2[c] = X_gen2[c].astype("category")
            
    # 【第3世代】 Model 12 現状 (第3世代までのドメイン知識パーサー構造化含む)
    gen3_cols = [c for c in tabular_cols if c not in domain_cols_gen4_new]
    X_gen3 = pd.concat([df[gen3_cols].copy(), X_pred, X_bert], axis=1)
    for c in cat_cols:
        if c in X_gen3.columns:
            X_gen3[c] = X_gen3[c].astype("category")

    # 【第4世代】 Model 13 現状 (「安い理由」ドメイン特徴量を組み込んだフル特徴量)
    X_gen4 = pd.concat([df[tabular_cols].copy(), X_pred, X_bert], axis=1)
    for c in cat_cols:
        if c in X_gen4.columns:
            X_gen4[c] = X_gen4[c].astype("category")
            
    print(f"  第1世代の特徴量数: {X_gen1.shape[1]} 個")
    print(f"  第2世代の特徴量数: {X_gen2.shape[1]} 個")
    print(f"  第3世代(Model 12)の特徴量数: {X_gen3.shape[1]} 個")
    print(f"  第4世代(Model 13)の特徴量数: {X_gen4.shape[1]} 個")
    
    print("\n[2] 歴代世代モデルの 5-Fold OOF 学習および比較推論を実行中...")
    
    p_gen1, res_gen1 = train_eval_lgb(X_gen1, y, sample_weights, cat_cols, "第1世代 (表データのみ)")
    p_gen2, res_gen2 = train_eval_lgb(X_gen2, y, sample_weights, cat_cols, "第2世代 (表+BERT+LLM)")
    p_gen3, res_gen3 = train_eval_lgb(X_gen3, y, sample_weights, cat_cols, "第3世代 (Model 12: 高値理由)")
    p_gen4, res_gen4 = train_eval_lgb(X_gen4, y, sample_weights, cat_cols, "第4世代 (Model 13: 安い理由追加)")
    
    # サマリーDataFrame作成
    summary_df = pd.DataFrame([
        {
            "世代": "第1世代 (表データのみ)",
            "金額誤差_MAE(円)": res_gen1["mae"],
            "割合誤差_MAPE(%)": res_gen1["mape"],
            "割合誤差_MdAPE(%)": res_gen1["mdape"],
            "誤差±20%以内適合率(%)": res_gen1["win_20pct"],
            "誤差1000円以内適合率(%)": res_gen1["win_1000yen"],
            "決定係数_R2": res_gen1["r2"]
        },
        {
            "世代": "第2世代 (表+BERT+LLM)",
            "金額誤差_MAE(円)": res_gen2["mae"],
            "割合誤差_MAPE(%)": res_gen2["mape"],
            "割合誤差_MdAPE(%)": res_gen2["mdape"],
            "誤差±20%以内適合率(%)": res_gen2["win_20pct"],
            "誤差1000円以内適合率(%)": res_gen2["win_1000yen"],
            "決定係数_R2": res_gen2["r2"]
        },
        {
            "世代": "第3世代 (Model 12: 高値理由)",
            "金額誤差_MAE(円)": res_gen3["mae"],
            "割合誤差_MAPE(%)": res_gen3["mape"],
            "割合誤差_MdAPE(%)": res_gen3["mdape"],
            "誤差±20%以内適合率(%)": res_gen3["win_20pct"],
            "誤差1000円以内適合率(%)": res_gen3["win_1000yen"],
            "決定係数_R2": res_gen3["r2"]
        },
        {
            "世代": "第4世代 (Model 13: 安い理由追加)",
            "金額誤差_MAE(円)": res_gen4["mae"],
            "割合誤差_MAPE(%)": res_gen4["mape"],
            "割合誤差_MdAPE(%)": res_gen4["mdape"],
            "誤差±20%以内適合率(%)": res_gen4["win_20pct"],
            "誤差1000円以内適合率(%)": res_gen4["win_1000yen"],
            "決定係数_R2": res_gen4["r2"]
        }
    ])
    csv_sum_path = os.path.join(OUTPUT_DIR, "model_generations_comparison_summary.csv")
    summary_df.to_csv(csv_sum_path, index=False, encoding="utf-8-sig")
    
    print("\n=================================================================================")
    print(" [歴代世代サマリー] 金額誤差(MAE) vs 割合誤差(MAPE) vs ±20%以内適合率(%)")
    print("=================================================================================")
    print(summary_df.to_string(index=False))
    print("=================================================================================")
    
    # ==========================================
    # 5,000円価格帯別: 歴代世代の MAPE(%) 推移比較
    # ==========================================
    true_price = np.expm1(y.values)
    max_p = int(np.ceil(true_price.max() / 5000.0) * 5000)
    intervals_data = []
    
    for low in range(0, max_p, 5000):
        high = low + 5000
        mask = (true_price > low) & (true_price <= high) if low > 0 else (true_price >= low) & (true_price <= high)
        if np.sum(mask) == 0:
            continue
            
        intervals_data.append({
            "label": f"{int(high/1000)}k",
            "low": low,
            "high": high,
            "total": np.sum(mask),
            "mape_gen1": np.mean(res_gen1["abs_pct_error"][mask]),
            "mape_gen2": np.mean(res_gen2["abs_pct_error"][mask]),
            "mape_gen3": np.mean(res_gen3["abs_pct_error"][mask]),
            "mape_gen4": np.mean(res_gen4["abs_pct_error"][mask]),
            "mae_gen1": np.mean(res_gen1["abs_error"][mask]),
            "mae_gen2": np.mean(res_gen2["abs_error"][mask]),
            "mae_gen3": np.mean(res_gen3["abs_error"][mask]),
            "mae_gen4": np.mean(res_gen4["abs_error"][mask]),
        })
        
    df_5k = pd.DataFrame(intervals_data)
    csv_5k_path = os.path.join(OUTPUT_DIR, "model_generations_5k_comparison.csv")
    df_5k.to_csv(csv_5k_path, index=False, encoding="utf-8-sig")
    
    print("\n=================================================================================")
    print(" [5,000円価格帯別サマリー表] データ件数 vs 各世代 MAPE(%) / MAE(円)")
    print("=================================================================================")
    for _, row in df_5k.iterrows():
        print(f" [{row['label']:>4}] ({int(row['total']):>4}件) | MAPE: Gen1 {row['mape_gen1']:>5.1f}% -> Gen2 {row['mape_gen2']:>5.1f}% -> Gen3 {row['mape_gen3']:>5.1f}% -> Gen4(M13) {row['mape_gen4']:>5.1f}% | MAE(M13): {row['mae_gen4']:>6,.0f}円")
    print("=================================================================================\n")

    # ==========================================
    # グラフ作成 (3つの視点で可視化: 全体サマリー / 価格帯別MAPE / 価格帯別MAE)
    # ==========================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 19))
    
    # --- 上段: 全体比較 ---
    x_pos = np.arange(4)
    gen_names = ["第1世代\n(表のみ)", "第2世代\n(+BERT/LLM)", "第3世代\n(Model 12:高値理由)", "第4世代\n(Model 13:安い理由)"]
    
    width = 0.25
    ax1_left = ax1
    ax1_right = ax1.twinx()
    
    b_mae = ax1_left.bar(x_pos - width, summary_df["金額誤差_MAE(円)"], width=width, color='#E74C3C', alpha=0.85, label="金額誤差: MAE(円) [左軸・低いほど良]")
    b_mape = ax1_right.bar(x_pos, summary_df["割合誤差_MAPE(%)"], width=width, color='#F39C12', alpha=0.85, label="割合誤差: MAPE(%) [右軸・低いほど良]")
    b_win = ax1_right.bar(x_pos + width, summary_df["誤差±20%以内適合率(%)"], width=width, color='#27AE60', alpha=0.9, label="精度率: 誤差±20%以内適合率(%) [右軸・高いほど良]")
    
    ax1_left.set_ylabel("平均金額誤差 MAE (円)", fontsize=12, color='#E74C3C')
    ax1_right.set_ylabel("割合(%) / 適合率(%)", fontsize=12)
    ax1_left.set_xticks(x_pos)
    ax1_left.set_xticklabels(gen_names, fontsize=11, fontweight='bold')
    ax1_left.set_title("[歴代AIモデル進化サマリー (全4世代)] 金額誤差(MAE) vs 割合誤差(MAPE) vs ±20%以内適合率の進化", fontsize=14, pad=15)
    ax1_left.grid(True, linestyle=':', alpha=0.5)
    
    ax1_left.set_ylim(0, 9500)
    ax1_right.set_ylim(0, 85)
    
    for b in b_mae:
        ax1_left.annotate(f"{b.get_height():,.0f}円", (b.get_x() + b.get_width()/2., b.get_height()), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=10, fontweight='bold', color='#C0392B')
    for b in b_mape:
        ax1_right.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2., b.get_height()), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=10, fontweight='bold', color='#D68910')
    for b in b_win:
        ax1_right.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2., b.get_height()), xytext=(0, 6), textcoords="offset points", ha='center', fontsize=10, fontweight='bold', color='#1E8449')
        
    lines_l, labels_l = ax1_left.get_legend_handles_labels()
    lines_r, labels_r = ax1_right.get_legend_handles_labels()
    ax1_left.legend(lines_l + lines_r, labels_l + labels_r, loc='upper center', ncol=3, fontsize=10, framealpha=0.95)
    
    # --- 中段: 5,000円価格帯別 MAPE(割合誤差 %) の歴代比較推移 ---
    x_idx = np.arange(len(df_5k))
    ax2.plot(x_idx, df_5k["mape_gen1"], color='#7F8C8D', marker='s', linewidth=1.8, linestyle='--', label='第1世代 (表データのみ)')
    ax2.plot(x_idx, df_5k["mape_gen2"], color='#2980B9', marker='^', linewidth=2.0, label='第2世代 (表+BERT+LLM)')
    ax2.plot(x_idx, df_5k["mape_gen3"], color='#E67E22', marker='d', linewidth=2.5, label='第3世代 (Model 12: 高値理由パーサー)')
    ax2.plot(x_idx, df_5k["mape_gen4"], color='#E74C3C', marker='o', linewidth=3.2, label='第4世代 (Model 13: 安い理由・平日パーサー追加)')
    
    ax2.set_xlabel("チケット価格帯 (上限額: k=1,000円)", fontsize=12)
    ax2.set_ylabel("価格帯内 MAPE (平均絶対％誤差)", fontsize=12)
    ax2.set_title("[価格帯別・割合誤差 MAPE %] 5,000円価格帯ごとの割合誤差推移 — Model 13 (第4世代) が低価格帯を改善", fontsize=13, pad=12)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(df_5k["label"], rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- 下段: 5,000円価格帯別 MAE(金額誤差 円) の歴代比較推移 ---
    ax3.plot(x_idx, df_5k["mae_gen1"], color='#7F8C8D', marker='s', linewidth=1.8, linestyle='--', label='第1世代 (表データのみ)')
    ax3.plot(x_idx, df_5k["mae_gen2"], color='#2980B9', marker='^', linewidth=2.0, label='第2世代 (表+BERT+LLM)')
    ax3.plot(x_idx, df_5k["mae_gen3"], color='#E67E22', marker='d', linewidth=2.5, label='第3世代 (Model 12: 高値理由パーサー)')
    ax3.plot(x_idx, df_5k["mae_gen4"], color='#E74C3C', marker='o', linewidth=3.2, label='第4世代 (Model 13: 安い理由・平日パーサー追加)')
    
    ax3.set_xlabel("チケット価格帯 (上限額: k=1,000円)", fontsize=12)
    ax3.set_ylabel("価格帯内 MAE (平均金額誤差 円)", fontsize=12)
    ax3.set_title("[価格帯別・金額誤差 MAE 円] 5,000円価格帯ごとの平均金額誤差(円) 推移 — 歴代モデルの誤差収束", fontsize=13, pad=12)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels(df_5k["label"], rotation=45, ha='right', fontsize=10)
    max_mae = max(df_5k["mae_gen1"].max(), df_5k["mae_gen2"].max(), df_5k["mae_gen3"].max(), df_5k["mae_gen4"].max())
    ax3.set_ylim(0, max_mae * 1.1)
    ax3.legend(loc='upper left', fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(pad=3.0, h_pad=4.5)
    img_path = os.path.join(OUTPUT_DIR, "model_generations_comparison.png")

    plt.savefig(img_path, dpi=150)
    print(f"\n[*] 歴代比較グラフを保存しました: {img_path}")
    print(f"[*] サマリーCSV: {csv_sum_path}")

if __name__ == "__main__":
    main()
