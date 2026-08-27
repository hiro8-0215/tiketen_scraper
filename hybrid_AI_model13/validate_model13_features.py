"""
Model 13 事前検証スクリプト
==========================
低価格帯の「安い理由」特徴量が実際のCSVデータに存在するか、
またそれが価格と相関しているかを検証する。
既存コードは一切変更せず、単体で動作する。
"""
import os
import sys
import re
import numpy as np
import pandas as pd

# --- Model 12 のデータローダーを流用してデータ取得 ---
MODEL12_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hybrid_AI_model12")
sys.path.insert(0, MODEL12_DIR)
from data_loader import prepare_dataset

def main():
    print("=" * 70)
    print(" [Model 13 事前検証] 低価格帯「安い理由」特徴量の存在確認")
    print("=" * 70)

    df, feature_cols, _ = prepare_dataset()
    
    # raw_description を保持
    texts = df["raw_description"].fillna("")
    prices = df["price"]
    
    print(f"\n全データ件数: {len(df)} 件")
    print(f"価格レンジ: {prices.min():,.0f} 〜 {prices.max():,.0f} 円")
    print(f"中央値: {prices.median():,.0f} 円")
    
    # ============================================================
    # 1. 各「安い理由」キーワードの出現数と価格分布を調査
    # ============================================================
    features_to_check = {
        "見切れ席": {
            "pattern": r"見切れ|見えにくい|見えづらい|視界不良|視界が悪|柱\s*(?:あり|有|が)",
            "desc": "見切れ・視界不良の席"
        },
        "制作開放": {
            "pattern": r"制作\s*(?:開放|解放|枠)",
            "desc": "制作開放/解放枠（見切れリスク大）"
        },
        "注釈付き": {
            "pattern": r"注釈\s*付[きけ]|注釈\s*席|※\s*座席\s*の\s*特性|ステージ\s*(?:の\s*)?一部\s*(?:が\s*)?見え",
            "desc": "注釈付き席（視界制限あり）"
        },
        "機材席": {
            "pattern": r"機材\s*(?:席|開放|解放)|機材\s*(?:の\s*)?(?:影響|関係)",
            "desc": "機材席・機材開放（視界制限）"
        },
        "バラ売り": {
            "pattern": r"バラ(?:売り|\s*で)|(?:1|１)\s*枚\s*(?:のみ|だけ|単独)|(?:1|１)\s*枚\s*(?:で\s*)?(?:の\s*)?出品|(?:連番\s*)?(?:の\s*)?(?:うち|中)\s*(?:1|１)\s*枚",
            "desc": "連番バラ売り（1枚のみ→1人参戦用→需要低）"
        },
        "平日公演": {
            "pattern": None,  # perf_day_of_week で判定（テキストではなくデータから）
            "desc": "平日公演（月〜金→需要低下）"
        },
        "急ぎ/投げ売り": {
            "pattern": r"急[いぎ]|至急|(?:どなた|誰)\s*か|(?:お\s*)?(?:譲り|引[きけ]取[りっ])\s*(?:先|手)\s*(?:を\s*)?(?:探|さが)|行[けか]な[いく]なっ|(?:仕事|急用|体調)\s*(?:の\s*)?(?:都合|ため|により)",
            "desc": "急ぎ/行けなくなった等（投げ売り傾向）"
        },
        "定価以下": {
            "pattern": r"定価\s*(?:以下|割れ|未満|より\s*安)|定価\s*(?:の\s*)?\d+\s*割|半額|お安く",
            "desc": "定価以下・安価キーワード"
        },
    }
    
    print("\n" + "=" * 70)
    print(" [検証1] 各「安い理由」キーワードの出現数と価格への影響")
    print("=" * 70)
    
    results = []
    
    for fname, finfo in features_to_check.items():
        if finfo["pattern"] is None:
            # 平日公演はデータカラムから判定
            if "perf_day_of_week" in df.columns:
                mask = df["perf_day_of_week"].isin([0, 1, 2, 3, 4])  # 月〜金
            else:
                print(f"  ⚠ {fname}: perf_day_of_week カラムが存在しません。スキップ。")
                continue
        else:
            mask = texts.str.contains(finfo["pattern"], na=False, regex=True)
        
        count = mask.sum()
        pct = count / len(df) * 100
        
        if count > 0:
            avg_with = prices[mask].mean()
            med_with = prices[mask].median()
            avg_without = prices[~mask].mean()
            med_without = prices[~mask].median()
            price_diff = avg_with - avg_without
            price_diff_pct = (avg_with / avg_without - 1) * 100
        else:
            avg_with = med_with = avg_without = med_without = price_diff = price_diff_pct = 0
        
        results.append({
            "特徴量名": fname,
            "説明": finfo["desc"],
            "該当件数": count,
            "全体比率(%)": round(pct, 2),
            "該当_平均価格(円)": round(avg_with, 0),
            "該当_中央値(円)": round(med_with, 0),
            "非該当_平均価格(円)": round(avg_without, 0),
            "非該当_中央値(円)": round(med_without, 0),
            "価格差(円)": round(price_diff, 0),
            "価格差(%)": round(price_diff_pct, 1),
        })
        
        status = "✅ 有効" if count >= 10 and abs(price_diff_pct) >= 5 else ("⚠ 少数" if count < 10 else "△ 差小")
        print(f"\n  [{status}] {fname} ({finfo['desc']})")
        print(f"    該当件数: {count:,} 件 ({pct:.1f}%)")
        if count > 0:
            print(f"    該当チケットの平均価格: {avg_with:,.0f}円 (中央値: {med_with:,.0f}円)")
            print(f"    非該当チケットの平均価格: {avg_without:,.0f}円 (中央値: {med_without:,.0f}円)")
            print(f"    → 価格差: {price_diff:+,.0f}円 ({price_diff_pct:+.1f}%)")
    
    # ============================================================
    # 2. 低価格帯（0〜15,000円）に絞った分析
    # ============================================================
    print("\n\n" + "=" * 70)
    print(" [検証2] 低価格帯（0〜15,000円）限定での出現率")
    print("=" * 70)
    
    low_mask = prices <= 15000
    low_df = df[low_mask]
    low_texts = texts[low_mask]
    low_prices = prices[low_mask]
    print(f"\n低価格帯 件数: {len(low_df)} 件 (全体の {len(low_df)/len(df)*100:.1f}%)")
    print(f"低価格帯 平均価格: {low_prices.mean():,.0f}円 / 中央値: {low_prices.median():,.0f}円")
    
    for fname, finfo in features_to_check.items():
        if finfo["pattern"] is None:
            if "perf_day_of_week" in df.columns:
                mask_low = low_df["perf_day_of_week"].isin([0, 1, 2, 3, 4])
            else:
                continue
        else:
            mask_low = low_texts.str.contains(finfo["pattern"], na=False, regex=True)
        
        count_low = mask_low.sum()
        pct_low = count_low / len(low_df) * 100 if len(low_df) > 0 else 0
        
        # 全体での出現率と比較
        if finfo["pattern"] is not None:
            mask_all = texts.str.contains(finfo["pattern"], na=False, regex=True)
        elif "perf_day_of_week" in df.columns:
            mask_all = df["perf_day_of_week"].isin([0, 1, 2, 3, 4])
        else:
            continue
        pct_all = mask_all.sum() / len(df) * 100
        
        enrichment = pct_low / pct_all if pct_all > 0 else 0
        
        marker = "🔴" if enrichment > 1.5 else ("🟡" if enrichment > 1.1 else "⚪")
        print(f"  {marker} {fname}: 低価格帯 {count_low}件 ({pct_low:.1f}%) vs 全体 ({pct_all:.1f}%) → 濃縮率 {enrichment:.2f}x")

    # ============================================================
    # 3. 実際のサンプル表示（低価格帯で各特徴に該当するテキスト例）
    # ============================================================
    print("\n\n" + "=" * 70)
    print(" [検証3] 低価格帯の該当テキストサンプル（各特徴量ごと最大3件）")
    print("=" * 70)
    
    for fname, finfo in features_to_check.items():
        if finfo["pattern"] is None:
            continue
        
        mask_low = low_texts.str.contains(finfo["pattern"], na=False, regex=True)
        samples = low_df[mask_low].head(3)
        
        if len(samples) == 0:
            print(f"\n  [{fname}] 該当なし")
            continue
        
        print(f"\n  [{fname}] {mask_low.sum()}件 — サンプル:")
        for i, (idx, row) in enumerate(samples.iterrows()):
            desc_short = str(row["raw_description"])[:120].replace("\n", " ")
            print(f"    [{i+1}] ¥{row['price']:,.0f} | {desc_short}...")
    
    # ============================================================
    # 4. 結果サマリーCSV出力
    # ============================================================
    df_results = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model13_feature_validation.csv")
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n\n[*] 検証結果CSV: {out_path}")
    
    # ============================================================
    # 5. 総合判定
    # ============================================================
    print("\n" + "=" * 70)
    print(" [総合判定] Model 13 に追加する価値のある特徴量")
    print("=" * 70)
    
    for r in results:
        fname = r["特徴量名"]
        count = r["該当件数"]
        diff_pct = r["価格差(%)"]
        
        if count >= 20 and abs(diff_pct) >= 5:
            verdict = "✅ 採用推奨（十分な件数 + 明確な価格差）"
        elif count >= 5 and abs(diff_pct) >= 10:
            verdict = "✅ 採用推奨（件数少ないが価格差大）"
        elif count >= 20 and abs(diff_pct) < 5:
            verdict = "△ 要検討（件数はあるが価格差が小さい）"
        elif count < 5:
            verdict = "❌ 見送り（出現数が少なすぎる）"
        else:
            verdict = "△ 要検討"
        
        print(f"  {verdict} | {fname}: {count}件, 価格差 {diff_pct:+.1f}%")
    
    print("\n[完了]")


if __name__ == "__main__":
    main()
