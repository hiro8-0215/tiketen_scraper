"""Print Model 14 metrics and create human-readable result tables/plots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def setup_font():
    for path in [Path("C:/Windows/Fonts/meiryo.ttc"), Path("C:/Windows/Fonts/YuGothM.ttc")]:
        if path.exists():
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break


def main():
    report = json.loads((ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ARTIFACTS / "oof_predictions.csv")
    df["error"] = df["pred_price"] - df["true_price"]
    df["abs_error"] = df["error"].abs()
    df["ape_pct"] = df["abs_error"] / df["true_price"].clip(lower=1) * 100

    bins = [0, 5_000, 10_000, 15_000, 20_000, 30_000, 50_000, 80_000, 150_000, np.inf]
    labels = ["0-5千", "5千-1万", "1-1.5万", "1.5-2万", "2-3万", "3-5万", "5-8万", "8-15万", "15万以上"]
    df["price_band"] = pd.cut(df["true_price"], bins=bins, labels=labels, right=False)
    band = df.groupby("price_band", observed=False).agg(
        件数=("true_price", "size"),
        MAE_円=("abs_error", "mean"),
        MAPE_pct=("ape_pct", "mean"),
        MdAPE_pct=("ape_pct", "median"),
        平均バイアス_円=("error", "mean"),
    ).reset_index()
    within = df.assign(within20=df["ape_pct"].le(20)).groupby("price_band", observed=False)["within20"].mean().mul(100)
    band["±20pct以内"] = band["price_band"].map(within).astype(float)
    for col in ["MAE_円", "MAPE_pct", "MdAPE_pct", "平均バイアス_円", "±20pct以内"]:
        band[col] = band[col].round(1)
    band.to_csv(ARTIFACTS / "price_band_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = report["blended"]
    print("\n" + "=" * 64)
    print("Model 14 — sold価格 OOF評価")
    print("=" * 64)
    print(f"対象: {metrics['count']:,}件 / {report['events']}公演")
    print(f"MAE       : {metrics['mae_yen']:,.0f}円")
    print(f"RMSE      : {metrics['rmse_yen']:,.0f}円")
    print(f"MAPE      : {metrics['mape_pct']:.2f}%")
    print(f"MdAPE     : {metrics['mdape_pct']:.2f}%")
    print(f"±20%以内  : {metrics['within_20_pct']:.2f}%")
    print(f"R²        : {metrics['r2']:.4f}")
    print(f"LGBM比率  : {report['blend_weight_lgbm']:.2f}")
    print("\n価格帯別:")
    print(band.to_string(index=False))

    setup_font()
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    limit = float(np.percentile(np.r_[df.true_price, df.pred_price], 99.5))
    axes[0, 0].scatter(df.true_price, df.pred_price, s=8, alpha=.25)
    axes[0, 0].plot([0, limit], [0, limit], "r--", linewidth=1.5)
    axes[0, 0].set(xlim=(0, limit), ylim=(0, limit), xlabel="実価格（円）", ylabel="予測価格（円）", title="実価格 vs OOF予測")

    axes[0, 1].hist(df.error.clip(-50_000, 50_000), bins=80, color="#4c78a8")
    axes[0, 1].axvline(0, color="red", linestyle="--")
    axes[0, 1].set(xlabel="予測誤差（予測−実価格、円）", ylabel="件数", title="誤差分布（±5万円で表示）")

    axes[1, 0].bar(band["price_band"].astype(str), band["MAE_円"], color="#f58518")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].set(ylabel="MAE（円）", title="価格帯別MAE")

    axes[1, 1].bar(band["price_band"].astype(str), band["±20pct以内"], color="#54a24b")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].set(ylim=(0, 100), ylabel="適合率（%）", title="価格帯別 ±20%以内")

    fig.suptitle(f"Model 14 | MAE {metrics['mae_yen']:,.0f}円 | MdAPE {metrics['mdape_pct']:.2f}% | R² {metrics['r2']:.3f}", fontsize=15)
    fig.tight_layout()
    output = ARTIFACTS / "model14_results.png"
    fig.savefig(output, dpi=160)
    print(f"\n保存: {output}")
    print(f"保存: {ARTIFACTS / 'price_band_metrics.csv'}")


if __name__ == "__main__":
    main()
