"""Create static PNG evaluation plots from completed Model16 OOF predictions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from config import ARTIFACT_DIR, PIPELINE_VERSION


MODEL_LABELS = {
    "lgbm_log_mae": "LightGBM log",
    "lgbm_raw_mape": "LightGBM MAPE",
    "catboost_raw_mae": "CatBoost",
    "bert_ridge": "BERT Ridge",
    "global_convex_ensemble": "固定アンサンブル",
}
PRICE_BINS = [2_000, 10_000, 20_000, 30_000, 50_000, 80_000, 150_001]
PRICE_LABELS = [
    "2–9千円", "10–19千円", "20–29千円",
    "30–49千円", "50–79千円", "80–150千円",
]


def load_results():
    report = json.loads(
        (ARTIFACT_DIR / "evaluation_model16.json").read_text(encoding="utf-8")
    )
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise RuntimeError("Model16 evaluation is missing or stale")
    oof = pd.read_csv(ARTIFACT_DIR / "oof_predictions_model16.csv")
    if len(oof) != report["rows"] or not np.isfinite(
        oof.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("Model16 OOF predictions are incomplete or invalid")
    return report, oof


def configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Noto Sans JP",
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def calculate_metrics(y, prediction):
    error = prediction - y
    return {
        "mae": float(np.abs(error).mean()),
        "mape": float(np.mean(np.abs(error) / np.maximum(y, 1)) * 100),
        "r2": float(1 - np.sum(error**2) / np.sum((y - y.mean()) ** 2)),
    }


def axis_limit(y, prediction):
    maximum = max(float(np.max(y)), float(np.max(prediction)))
    return max(10_000, int(np.ceil(maximum / 10_000)) * 10_000)


def draw_scatter(ax, y, prediction, metrics, panel_title=None):
    limit = axis_limit(y, prediction)
    ax.scatter(
        y,
        prediction,
        s=18,
        alpha=0.27,
        color="#2f89bd",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        linewidth=2.5,
        color="#ed4939",
        label="完全一致線（予測＝実測）",
    )
    title = panel_title or "実際の売却価格 vs Model16予測価格"
    ax.set_title(
        f"{title}\n（R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:,.1f}円, "
        f"MAPE = {metrics['mape']:.2f}%）"
    )
    ax.set_xlabel("実際の売却価格（円）")
    ax.set_ylabel("Model16 OOF予測価格（円）")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)


def save_scatter(y, prediction, metrics):
    fig, ax = plt.subplots(figsize=(13.5, 9))
    draw_scatter(ax, y, prediction, metrics, "実際の売却価格 vs Model16予測価格 散布図")
    fig.tight_layout()
    output = ARTIFACT_DIR / "actual_vs_predicted_model16.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def draw_candidate_comparison(ax, report):
    rows = sorted(
        [
            {
                "name": MODEL_LABELS.get(name, name),
                "mae": value["mae_yen"],
                "mape": value["mape_pct"],
                "primary": name == report["primary_candidate"],
            }
            for name, value in report["candidates"].items()
        ],
        key=lambda row: row["mae"],
        reverse=True,
    )
    colors = ["#ed8b2f" if row["primary"] else "#4f91bc" for row in rows]
    bars = ax.barh([row["name"] for row in rows], [row["mae"] for row in rows], color=colors)
    maximum = max(row["mae"] for row in rows)
    ax.set_xlim(0, maximum * 1.25)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_width() + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mae']:,.0f}円 / {row['mape']:.2f}%",
            va="center",
            fontsize=10,
        )
    ax.set_title("候補モデル比較（MAE / MAPE）")
    ax.set_xlabel("平均絶対誤差 MAE（円）")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(axis="x", linestyle=":", alpha=0.65)
    ax.grid(axis="y", visible=False)


def draw_residuals(ax, y, prediction):
    residual = prediction - y
    ax.scatter(y, residual, s=14, alpha=0.2, color="#2f89bd", edgecolors="none", rasterized=True)
    ax.axhline(0, color="#ed4939", linestyle="--", linewidth=2, label="誤差0円")
    bin_id = pd.qcut(y, q=20, labels=False, duplicates="drop")
    trend = pd.DataFrame({"actual": y, "residual": residual, "bin": bin_id}).groupby("bin").median()
    ax.plot(
        trend["actual"],
        trend["residual"],
        color="#ef8a28",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="実価格分位ごとの中央値",
    )
    ax.set_title("実価格と予測残差")
    ax.set_xlabel("実際の売却価格（円）")
    ax.set_ylabel("予測－実測（円）")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.65)


def draw_price_diagnostics(ax, y, prediction):
    frame = pd.DataFrame({"actual": y, "prediction": prediction})
    frame["price_band"] = pd.cut(
        frame["actual"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False
    )
    rows = []
    for label, group in frame.groupby("price_band", observed=False):
        error = np.abs(group["prediction"] - group["actual"])
        rows.append(
            {
                "label": str(label),
                "n": len(group),
                "mae": float(error.mean()),
                "mape": float((error / group["actual"]).mean() * 100),
            }
        )
    x = np.arange(len(rows))
    bars = ax.bar(x, [row["mae"] for row in rows], color="#4f91bc", alpha=0.85)
    ax.set_xticks(x, [row["label"] for row in rows], rotation=18, ha="right")
    ax.set_xlabel("実際の売却価格帯")
    ax.set_ylabel("MAE（円）", color="#286f9d")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.tick_params(axis="y", labelcolor="#286f9d")
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={row['n']:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    relative = ax.twinx()
    relative.plot(
        x,
        [row["mape"] for row in rows],
        color="#ed6a1e",
        marker="o",
        linewidth=2.4,
        label="MAPE",
    )
    relative.set_ylabel("MAPE（%）", color="#cc5311")
    relative.tick_params(axis="y", labelcolor="#cc5311")
    ax.set_title("実価格帯別の誤差診断（モデル分割なし）")
    ax.grid(axis="y", linestyle=":", alpha=0.65)
    ax.grid(axis="x", visible=False)


def save_dashboard(report, y, prediction, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)
    draw_scatter(axes[0, 0], y, prediction, metrics, "[図1] 実測価格とOOF予測価格")
    draw_candidate_comparison(axes[0, 1], report)
    draw_residuals(axes[1, 0], y, prediction)
    draw_price_diagnostics(axes[1, 1], y, prediction)
    fig.suptitle(
        "チケット価格予測 Model16 評価結果（nested OOF・7,313件）",
        fontsize=21,
        fontweight="bold",
    )
    output = ARTIFACT_DIR / "evaluation_dashboard_model16.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    configure_style()
    report, oof = load_results()
    y = oof["true_price"].to_numpy(float)
    prediction = oof["pred_primary"].to_numpy(float)
    metrics = calculate_metrics(y, prediction)
    scatter = save_scatter(y, prediction, metrics)
    dashboard = save_dashboard(report, y, prediction, metrics)
    print(f"作成: {scatter}")
    print(f"作成: {dashboard}")


if __name__ == "__main__":
    main()
