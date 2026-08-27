"""Display Model 15's joint feature/family selection result."""
import json
from pathlib import Path

import pandas as pd

from config import PIPELINE_VERSION, QWEN_OOF_SCHEMA_VERSION


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def main():
    report = json.loads(
        (ARTIFACTS / "evaluation_model15.json").read_text(encoding="utf-8")
    )
    if report.get("cleaning_policy") != "model13_exact":
        raise RuntimeError(
            "This result predates Model13-equivalent cleansing. "
            "Run the rebuilt Model15 pipeline before viewing it."
        )
    if report.get("qwen_oof_schema_version") != QWEN_OOF_SCHEMA_VERSION:
        raise RuntimeError(
            "This result predates ordered Qwen OOF repair. Run repair_model15.py."
        )
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise RuntimeError(
            "This result predates joint feature/family selection. "
            "Run retrain_meta_model15.py (Qwen retraining is not required)."
        )

    rows = []
    for candidate, scopes in report["candidates"].items():
        metric = scopes["clean_sold"]
        family, profile = candidate.split("__", 1)
        rows.append(
            {
                "candidate": candidate,
                "profile": profile,
                "family": family,
                "MAE円": metric["mae_yen"],
                "MAPE%": metric["mape_pct"],
                "MdAPE%": metric["mdape_pct"],
                "WMAPE%": metric["wmape_pct"],
                "±20%": metric["within_20_pct"],
                "R2": metric["r2"],
                "bias円": metric["bias_yen"],
            }
        )
    table = pd.DataFrame(rows).sort_values("MAE円")
    table.to_csv(
        ARTIFACTS / "candidate_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Model 15 primary: {report['primary_candidate']}")
    print(
        f"selected profile={report['primary_feature_profile']}, "
        f"family={report['primary_model_family']}"
    )
    print(
        f"Joint Optuna: {report['lgbm_optuna_trials']}/"
        f"{report['lgbm_optuna_target_trials']} completed, "
        f"best internal MAE={report['lgbm_optuna_best_mae_yen']:,.0f}円"
    )
    print(
        f"semantic coverage: {report['semantic_coverage_pct']:.1f}% "
        f"(Qwen15 enriched: {report['semantic_qwen15_pct']:.1f}%)"
    )
    print(
        f"LLM JSON MAE effect: {report['llm_json_effect_mae_yen']:+,.0f}円"
        "（正なら改善）"
    )
    print(
        f"Qwen optimized-profile effect: {report['qwen_effect_mae_yen']:+,.0f}円"
        "（正なら改善）"
    )
    print(
        f"BERT optimized-profile effect: {report['bert_effect_mae_yen']:+,.0f}円"
        "（正なら改善）"
    )
    print("\nBest MAE by feature profile:")
    for profile, mae in sorted(
        report["feature_profile_best_mae_yen"].items(), key=lambda item: item[1]
    ):
        print(f"  {profile}: {mae:,.1f}円")
    print("\nAll jointly optimized candidates:")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
