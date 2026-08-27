"""Display Model16's strict nested OOF result."""
import json
from pathlib import Path

import pandas as pd

from config import ARTIFACT_DIR, PIPELINE_VERSION


def main():
    report = json.loads(
        (ARTIFACT_DIR / "evaluation_model16.json").read_text(encoding="utf-8")
    )
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise RuntimeError("Model16 result is stale; run run_model16.py")
    if report.get("price_band_routing") is not False:
        raise RuntimeError("Model16 must not use price-band routing")
    rows = []
    for name, value in report["candidates"].items():
        rows.append({"candidate": name, **value})
    table = pd.DataFrame(rows).sort_values("mae_yen")
    table.to_csv(ARTIFACT_DIR / "candidate_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"Model16 primary: {report['primary_candidate']}")
    print("price-band routing: disabled")
    print("Qwen: excluded")
    print("production weights:")
    for name, weight in report["production_weights"].items():
        print(f"  {name}: {weight:.6f}")
    print(table[[
        "candidate", "mae_yen", "mape_pct", "mdape_pct", "within_20_pct", "r2", "bias_yen"
    ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
