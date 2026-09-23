"""Generate only missing semantics, then evaluate the frozen Model16."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


MODEL16_DIR = Path(__file__).resolve().parent
MODEL15_DIR = MODEL16_DIR.parent / "hybrid_AI_model15"
BASELINE_JSON = MODEL16_DIR / "artifacts" / "incremental_baseline.json"


def main() -> None:
    if not BASELINE_JSON.exists():
        raise SystemExit(
            "追加評価基準がありません。先に[追加評価][基準固定]を1回実行してください"
        )
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    cutoff = baseline.get("training_cutoff")
    if not cutoff:
        raise SystemExit("追加評価基準にtraining_cutoffがありません")
    print("新規soldのLLM JSON意味特徴だけを差分生成します。モデル学習は行いません。", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(MODEL15_DIR / "1_extract_semantic_json.py"),
            "--historical",
            "--refresh-legacy",
            "--first-observed-after", cutoff,
            "--batch-size", "4",
        ],
        cwd=MODEL15_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(MODEL16_DIR / "incremental_evaluation.py")],
        cwd=MODEL16_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
