"""Re-select Model15's feature profile and LightGBM family from cached OOF data.

This entry point never trains or runs Qwen and never extracts BERT.  It backs up
the completed v2 result, validates the cached inputs, runs CPU-only meta-model
selection, and displays the new comparison.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("\n実行:", " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"完了: {script} ({(time.time() - started) / 60:.1f}分)", flush=True)


def main():
    run("preflight.py", "--meta-only")
    backup_dir = ARTIFACTS / "pre_joint_feature_selection_v3"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "evaluation_model15.json",
        "oof_predictions_model15.csv",
        "candidate_comparison.csv",
        "model15.joblib",
    ]:
        source = ARTIFACTS / name
        target = backup_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    print(f"変更前のModel15成果物を保存: {backup_dir}", flush=True)
    print(
        "Qwen OOF/BERTは読み取り専用で再利用します。"
        "Qwen学習・Qwen推論・BERT抽出は行いません。",
        flush=True,
    )
    run("2_train_model15.py")
    run("view_results.py")


if __name__ == "__main__":
    main()
