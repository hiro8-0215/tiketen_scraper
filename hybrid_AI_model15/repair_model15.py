"""Repair the completed Model15 run without repeating Qwen adapter training."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("\n実行:", " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"完了: {script} ({(time.time() - started) / 60:.1f}分)", flush=True)


def main():
    # This path intentionally does not call 2_train_qwen_oof.py. The existing
    # five best adapters are loaded for ordered FP16 inference only.
    run("preflight.py", "--repair")
    backup_dir = ARTIFACTS / "pre_order_repair"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "qwen_oof.csv", "evaluation_model15.json", "oof_predictions_model15.csv",
        "candidate_comparison.csv", "model15.joblib",
    ]:
        source = ARTIFACTS / name
        target = backup_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    print(f"修復前成果物を保存: {backup_dir}", flush=True)
    run("2_repair_qwen_oof.py")
    run("2_train_model15.py")
    run("view_results.py")


if __name__ == "__main__":
    main()
