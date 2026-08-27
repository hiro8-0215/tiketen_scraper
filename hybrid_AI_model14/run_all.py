"""Run the complete Model 14 training pipeline sequentially and resumably."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    print("\n" + "=" * 72, flush=True)
    print("実行:", " ".join(command), flush=True)
    print("=" * 72, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    run("make_folds.py")
    run("1_extract_bert.py")
    # The Qwen runner skips folds already present in qwen_oof.csv.
    run("2_train_qwen_oof.py")
    run("3_train_meta.py")
    print("\nModel 14 の全学習が完了しました。", flush=True)


if __name__ == "__main__":
    main()
