"""User entry point for Model16."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("\n実行:", " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"完了: {script} ({(time.time() - started) / 60:.1f}分)", flush=True)


def main():
    run("prepare_inputs.py")
    print(
        "Model15本体は学習せず、準備済みの意味特徴/BERT/foldを使って"
        "価格帯分割なし・全件共通固定式のModel16を学習します。",
        flush=True,
    )
    run("train_model16.py")
    run("incremental_evaluation.py", "--freeze")
    run("view_results.py")


if __name__ == "__main__":
    main()
