"""User entry point for Model16."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script):
    command = [sys.executable, str(ROOT / script)]
    print("\n実行:", " ".join(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"完了: {script} ({(time.time() - started) / 60:.1f}分)", flush=True)


def main():
    run("preflight.py")
    print("価格帯分割なし・Qwenなし・全件共通固定式で学習します。", flush=True)
    run("train_model16.py")
    run("view_results.py")


if __name__ == "__main__":
    main()
