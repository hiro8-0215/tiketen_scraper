import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("\n実行:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    run("bootstrap_artifacts.py")
    run("2_train_qwen_oof.py")
    run("3_train_meta.py")


if __name__ == "__main__":
    main()

