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
    run("preflight.py")
    run("bootstrap_artifacts.py")
    run("1_extract_bert.py")
    run("make_folds.py")
    # A full Model15 run enriches every sold-description with the extended,
    # target-free semantic schema.  The extractor is resumable, so completed
    # descriptions are skipped after interruption.
    run("1_extract_semantic_json.py", "--refresh-legacy")
    # Qwen must be retrained after Model13 cleansing; reusing the all-sold
    # Model14 regressor would mix excluded noisy rows back into the base learner.
    run("2_train_qwen_oof.py")
    run("2_train_model15.py")
    run("view_results.py")


if __name__ == "__main__":
    main()
