"""Explicitly copy sibling OOF files as data inputs (no code dependency)."""
from pathlib import Path
import shutil

from config import ALTERNATIVE_OOF, DEMAND_OOF, MODEL_DIR


if __name__ == "__main__":
    sources = {
        DEMAND_OOF: MODEL_DIR.parent / "demand_state_model" / "artifacts" / "oof_predictions.csv",
        ALTERNATIVE_OOF: MODEL_DIR.parent / "alternative_arrival_model" / "artifacts" / "oof_predictions.csv",
    }
    for destination, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"Run the source model first: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"copied {source} -> {destination}")
