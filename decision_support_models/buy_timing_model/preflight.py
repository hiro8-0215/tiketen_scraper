"""Check OOF inputs and overlap without fitting a policy."""
import json
import sys

from config import ALTERNATIVE_OOF, DEMAND_OOF, MODEL_DIR
from data_loader import load_oof

sys.path.insert(0, str(MODEL_DIR.parent / "pipeline_runner"))
from snapshot_audit import latest_snapshot, validate_snapshot


def check():
    selected = latest_snapshot(MODEL_DIR.parents[1] / "tiketen_date_data")
    try:
        snapshot_report = validate_snapshot(selected)
    except RuntimeError as error:
        return {
            "ok": False, "snapshot": str(selected), "error": str(error),
            "next": "Retrain the source models from a clean snapshot first.",
            "note": "No policy was trained.",
        }
    absent = [str(path) for path in (DEMAND_OOF, ALTERNATIVE_OOF) if not path.exists()]
    if absent:
        return {"ok": False, "missing_inputs": absent, "next": "Run both source models, then python prepare_inputs.py", "note": "No policy was trained."}
    frame = load_oof()
    return {"ok": True, "snapshot_quality": snapshot_report, "aligned_oof_rows": len(frame), "horizons": sorted(frame.horizon_days.unique().tolist()), "first_landmark": str(frame.landmark_at.min()), "last_landmark": str(frame.landmark_at.max()), "note": "No policy was trained."}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
