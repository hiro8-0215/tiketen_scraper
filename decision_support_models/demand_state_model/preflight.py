"""Read-only input and environment checks. Does not train a model."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys

from config import HORIZONS_DAYS, MODEL_DIR, SEMANTIC_FEATURES_FILE, SEMANTIC_MANIFEST_FILE
from data_loader import latest_data_dir, load_tickets
from features import add_market_features, feature_profiles
from timeline import add_end_times, build_landmarks

sys.path.insert(0, str(MODEL_DIR.parent / "pipeline_runner"))
from snapshot_audit import validate_snapshot


def check() -> dict:
    missing_packages = [name for name in ("lightgbm", "sklearn", "scipy", "joblib") if importlib.util.find_spec(name) is None]
    if missing_packages:
        raise RuntimeError(f"Missing packages: {missing_packages}")
    selected = latest_data_dir()
    try:
        snapshot_report = validate_snapshot(selected)
    except RuntimeError as error:
        return {
            "ok": False, "snapshot": str(selected),
            "error": str(error),
            "next": "Collect a clean listing-containing snapshot before training.",
            "note": "No training or LLM extraction was executed.",
        }
    if not SEMANTIC_FEATURES_FILE.exists() or not SEMANTIC_MANIFEST_FILE.exists():
        return {
            "ok": False, "snapshot": str(selected), "semantic_ready": False,
            "next": "Run decision_support_models/semantic_data_builder/extract_semantic_json.py",
            "note": "No training or LLM extraction was executed.",
        }
    try:
        tickets = load_tickets(selected)
    except (FileNotFoundError, ValueError) as error:
        return {
            "ok": False, "snapshot": str(selected), "semantic_ready": False,
            "error": str(error), "next": "Complete semantic_data_builder first.",
            "note": "No training or LLM extraction was executed.",
        }
    tickets, _ = add_end_times(tickets)
    sample_tickets = tickets.head(min(200, len(tickets)))
    sample = add_market_features(
        build_landmarks(sample_tickets, horizons=HORIZONS_DAYS), sample_tickets
    )
    profiles = feature_profiles(sample)
    numeric, categorical = profiles["semantic"]
    projected = int(round(len(sample) / len(sample_tickets) * len(tickets)))
    row_bytes = sample.memory_usage(index=True, deep=True).sum() / max(len(sample), 1)
    free_gib = shutil.disk_usage(MODEL_DIR).free / 1024 ** 3
    projected_gib = row_bytes * projected / 1024 ** 3
    required_free_gib = max(10.0, projected_gib * 2.5 + 3.0)
    disk_ok = free_gib >= required_free_gib
    return {
        "ok": disk_ok,
        "snapshot": str(selected),
        "tickets": len(tickets),
        "status_counts": tickets["status"].value_counts().to_dict(),
        "snapshot_quality": snapshot_report,
        "excluded_temporal_anomalies": int(
            tickets.attrs.get("excluded_temporal_anomalies", 0)
        ),
        "excluded_temporal_anomaly_ticket_ids": list(
            tickets.attrs.get("excluded_temporal_anomaly_ticket_ids", [])
        ),
        "active_listings": int(tickets["status"].eq("listing").sum()),
        "sample_landmarks": len(sample),
        "projected_landmarks": projected,
        "projected_landmark_table_gib": round(projected_gib, 2),
        "feature_count": len(numeric) + len(categorical),
        "feature_profile_counts": {
            name: len(columns[0]) + len(columns[1]) for name, columns in profiles.items()
        },
        "semantic_ready": True,
        "sample_label_counts": {
            f"{horizon}d": {str(key): int(value) for key, value in sample[f"state_{horizon}d"].value_counts().items()}
            for horizon in HORIZONS_DAYS
        },
        "free_disk_gib": round(free_gib, 2),
        "required_free_disk_gib": round(required_free_gib, 2),
        "disk_warning": (
            f"At least {required_free_gib:.1f} GiB is required for atomic cache/model/OOF writes."
            if not disk_ok else None
        ),
        "invalid_listing_price_rows": int(
            tickets.attrs.get("invalid_listing_price_rows", 0)
        ),
        "invalid_listing_price_policy": tickets.attrs.get(
            "invalid_listing_price_policy"
        ),
        "fair_price_enabled": bool(tickets["fair_price"].notna().all()),
        "note": "No training was executed.",
        "inference_warning": "Current inference needs a snapshot containing listing rows." if not tickets["status"].eq("listing").any() else None,
    }


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
