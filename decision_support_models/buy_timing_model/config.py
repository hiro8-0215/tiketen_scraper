"""Standalone configuration for the decision policy layer."""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
INPUT_DIR = MODEL_DIR / "inputs"
ARTIFACT_DIR = MODEL_DIR / "artifacts"

DEMAND_OOF = INPUT_DIR / "demand_oof.csv"
ALTERNATIVE_OOF = INPUT_DIR / "alternative_oof.csv"
PIPELINE_VERSION = "buy_timing_policy_v1"

PROFILES = {
    # Opportunity loss when a listing disappears before a cheaper replacement.
    "safety": {"sold_miss_penalty": 15000.0, "deleted_multiplier": 0.50},
    "balanced": {"sold_miss_penalty": 7500.0, "deleted_multiplier": 0.35},
    "savings": {"sold_miss_penalty": 2500.0, "deleted_multiplier": 0.20},
}

GRID = {
    "disappear_weight": (0.10, 0.25, 0.50, 0.75),
    "alternative_weight": (0.10, 0.25, 0.50, 0.75),
    "buy_threshold": (-0.05, 0.00, 0.05, 0.10, 0.20),
    "wait_threshold": (0.25, 0.40, 0.55, 0.70),
}
