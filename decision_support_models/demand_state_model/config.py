"""Standalone configuration for the demand-state competing-risk model."""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "tiketen_date_data"
MANUAL_DIR = PROJECT_ROOT / "手動_data"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
SEMANTIC_DATA_DIR = PROJECT_ROOT / "semantic_feature_data"
SEMANTIC_FEATURES_FILE = SEMANTIC_DATA_DIR / "semantic_features.csv"
SEMANTIC_MANIFEST_FILE = SEMANTIC_DATA_DIR / "semantic_manifest.json"
SEMANTIC_SCHEMA_VERSION = "target_free_semantic_v1"
REQUIRE_SEMANTIC_FEATURES = True

SEMANTIC_CATEGORICAL_FEATURES = [
    "semantic_seat_level", "semantic_row_position", "semantic_winning_route",
    "semantic_name_status", "semantic_identity_check",
    "semantic_distribution_type", "semantic_visibility",
]
SEMANTIC_NUMERIC_FEATURES = ["semantic_is_fc_early", "semantic_is_random"]
SEMANTIC_FEATURES = SEMANTIC_CATEGORICAL_FEATURES + SEMANTIC_NUMERIC_FEATURES
SEMANTIC_MIN_LOGLOSS_IMPROVEMENT = 0.001
SEMANTIC_MAX_PARSE_ERROR_RATE = 0.01

# Optional, but accepted only when every ticket in the selected snapshot is covered.
FAIR_PRICE_CACHE = ARTIFACT_DIR / "fair_price_all_tickets.csv"

PIPELINE_VERSION = "demand_state_semantic_selection_v4_logical_identity"
SEED = 42
HORIZONS_DAYS = (1, 3, 7)
LANDMARK_STEP_DAYS = 1
N_TEMPORAL_FOLDS = 4

STATUS_CLASSES = ("active", "sold", "deleted")
CLASS_TO_ID = {name: index for index, name in enumerate(STATUS_CLASSES)}

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 700,
    "learning_rate": 0.035,
    "num_leaves": 48,
    "max_depth": 9,
    "min_child_samples": 40,
    "colsample_bytree": 0.78,
    "subsample": 0.85,
    "subsample_freq": 1,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "random_state": SEED,
    # Explicit logical-core count avoids joblib's broken physical-core probe on
    # this Windows host while retaining the same all-CPU training policy.
    "n_jobs": max(os.cpu_count() or 1, 1),
    "verbosity": -1,
    "force_col_wise": True,
}

REQUIRED_COLUMNS = {
    "ticket_id", "event_id", "first_observed_at", "last_observed_at",
    "sold_at", "status", "price", "perf_date",
}
ALLOWED_STATUS = {"listing", "sold", "deleted"}
