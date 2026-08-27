"""Model16: honest global ensemble configuration."""
from pathlib import Path
import sys

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
MODEL15_DIR = ROOT / "hybrid_AI_model15"
SOURCE_ARTIFACT_DIR = MODEL15_DIR / "artifacts"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
DATA_ROOT = ROOT / "tiketen_date_data"
MANUAL_DIR = ROOT / "手動_data"
SEMANTIC_FEATURES_FILE = SOURCE_ARTIFACT_DIR / "semantic_features.json"

PIPELINE_VERSION = "model16_global_nested_ensemble_v2"
SEED = 42
TARGET = "price"
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
RAW_PRICE_SCALE = 10_000.0

# Nested tuning is intentionally resumable.  Each trial reuses fold-local OHE
# matrices; CatBoost uses CPU Ordered boosting for deterministic category CTRs.
OUTER_LGBM_TRIALS = 24
OUTER_CATBOOST_TRIALS = 24
FINAL_LGBM_TRIALS = 40
FINAL_CATBOOST_TRIALS = 40
MAX_BOOST_ROUNDS = 3500
EARLY_STOPPING_ROUNDS = 120
CATBOOST_THREAD_COUNT = -1

BERT_PCA_DIMS = (16, 32, 48, 64, 96)
BERT_PCA_MAX_DIM = max(BERT_PCA_DIMS)
BERT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

# The same constant convex weights are applied to every ticket.  No price-band
# routing or target-dependent selection is permitted.
BASE_EXPERTS = (
    "lgbm_log_mae",
    "lgbm_raw_mape",
    "catboost_raw_mae",
    "bert_ridge",
)
BLEND_MAPE_TOLERANCE_PCT_POINTS = 0.10
WEIGHT_EPSILON = 1e-8

# Model13-equivalent sold cleaning inherited by the Model15 loader.
EXCLUDE_GROUPS = {"ambitious", "b-and-zai", "banzai", "boys-be"}
MIN_PRICE = 2_000
MAX_PRICE = 150_000
MIN_DESCRIPTION_LENGTH = 5
EVENT_PRICE_LOW_QUANTILE = 0.05
EVENT_PRICE_HIGH_QUANTILE = 0.98
NOISE_DESCRIPTION_PATTERN = (
    r"専用|代理|取り置き|ダミー|相場理解|即決額|手渡し|"
    r"別途\s*(?:定価|支払|決済|負担)|即\s*\d(?:\.\d)?|当日\s*\d(?:\.\d)?"
)
UNRELIABLE_TABULAR_FEATURES = {
    "seller_rating", "row_number", "block_rank", "ticket_count_offered",
}
SEMANTIC_CATEGORICAL_FEATURES = [
    "semantic_seat_level", "semantic_row_position", "semantic_winning_route",
    "semantic_name_status", "semantic_identity_check", "semantic_distribution_type",
    "semantic_visibility", "semantic_source",
]
SEMANTIC_NUMERIC_FEATURES = [
    "semantic_is_fc_early", "semantic_is_random", "semantic_confidence",
    "semantic_available",
]
FORBIDDEN_MODEL_COLUMNS = {
    "price", "status", "sold_at", "last_observed_at", "listing_duration_days",
    "days_listed_before_sold", "sold_timing_rank", "event_sold_total",
    "event_final_sold_ratio", "event_future_sales", "price_estimate",
    "semantic_price_estimate",
}
