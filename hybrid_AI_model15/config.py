"""Model 15: leakage-safe semantic restoration and price-tail correction."""
from pathlib import Path
import sys

for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "tiketen_date_data"
MANUAL_DIR = ROOT / "手動_data"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
SEMANTIC_FEATURES_FILE = ARTIFACT_DIR / "semantic_features.json"

SEED = 42
N_FOLDS = 5
TARGET = "price"
EXCLUDE_GROUPS = {"ambitious", "b-and-zai", "banzai", "boys-be"}
PIPELINE_VERSION = "model15_joint_feature_selection_v3"
QWEN_OOF_SCHEMA_VERSION = "model15_qwen_oof_ordered_v2"

# Model 13 clean-data policy.  These exclusions remove listings whose recorded
# price is not a meaningful sold-ticket market price.
MIN_PRICE = 2_000
MAX_PRICE = 150_000
MIN_DESCRIPTION_LENGTH = 5
EVENT_PRICE_LOW_QUANTILE = 0.05
EVENT_PRICE_HIGH_QUANTILE = 0.98
NOISE_DESCRIPTION_PATTERN = (
    r"専用|代理|取り置き|ダミー|相場理解|即決額|手渡し|"
    r"別途\s*(?:定価|支払|決済|負担)|即\s*\d(?:\.\d)?|当日\s*\d(?:\.\d)?"
)

BERT_MODEL = "cl-tohoku/bert-base-japanese-v3"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 64
BERT_PCA_DIM = 64
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_MAX_LENGTH = 384
QWEN_EPOCHS = 5
QWEN_BATCH_SIZE = 4
QWEN_GRAD_ACCUM = 8
QWEN_EVAL_BATCH_SIZE = 8
QWEN_LR = 1e-4
QWEN_DATALOADER_WORKERS = 2
QWEN_CPU_THREADS = 8
QWEN_GPU_MEMORY_FRACTION = 0.90
QWEN_GPU_INDEX = 0
# One joint search covers 4 feature profiles x 3 objective families.  TPE can
# concentrate trials on promising combinations instead of spending 50 trials
# independently on all 12 combinations (600 trials total).
LGBM_OPTUNA_TRIALS = 120
RAW_PRICE_SCALE = 10_000.0

# Model13 deliberately excluded these columns because their observed coverage
# is too low or not stable at listing time. They remain in the raw dataset and
# text, but are not direct LightGBM columns.
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
