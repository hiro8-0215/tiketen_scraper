"""Configuration for target-free semantic data generation."""
from pathlib import Path

BUILDER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUILDER_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "tiketen_date_data"
OUTPUT_DIR = PROJECT_ROOT / "semantic_feature_data"
OUTPUT_FILE = OUTPUT_DIR / "semantic_features.csv"
MANIFEST_FILE = OUTPUT_DIR / "semantic_manifest.json"
FAILURE_LOG_FILE = OUTPUT_DIR / "parse_failures.jsonl"

LEGACY_MODEL15_FILE = PROJECT_ROOT / "hybrid_AI_model15" / "artifacts" / "semantic_features.json"
SCHEMA_VERSION = "target_free_semantic_v1"
COMPATIBLE_LEGACY_SCHEMA = "model15_semantic_v1"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CUDA_DEVICE = 0
GPU_MEMORY_FRACTION = 0.90
DEFAULT_BATCH_SIZE = 8
MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 36
RETRY_OUTPUT_TOKENS = 36
SAVE_EVERY = 100
MAX_PARSE_ERROR_RATE = 0.01

SEMANTIC_CATEGORICAL = [
    "semantic_seat_level", "semantic_row_position", "semantic_winning_route",
    "semantic_name_status", "semantic_identity_check",
    "semantic_distribution_type", "semantic_visibility",
]
SEMANTIC_NUMERIC = ["semantic_is_fc_early", "semantic_is_random"]
SEMANTIC_FEATURES = SEMANTIC_CATEGORICAL + SEMANTIC_NUMERIC
