"""Model 14: sold-price maximization without post-sale leakage."""
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

SEED = 42
N_FOLDS = 5
TARGET = "price"
EXCLUDE_GROUPS = {"ambitious", "b-and-zai", "banzai", "boys-be"}

BERT_MODEL = "cl-tohoku/bert-base-japanese-v3"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 64
BERT_PCA_DIM = 64

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_MAX_LENGTH = 384
QWEN_EPOCHS = 5
QWEN_BATCH_SIZE = 2
QWEN_GRAD_ACCUM = 16
QWEN_LR = 1e-4

# Never available at listing-time prediction. `sold_at` is only allowed inside
# the as-of history builder, where strictly earlier transactions are queried.
FORBIDDEN_MODEL_COLUMNS = {
    "price", "status", "sold_at", "last_observed_at", "listing_duration_days",
    "days_listed_before_sold", "sold_timing_rank", "event_sold_total",
    "event_final_sold_ratio", "event_future_sales",
}
