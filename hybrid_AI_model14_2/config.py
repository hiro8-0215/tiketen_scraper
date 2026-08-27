"""Model 14-2: same learning volume as Model 14, optimized execution."""
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
N_FOLDS = 5                 # Model 14と同一
TARGET = "price"
EXCLUDE_GROUPS = {"ambitious", "b-and-zai", "banzai", "boys-be"}

BERT_MODEL = "cl-tohoku/bert-base-japanese-v3"
BERT_MAX_LENGTH = 256
BERT_BATCH_SIZE = 64
BERT_PCA_DIM = 64

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 小型化しない
QWEN_MAX_LENGTH = 384                     # 短縮しない
QWEN_EPOCHS = 5                           # 削減しない
QWEN_BATCH_SIZE = 4                       # 2 -> 4: GPU並列性を高める
QWEN_GRAD_ACCUM = 8                       # 16 -> 8: 実効batch 32を維持
QWEN_EVAL_BATCH_SIZE = 8                  # backward不要なので拡大
QWEN_LR = 1e-4
QWEN_DATALOADER_WORKERS = 2
QWEN_CPU_THREADS = 8                       # 24論理CPUをPyTorchだけで占有しない
QWEN_GPU_MEMORY_FRACTION = 0.90            # WDDM共有メモリへ溢れる前にOOMにする
QWEN_GPU_INDEX = 0

FORBIDDEN_MODEL_COLUMNS = {
    "price", "status", "sold_at", "last_observed_at", "listing_duration_days",
    "days_listed_before_sold", "sold_timing_rank", "event_sold_total",
    "event_final_sold_ratio", "event_future_sales",
}
