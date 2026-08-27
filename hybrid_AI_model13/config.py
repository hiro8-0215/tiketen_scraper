# ============================================================
# [Model 13] 低価格帯「安い理由」特徴量追加 — 設定ファイル
# ============================================================
# Model 12 からの変更点:
#   - description_parser.py に「安い理由」特徴量を6個追加
#     (制作開放, 急ぎ/投げ売り, バラ売り, 定価以下, 見切れ/注釈付き統合)
#   - engineer_features() に平日公演フラグを追加
# ============================================================
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_DATE = "latest"

if TRAIN_DATA_DATE == "latest":
    date_data_dir = os.path.join(PROJECT_ROOT, "tiketen_date_data")
    dirs = [d for d in os.listdir(date_data_dir) if os.path.isdir(os.path.join(date_data_dir, d)) and d.startswith("data_")]
    if not dirs:
        DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    else:
        def parse_date(d):
            parts = d.split('_')
            try: return (int(parts[1]), int(parts[2]))
            except: return (0, 0)
        latest_dir = sorted(dirs, key=parse_date)[-1]
        DATA_DIR = os.path.join(date_data_dir, latest_dir)
        print(f"[*] Latest data dir: {latest_dir}")
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, "tiketen_date_data", TRAIN_DATA_DATE)

MANUAL_DATA_DIR = os.path.join(PROJECT_ROOT, "手動_data")
MASTER_ARTIST = os.path.join(MANUAL_DATA_DIR, "master_artist.csv")
MASTER_VENUE  = os.path.join(MANUAL_DATA_DIR, "master_venue.csv")
MASTER_TOUR   = os.path.join(MANUAL_DATA_DIR, "master_tour.csv")

AI_DEV_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(AI_DEV_DIR, "train_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# データ設定
# ============================================================
TARGET_GROUPS = "all"
EXCLUDE_GROUPS = ["ambitious", "b-and-zai", "banzai", "boys-be"]  # SixTONES復活
TARGET_COLUMN    = "price"
TEST_SPLIT_RATIO = 0.3
RANDOM_SEED      = 42
USE_LOG_TRANSFORM = True
FILTER_ONLY_SOLD = True
MIN_PRICE = 2000
MAX_PRICE = 150000  # 200K→150Kに調整

# 不要特徴量（データがほぼ無いものを追加）
DROP_FEATURES = [
    "seller_rating", "perf_hour", "listing_duration_days",
    "row_number", "block_rank", "ticket_count_offered",
]

# ============================================================
# Expert 1: LightGBM (CPU全スレッド)
# ============================================================
LGBM_OPTUNA_TRIALS = 50
LGBM_THREAD_COUNT  = -1
CATEGORICAL_FEATURES = ["event_id", "venue", "ticket_type", "name_type", "gate_info", "seat_level", "row_position", "is_fc_early", "is_random"]

# ============================================================
# Expert 2: BERT → PCA → LightGBM (GPU)
# ============================================================
BERT_MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 128  # BERT推論は軽いのでバッチを大きく
BERT_PCA_DIM    = 48   # 768d → 48d PCA圧縮
BERT_DIM        = BERT_PCA_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LLM (Qwen) 設定
# ============================================================
LLM_MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_LENGTH  = 256
LLM_BATCH_SIZE_TRAIN = 2
LLM_BATCH_SIZE_INFER = 8
LLM_EPOCHS      = 8
LLM_LEARNING_RATE = 2e-4
LLM_LORA_R      = 16
LLM_LORA_ALPHA  = 32
LLM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ============================================================
# NN アーキテクチャ（不使用だが互換性のため残す）
# ============================================================
NN_HIDDEN_1     = 256
NN_HIDDEN_2     = 64
NN_DROPOUT_1    = 0.3
NN_DROPOUT_2    = 0.2
NN_LEARNING_RATE = 1e-4
NN_EPOCHS       = 50
NN_BATCH_SIZE   = 64
NN_PATIENCE     = 10
GROUP_EMBED_DIM = 8
TTYPE_EMBED_DIM = 4

# ============================================================
# ユーティリティ
# ============================================================
def get_data_files():
    import glob
    all_csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*_master.csv")))
    if TARGET_GROUPS == "all":
        return [f for f in all_csvs if os.path.basename(f).replace("_master.csv", "") not in EXCLUDE_GROUPS]
    return sorted([os.path.join(DATA_DIR, f"{slug}_master.csv") for slug in TARGET_GROUPS if slug not in EXCLUDE_GROUPS])
