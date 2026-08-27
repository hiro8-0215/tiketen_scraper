# ============================================================
# [Model 10] データローダー — 需給特徴量エンジニアリング
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    DATA_DIR, MASTER_ARTIST, MASTER_VENUE, MASTER_TOUR,
    TARGET_COLUMN, CATEGORICAL_FEATURES, RANDOM_SEED, get_data_files,
    MIN_PRICE, MAX_PRICE, FILTER_ONLY_SOLD, DROP_FEATURES, OUTPUT_DIR
)
import json
from description_parser import parse_description


def load_raw_data() -> pd.DataFrame:
    """全データ (sold/deleted/listing) を読み込む"""
    data_files = get_data_files()
    frames = []
    for fpath in data_files:
        slug = os.path.basename(fpath).replace("_master.csv", "")
        df = pd.read_csv(fpath, low_memory=False)
        df["group_slug"] = slug
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"[データ読み込み完了] 合計 {len(combined)} 件 ({len(data_files)} グループ)")
    return combined


def compute_demand_supply_features(df_all: pd.DataFrame, df_sold: pd.DataFrame) -> pd.DataFrame:
    """
    需給特徴量を計算する。
    df_all:  全データ (sold + deleted + listing) — 供給量の計算用
    df_sold: soldデータのみ — 学習対象。ここに特徴量を追加して返す。

    Target leakage対策: leave-one-out で自分自身を除外する。
    """
    df = df_sold.copy()
    print(f"\n  [需給特徴量 (リーク無し)] 計算中...")

    # --------------------------------------------------
    # カテゴリ1: 需給バランス (全データから。priceを使わない)
    # --------------------------------------------------
    if "event_id" in df_all.columns and "event_id" in df.columns:
        # イベントごとの全出品数 (sold + deleted + listing)
        event_total = df_all.groupby("event_id").size().rename("event_listing_total")
        df = df.merge(event_total, on="event_id", how="left")

        # イベントごとのsold数
        if "status" in df_all.columns:
            event_sold_all = df_all[df_all["status"] == "sold"].groupby("event_id").size().rename("event_sold_total")
            df = df.merge(event_sold_all, on="event_id", how="left")
            df["event_sold_total"] = df["event_sold_total"].fillna(0)

            # 売却率 = 需要の強さ (priceを使わない)
            df["event_sold_ratio"] = df["event_sold_total"] / df["event_listing_total"].clip(lower=1)
        else:
            df["event_sold_ratio"] = np.nan

        # キャパに対する供給圧
        if "capacity" in df.columns:
            df["event_supply_per_cap"] = df["event_listing_total"] / df["capacity"].clip(lower=1)

        # 同イベントの売却件数
        df["event_sold_count"] = df.groupby("event_id")["event_id"].transform("count")

        print(f"    event_listing_total: mean={df['event_listing_total'].mean():.0f}")
        print(f"    event_sold_ratio: mean={df['event_sold_ratio'].mean():.3f}")

    # --------------------------------------------------
    # カテゴリ2: 時間系の需給 (priceを使わない)
    # --------------------------------------------------
    if "first_observed_at" in df.columns and "sold_at" in df.columns:
        df["days_listed_before_sold"] = (df["sold_at"] - df["first_observed_at"]).dt.days
        df["days_listed_before_sold"] = df["days_listed_before_sold"].clip(lower=0)

    if "sold_at" in df.columns and "event_id" in df.columns:
        # イベント内で何番目に売れたか (0=最初, 1=最後)
        df["sold_timing_rank"] = df.groupby("event_id")["sold_at"].rank(pct=True)

    # グループ内の売却件数 (priceを使わない)
    if "group_slug" in df.columns:
        df["group_sold_count"] = df.groupby("group_slug")["group_slug"].transform("count")

    new_cols = [
        "event_listing_total", "event_sold_total", "event_sold_ratio",
        "event_supply_per_cap", "event_sold_count",
        "days_listed_before_sold", "sold_timing_rank",
        "group_sold_count",
    ]
    existing_new = [c for c in new_cols if c in df.columns]
    print(f"  [需給特徴量 (リーク無し)] 完了 — {len(existing_new)} 個")

    return df


def compute_target_features_for_fold(df_full, train_idx, val_idx):
    """
    CV内部で呼ぶ: 学習データのpriceのみを使って需給特徴量を計算し、
    検証データにはルックアップで適用する。これによりリークを防ぐ。

    Returns: (train_target_features_df, val_target_features_df)
    """
    df_train = df_full.iloc[train_idx]
    df_val = df_full.iloc[val_idx]
    train_prices = df_train["price"].values
    global_mean_price = train_prices.mean()
    global_log_mean = np.log1p(train_prices).mean()

    result_cols = []

    # --- イベント別の平均売却価格 (学習データのみから計算) ---
    if "event_id" in df_train.columns:
        event_stats = df_train.groupby("event_id")["price"].agg(["mean", "median", "std"]).reset_index()
        event_stats.columns = ["event_id", "event_sold_mean_train", "event_sold_median_train", "event_sold_std_train"]

        event_log_stats = df_train.assign(_lp=np.log1p(df_train["price"])).groupby("event_id")["_lp"].mean().reset_index()
        event_log_stats.columns = ["event_id", "event_sold_logmean_train"]

        # 学習データ: leave-one-out (学習データ内のみ)
        train_event_sum = df_train.groupby("event_id")["price"].transform("sum")
        train_event_cnt = df_train.groupby("event_id")["price"].transform("count")
        train_event_mean_loo = (train_event_sum - df_train["price"]) / (train_event_cnt - 1)
        train_event_mean_loo = train_event_mean_loo.replace([np.inf, -np.inf], np.nan).fillna(global_mean_price)

        train_log_prices = np.log1p(df_train["price"])
        train_log_sum = train_log_prices.groupby(df_train["event_id"]).transform("sum")
        train_log_cnt = train_log_prices.groupby(df_train["event_id"]).transform("count")
        train_log_mean_loo = (train_log_sum - train_log_prices) / (train_log_cnt - 1)
        train_log_mean_loo = train_log_mean_loo.replace([np.inf, -np.inf], np.nan).fillna(global_log_mean)

        train_std = df_train.groupby("event_id")["price"].transform("std").fillna(0)

        train_feat = pd.DataFrame({
            "event_sold_mean": train_event_mean_loo.values,
            "event_sold_logmean": train_log_mean_loo.values,
            "event_sold_std": train_std.values,
        }, index=df_train.index)

        # 検証データ: 学習データの統計量をルックアップ (検証データのpriceは一切使わない)
        val_feat = df_val[["event_id"]].merge(event_stats, on="event_id", how="left")
        val_feat = val_feat.merge(event_log_stats, on="event_id", how="left")
        val_feat = val_feat.rename(columns={
            "event_sold_mean_train": "event_sold_mean",
            "event_sold_logmean_train": "event_sold_logmean",
            "event_sold_std_train": "event_sold_std",
        })
        val_feat["event_sold_mean"] = val_feat["event_sold_mean"].fillna(global_mean_price)
        val_feat["event_sold_logmean"] = val_feat["event_sold_logmean"].fillna(global_log_mean)
        val_feat["event_sold_std"] = val_feat["event_sold_std"].fillna(0)
        val_feat = val_feat.drop(columns=["event_id", "event_sold_median_train"], errors="ignore")
        val_feat.index = df_val.index

        result_cols.extend(["event_sold_mean", "event_sold_logmean", "event_sold_std"])
    else:
        train_feat = pd.DataFrame(index=df_train.index)
        val_feat = pd.DataFrame(index=df_val.index)

    # --- グループ別の定価倍率 (学習データのみから計算) ---
    if "base_price" in df_train.columns and "group_slug" in df_train.columns:
        bp_train = df_train["base_price"].clip(lower=1)
        markup_train = df_train["price"] / bp_train

        group_stats = pd.DataFrame({
            "group_slug": df_train["group_slug"],
            "markup": markup_train
        }).groupby("group_slug")["markup"].agg(["mean", "median"]).reset_index()
        group_stats.columns = ["group_slug", "group_avg_markup_train", "group_med_markup_train"]

        # 学習データ: leave-one-out
        grp_sum = markup_train.groupby(df_train["group_slug"]).transform("sum")
        grp_cnt = markup_train.groupby(df_train["group_slug"]).transform("count")
        grp_loo = (grp_sum - markup_train) / (grp_cnt - 1)
        grp_loo = grp_loo.replace([np.inf, -np.inf], np.nan).fillna(markup_train.mean())
        train_feat["group_avg_markup"] = grp_loo.values

        # 検証データ: ルックアップ
        val_grp = df_val[["group_slug"]].merge(group_stats, on="group_slug", how="left")
        val_feat["group_avg_markup"] = val_grp["group_avg_markup_train"].fillna(markup_train.mean()).values

        result_cols.append("group_avg_markup")

    return train_feat[result_cols], val_feat[result_cols]


def clean_data(df: pd.DataFrame, keep_all_status=False) -> pd.DataFrame:
    """データクレンジング。keep_all_status=True なら全ステータスを保持"""
    df = df.copy()

    if not keep_all_status and FILTER_ONLY_SOLD and "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "sold"]
        print(f"  [クレンジング] 売却済み(sold)データのみ抽出: {before} -> {len(df)} 件")

    # ---------------------------------------------------------
    # [Model 9-2 追加] テキストから予測不可能なノイズデータの徹底除外
    # ---------------------------------------------------------
    if "raw_description" in df.columns:
        before = len(df)
        
        # 1. 空・極端に短いテキスト（5文字未満）を除外
        mask_valid_desc = df["raw_description"].notna() & (df["raw_description"].str.strip().str.len() >= 5)
        df = df[mask_valid_desc]
        print(f"  [クレンジング] 説明文が空/極短(5文字未満)を除外: {before} -> {len(df)} 件")
        
        before = len(df)
        # 2. クローズドな取引・価格偽装ワードを除外
        # （手渡し、専用、代理、取り置き、ダミー、相場理解、即決額など）
        pattern = r"専用|代理|取り置き|ダミー|相場理解|即決額|手渡し|別途\s*(?:定価|支払|決済|負担)|即\s*\d(?:\.\d)?|当日\s*\d(?:\.\d)?"
        mask_clean = ~df["raw_description"].str.contains(pattern, na=False, regex=True)
        df = df[mask_clean]
        print(f"  [クレンジング] 事前交渉/専用/手渡し等のノイズ除外: {before} -> {len(df)} 件")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    before = len(df)
    df = df[(df[TARGET_COLUMN] >= MIN_PRICE) & (df[TARGET_COLUMN] <= MAX_PRICE)]
    if before != len(df):
        print(f"  [クレンジング] price異常値を除外: {before} -> {len(df)} 件")

    if "event_id" in df.columns:
        before = len(df)
        q_low = df.groupby("event_id")[TARGET_COLUMN].transform(lambda x: x.quantile(0.05))
        q_hi = df.groupby("event_id")[TARGET_COLUMN].transform(lambda x: x.quantile(0.98))
        df = df[(df[TARGET_COLUMN] >= q_low) & (df[TARGET_COLUMN] <= q_hi)]
        print(f"  [クレンジング] 異常値除外(下位5%/上位2%): {before} -> {len(df)} 件")

    for col in ["perf_date", "first_observed_at", "last_observed_at", "sold_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).astype(int)

    for col in DROP_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def load_manual_data() -> dict:
    masters = {}
    for name, path in [("artist", MASTER_ARTIST), ("venue", MASTER_VENUE), ("tour", MASTER_TOUR)]:
        if os.path.exists(path):
            masters[name] = pd.read_csv(path, encoding="utf-8")
        else:
            masters[name] = pd.DataFrame()
    df_tour = masters.get("tour", pd.DataFrame())
    if not df_tour.empty:
        for col in ["lottery_date", "first_day", "last_day"]:
            if col in df_tour.columns:
                df_tour[col] = pd.to_datetime(df_tour[col], errors="coerce")
    return masters


def merge_manual_data(df: pd.DataFrame, masters: dict) -> pd.DataFrame:
    if not masters or all(v.empty for v in masters.values()):
        return df
    df = df.copy()
    if not masters["venue"].empty:
        df_venue = masters["venue"].drop_duplicates(subset=["venue"])
        df_venue["capacity"] = pd.to_numeric(df_venue["capacity"], errors="coerce")
        df = df.merge(df_venue[["venue", "capacity"]], on="venue", how="left")
    if not masters["tour"].empty:
        df_tour = masters["tour"].drop_duplicates(subset=["event_id", "venue"])
        use_cols = ["event_id", "venue", "base_price", "lottery_date", "seat_rule", "first_day", "last_day", "total_stages"]
        df = df.merge(df_tour[[c for c in use_cols if c in df_tour.columns]], on=["event_id", "venue"], how="left")
    if not masters["artist"].empty:
        df_artist = masters["artist"].drop_duplicates(subset=["artist_id"])
        df_artist["fc_members"] = pd.to_numeric(df_artist["fc_members"], errors="coerce")
        df = df.merge(df_artist[["artist_id", "fc_members"]], left_on="group_slug", right_on="artist_id", how="left")

    if "base_price" in df.columns:
        df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce")
    if "fc_members" in df.columns and "capacity" in df.columns:
        total_stages = pd.to_numeric(df.get("total_stages", 1), errors="coerce").fillna(1)
        df["ticket_multiplier"] = df["fc_members"] / (df["capacity"] * total_stages).replace(0, np.nan)
    if "lottery_date" in df.columns and "first_observed_at" in df.columns:
        df["days_since_lottery"] = (df["first_observed_at"] - df["lottery_date"]).dt.days
    if "first_day" in df.columns and "perf_date" in df.columns:
        df["is_tour_first_day"] = (df["perf_date"] == df["first_day"]).astype(int)
    if "last_day" in df.columns and "perf_date" in df.columns:
        df["is_tour_last_day"] = (df["perf_date"] == df["last_day"]).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "perf_date" in df.columns and "first_observed_at" in df.columns:
        df["days_until_event"] = (df["perf_date"] - df["first_observed_at"]).dt.days
        df.loc[df["days_until_event"] < -30, "days_until_event"] = np.nan
    if "first_observed_at" in df.columns and "last_observed_at" in df.columns:
        df["listing_duration_days"] = (df["last_observed_at"] - df["first_observed_at"]).dt.days
    if "perf_date" in df.columns:
        df["perf_day_of_week"] = df["perf_date"].dt.dayofweek
        df["is_heijitsu"] = df["perf_day_of_week"].isin([0, 1, 2, 3, 4]).astype(int)

    if "ticket_tags" in df.columns:
        tags = df["ticket_tags"].fillna("")
        df["tag_doukkou"] = tags.str.contains("同行", na=False).astype(int)
        df["tag_jyouken_ari"] = tags.str.contains("条件あり", na=False).astype(int)

    if "raw_description" in df.columns:
        df = parse_description(df, text_col="raw_description")

    # LabelEncoding
    label_encoders = {}
    for col in CATEGORICAL_FEATURES + ["group_slug"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].fillna("__UNKNOWN__").astype(str))
            label_encoders[col] = le

    if "bante" in df.columns and "base_price" in df.columns:
        df["bante_x_baseprice"] = df["bante"].fillna(0) * df["base_price"].fillna(0)
    if "surikae_nashi" in df.columns and "bante" in df.columns:
        df["surikae_x_bante"] = df["surikae_nashi"] * df["bante"].fillna(0)
    return df, label_encoders


def merge_json_features(df: pd.DataFrame) -> pd.DataFrame:
    """LLMが抽出したJSONをロードして特徴量として結合する"""
    json_path = os.path.join(OUTPUT_DIR, "llm_extracted_features.json")
    if not os.path.exists(json_path):
        print(f"[!] {json_path} が見つかりません。先に 1_extract_json_llm.py を実行してください。")
        return df

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            extracted_dict = json.load(f)
    except Exception as e:
        print(f"[!] JSON読み込みエラー: {e}")
        return df

    # dictからDataFrameへ変換
    df_json = pd.DataFrame.from_dict(extracted_dict, orient="index").reset_index()
    df_json = df_json.rename(columns={"index": "raw_description"})

    # マージ
    before_len = len(df)
    df = df.merge(df_json, on="raw_description", how="left")
    
    # 欠損値埋め
    df["seat_level"] = df["seat_level"].fillna("不明")
    df["row_position"] = df["row_position"].fillna("不明")
    df["is_fc_early"] = df["is_fc_early"].fillna(False).astype(int)
    df["is_random"] = df["is_random"].fillna(False).astype(int)
    df["price_estimate"] = pd.to_numeric(df["price_estimate"], errors="coerce").fillna(0)
    
    print(f"  [JSON特徴量] 結合完了 (欠損なし: {len(df.dropna(subset=['seat_level']))}/{len(df)})")
    return df


def get_tabular_feature_columns(df: pd.DataFrame) -> list:
    """特徴量カラムを返す（需給特徴量含む）"""
    exclude = {
        TARGET_COLUMN, "ticket_id", "created_at_unix", "seller_name",
        "order_num", "raw_description", "details_fetched", "perf_date",
        "first_observed_at", "last_observed_at", "sold_at", "perf_time",
        "ticket_tags", "delivery_method",
        "event_id", "venue", "ticket_type", "name_type", "group_slug", "status",
        "lottery_date", "first_day", "last_day", "artist_id", "artist_name",
        "seller_rating", "perf_hour",
    }
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.int64, np.float64, np.int32, np.float32, int, float]]


def prepare_dataset():
    """メインのデータセット準備関数。需給特徴量も含む。"""
    # Step 1: 全データを読み込む（供給量の計算に必要）
    df_all = load_raw_data()

    # Step 2: soldデータのクレンジング
    df_sold = clean_data(df_all.copy())

    # Step 3: 日付変換（df_allのsold_at等も変換しておく）
    # df_sold は clean_data の中で変換済みだが、df_all は未変換のためここで処理
    for col in ["perf_date", "first_observed_at", "last_observed_at", "sold_at"]:
        if col in df_all.columns:
            df_all[col] = pd.to_datetime(df_all[col], errors="coerce")

    # Step 4: マスターデータのマージ
    masters = load_manual_data()
    df_all = merge_manual_data(df_all, masters)  # 全データにもマージ（capacity等が必要）
    df_sold = merge_manual_data(df_sold, masters)

    # Step 5: 需給特徴量の計算
    df_sold = compute_demand_supply_features(df_all, df_sold)

    # Step 5.5: LLM JSON特徴量のマージ
    df_sold = merge_json_features(df_sold)

    # Step 6: 基本特徴量エンジニアリング
    df_sold, label_encoders = engineer_features(df_sold)
    feature_cols = get_tabular_feature_columns(df_sold)
    print(f"\n[特徴量] {len(feature_cols)} 個")

    return df_sold, feature_cols, label_encoders
