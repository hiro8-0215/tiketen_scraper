import os
import sys
import glob
import pandas as pd

# Windowsのエンコーディング対策
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding="utf-8")

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "tiketen_date_data")
MANUAL_DIR = os.path.join(BASE_DIR, "手動_data")

MASTER_ARTIST = os.path.join(MANUAL_DIR, "master_artist.csv")
MASTER_VENUE = os.path.join(MANUAL_DIR, "master_venue.csv")
MASTER_TOUR = os.path.join(MANUAL_DIR, "master_tour.csv")

def ensure_master_files():
    """マスタ用CSVファイルが存在しない場合は、ヘッダーのみの空ファイルを作成する"""
    os.makedirs(MANUAL_DIR, exist_ok=True)
    
    if not os.path.exists(MASTER_ARTIST):
        pd.DataFrame(columns=["artist_id", "artist_name", "fc_members"]).to_csv(MASTER_ARTIST, index=False, encoding="utf-8-sig")
        print(f"[{MASTER_ARTIST}] を新規作成しました。")
        
    if not os.path.exists(MASTER_VENUE):
        pd.DataFrame(columns=["venue", "capacity"]).to_csv(MASTER_VENUE, index=False, encoding="utf-8-sig")
        print(f"[{MASTER_VENUE}] を新規作成しました。")
        
    if not os.path.exists(MASTER_TOUR):
        pd.DataFrame(columns=[
            "event_id", "venue", "artist_id", "base_price", "lottery_date", 
            "seat_rule", "first_day", "last_day", "total_stages"
        ]).to_csv(MASTER_TOUR, index=False, encoding="utf-8-sig")
        print(f"[{MASTER_TOUR}] を新規作成しました。")

def get_latest_data_dir():
    """tiketen_date_data 内の最新の日付フォルダを取得する"""
    dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("data_")]
    if not dirs:
        return None
    # フォルダ名 (例: data_7_3) から (月, 日) のタプルを作成してソート
    def parse_date(d):
        parts = d.split('_')
        if len(parts) >= 3:
            try:
                return (int(parts[1]), int(parts[2]))
            except ValueError:
                return (0, 0)
        return (0, 0)
        
    latest_dir = sorted(dirs, key=parse_date)[-1]
    return os.path.join(DATA_DIR, latest_dir)

def update_masters():
    """スクレイピングデータから新しい要素を抽出し、マスタCSVに追記する"""
    ensure_master_files()
    
    latest_dir = get_latest_data_dir()
    if not latest_dir:
        print("エラー: スクレイピングデータが見つかりません。")
        return
        
    print(f"最新データフォルダを解析中: {os.path.basename(latest_dir)}")
    
    # 全データを一時的に格納するセット
    scraped_artists = set()
    scraped_venues = set()
    scraped_tours = set() # (event_id, venue, artist_id)
    
    csv_files = glob.glob(os.path.join(latest_dir, "*_master.csv"))
    
    for f in csv_files:
        filename = os.path.basename(f)
        artist_id = filename.replace("_master.csv", "")
        scraped_artists.add(artist_id)
        
        try:
            df = pd.read_csv(f, usecols=["event_id", "venue"])
            # venueの収集
            for venue in df["venue"].dropna().unique():
                scraped_venues.add(venue)
            # tourの収集
            for row in df[["event_id", "venue"]].dropna().drop_duplicates().itertuples(index=False):
                scraped_tours.add((row.event_id, row.venue, artist_id))
        except Exception as e:
            print(f"ファイル読み込みエラー {filename}: {e}")
            continue

    # 1. アーティストマスタの更新
    df_artist = pd.read_csv(MASTER_ARTIST, encoding="utf-8")
    existing_artists = set(df_artist["artist_id"].dropna().astype(str))
    new_artists = scraped_artists - existing_artists
    if new_artists:
        new_rows = pd.DataFrame([{"artist_id": a, "artist_name": "", "fc_members": ""} for a in new_artists])
        new_rows.to_csv(MASTER_ARTIST, mode='a', header=False, index=False, encoding="utf-8-sig")
        print(f"  追加: アーティスト {len(new_artists)} 件")
    else:
        print("  追加: アーティスト 0 件 (すべて登録済み)")

    # 2. 会場マスタの更新
    df_venue = pd.read_csv(MASTER_VENUE, encoding="utf-8")
    existing_venues = set(df_venue["venue"].dropna().astype(str))
    new_venues = scraped_venues - existing_venues
    if new_venues:
        new_rows = pd.DataFrame([{"venue": v, "capacity": ""} for v in new_venues])
        new_rows.to_csv(MASTER_VENUE, mode='a', header=False, index=False, encoding="utf-8-sig")
        print(f"  追加: 会場 {len(new_venues)} 件")
    else:
        print("  追加: 会場 0 件 (すべて登録済み)")

    # 3. ツアーマスタの更新
    df_tour = pd.read_csv(MASTER_TOUR, encoding="utf-8-sig")
    existing_tours = set(tuple(x) for x in df_tour[["event_id", "venue"]].dropna().values.astype(str))
    new_tours = [t for t in scraped_tours if (t[0], t[1]) not in existing_tours]
    if new_tours:
        new_rows = pd.DataFrame([{
            "event_id": t[0], "venue": t[1], "artist_id": t[2], "base_price": "", 
            "lottery_date": "", "seat_rule": "", "first_day": "", "last_day": "", "total_stages": ""
        } for t in new_tours])
        new_rows.to_csv(MASTER_TOUR, mode='a', header=False, index=False, encoding="utf-8-sig")
        print(f"  追加: ツアー {len(new_tours)} 件")
    else:
        print("  追加: ツアー 0 件 (すべて登録済み)")

    print("="*50)
    print("マスタファイルの更新が完了しました！")
    print(f"手動データを開き、空欄を埋めてください: {MANUAL_DIR}")

if __name__ == "__main__":
    update_masters()
