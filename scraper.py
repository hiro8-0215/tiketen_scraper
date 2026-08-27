import os
import re
import json
import csv
import urllib.request
import time
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# === タイムリミット設定 ===
# 25分経過で途中保存して正常終了。次回トリガーで続きを自動再開。
SCRAPE_START_TIME = time.time()
MAX_RUNTIME_SECONDS = 25 * 60  # 25分

def is_time_remaining():
    """残り時間があるかチェック"""
    return (time.time() - SCRAPE_START_TIME) < MAX_RUNTIME_SECONDS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SNAPSHOT_DIR = os.path.join(DATA_DIR, 'snapshot')
MARKET_DIR = os.path.join(DATA_DIR, 'market_snapshot')

KNOWN_API_STATUSES = {'active', 'sold'}
MAX_UNEXPLAINED_DISAPPEARANCE_FRACTION = 0.80
MIN_LISTINGS_FOR_FRACTION_GUARD = 20


class ScrapeIntegrityError(RuntimeError):
    """Raised when a response is unsafe to use for state transitions."""


def fetch_html(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        return urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
    except Exception as error:
        raise ScrapeIntegrityError(f"HTML fetch failed for {url}: {error}") from error

def get_events(performer):
    html = fetch_html(f'https://ticketen.jp/performers/{performer}')
    soup = BeautifulSoup(html, 'html.parser')
    events = []
    for a in soup.find_all('a', href=True):
        m = re.match(r'^/events/([^/]+)$', a['href'])
        if m and m.group(1) not in events:
            events.append(m.group(1))
    if not events:
        raise ScrapeIntegrityError(
            f"No events were found for {performer}; preserving existing listings"
        )
    return events

def get_event_id_from_slug(slug):
    html = fetch_html(f'https://ticketen.jp/events/{slug}')
    html = html.replace('\\"', '"')
    
    match = re.search(rf'"id":"([a-zA-Z0-9]{{20}})","name":"[^"]+","slug":"{slug}"', html)
    if match: return match.group(1)
    
    match = re.search(rf'"slug":"{slug}","id":"([a-zA-Z0-9]{{20}})"', html)
    if match: return match.group(1)
    
    match = re.search(rf'"id":"([a-zA-Z0-9]{{20}})","slug":"{slug}"', html)
    if match: return match.group(1)
    
    return None

def fetch_all_tickets(event_id):
    tickets = []
    offset = 0
    limit = 1000
    while True:
        url = f"https://ticketen.jp/api/tickets/all?context=event&eventId={event_id}&activeOnly=0&limit={limit}&offset={offset}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            res = urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
            data = json.loads(res)
            batch = data.get('tickets', [])
            if not isinstance(batch, list):
                raise ScrapeIntegrityError(
                    f"Invalid ticket page for {event_id} offset {offset}"
                )
            tickets.extend(batch)
            has_more = data.get('hasMore')
            time.sleep(1)  # サーバー負荷軽減: 最終ページ含め全リクエスト後に1秒待機
            if not has_more:
                break
            next_offset = data.get('nextOffset')
            if next_offset is None or next_offset == offset:
                raise ScrapeIntegrityError(
                    f"Invalid pagination for {event_id} offset {offset}"
                )
            offset = next_offset
        except ScrapeIntegrityError:
            raise
        except Exception as e:
            raise ScrapeIntegrityError(
                f"API fetch failed for {event_id} offset {offset}: {e}"
            ) from e
    return tickets


def _ticket_match_key(row, event_id=None):
    event = event_id if event_id is not None else row.get('event_id', '')
    return (
        f"{str(event)}_{str(row.get('created_at_unix', ''))}_"
        f"{str(row.get('price', ''))}"
    )


def _sold_ticket_id(event_id, created_at_unix, price):
    """Build a stable ID without cross-event timestamp collisions."""
    return f"sold_{event_id}_{created_at_unix}_{price}"


def _all_performances_finished(rows, now):
    dates = []
    for row in rows:
        value = str(row.get('perf_date', '')).strip()[:10]
        try:
            dates.append(datetime.fromisoformat(value).date())
        except ValueError:
            return False
    return bool(dates) and max(dates) < now.date()


def validate_event_snapshot(slug, tickets, prior_active, master, now):
    """Validate a complete event response before changing stored states."""
    if not isinstance(tickets, list):
        raise ScrapeIntegrityError(f"Ticket response for {slug} is not a list")
    if prior_active and not tickets:
        raise ScrapeIntegrityError(
            f"Empty ticket response for {slug} with {len(prior_active)} stored listings"
        )

    statuses = {str(ticket.get('status', '')) for ticket in tickets}
    unknown = statuses - KNOWN_API_STATUSES
    if unknown:
        raise ScrapeIntegrityError(
            f"Unknown API statuses for {slug}: {sorted(unknown)}"
        )

    active_codes = set()
    sold_keys = set()
    for ticket in tickets:
        status = ticket.get('status')
        if status == 'active':
            share_code = ticket.get('shareCode')
            if not share_code:
                raise ScrapeIntegrityError(
                    f"Active ticket without shareCode for {slug}"
                )
            existing = master.get(share_code)
            if existing and existing.get('status') == 'sold':
                raise ScrapeIntegrityError(
                    f"Confirmed sold ticket returned as active: {share_code}"
                )
            active_codes.add(share_code)
        elif status == 'sold':
            sold_keys.add(_ticket_match_key({
                'created_at_unix': ticket.get('createdAt', ''),
                'price': ticket.get('pricePerTicket', ''),
            }, slug))

    unexplained = [
        ticket_id for ticket_id, row in prior_active.items()
        if ticket_id not in active_codes and _ticket_match_key(row) not in sold_keys
    ]
    prior_count = len(prior_active)
    future_or_unknown = not _all_performances_finished(
        list(prior_active.values()), now
    )
    if prior_count and future_or_unknown and not active_codes and unexplained:
        raise ScrapeIntegrityError(
            f"All active tickets disappeared without sold confirmation for {slug}: "
            f"{len(unexplained)}/{prior_count}; preserving listings"
        )
    if (
        prior_count >= MIN_LISTINGS_FOR_FRACTION_GUARD
        and future_or_unknown
        and len(unexplained) / prior_count
        > MAX_UNEXPLAINED_DISAPPEARANCE_FRACTION
    ):
        raise ScrapeIntegrityError(
            f"Unexplained listing disappearance for {slug}: "
            f"{len(unexplained)}/{prior_count}; preserving listings"
        )
    return active_codes


def mark_confirmed_absences_deleted(master, active_codes_by_event, now_str):
    """Delete only listings belonging to fully validated event responses."""
    changed = 0
    for ticket_id, row in master.items():
        event_id = row.get('event_id')
        if event_id not in active_codes_by_event:
            continue
        if (
            row.get('status') == 'listing'
            and ticket_id not in active_codes_by_event[event_id]
        ):
            row['status'] = 'deleted'
            row['last_observed_at'] = now_str
            changed += 1
    return changed

def parse_ticket_details(page, share_code):
    url = f"https://ticketen.jp/ticket/{share_code}"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=5000)
        page.wait_for_selector('text=チケット概要', timeout=5000)
    except Exception as e:
        print(f"Failed to load details for {share_code}: {e}")
        return None
        
    html = page.content()
    soup = BeautifulSoup(html, 'html.parser')
    
    data = {
        'seller_name': '',
        'seller_rating': '',
        'order_num': '',
        'ticket_tags': '',
        'raw_description': ''
    }
    
    text_blocks = [elem.get_text(strip=True) for elem in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3'])]
    
    def find_next_text(label):
        for i, text in enumerate(text_blocks):
            if label in text and text == label:
                if i + 1 < len(text_blocks):
                    return text_blocks[i+1]
            elif text.startswith(label):
                return text.replace(label, '').strip()
        return ''
            
    # ---- 関連タグ（同行・QRなどの抽出）----
    tags_str = find_next_text('関連タグ')
    if tags_str:
        data['ticket_tags'] = tags_str

    # ---- 詳細・備考: ページ全体テキストから正確に切り出す ----
    # 旧方式(text_blocks)は出品者情報が混入するバグがあったため、全文splitで境界検出する方式に変更
    full_text = soup.get_text(separator='\n')
    all_lines = [l.strip() for l in full_text.split('\n') if l.strip()]
    
    desc_start_idx = None
    desc_end_idx = None
    for i, line in enumerate(all_lines):
        if '詳細・備考' in line and desc_start_idx is None:
            desc_start_idx = i + 1
        if desc_start_idx is not None and line == '出品者':
            desc_end_idx = i
            break
    
    if desc_start_idx is not None:
        end = desc_end_idx if desc_end_idx else min(desc_start_idx + 60, len(all_lines))
        raw_lines = all_lines[desc_start_idx:end]
        # 連続重複行を除去
        unique_desc = []
        for line in raw_lines:
            if not unique_desc or unique_desc[-1] != line:
                unique_desc.append(line)
        data['raw_description'] = "\n".join(unique_desc).strip()
    
    # 同行・同行が先にないかticket_tagsに記録
    if not data['ticket_tags'] and data['raw_description']:
        if '同行' in data['raw_description']:
            data['ticket_tags'] = '同行記載あり'
        elif 'ランダム' in data['raw_description']:
            data['ticket_tags'] = 'ランダム記載あり'
    
    # ---- 出品者情報: 全文splitから正確に切り出す ----
    # 「出品者」〜「購入リクエスト」間を抽出して名前・評価を分離
    # ---- 出品者情報: 全文splitから正確に切り出す ----
    # 構造: 出品者 → 名前 → 評価（X.X（N件）） → 登録情報 → 購入リクエスト
    import re as _re
    seller_start_idx = None
    seller_end_idx = None
    for i, line in enumerate(all_lines):
        if line == '出品者' and seller_start_idx is None:
            seller_start_idx = i + 1
        if seller_start_idx is not None and ('購入リクエスト' in line or 'ログインして' in line):
            seller_end_idx = i
            break
    
    if seller_start_idx is not None:
        end = seller_end_idx if seller_end_idx else min(seller_start_idx + 6, len(all_lines))
        seller_block = all_lines[seller_start_idx:end]
        # 重複除去
        unique_seller = []
        for line in seller_block:
            if not unique_seller or unique_seller[-1] != line:
                unique_seller.append(line)
        
        if unique_seller:
            # 先頭行が名前（評価を含まない行）
            first_line = unique_seller[0]
            rating_in_first = _re.search(r'(\d+\.\d+)', first_line)
            if rating_in_first:
                # 名前と評価が同一行の場合: 評価の前を名前とする
                name_part = first_line[:rating_in_first.start()].strip()
                data['seller_name'] = name_part if name_part else first_line
                data['seller_rating'] = rating_in_first.group(1)
            else:
                data['seller_name'] = first_line
                # 2行目以降から評価（X.X形式）を探す
                for line in unique_seller[1:]:
                    if '誠意' in line or '登録' in line or '時間' in line:
                        break
                    m = _re.search(r'(\d+\.\d+)', line)
                    if m:
                        data['seller_rating'] = m.group(1)
                        break
            
    return data



def load_master(performer):
    master_file = os.path.join(DATA_DIR, f'{performer}_master.csv')
    if not os.path.exists(master_file):
        return {}
    
    master = {}
    with open(master_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            master[row['ticket_id']] = row
    return master

def save_master(performer, master):
    os.makedirs(DATA_DIR, exist_ok=True)
    master_file = os.path.join(DATA_DIR, f'{performer}_master.csv')
    fieldnames = ['ticket_id', 'created_at_unix', 'event_id', 'perf_date', 'perf_time', 'venue', 
                  'ticket_type', 'name_type', 'delivery_method', 'seller_name', 
                  'seller_rating', 'order_num', 'ticket_tags', 'first_observed_at', 'last_observed_at', 
                  'sold_at', 'status', 'quantity', 'price', 'raw_description', 'details_fetched']
                  
    with open(master_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t_id, row in master.items():
            row_out = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(row_out)

def save_snapshots(performer, master):
    import pandas as pd
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(MARKET_DIR, exist_ok=True)
    
    df = pd.DataFrame(list(master.values()))
    if df.empty: return
        
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['first_observed_at'] = pd.to_datetime(df['first_observed_at'], errors='coerce')
    # Filter out nat
    df = df.dropna(subset=['first_observed_at']).copy()
    df['year_month'] = df['first_observed_at'].dt.strftime('%Y-%m')
    
    for ym, group in df.groupby('year_month'):
        group.to_csv(os.path.join(SNAPSHOT_DIR, f'{performer}_{ym}.csv'), index=False, encoding='utf-8-sig')
        
    market_records = []
    for ym, group in df.groupby('year_month'):
        for (ev_id, p_date, p_time), sub in group.groupby(['event_id', 'perf_date', 'perf_time']):
            valid_prices = sub['price'].dropna()
            market_records.append({
                'year_month': ym,
                'event_id': ev_id,
                'perf_date': p_date,
                'perf_time': p_time,
                'venue': sub['venue'].iloc[0] if not sub.empty else '',
                'total_tickets': len(sub),
                'active_tickets': len(sub[sub['status'] == 'listing']),
                'sold_tickets': len(sub[sub['status'] == 'sold']),
                'deleted_tickets': len(sub[sub['status'] == 'deleted']),
                'avg_price': valid_prices.mean() if not valid_prices.empty else 0,
                'min_price': valid_prices.min() if not valid_prices.empty else 0,
                'max_price': valid_prices.max() if not valid_prices.empty else 0,
            })
            
    if market_records:
        mdf = pd.DataFrame(market_records)
        for ym, group in mdf.groupby('year_month'):
            group.to_csv(os.path.join(MARKET_DIR, f'{performer}_{ym}.csv'), index=False, encoding='utf-8-sig')

def main():
    targets_file = os.path.join(DATA_DIR, 'targets.json')
    if not os.path.exists(targets_file):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(targets_file, 'w') as f:
            json.dump(["snow-man"], f)
            
    with open(targets_file, 'r') as f:
        performers = json.load(f)

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    time_limit_reached = False
    for performer in performers:
        if not is_time_remaining():
            print(f"[TIME LIMIT] 25分経過のため残りのperformerをスキップします。次回実行で継続します。")
            time_limit_reached = True
            break
        print(f"=== Processing {performer} ===")
        master = load_master(performer)
        
        by_share_code = {t['ticket_id']: t for t in master.values() if not t['ticket_id'].startswith('sold_')}
        by_created_at = {
            _ticket_match_key(t): t
            for t in master.values() if t.get('created_at_unix')
        }
        
        new_active_tickets = []

        print(f"Fetching events for {performer}...")
        try:
            events = get_events(performer)
        except ScrapeIntegrityError as error:
            print(f"[INTEGRITY] {error}")
            continue
        active_codes_by_event = {}
        for slug in events:
            prior_active = {
                ticket_id: row for ticket_id, row in master.items()
                if row.get('status') == 'listing' and row.get('event_id') == slug
            }
            try:
                ev_firestore_id = get_event_id_from_slug(slug)
                if not ev_firestore_id:
                    raise ScrapeIntegrityError(
                        f"Could not find firestore ID for {slug}"
                    )

                print(f"Fetching API tickets for {slug}...")
                tickets = fetch_all_tickets(ev_firestore_id)
                event_active_codes = validate_event_snapshot(
                    slug, tickets, prior_active, master, datetime.now()
                )
            except ScrapeIntegrityError as error:
                print(f"[INTEGRITY] {error}")
                continue

            active_codes_by_event[slug] = event_active_codes
            for t in tickets:
                status = t.get('status')
                created_at_unix = str(t.get('createdAt', ''))
                price_val = str(t.get('pricePerTicket', ''))
                match_key = _ticket_match_key({
                    'created_at_unix': created_at_unix,
                    'price': price_val,
                }, slug)
                
                if status == 'active':
                    share_code = t.get('shareCode')
                    if not share_code: continue
                    if share_code in by_share_code:
                        row = by_share_code[share_code]
                        if row.get('status') == 'deleted':
                            row['status'] = 'listing'
                            row['sold_at'] = ''
                        row['created_at_unix'] = created_at_unix
                        row['last_observed_at'] = now_str
                        if str(row.get('details_fetched', 'False')) != 'True':
                            new_active_tickets.append(share_code)
                        by_created_at[match_key] = row
                    else:
                        row = {
                            'ticket_id': share_code,
                            'created_at_unix': created_at_unix,
                            'event_id': slug,
                            'perf_date': t.get('eventDate', ''),
                            'perf_time': t.get('eventStartTime', ''),
                            'venue': t.get('venue', ''),
                            'status': 'listing',
                            'price': t.get('pricePerTicket', 0),
                            'quantity': t.get('quantity', 0),
                            'delivery_method': t.get('deliveryMethod', ''),
                            'ticket_type': t.get('ticketType', ''),
                            'name_type': t.get('nameGender', ''),
                            'raw_description': t.get('description', ''),
                            'first_observed_at': now_str,
                            'last_observed_at': now_str,
                            'details_fetched': 'False',
                        }
                        try:
                            row['first_observed_at'] = datetime.fromtimestamp(int(created_at_unix)/1000.0).strftime('%Y-%m-%d %H:%M:%S')
                        except: pass
                            
                        by_share_code[share_code] = row
                        by_created_at[match_key] = row
                        master[share_code] = row
                        new_active_tickets.append(share_code)
                        
                elif status == 'sold':
                    if match_key in by_created_at:
                        row = by_created_at[match_key]
                        if row.get('status') != 'sold':
                            row['status'] = 'sold'
                            if not row.get('sold_at'):
                                row['sold_at'] = now_str
                        row['last_observed_at'] = now_str
                    else:
                        t_id = _sold_ticket_id(slug, created_at_unix, price_val)
                        row = {
                            'ticket_id': t_id,
                            'created_at_unix': created_at_unix,
                            'event_id': slug,
                            'perf_date': t.get('eventDate', ''),
                            'perf_time': t.get('eventStartTime', ''),
                            'venue': t.get('venue', ''),
                            'status': 'sold',
                            'price': t.get('pricePerTicket', 0),
                            'quantity': t.get('quantity', 0),
                            'delivery_method': t.get('deliveryMethod', ''),
                            'ticket_type': t.get('ticketType', ''),
                            'name_type': t.get('nameGender', ''),
                            'raw_description': t.get('description', ''),
                            'first_observed_at': now_str,
                            'last_observed_at': now_str,
                            'sold_at': now_str,
                            'details_fetched': 'False',
                        }
                        try:
                            row['first_observed_at'] = datetime.fromtimestamp(int(created_at_unix)/1000.0).strftime('%Y-%m-%d %H:%M:%S')
                        except: pass
                        by_created_at[match_key] = row
                        master[t_id] = row

        if new_active_tickets:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                fetched_count = 0
                for i, share_code in enumerate(new_active_tickets):
                    if not is_time_remaining():
                        print(f"[TIME LIMIT] 25分経過のため詳細取得を中断します。残り{len(new_active_tickets) - i}件は次回実行で継続します。")
                        # 途中まで取得した分を保存してからbreak
                        save_master(performer, master)
                        time_limit_reached = True
                        break
                    print(f"Fetching details for NEW active ticket {share_code}...")
                    details = parse_ticket_details(page, share_code)
                    time.sleep(1)  # サーバー負荷軽減: 全ページアクセス後に1秒待機
                    if not details: continue
                    
                    row = master[share_code]
                    if details.get('raw_description'):
                        row['raw_description'] = details['raw_description']
                    row['seller_name'] = details.get('seller_name', '')
                    row['seller_rating'] = details.get('seller_rating', '')
                    row['order_num'] = details.get('order_num', '')
                    row['ticket_tags'] = details.get('ticket_tags', '')
                    row['details_fetched'] = 'True'
                    fetched_count += 1
                    
                    # インクリメンタル保存: 50件ごと + 最後の1件はかならず保存
                    if fetched_count % 50 == 0 or (i + 1) == len(new_active_tickets):
                        print(f"[CHECKPOINT] Saving after {fetched_count} detail fetches...")
                        save_master(performer, master)
                    
                browser.close()
        
        if time_limit_reached:
            # タイムリミットに達した場合、現在のperformerまでの結果を保存して終了
            save_master(performer, master)
            save_snapshots(performer, master)
            print(f"[TIME LIMIT] {performer} まで保存完了。残りは次回実行で処理します。")
            break

        deleted_count = mark_confirmed_absences_deleted(
            master, active_codes_by_event, now_str
        )
        print(
            f"Validated {len(active_codes_by_event)}/{len(events)} events; "
            f"marked {deleted_count} confirmed absences deleted."
        )

        save_master(performer, master)
        save_snapshots(performer, master)
        print(f"Saved {len(master)} tickets to master for {performer}.")

if __name__ == '__main__':
    main()
