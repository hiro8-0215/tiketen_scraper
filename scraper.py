import os
import re
import json
import csv
import hashlib
import urllib.request
import urllib.error
import urllib.parse
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# === タイムリミット設定 ===
# 25分経過で途中保存して正常終了。次回トリガーで続きを自動再開。
SCRAPE_START_TIME = time.time()
MAX_RUNTIME_SECONDS = 25 * 60  # 25分
SCRAPE_MODE = os.environ.get('SCRAPE_MODE', 'full').strip().lower()
if SCRAPE_MODE not in {'full', 'api', 'details'}:
    raise RuntimeError(
        "SCRAPE_MODE must be one of: full, api, details"
    )

def is_time_remaining():
    """残り時間があるかチェック"""
    return (time.time() - SCRAPE_START_TIME) < MAX_RUNTIME_SECONDS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SNAPSHOT_DIR = os.path.join(DATA_DIR, 'snapshot')
MARKET_DIR = os.path.join(DATA_DIR, 'market_snapshot')
ANONYMOUS_SOLD_INVENTORY = 'anonymous_sold_inventory.jsonl'

KNOWN_API_STATUSES = {'active', 'sold'}
MAX_UNEXPLAINED_DISAPPEARANCE_FRACTION = 0.80
MIN_LISTINGS_FOR_FRACTION_GUARD = 20


class ScrapeIntegrityError(RuntimeError):
    """Raised when a response is unsafe to use for state transitions."""


class NoEventsFound(ScrapeIntegrityError):
    """The performer page is valid but currently has no linked events."""


class UnexplainedDisappearance(ScrapeIntegrityError):
    """A complete API poll with active shareCodes that cannot be reconciled."""

    def __init__(self, message, active_codes):
        super().__init__(message)
        self.active_codes = active_codes


def fetch_html(url, redirects=0):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        return urllib.request.urlopen(req, timeout=15).read().decode('utf-8')
    except urllib.error.HTTPError as error:
        if error.code == 308 and redirects < 5:
            location = error.headers.get('Location')
            target = urllib.parse.urljoin(url, location or '')
            original = urllib.parse.urlparse(url)
            redirected = urllib.parse.urlparse(target)
            if (location and redirected.scheme == 'https'
                    and redirected.hostname == original.hostname):
                return fetch_html(target, redirects + 1)
        raise ScrapeIntegrityError(f"HTML fetch failed for {url}: {error}") from error
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
        raise NoEventsFound(
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


def _identifier_text(value):
    text = str(value).strip() if value is not None else ''
    return '' if text.lower() in {'none', 'null', 'nan'} else text


def _ticket_match_key(row, event_id=None):
    event = event_id if event_id is not None else row.get('event_id', '')
    created = _identifier_text(row.get('created_at_unix'))
    price = _identifier_text(row.get('price'))
    if not event or not created or not price:
        return None
    return f"{event}_{created}_{price}"


def _listing_identity_key(row, event_id=None):
    """Stable listing identity across shareCode and price edits."""
    event = event_id if event_id is not None else row.get('event_id', '')
    created = _identifier_text(row.get('created_at_unix'))
    if not event or not created:
        return None
    return f"{event}_{created}"


def _rekey_active_listing(
    master, by_share_code, by_identity, share_code, identity_key
):
    """Reuse an edited active listing instead of creating a false deletion."""
    row = by_share_code.get(share_code)
    if row is None and identity_key is not None:
        row = by_identity.get(identity_key)
        # A different shareCode after a confirmed sale is a new lifecycle,
        # not a rotation of the sold listing.
        if row is not None and row.get('status') == 'sold':
            return None, False
    if row is None:
        return None, False
    old_id = str(row.get('ticket_id', ''))
    changed = old_id != share_code
    if changed:
        master.pop(old_id, None)
        by_share_code.pop(old_id, None)
        row['ticket_id'] = share_code
        master[share_code] = row
    by_share_code[share_code] = row
    if identity_key is not None:
        by_identity[identity_key] = row
    return row, changed


def _sold_match(ticket, event, by_share_code, by_created_at, by_identity):
    """Never identify a sold listing by event and price alone."""
    share_code = _identifier_text(ticket.get('shareCode'))
    if (share_code and share_code in by_share_code
            and by_share_code[share_code].get('event_id') == event):
        return by_share_code[share_code]
    created = _identifier_text(ticket.get('createdAt'))
    price = _identifier_text(ticket.get('pricePerTicket'))
    match_key = _ticket_match_key({
        'created_at_unix': created, 'price': price,
    }, event)
    if match_key is not None and match_key in by_created_at:
        return by_created_at[match_key]
    identity_key = _listing_identity_key({'created_at_unix': created}, event)
    if identity_key is not None:
        return by_identity.get(identity_key)
    return None


def _sold_ticket_id(event_id, created_at_unix, price):
    """Build a stable ID without cross-event timestamp collisions."""
    if not created_at_unix:
        raise ValueError('Cannot create a stable sold ID without createdAt')
    return f"sold_{event_id}_{created_at_unix}_{price}"


def _anonymous_sold_record(event, ticket):
    """Retain public sold data without inventing a ticket identity or sale time."""
    def clean(value):
        if isinstance(value, str):
            return sanitize_unicode(value)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        return value

    fields = (
        'dateId', 'eventDate', 'eventStartTime', 'venue',
        'pricePerTicket', 'quantity', 'ticketType', 'nameGender',
        'deliveryMethod', 'seatType', 'description', 'tags',
    )
    values = {key: clean(ticket.get(key, '')) for key in fields}
    canonical = json.dumps(
        {'event_id': event, 'ticket': values},
        ensure_ascii=False, sort_keys=True, default=str,
    )
    fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    return fingerprint, values


def save_anonymous_sold_inventory(events, observed_at):
    """Save anonymous sold observations separately from per-ticket master CSVs.

    Identical API rows may represent multiple sales; max_observed_count keeps
    multiplicity without asserting a stable per-ticket identity.
    """
    if not any(events.values()):
        return
    path = os.path.join(DATA_DIR, ANONYMOUS_SOLD_INVENTORY)
    inventory = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    inventory[record['fingerprint']] = record
    for event, tickets in events.items():
        counts = Counter()
        values_by_fingerprint = {}
        for ticket in tickets:
            fingerprint, values = _anonymous_sold_record(event, ticket)
            counts[fingerprint] += 1
            values_by_fingerprint[fingerprint] = values
        for fingerprint, count in counts.items():
            record = inventory.get(fingerprint)
            if record is None:
                record = {
                    'fingerprint': fingerprint,
                    'event_id': event,
                    'ticket': values_by_fingerprint[fingerprint],
                    'first_observed_at': observed_at,
                    'identity_source': 'anonymous_sold_api',
                    'sold_at': None,
                }
                inventory[fingerprint] = record
            record['last_observed_at'] = observed_at
            record['max_observed_count'] = max(
                count, int(record.get('max_observed_count', 0))
            )
    temporary = path + '.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as stream:
            for fingerprint in sorted(inventory):
                stream.write(json.dumps(
                    inventory[fingerprint], ensure_ascii=True, default=str,
                ) + '\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _all_performances_finished(rows, now):
    dates = []
    for row in rows:
        value = str(row.get('perf_date', '')).strip()[:10]
        try:
            dates.append(datetime.fromisoformat(value).date())
        except ValueError:
            return False
    return bool(dates) and max(dates) < now.date()


def _events_to_poll(discovered, master, now):
    """Keep checking previously seen future events even if performer links change."""
    events = list(dict.fromkeys(discovered))
    for row in master.values():
        if row.get('status') != 'listing':
            continue
        slug = str(row.get('event_id') or '').strip()
        if not slug or slug in events:
            continue
        date_text = str(row.get('perf_date') or '').strip()[:10]
        try:
            performance_date = datetime.fromisoformat(date_text).date()
        except ValueError:
            performance_date = None
        if performance_date is None or performance_date >= now.date() - timedelta(days=1):
            events.append(slug)
    return events


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
    active_identities = set()
    sold_keys = set()
    sold_identities = set()
    sold_codes = set()
    for ticket in tickets:
        status = ticket.get('status')
        if status == 'active':
            share_code = ticket.get('shareCode')
            if not share_code:
                raise ScrapeIntegrityError(
                    f"Active ticket without shareCode for {slug}"
                )
            existing = master.get(share_code)
            if existing and existing.get('event_id') != slug:
                raise ScrapeIntegrityError(
                    f"shareCode {share_code} belongs to another event"
                )
            active_codes.add(share_code)
            identity_key = _listing_identity_key({
                'created_at_unix': ticket.get('createdAt', ''),
            }, slug)
            if identity_key is not None:
                if identity_key in active_identities:
                    raise ScrapeIntegrityError(
                        f"Duplicate active createdAt for {slug}; cannot rekey safely"
                    )
                active_identities.add(identity_key)
        elif status == 'sold':
            share_code = _identifier_text(ticket.get('shareCode'))
            if share_code:
                sold_codes.add(share_code)
            match_key = _ticket_match_key({
                'created_at_unix': ticket.get('createdAt', ''),
                'price': ticket.get('pricePerTicket', ''),
            }, slug)
            if match_key is not None:
                sold_keys.add(match_key)
            identity_key = _listing_identity_key({
                'created_at_unix': ticket.get('createdAt', ''),
            }, slug)
            if identity_key is not None:
                sold_identities.add(identity_key)

    unexplained = [
        ticket_id for ticket_id, row in prior_active.items()
        if ticket_id not in active_codes
        and ticket_id not in sold_codes
        and _listing_identity_key(row) not in active_identities
        and _listing_identity_key(row) not in sold_identities
        and _ticket_match_key(row) not in sold_keys
    ]
    prior_count = len(prior_active)
    future_or_unknown = not _all_performances_finished(
        list(prior_active.values()), now
    )
    if prior_count and future_or_unknown and not active_codes and unexplained:
        raise UnexplainedDisappearance(
            f"All active tickets disappeared without sold confirmation for {slug}: "
            f"{len(unexplained)}/{prior_count}; preserving listings",
            active_codes,
        )
    if (
        prior_count >= MIN_LISTINGS_FOR_FRACTION_GUARD
        and future_or_unknown
        and len(unexplained) / prior_count
        > MAX_UNEXPLAINED_DISAPPEARANCE_FRACTION
    ):
        raise UnexplainedDisappearance(
            f"Unexplained listing disappearance for {slug}: "
            f"{len(unexplained)}/{prior_count}; preserving listings",
            active_codes,
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


def canonicalize_master(master):
    """Collapse historical shareCode rotations before applying a new snapshot."""
    groups = {}
    for row in master.values():
        identity = _listing_identity_key(row)
        if identity is None:
            identity = f"ticket:{row.get('ticket_id', '')}"
        groups.setdefault(identity, []).append(row)

    result = {}
    priority = {'deleted': 0, 'listing': 1, 'sold': 2}
    detail_fields = (
        'seller_name', 'seller_rating', 'order_num', 'ticket_tags',
    )
    for rows in groups.values():
        def rank(row):
            try:
                observed = datetime.fromisoformat(str(row.get('last_observed_at', '')))
            except ValueError:
                observed = datetime.min
            return (
                observed,
                priority.get(str(row.get('status', '')).lower(), -1),
                str(row.get('ticket_id', '')),
            )

        winner = max(rows, key=rank).copy()
        for source in rows:
            for field in detail_fields:
                if not winner.get(field) and source.get(field):
                    winner[field] = source[field]
            if str(source.get('details_fetched', 'False')) == 'True':
                winner['details_fetched'] = 'True'
        result[str(winner.get('ticket_id', ''))] = winner
    return result, len(master) - len(result)

def sanitize_unicode(value):
    """Preserve valid Unicode and replace only malformed UTF-16 surrogates.

    Some upstream descriptions contain a lone half of an emoji (for example
    ``\\ud83c``).  Python can keep that value in memory, but UTF-8 cannot encode
    it.  Valid surrogate pairs are combined into their actual Unicode code
    point; a lone surrogate is represented explicitly with U+FFFD instead of
    aborting the entire scrape or silently dropping text.
    """
    if not isinstance(value, str):
        return value

    output = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(value):
                low = ord(value[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    combined = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                    output.append(chr(combined))
                    index += 2
                    continue
            output.append('\ufffd')
        elif 0xDC00 <= codepoint <= 0xDFFF:
            output.append('\ufffd')
        else:
            output.append(value[index])
        index += 1
    return ''.join(output)


def _sanitized_row(row, fieldnames=None):
    keys = fieldnames if fieldnames is not None else row.keys()
    return {key: sanitize_unicode(row.get(key, '')) for key in keys}

def save_master(performer, master):
    os.makedirs(DATA_DIR, exist_ok=True)
    master_file = os.path.join(DATA_DIR, f'{performer}_master.csv')
    fieldnames = ['ticket_id', 'created_at_unix', 'event_id', 'perf_date', 'perf_time', 'venue', 
                  'ticket_type', 'name_type', 'delivery_method', 'seller_name', 
                  'seller_rating', 'order_num', 'ticket_tags', 'first_observed_at', 'first_observed_source', 'last_observed_at',
                  'sold_at', 'sold_at_source', 'status', 'quantity', 'price',
                  'raw_description', 'details_fetched']
                  
    temporary_file = master_file + '.tmp'
    try:
        with open(temporary_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in master.values():
                writer.writerow(_sanitized_row(row, fieldnames))
        # Never leave a partially written master if encoding or I/O fails.
        os.replace(temporary_file, master_file)
    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)

def save_snapshots(performer, master):
    import pandas as pd
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(MARKET_DIR, exist_ok=True)
    
    # Snapshot/market CSVs must use the same malformed-Unicode handling as the
    # master file; otherwise the later snapshot stage can fail on the same row.
    df = pd.DataFrame([_sanitized_row(row) for row in master.values()])
    if df.empty: return
        
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['first_observed_at'] = pd.to_datetime(df['first_observed_at'], errors='coerce')
    # Filter out nat
    df = df.dropna(subset=['first_observed_at']).copy()
    df['year_month'] = df['first_observed_at'].dt.strftime('%Y-%m')
    
    for ym, group in df.groupby('year_month'):
        group.to_csv(os.path.join(SNAPSHOT_DIR, f'{performer}_{ym}.csv'), index=False, encoding='utf-8-sig')
        
    market_records = []
    freshness_cutoff = pd.Timestamp.now() - pd.Timedelta(hours=2)
    for ym, group in df.groupby('year_month'):
        for (ev_id, p_date, p_time), sub in group.groupby(['event_id', 'perf_date', 'perf_time']):
            valid_prices = sub['price'].dropna()
            last_seen = pd.to_datetime(sub['last_observed_at'], errors='coerce')
            current = sub[(sub['status'] == 'listing') & last_seen.ge(freshness_cutoff)]
            # Zero is used for "price on request", not a market price.
            current_prices = current.loc[current['price'] > 0, 'price'].dropna()
            market_records.append({
                'year_month': ym,
                'event_id': ev_id,
                'perf_date': p_date,
                'perf_time': p_time,
                'venue': sub['venue'].iloc[0] if not sub.empty else '',
                'total_tickets': len(sub),
                'active_tickets': len(sub[sub['status'] == 'listing']),
                'current_active_tickets': len(current),
                'current_avg_price': current_prices.mean() if not current_prices.empty else 0,
                'current_min_price': current_prices.min() if not current_prices.empty else 0,
                'current_max_price': current_prices.max() if not current_prices.empty else 0,
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

def enrich_ticket_details(performer, master, ticket_ids):
    """Incrementally enrich active tickets, returning True on time limit."""
    pending = list(dict.fromkeys(ticket_ids))
    if not pending:
        return False
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        fetched_count = 0
        for index, share_code in enumerate(pending):
            if not is_time_remaining():
                print(
                    f"[TIME LIMIT] 詳細取得を中断します。"
                    f"残り{len(pending) - index}件は次回実行で継続します。"
                )
                save_master(performer, master)
                browser.close()
                return True
            print(f"Fetching details for NEW active ticket {share_code}...")
            details = parse_ticket_details(page, share_code)
            time.sleep(1)
            if not details:
                continue
            row = master.get(share_code)
            if row is None or row.get('status') != 'listing':
                continue
            if details.get('raw_description'):
                row['raw_description'] = details['raw_description']
            row['seller_name'] = details.get('seller_name', '')
            row['seller_rating'] = details.get('seller_rating', '')
            row['order_num'] = details.get('order_num', '')
            row['ticket_tags'] = details.get('ticket_tags', '')
            row['details_fetched'] = 'True'
            fetched_count += 1
            if fetched_count % 50 == 0 or (index + 1) == len(pending):
                print(
                    f"[CHECKPOINT] Saving after {fetched_count} detail fetches..."
                )
                save_master(performer, master)
        browser.close()
    return False


def normalize_targets(raw_targets):
    """Return stable output names paired with current Ticketen performer IDs.

    String entries remain backward compatible. Object entries allow Ticketen's
    URL identifier to change without renaming historical master CSVs.
    """
    targets = []
    seen_names = set()
    for entry in raw_targets:
        if isinstance(entry, str):
            name = source = entry.strip()
        elif isinstance(entry, dict):
            name = str(entry.get('name', '')).strip()
            source = str(entry.get('source', '')).strip()
        else:
            raise ValueError(f"Invalid target entry: {entry!r}")
        if not name or not source:
            raise ValueError(f"Target name/source must not be empty: {entry!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate target output name: {name}")
        seen_names.add(name)
        targets.append({'name': name, 'source': source})
    if not targets:
        raise ValueError("At least one scrape target is required")
    return targets


def main():
    targets_file = os.path.join(DATA_DIR, 'targets.json')
    if not os.path.exists(targets_file):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(targets_file, 'w') as f:
            json.dump(["snow-man"], f)
            
    with open(targets_file, 'r', encoding='utf-8') as f:
        targets = normalize_targets(json.load(f))

    now_aware = datetime.now().astimezone()
    now_str = now_aware.strftime('%Y-%m-%d %H:%M:%S')
    now_utc = now_aware.astimezone(timezone.utc).isoformat()
    now_timezone = str(now_aware.tzinfo)

    time_limit_reached = False
    print(f"Scrape mode: {SCRAPE_MODE}")
    for target in targets:
        performer = target['name']
        source_performer = target['source']
        if SCRAPE_MODE != 'api' and not is_time_remaining():
            print(f"[TIME LIMIT] 25分経過のため残りのperformerをスキップします。次回実行で継続します。")
            time_limit_reached = True
            break
        print(f"=== Processing {performer} ===")
        master = load_master(performer)
        master, collapsed = canonicalize_master(master)
        if collapsed:
            print(
                f"[IDENTITY] Collapsed {collapsed} historical shareCode "
                f"duplicates for {performer}."
            )

        if SCRAPE_MODE == 'details':
            pending = [
                ticket_id for ticket_id, row in master.items()
                if row.get('status') == 'listing'
                and str(row.get('details_fetched', 'False')) != 'True'
            ]
            time_limit_reached = enrich_ticket_details(
                performer, master, pending
            )
            save_master(performer, master)
            save_snapshots(performer, master)
            if time_limit_reached:
                print(
                    f"[TIME LIMIT] {performer}までの詳細を保存しました。"
                )
                break
            continue
        
        by_share_code = {t['ticket_id']: t for t in master.values() if not t['ticket_id'].startswith('sold_')}
        by_created_at = {
            key: t
            for t in master.values()
            if (key := _ticket_match_key(t)) is not None
        }
        by_identity = {
            key: t
            for t in master.values()
            if (key := _listing_identity_key(t)) is not None
        }
        
        new_active_tickets = []

        print(f"Fetching events for {performer} ({source_performer})...")
        try:
            discovered_events = get_events(source_performer)
        except NoEventsFound as error:
            print(f"[NO EVENTS] {error}")
            discovered_events = []
        except ScrapeIntegrityError as error:
            print(f"[INTEGRITY] {error}")
            continue
        events = _events_to_poll(discovered_events, master, datetime.now())
        if not events:
            save_master(performer, master)
            save_snapshots(performer, master)
            print(
                f"Saved {len(master)} existing tickets for {performer}; "
                "no status transitions applied."
            )
            continue
        historical_only = set(events) - set(discovered_events)
        if historical_only:
            print(
                f"[COVERAGE] {performer}: checking {len(historical_only)} "
                "previously observed future events absent from performer page."
            )
        active_codes_by_event = {}
        absence_safe_events = set()
        event_diagnostics = {}
        anonymous_sold_by_event = {}
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
                uncertain_identity = False
                try:
                    event_active_codes = validate_event_snapshot(
                        slug, tickets, prior_active, master, datetime.now()
                    )
                except UnexplainedDisappearance as warning:
                    # The API still provides current active and aggregate sold
                    # rows. Save those, but do not infer what happened to the
                    # old shareCodes or certify the interval for model labels.
                    print(f"[IDENTITY UNCERTAIN] {warning}")
                    event_active_codes = warning.active_codes
                    uncertain_identity = True
            except ScrapeIntegrityError as error:
                print(f"[INTEGRITY] {error}")
                continue

            active_codes_by_event[slug] = event_active_codes
            unresolved_sold = 0
            anonymous_sold_by_event[slug] = []
            for t in tickets:
                status = t.get('status')
                created_at_unix = _identifier_text(t.get('createdAt'))
                price_val = _identifier_text(t.get('pricePerTicket'))
                match_key = _ticket_match_key({
                    'created_at_unix': created_at_unix,
                    'price': price_val,
                }, slug)
                
                if status == 'active':
                    share_code = t.get('shareCode')
                    if not share_code: continue
                    identity_key = _listing_identity_key({
                        'created_at_unix': created_at_unix,
                    }, slug)
                    row, rekeyed = _rekey_active_listing(
                        master, by_share_code, by_identity,
                        share_code, identity_key,
                    )
                    if row is not None:
                        row['status'] = 'listing'
                        row['sold_at'] = ''
                        row['sold_at_source'] = ''
                        # The API may omit createdAt on a later poll. Retain
                        # the identity previously observed for this shareCode.
                        if created_at_unix:
                            row['created_at_unix'] = created_at_unix
                        row['last_observed_at'] = now_str
                        row['price'] = t.get('pricePerTicket', row.get('price', 0))
                        row['quantity'] = t.get('quantity', row.get('quantity', 0))
                        row['delivery_method'] = t.get(
                            'deliveryMethod', row.get('delivery_method', '')
                        )
                        row['ticket_type'] = t.get(
                            'ticketType', row.get('ticket_type', '')
                        )
                        row['name_type'] = t.get(
                            'nameGender', row.get('name_type', '')
                        )
                        if t.get('description'):
                            row['raw_description'] = t['description']
                        if str(row.get('details_fetched', 'False')) != 'True':
                            new_active_tickets.append(share_code)
                        if match_key is not None:
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
                            'first_observed_source': 'scrape_observed',
                            'last_observed_at': now_str,
                            'sold_at_source': '',
                            'details_fetched': 'False',
                        }
                        # created_at_unix records seller creation; first_observed_at
                        # must remain the actual first collection time.
                            
                        by_share_code[share_code] = row
                        if match_key is not None:
                            by_created_at[match_key] = row
                        if identity_key is not None:
                            by_identity[identity_key] = row
                        master[share_code] = row
                        new_active_tickets.append(share_code)
                        
                elif status == 'sold':
                    row = _sold_match(t, slug, by_share_code, by_created_at, by_identity)
                    if row is not None:
                        if row.get('status') != 'sold':
                            row['status'] = 'sold'
                            if not row.get('sold_at'):
                                row['sold_at'] = now_str
                            row['sold_at_source'] = 'transition_observed'
                        row['last_observed_at'] = now_str
                    else:
                        share_code = _identifier_text(t.get('shareCode'))
                        if not share_code and not created_at_unix:
                            unresolved_sold += 1
                            anonymous_sold_by_event[slug].append(t)
                            continue
                        t_id = share_code or _sold_ticket_id(slug, created_at_unix, price_val)
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
                            'first_observed_source': 'scrape_observed',
                            'last_observed_at': now_str,
                            'sold_at': '',
                            'sold_at_source': 'historical_unknown',
                            'details_fetched': 'False',
                        }
                        # Historical sales do not imply historical observations.
                        if match_key is not None:
                            by_created_at[match_key] = row
                        identity_key = _listing_identity_key(row)
                        if identity_key is not None:
                            by_identity[identity_key] = row
                        if share_code:
                            by_share_code[share_code] = row
                        master[t_id] = row

            event_diagnostics[slug] = {
                'api_active_count': sum(t.get('status') == 'active' for t in tickets),
                'api_sold_count': sum(t.get('status') == 'sold' for t in tickets),
                'missing_created_at_count': sum(
                    not _identifier_text(t.get('createdAt')) for t in tickets
                ),
                'unresolved_sold_count': unresolved_sold,
                'identity_uncertain': uncertain_identity or (
                    any(not _identifier_text(t.get('createdAt')) for t in tickets)
                    and any(ticket_id not in event_active_codes for ticket_id in prior_active)
                ),
                'api_ticket_keys': sorted({key for t in tickets for key in t}),
            }
            if unresolved_sold:
                print(
                    f"[INTEGRITY] {slug}: {unresolved_sold} sold API rows lack "
                    "both shareCode and createdAt; preserving unmatched listings."
                )
            elif not event_diagnostics[slug]['identity_uncertain']:
                absence_safe_events.add(slug)

        deleted_count = mark_confirmed_absences_deleted(
            master,
            {slug: active_codes_by_event[slug] for slug in absence_safe_events},
            now_str,
        )
        print(
            f"Fetched {len(active_codes_by_event)}/{len(events)} events; "
            f"label-safe {len(absence_safe_events)}/{len(events)}; "
            f"marked {deleted_count} confirmed absences deleted."
        )
        if event_diagnostics:
            print(
                f"[API AUDIT] {performer}: "
                f"active={sum(d['api_active_count'] for d in event_diagnostics.values())}, "
                f"sold={sum(d['api_sold_count'] for d in event_diagnostics.values())}, "
                f"missing_createdAt={sum(d['missing_created_at_count'] for d in event_diagnostics.values())}, "
                f"unresolved_sold={sum(d['unresolved_sold_count'] for d in event_diagnostics.values())}"
            )

        save_master(performer, master)
        save_snapshots(performer, master)
        save_anonymous_sold_inventory(anonymous_sold_by_event, now_str)
        print(f"Saved {len(master)} tickets to master for {performer}.")
        # Persist only after the corresponding master has been saved.
        coverage_path = os.path.join(DATA_DIR, 'observation_' + datetime.now().strftime('%Y%m%d') + '.jsonl')
        with open(coverage_path, 'a', encoding='utf-8') as coverage:
            for event in events:
                rows = [r for r in master.values() if r.get('event_id') == event]
                coverage.write(json.dumps({
                    'event_id': event, 'performer': performer,
                    'observed_at': now_str,
                    'observed_at_utc': now_utc,
                    'observed_at_timezone': now_timezone,
                    'api_fetch_complete': event in active_codes_by_event,
                    'complete': event in absence_safe_events,
                    'listing_count': sum(r.get('status') == 'listing' for r in rows),
                    'sold_count': sum(r.get('status') == 'sold' for r in rows),
                    **event_diagnostics.get(event, {}),
                    'absence_classification_complete': event in absence_safe_events,
                    'schema_version': 'event_poll_v2',
                }, ensure_ascii=False) + '\n')

        if SCRAPE_MODE != 'api':
            time_limit_reached = enrich_ticket_details(
                performer, master, new_active_tickets
            )
            save_master(performer, master)
            save_snapshots(performer, master)
            if time_limit_reached:
                print(
                    f"[TIME LIMIT] {performer}までの状態・詳細を保存しました。"
                )
                break

    if SCRAPE_MODE == 'api':
        missing = [
            target['name'] for target in targets
            if not os.path.exists(
                os.path.join(DATA_DIR, f"{target['name']}_master.csv")
            )
        ]
        if missing:
            raise RuntimeError(
                f"API pass did not create all target masters: {missing}"
            )
        print(
            f"API pass saved all {len(targets)} target master files. "
            "Check per-event completeness in observation logs."
        )

if __name__ == '__main__':
    main()
