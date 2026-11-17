import os
import re
import json
import csv
import hashlib
import html as html_entities
import urllib.request
import urllib.error
import urllib.parse
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from collection_evidence import EVIDENCE_COLUMNS, reconcile_public_listings
from collection_evidence import (
    fetch_public_ticket, apply_public_evidence, PublicRequestPolicy,
    public_error_label, is_deferred_public_error, flatten_canonical_links,
    prepare_public_listings, query_public_listings, apply_public_listing_check,
)
from collection_content import apply_event_content, content_refresh_reason, has_public_full_text
from concurrent.futures import ThreadPoolExecutor
from ticket_change_log import append_changes
from public_ticket_cache import load_run_cache, save_run_cache, remember_verified

# === タイムリミット設定 ===
# 40分で取得を止めて途中保存。Actionsの品質判定が未取得分を検知する。
SCRAPE_START_TIME = time.time()
SCRAPE_MODE = os.environ.get('SCRAPE_MODE', 'full').strip().lower()
if SCRAPE_MODE not in {'full', 'api', 'details'}:
    raise RuntimeError(
        "SCRAPE_MODE must be one of: full, api, details"
    )

# Ordinary runs checkpoint within 40 minutes. Details-only recovery can
# continue for longer under the same request pacing.
DETAIL_RUNTIME_MINUTES = int(os.environ.get('PUBLIC_DETAIL_RUNTIME_MINUTES', '40')) if SCRAPE_MODE == 'details' else 40
if not 1 <= DETAIL_RUNTIME_MINUTES <= 350:
    raise ValueError('PUBLIC_DETAIL_RUNTIME_MINUTES must be between 1 and 350')
MAX_RUNTIME_SECONDS = DETAIL_RUNTIME_MINUTES * 60
PUBLIC_DETAIL_DEADLINE = time.monotonic() + MAX_RUNTIME_SECONDS


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


class UnverifiedEmptyInventory(ScrapeIntegrityError):
    """An empty response cannot establish outcomes of previously seen listings."""


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
    page = f'https://ticketen.jp/performers/{performer}'
    soup = BeautifulSoup(fetch_html(page), 'html.parser')
    events = {}
    for link in soup.find_all('a', href=True):
        try:
            url = urllib.parse.urlsplit(urllib.parse.urljoin(page, link['href']))
        except ValueError:
            continue
        match = re.fullmatch(r'/events/([A-Za-z0-9_-]{1,128})/?', url.path)
        if url.scheme == 'https' and url.netloc == 'ticketen.jp' and match:
            events[match[1]] = None
    if not events:
        raise NoEventsFound(
            f"No events were found for {performer}; preserving existing listings"
        )
    return list(events)

def get_event_id_from_slug(slug):
    return extract_event_id(fetch_html(f'https://ticketen.jp/events/{slug}'), slug)


def extract_event_id(document, slug):
    """Resolve only the requested event, not a performer or recommendation."""
    document = html_entities.unescape(document)
    decoder = json.JSONDecoder()
    candidates = set()
    # Next.js serializes JSON inside script strings. Decode that string layer
    # rather than destroying the escaping around quotes inside event names.
    serialized = re.compile(r'"(?:[^"\\]|\\.)*"')
    embedded = []
    for script in re.findall(r'<script\b[^>]*>(.*?)</script>', document, re.S | re.I):
        if r'\"event\"' not in script:
            continue
        for token in serialized.finditer(script):
            if r'\"event\"' in token.group():
                try:
                    embedded.append(json.loads(token.group()))
                except ValueError:
                    continue
    if document.startswith('"') and document.endswith('"'):
        try:
            embedded.append(json.loads(document))
        except ValueError:
            pass
    for value in (document, *embedded, document.replace('\\"', '"')):
        if not isinstance(value, str):
            continue
        for match in re.finditer(r'"event"\s*:\s*(?=\{)', value):
            try:
                event, _ = decoder.raw_decode(value, match.end())
            except ValueError:
                continue
            event_id = event.get('id')
            if (isinstance(event_id, str)
                    and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', event_id)
                    and (event.get('slug') == slug or event_id == slug)):
                candidates.add(event_id)
    if len(candidates) > 1:
        raise ScrapeIntegrityError(f'Ambiguous event ID for {slug}')
    return next(iter(candidates), None)

def fetch_all_tickets(event_id, *, active_only=False):
    tickets = []
    offset = 0
    limit = 1000
    while True:
        query = urllib.parse.urlencode(dict(context='event', eventId=event_id,
                                            activeOnly=int(active_only), limit=limit, offset=offset))
        url = f"https://ticketen.jp/api/tickets/all?{query}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                res = response.read().decode('utf-8')
            data = json.loads(res)
            if not isinstance(data, dict) or data.get('success') is not True:
                raise ScrapeIntegrityError(f'Unsuccessful ticket response for {event_id} offset {offset}')
            batch = data.get('tickets')
            if not isinstance(batch, list):
                raise ScrapeIntegrityError(
                    f"Invalid ticket page for {event_id} offset {offset}"
                )
            tickets.extend(batch)
            has_more = data.get('hasMore')
            if type(has_more) is not bool:
                raise ScrapeIntegrityError(f'Missing or invalid hasMore for {event_id} offset {offset}')
            if active_only and not batch and data.get('totalCount') != 0:
                raise ScrapeIntegrityError(f'Empty active response without zero total for {event_id}')
            time.sleep(1)  # サーバー負荷軽減: 最終ページ含め全リクエスト後に1秒待機
            if not has_more:
                break
            next_offset = data.get('nextOffset')
            if type(next_offset) is not int or next_offset <= offset or not batch:
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
        historical = row.copy()
        historical['canonical_ticket_id'] = share_code
        historical['observation_state'] = 'alias'
        master[old_id] = historical
        by_share_code[old_id] = historical
        row['ticket_id'] = share_code
        row['canonical_ticket_id'] = share_code
        row['first_observed_source'] = 'created_at_alias_observed'
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
        row = by_share_code[share_code]
        # Do not convert an archived URL into an independent sale. The current
        # code may still be active; only its own evidence can conclude its state.
        if row.get('observation_state') == 'alias':
            return None
        return row
    created = _identifier_text(ticket.get('createdAt'))
    price = _identifier_text(ticket.get('pricePerTicket'))
    match_key = _ticket_match_key({
        'created_at_unix': created, 'price': price,
    }, event)
    if match_key is not None and match_key in by_created_at:
        row = by_created_at[match_key]
        if row.get('observation_state') != 'alias':
            return row
    identity_key = _listing_identity_key({'created_at_unix': created}, event)
    if identity_key is not None:
        row = by_identity.get(identity_key)
        if row is not None and row.get('observation_state') != 'alias':
            return row
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


def validate_event_snapshot(slug, tickets, prior_active, master, now, *, public_event_id=None):
    """Validate a complete event response before changing stored states."""
    if not isinstance(tickets, list):
        raise ScrapeIntegrityError(f"Ticket response for {slug} is not a list")
    if any(not isinstance(ticket, dict) for ticket in tickets):
        raise ScrapeIntegrityError(f'Invalid ticket row for {slug}')
    if prior_active and not tickets:
        raise UnverifiedEmptyInventory(
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
        if public_event_id and ticket.get('eventId') not in (None, '', public_event_id):
            raise ScrapeIntegrityError(f'Ticket response contains an unrelated public event for {slug}')
        status = ticket.get('status')
        if status == 'active':
            share_code = ticket.get('shareCode')
            if not share_code:
                raise ScrapeIntegrityError(
                    f"Active ticket without shareCode for {slug}"
                )
            existing = master.get(share_code)
            if existing:
                stored_public_id = existing.get('public_event_id')
                same_event = (stored_public_id == public_event_id if stored_public_id and public_event_id
                              else existing.get('event_id') == slug)
                if not same_event:
                    raise ScrapeIntegrityError(f"shareCode {share_code} belongs to another event")
            if share_code in active_codes:
                raise ScrapeIntegrityError(f'Duplicate active shareCode for {slug}: {share_code}')
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
        and _listing_identity_key(row, slug) not in active_identities
        and _listing_identity_key(row, slug) not in sold_identities
        and _ticket_match_key(row, slug) not in sold_keys
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
            and row.get('observation_state') != 'alias'
            and row.get('observation_state') not in {
                'absent_unknown', 'absent_unverified', 'inactive', 'paused',
                'public_unclassified', 'expired'}
            and not (row.get('observation_state') == 'active'
                     and row.get('status_source') == 'public_detail'
                     and (row.get('state_checked_at', '') >= now_str
                          or row.get('absence_first_observed_at')))
            and ticket_id not in active_codes_by_event[event_id]
        ):
            row['status'] = 'deleted'
            row['status_source'] = 'complete_api_absence'
            row['observation_state'] = 'deleted_confirmed'
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

_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def sanitize_unicode(value):
    """Preserve valid Unicode and replace only malformed UTF-16 surrogates.

    Some upstream descriptions contain a lone half of an emoji (for example
    ``\\ud83c``).  Python can keep that value in memory, but UTF-8 cannot encode
    it.  Valid surrogate pairs are combined into their actual Unicode code
    point; a lone surrogate is represented explicitly with U+FFFD instead of
    aborting the entire scrape or silently dropping text.
    """
    if not isinstance(value, str) or not _SURROGATE_RE.search(value):
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
                  'raw_description', 'details_fetched'] + EVIDENCE_COLUMNS
                  
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


def save_ticket_changes(performer, before, master, observed_at, observed_at_utc):
    """Append changed facts, preserving intraday price/state/identity history."""
    fields = ['event_id', 'status', 'price', 'quantity'] + EVIDENCE_COLUMNS
    fields += ['raw_description', 'seller_name', 'seller_rating', 'ticket_tags']
    compared = [field for field in fields if field not in
                {'state_checked_at', 'description_checked_at', 'content_checked_at',
                 'status_source', 'price_source'}]
    compared_without_canonical = [field for field in compared if field != 'canonical_ticket_id']
    records = []
    for code, row in master.items():
        old = before.get(code)
        if old is not None and all(str(old.get(f, '')) == str(row.get(f, '')) for f in compared):
            continue
        if (old is not None and old.get('observation_state') == 'alias'
                and row.get('observation_state') == 'alias'
                and all(str(old.get(f, '')) == str(row.get(f, '')) for f in compared_without_canonical)):
            continue
        first = row.get('first_observed_at')
        origin = row.get('identity_first_observed_at')
        records.append({'schema_version': 'ticket_change_v1',
                        'performer': performer, 'ticket_id': code,
                        'observed_at': observed_at, 'observed_at_utc': observed_at_utc,
                        'newly_observed_id': old is None,
                        'newly_observed_listing': bool(
                            old is None and row.get('observation_state') == 'active'
                            and row.get('first_observed_source') == 'scrape_observed'
                            and (not origin or origin == first)),
                        'before': _sanitized_row(old or {}, fields),
                        'after': _sanitized_row(row, fields)})
    if records:
        append_changes(DATA_DIR, datetime.now().strftime('%Y%m%d'), records)

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
            states = sub.get('observation_state', pd.Series('', index=sub.index)).fillna('')
            alias_count = int(states.eq('alias').sum())
            sub = sub[~states.eq('alias')]
            valid_prices = sub.loc[sub['price'] > 0, 'price'].dropna()
            last_seen = pd.to_datetime(sub['last_observed_at'], errors='coerce')
            current = sub[(sub['status'] == 'listing') & last_seen.ge(freshness_cutoff)]
            if 'observation_state' in current:
                current = current[current['observation_state'].eq('active')]
            # Zero is used for "price on request", not a market price.
            current_prices = current.loc[current['price'] > 0, 'price'].dropna()
            market_records.append({
                'year_month': ym,
                'event_id': ev_id,
                'perf_date': p_date,
                'perf_time': p_time,
                'venue': sub['venue'].iloc[0] if not sub.empty else '',
                'total_tickets': len(sub),
                'active_tickets': len(current),
                'current_active_tickets': len(current),
                'current_active_quantity': int(pd.to_numeric(
                    current.get('quantity', pd.Series(0, index=current.index)),
                    errors='coerce').fillna(0).sum()),
                'historical_listing_rows': int(sub['status'].eq('listing').sum()),
                'unknown_listing_tickets': len(sub[sub.get('observation_state', pd.Series('', index=sub.index)).isin(['absent_unknown', 'absent_unverified'])]),
                'alias_rows': alias_count,
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

def reusable_public_content(row, code, reply):
    if reply is None:
        return False
    ticket, error, checked_at = reply
    return (not error and ticket is not None and isinstance(ticket.get('description'), str)
            and ticket.get('shareCode') == code and ticket.get('eventId') == row.get('public_event_id')
            and checked_at >= max(row.get('state_checked_at', ''), row.get('last_observed_at', '')))


def write_content_observation(performer, selected, counts, errors=None):
    now = datetime.now().astimezone()
    log = os.path.join(DATA_DIR, 'content_observation_' + now.strftime('%Y%m%d') + '.jsonl')
    with open(log, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'performer': performer, 'observed_at_utc': now.isoformat(),
                                 'scope': 'observed_active', 'selected': selected, **counts,
                                 'request_errors': dict(errors or {})}) + '\n')
    print(f'[PUBLIC CONTENT] {performer}: {counts}')
    if errors:
        print(f'[PUBLIC CONTENT HTTP SUMMARY] {performer}: {dict(errors)}', flush=True)


def enrich_ticket_details(performer, master, ticket_ids, cache=None, policy=None, *, responses=None):
    """Refresh due public content with the shared request budget and atomic checkpoints.
    Failed/null/rotated replies never erase earlier full descriptions.
    """
    cache = {} if cache is None else cache
    before = {code: row.copy() for code, row in master.items()}
    pending = sorted(set(ticket_ids), key=lambda code: (
        has_public_full_text(master[code]), master[code].get('description_checked_at', ''), code,
    ))
    events = {}
    counts = {'checked':0, 'full':0, 'failed':0, 'null':0, 'pending':0}
    request_errors = Counter()
    reusable = {code for code in pending if reusable_public_content(master[code], code, cache.get(code))}
    if responses is None:
        print(f'[PUBLIC CONTENT CACHE] {performer}: {len(reusable)}/{len(pending)} replies reusable', flush=True)
    for code in pending:
        if responses is not None and code not in responses:
            continue
        row = master[code]
        event = row.get('event_id')
        if event not in events:
            try:
                events[event] = row.get('public_event_id') or (
                    get_event_id_from_slug(event) if responses is not None or is_time_remaining() else None)
            except ScrapeIntegrityError:
                events[event] = None

    def query(code):
        if responses is not None:
            return code, *responses.get(code, (None, 'budget', ''))
        if not is_time_remaining():
            return code, None, 'time_limit', ''
        if code not in reusable:
            try:
                ticket = (fetch_public_ticket(code) if policy is None else
                          fetch_public_ticket(code, policy=policy, deadline=PUBLIC_DETAIL_DEADLINE))
                cache[code] = (ticket, None, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            except Exception as error:
                label = public_error_label(error)
                cache[code] = (None, label, '')
                if not is_deferred_public_error(label):
                    print(f'[PUBLIC CONTENT FETCH ERROR] {code}: {label}', flush=True)
            if not is_deferred_public_error(cache[code][1]):
                time.sleep(0.5)
        value, error, checked_at = cache[code]
        return code, value, error, checked_at
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, (code, ticket, error, checked_at) in enumerate(pool.map(query, pending)):
            if error:
                if error in {'time_limit', 'budget'}:
                    counts['pending'] += 1
                elif str(error).startswith('deferred:'):
                    counts['pending'] += 1
                    request_errors[error.removeprefix('deferred:')] += 1
                else:
                    counts['failed'] += 1
                    request_errors[error] += 1
                continue
            if ticket is None:
                counts['null'] += 1
                continue
            try:
                if not events[master[code]['event_id']]:
                    raise ValueError('Missing public event identity')
                result = apply_public_evidence(master, code, ticket,
                                              events[master[code]['event_id']], checked_at)
                if result == 'stale':
                    counts['pending'] += 1
                    continue
                remember_verified(cache, ticket, error, checked_at)
                counts['checked'] += 1
                canonical = master[code].get('canonical_ticket_id') or code
                if master[canonical].get('description_checked_at') == checked_at:
                    counts['full'] += 1
                else:
                    counts['failed'] += 1
            except (ValueError, TypeError, KeyError) as exc:
                print(f'[CONTENT ERROR] {code}: {type(exc).__name__}')
                counts['failed'] += 1
            if (index + 1) % 100 == 0:
                save_master(performer, master)
                print(f'[CONTENT CHECKPOINT] {performer}: {index + 1}/{len(pending)}')
    # Detail lookups can establish new rotations after the event API phase.
    flatten_canonical_links(master)
    now = datetime.now().astimezone()
    save_ticket_changes(performer, before, master, now.strftime('%Y-%m-%d %H:%M:%S'),
                        now.astimezone(timezone.utc).isoformat())
    save_master(performer, master)
    write_content_observation(performer, len(pending), counts, request_errors)
    return counts['pending'] > 0


def refresh_priority_content(targets, now, cache, policy):
    plans = {}
    ranked = []
    for target in targets:
        performer = target['name']
        master = load_master(performer)
        due = []
        for code, row in master.items():
            if row.get('status') != 'listing' or row.get('observation_state') != 'active':
                continue
            reason = content_refresh_reason(row, now)
            if not reason:
                continue
            if code in cache and not reusable_public_content(row, code, cache[code]):
                cache.pop(code)
            order = {'public_description_preview_changed': 0, 'expired_full_description': 2}.get(reason, 1)
            due.append((order, row.get('description_checked_at', ''), code))
        due.sort()
        plans[performer] = [code for _, _, code in due]
        ranks = Counter()
        for order, _, code in due:
            ranked.append((order, ranks[order], code))
            ranks[order] += 1

    ranked.sort(key=lambda item: item[:2])
    codes = list(dict.fromkeys(code for _, _, code in ranked))
    priority = {code for order, _, code in ranked if order < 2}
    changed = {code for order, _, code in ranked if order == 0}
    print(f'[PUBLIC CONTENT PRIORITY] {len(priority)} new or changed ({len(changed)} visible changes); '
          f'{len(codes) - len(priority)} due for periodic verification', flush=True)
    limit = max(0, int(os.environ.get('PUBLIC_CONTENT_CHECK_LIMIT', str(len(codes)))))
    responses = query_public_listings(codes, cache, [limit], PUBLIC_DETAIL_DEADLINE, policy)
    resolved = {code for code, (_, error, _) in responses.items()
                if not is_deferred_public_error(error)}
    priority_pending = sum(code not in resolved for code in priority)
    changed_pending = sum(code not in resolved for code in changed)
    print(f'[PUBLIC CONTENT PRIORITY] {len(priority)} priority lookups, {priority_pending} pending '
          f'({changed_pending} visible changes pending)', flush=True)

    for performer, pending in plans.items():
        if any(code in responses and not is_deferred_public_error(responses[code][1]) for code in pending):
            master = load_master(performer)
            enrich_ticket_details(performer, master, pending, cache, policy, responses=responses)
            save_snapshots(performer, master)
        else:
            deferred_errors = Counter(
                responses[code][1].removeprefix('deferred:')
                for code in pending
                if code in responses and str(responses[code][1] or '').startswith('deferred:')
            )
            write_content_observation(performer, len(pending), {
                'checked': 0, 'full': 0, 'failed': 0, 'null': 0, 'pending': len(pending),
            }, deferred_errors)


def enrich_ticket_details_browser(performer, master, ticket_ids):
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
            if (row is None or row.get('status') != 'listing'
                    or row.get('observation_state', '') not in {'', 'active'}):
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


def detail_refresh_order(targets):
    """Give skipped groups the first chance on the following run."""
    last_attempt = {}
    paths = sorted(Path(DATA_DIR).glob('content_observation_*.jsonl'))[-3:]
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if (item.get('scope') == 'observed_active'
                        and any(item.get(key) for key in ('checked', 'failed', 'null'))):
                    last_attempt[item['performer']] = item['observed_at_utc']
    # Rotate equal-priority groups, including groups with no successful pass.
    offset = datetime.now(timezone.utc).hour % len(targets)
    rotated = targets[offset:] + targets[:offset]
    return sorted(rotated, key=lambda target: last_attempt.get(target['name'], ''))


def public_state_attempts():
    attempts = {}
    for path in sorted(Path(DATA_DIR).glob('observation_*.jsonl'))[-3:]:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                checks = item.get('public_status_checks', {})
                when = item.get('observed_at_utc')
                if when and (checks.get('checked') or checks.get('failed')):
                    events = attempts.setdefault(item['performer'], {})
                    slug = item['event_id']
                    events[slug] = max(when, events.get(slug, ''))
    return attempts


def set_observed_counts(record, master):
    rows = [row for row in master.values() if row.get('event_id') == record['event_id']]
    record.update(
        listing_count=sum(row.get('status') == 'listing' and row.get('observation_state') != 'alias'
                          for row in rows),
        confirmed_active_count=sum(record['api_fetch_complete'] and row.get('observation_state') == 'active'
                                   and row.get('last_observed_at') == record['observed_at'] for row in rows),
        sold_count=sum(row.get('status') == 'sold' for row in rows),
    )


def write_observations(records):
    path = os.path.join(DATA_DIR, 'observation_' + datetime.now().strftime('%Y%m%d') + '.jsonl')
    with open(path, 'a', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def public_state_order(batches, unverified_codes=()):
    ranked = []
    for events in batches.values():
        for _, _, candidates, _ in events:
            ranks = Counter()
            for code, newly_missing in candidates:
                band = 0 if newly_missing else (1 if code in unverified_codes else 2)
                ranked.append((band, ranks[band], code))
                ranks[band] += 1
    ranked.sort(key=lambda item: item[:2])
    return list(dict.fromkeys(code for _, _, code in ranked))


def finish_public_state_poll(batches, observations, cache, budget, deadline, policy, unverified_codes=()):
    responses = query_public_listings(
        public_state_order(batches, unverified_codes), cache, budget, deadline, policy
    )
    for performer, events in batches.items():
        checked = [
            (event_id, counts, code)
            for _, event_id, candidates, counts in events
            for code, _ in candidates if code in responses
        ]
        if checked:
            has_evidence = any(responses[code][1] is None for _, _, code in checked)
            master = load_master(performer) if has_evidence else {}
            before = {code: row.copy() for code, row in master.items()} if has_evidence else {}
            for event_id, counts, code in checked:
                apply_public_listing_check(master, code, event_id, responses[code], cache, counts)
            if has_evidence:
                flatten_canonical_links(master)
                now = datetime.now().astimezone()
                save_master(performer, master)
                save_snapshots(performer, master)
                save_ticket_changes(performer, before, master, now.strftime('%Y-%m-%d %H:%M:%S'),
                                    now.astimezone(timezone.utc).isoformat())
                for record in observations[performer]:
                    set_observed_counts(record, master)
        for slug, _, candidates, counts in events:
            new_absences = [code for code, newly_missing in candidates if newly_missing]
            counts['newly_missing'] = len(new_absences)
            counts['newly_missing_pending'] = sum(
                code not in responses or is_deferred_public_error(responses[code][1])
                for code in new_absences
            )
            print(f'[PUBLIC EVIDENCE] {slug}: {counts}')
    write_observations(record for records in observations.values() for record in records)


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
    content_now = now_aware.replace(tzinfo=None)
    now_utc = now_aware.astimezone(timezone.utc).isoformat()
    now_timezone = str(now_aware.tzinfo)

    time_limit_reached = False
    coverage_issues = []
    expected_events = fetched_events = 0
    # A new API observation starts a new cache. Only the following details
    # phase can reuse the verified responses associated with that observation.
    public_cache = load_run_cache() if SCRAPE_MODE == 'details' else {}
    has_prior_success = any(
        not error and ticket is not None for ticket, error, _ in public_cache.values()
    )
    min_interval = max(0.0, float(os.environ.get('PUBLIC_API_MIN_INTERVAL', '4.5')))
    public_policy = PublicRequestPolicy(
        os.environ.get('PUBLIC_API_COOLDOWN_PATH'),
        min_interval=min_interval,
        has_prior_success=has_prior_success,
    )
    public_budget = [max(0, int(os.environ.get('PUBLIC_STATUS_CHECK_LIMIT', '6000')))]
    status_minutes = int(os.environ.get('PUBLIC_STATUS_RUNTIME_MINUTES', '35'))
    if not 0 <= status_minutes <= 35:
        raise ValueError('PUBLIC_STATUS_RUNTIME_MINUTES must be between 0 and 35')
    recheck_hours = max(0.0, float(os.environ.get('PUBLIC_UNAVAILABLE_RECHECK_HOURS', '0')))
    public_deadline = time.monotonic() + status_minutes * 60
    state_attempts = {}
    public_state_batches = {}
    unverified_state_codes = set()
    observations = {}
    print(f"Scrape mode: {SCRAPE_MODE}")
    if SCRAPE_MODE == 'details':
        targets = detail_refresh_order(targets)
        print(f'[PUBLIC CONTENT BUDGET] {DETAIL_RUNTIME_MINUTES} minutes', flush=True)
        print('[PUBLIC CONTENT ORDER] ' + ', '.join(target['name'] for target in targets), flush=True)
        if DETAIL_RUNTIME_MINUTES <= 40:
            refresh_priority_content(targets, content_now, public_cache, public_policy)
            save_run_cache(public_cache)
            public_policy.save_cooldown()
            return
    else:
        state_attempts = public_state_attempts()
        targets.sort(key=lambda target: max(state_attempts.get(target['name'], {}).values(), default=''))
        print(f'[PUBLIC STATE BUDGET] {status_minutes} minutes', flush=True)
    for target in targets:
        performer = target['name']
        source_performer = target['source']
        if SCRAPE_MODE != 'api' and not is_time_remaining():
            print(f"[TIME LIMIT] {MAX_RUNTIME_SECONDS // 60}分経過のため残りのperformerをスキップします。次回実行で継続します。")
            time_limit_reached = True
            break
        print(f"=== Processing {performer} ===")
        master = load_master(performer)
        if SCRAPE_MODE == 'details':
            pending = [
                ticket_id for ticket_id, row in master.items()
                if row.get('status') == 'listing'
                and row.get('observation_state') == 'active'
                and content_refresh_reason(row, content_now)
            ]
            time_limit_reached = enrich_ticket_details(
                performer, master, pending, public_cache, public_policy
            )
            save_snapshots(performer, master)
            if time_limit_reached and not is_time_remaining():
                print(
                    f"[TIME LIMIT] {performer}までの詳細を保存しました。"
                )
                break
            continue
        # Retain all original IDs; proven aliases are annotated, not deleted.
        before_poll = {code: row.copy() for code, row in master.items()}
        
        by_share_code = {t['ticket_id']: t for t in master.values() if not t['ticket_id'].startswith('sold_')}
        by_created_at = {
            key: t
            for t in master.values()
            if t.get('observation_state') != 'alias'
            and (key := _ticket_match_key(t)) is not None
        }
        by_identity = {
            key: t
            for t in master.values()
            if t.get('observation_state') != 'alias'
            and (key := _listing_identity_key(t)) is not None
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
            coverage_issues.append(dict(performer=performer, event_id=None,
                                        reason='performer_discovery_failed', detail=str(error)))
            continue
        events = _events_to_poll(discovered_events, master, datetime.now())
        last_attempt = state_attempts.get(performer, {})
        events.sort(key=lambda slug: last_attempt.get(slug, ''))
        expected_events += len(events)
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
        event_errors = {}
        anonymous_sold_by_event = {}
        stored_event_slugs = {}
        for row in master.values():
            if row.get('public_event_id') and row.get('event_id'):
                stored_event_slugs.setdefault(row['public_event_id'], Counter())[row['event_id']] += 1
        polled_public_events = set()
        for requested_slug in events:
            slug = requested_slug
            try:
                ev_firestore_id = get_event_id_from_slug(requested_slug)
                if not ev_firestore_id:
                    raise ScrapeIntegrityError(
                        f"Could not find firestore ID for {requested_slug}"
                    )
                if ev_firestore_id in polled_public_events:
                    expected_events -= 1
                    print(f'[EVENT URL ALIAS] {requested_slug}: already polled public event {ev_firestore_id}')
                    continue
                polled_public_events.add(ev_firestore_id)
                if ev_firestore_id in stored_event_slugs:
                    slug = stored_event_slugs[ev_firestore_id].most_common(1)[0][0]
                if slug != requested_slug:
                    print(f'[EVENT URL ALIAS] {requested_slug}: retaining stored event key {slug}')
                prior_active = {
                    ticket_id: row for ticket_id, row in master.items()
                    if row.get('status') == 'listing' and row.get('observation_state') != 'alias'
                    and (row.get('public_event_id') == ev_firestore_id
                         or (not row.get('public_event_id') and row.get('event_id') == slug))
                }

                print(f"Fetching API tickets for {slug}...")
                tickets = fetch_all_tickets(ev_firestore_id)
                uncertain_identity = False
                empty_inventory_confirmed = False
                try:
                    event_active_codes = validate_event_snapshot(
                        slug, tickets, prior_active, master, datetime.now(),
                        public_event_id=ev_firestore_id,
                    )
                except UnexplainedDisappearance as warning:
                    # The API still provides current active and aggregate sold
                    # rows. Save those, but do not infer what happened to the
                    # old shareCodes or certify the interval for model labels.
                    print(f"[IDENTITY UNCERTAIN] {warning}")
                    event_active_codes = warning.active_codes
                    uncertain_identity = True
                except UnverifiedEmptyInventory:
                    # The all-ticket and active-only lists must agree. This
                    # confirms zero public inventory, never a sale/withdrawal.
                    if fetch_all_tickets(ev_firestore_id, active_only=True):
                        raise
                    print(f'[EMPTY INVENTORY] {slug}: both API views returned zero; prior outcomes remain unverified.', flush=True)
                    event_active_codes = set()
                    uncertain_identity = True
                    empty_inventory_confirmed = True
            except ScrapeIntegrityError as error:
                print(f"[INTEGRITY] {error}")
                failure_time = datetime.now().astimezone()
                reason = ('empty_inventory_unverified' if isinstance(error, UnverifiedEmptyInventory)
                          else 'event_poll_failed')
                event_errors[slug] = {
                    'observed_at': failure_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'observed_at_utc': failure_time.astimezone(timezone.utc).isoformat(),
                    'observed_at_timezone': str(failure_time.tzinfo),
                    'failure_reason': reason,
                }
                coverage_issues.append(dict(performer=performer, event_id=slug,
                                            reason=reason, detail=str(error)))
                continue

            # Each event is observed when its own API response arrives, not at
            # the start of a potentially 15-minute multi-event run.
            event_now = datetime.now().astimezone()
            now_str = event_now.strftime('%Y-%m-%d %H:%M:%S')
            now_utc = event_now.astimezone(timezone.utc).isoformat()
            now_timezone = str(event_now.tzinfo)
            for row in prior_active.values():
                if row.get('public_event_id') == ev_firestore_id:
                    row['event_id'] = slug
            active_codes_by_event[slug] = event_active_codes
            fetched_events += 1
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
                        row['event_id'] = slug
                        row['sold_at'] = ''
                        row['sold_at_source'] = ''
                        # The API may omit createdAt on a later poll. Retain
                        # the identity previously observed for this shareCode.
                        if created_at_unix:
                            row['created_at_unix'] = created_at_unix
                        row['last_observed_at'] = now_str
                        row['canonical_ticket_id'] = share_code
                        row['observation_state'] = 'active'
                        row['state_checked_at'] = now_str
                        row['absence_first_observed_at'] = ''
                        row['status_source'] = 'event_api'
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
                        apply_event_content(row, t)
                        row['public_event_id'] = ev_firestore_id
                        if content_refresh_reason(row, content_now):
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
                            'raw_description': '',
                            'first_observed_at': now_str,
                            'first_observed_source': 'scrape_observed',
                            'last_observed_at': now_str,
                            'sold_at_source': '',
                            'details_fetched': 'False',
                            'canonical_ticket_id': share_code,
                            'identity_first_observed_at': now_str,
                            'observation_state': 'active',
                            'state_checked_at': now_str,
                            'status_source': 'event_api',
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
                        apply_event_content(row, t)
                        row['public_event_id'] = ev_firestore_id
                    if 'isPriceOnRequest' in t:
                        row['is_price_on_request'] = str(t['isPriceOnRequest'] is True)
                    row['price_source'] = 'on_request' if t.get('isPriceOnRequest') is True else ('event_api' if price_val and float(price_val) > 0 else 'unknown')
                        
                elif status == 'sold':
                    row = _sold_match(t, slug, by_share_code, by_created_at, by_identity)
                    if row is not None:
                        if row.get('status') != 'sold':
                            row['status'] = 'sold'
                            if not row.get('sold_at'):
                                row['sold_at'] = now_str
                            row['sold_at_source'] = 'transition_observed'
                        row['last_observed_at'] = now_str
                        row['observation_state'] = 'sold_confirmed'
                        row['state_checked_at'] = now_str
                        row['status_source'] = 'event_api'
                    else:
                        share_code = _identifier_text(t.get('shareCode'))
                        if share_code and share_code in by_share_code and by_share_code[
                            share_code].get('observation_state') == 'alias':
                            # Inconsistent with the canonical code's current
                            # lifecycle. Retain the sale only as unresolved.
                            unresolved_sold += 1
                            anonymous_sold_by_event[slug].append(t)
                            continue
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
                            'observation_state': 'sold_historical',
                            'status_source': 'event_api',
                            'is_price_on_request': str(t.get('isPriceOnRequest') is True),
                            'price_source': 'on_request' if t.get('isPriceOnRequest') is True else 'event_api',
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

            if SCRAPE_MODE == 'api':
                public_counts, candidates = prepare_public_listings(
                    master, prior_active, event_active_codes, now_str,
                    recheck_after=timedelta(hours=recheck_hours),
                )
                unverified_state_codes.update(
                    code for code, _ in candidates
                    if master.get(code, {}).get('observation_state') == 'absent_unverified'
                )
                public_state_batches.setdefault(performer, []).append(
                    (slug, ev_firestore_id, candidates, public_counts))
            else:
                public_counts = reconcile_public_listings(
                    master, prior_active, event_active_codes, ev_firestore_id,
                    now_str, public_cache, public_budget, public_deadline, public_policy,
                )
                print(f"[PUBLIC EVIDENCE] {slug}: {public_counts}")
            event_diagnostics[slug] = {
                'observed_at': now_str,
                'observed_at_utc': now_utc,
                'observed_at_timezone': now_timezone,
                'public_status_checks': public_counts,
                'api_active_count': sum(t.get('status') == 'active' for t in tickets),
                'api_active_ids': sorted(event_active_codes),
                'api_sold_count': sum(t.get('status') == 'sold' for t in tickets),
                'missing_created_at_count': sum(
                    not _identifier_text(t.get('createdAt')) for t in tickets
                ),
                'unresolved_sold_count': unresolved_sold,
                'empty_inventory_confirmed': empty_inventory_confirmed,
                'identity_uncertain': uncertain_identity or (
                    any(not _identifier_text(t.get('createdAt')) for t in tickets)
                    and any(ticket_id not in event_active_codes for ticket_id in prior_active)
                ),
                'api_ticket_keys': sorted({key for t in tickets for key in t}),
            }
            if unresolved_sold:
                print(
                    f"[INTEGRITY] {slug}: {unresolved_sold} sold API rows cannot be "
                    "linked reliably to a listing; preserving unmatched listings."
                )
            elif not event_diagnostics[slug]['identity_uncertain']:
                absence_safe_events.add(slug)

        deleted_count = sum(mark_confirmed_absences_deleted(
            master, {slug: active_codes_by_event[slug]},
            event_diagnostics[slug]['observed_at'],
        ) for slug in absence_safe_events)
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
        written_at = datetime.now().astimezone()
        save_ticket_changes(performer, before_poll, master,
                            written_at.strftime('%Y-%m-%d %H:%M:%S'),
                            written_at.astimezone(timezone.utc).isoformat())
        save_anonymous_sold_inventory(anonymous_sold_by_event, now_str)
        print(f"Saved {len(master)} tickets to master for {performer}.")
        # Retain only actual polls, including failures, when several URLs resolve
        # to the same event. In API mode, publish the final reconciliation counts.
        records = []
        for event, result in {**event_errors, **event_diagnostics}.items():
            record = {
                'event_id': event, 'performer': performer,
                'api_fetch_complete': event in active_codes_by_event,
                'complete': event in absence_safe_events,
                **result,
                'absence_classification_complete': event in absence_safe_events,
                'schema_version': 'event_poll_v2',
            }
            set_observed_counts(record, master)
            records.append(record)
        if SCRAPE_MODE == 'api':
            observations[performer] = records
        else:
            write_observations(records)

        if SCRAPE_MODE != 'api':
            time_limit_reached = enrich_ticket_details(
                performer, master, [code for code, row in master.items()
                                    if row.get('observation_state') == 'active'], public_cache,
                public_policy,
            )
            save_snapshots(performer, master)
            if time_limit_reached:
                print(
                    f"[TIME LIMIT] {performer}までの状態・詳細を保存しました。"
                )
                break

    if SCRAPE_MODE == 'api':
        print('[PUBLIC STATE] Inventory saved; checking newly absent listings first.', flush=True)
        finish_public_state_poll(public_state_batches, observations, public_cache, public_budget,
                                 time.monotonic() + status_minutes * 60, public_policy,
                                 unverified_state_codes)
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
        report = dict(expected_events=expected_events, fetched_events=fetched_events,
                      issues=coverage_issues)
        print('[COLLECTION COVERAGE] ' + json.dumps(report, ensure_ascii=True), flush=True)
        path = os.environ.get('COLLECTION_COVERAGE_PATH')
        if path:
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=True)
    public_policy.save_cooldown()
    save_run_cache(public_cache)

if __name__ == '__main__':
    main()
