"""Public listing evidence; never identify listings by matching descriptions."""
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from collection_content import CONTENT_COLUMNS, public_updates
from public_ticket_cache import remember_verified

PUBLIC_QUERY_URL = 'https://asia-northeast1-ticketen-prod.cloudfunctions.net/ticket_queries_get'
EVIDENCE_COLUMNS = [
    'canonical_ticket_id', 'identity_first_observed_at', 'observation_state',
    'state_checked_at', 'absence_first_observed_at', 'status_source',
    'is_price_on_request', 'price_source', 'public_ticket_status',
] + CONTENT_COLUMNS


RETRYABLE_PUBLIC_HTTP = {408, 429, 500, 502, 503, 504}
PUBLIC_IP_BLOCK = 'ticket_detail_ip_blocked'


class PublicLookupDeferred(RuntimeError):
    """No request can be made safely within the remaining phase budget."""

    def __init__(self, message, *, status_code=None, ip_blocked=False):
        super().__init__(message)
        self.status_code = status_code
        self.ip_blocked = ip_blocked


def is_deferred_public_error(error):
    return error in {'budget', 'time_limit'} or str(error or '').startswith('deferred:')


def retry_after_seconds(value):
    if not value:
        return None
    try:
        delay = float(value)
        return max(0.0, delay) if math.isfinite(delay) else None
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def public_ip_blocked(error):
    if (error.code != 403 or error.fp is None
            or 'application/json' not in (error.headers or {}).get('Content-Type', '').lower()):
        return False
    try:
        payload = json.loads(error.read(4096))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get('error') == PUBLIC_IP_BLOCK


class PublicRequestPolicy:
    """Share server backoff and repeated access denials across both workers."""

    def __init__(self, cooldown_path=None, *, min_interval=0, has_prior_success=False,
                 retry_access_denial=False, server_cooldown_path=None):
        self._lock = threading.Lock()
        self._resume_at = 0.0
        self._next_request_at = 0.0
        self._min_interval = min_interval
        self._denied = 0
        self._blocked = False
        self._succeeded = 1 if has_prior_success else 0
        self._started_requests = 0
        self._reported_responses = Counter()
        self._ip_blocks = 0
        self.retry_access_denial = retry_access_denial
        self._server_cooldown_path = Path(server_cooldown_path) if server_cooldown_path else None
        self._server_delayed = False
        if self._server_cooldown_path and self._server_cooldown_path.exists():
            retry_epoch = float(self._server_cooldown_path.read_text(encoding='ascii'))
            if math.isfinite(retry_epoch) and retry_epoch > time.time():
                self._resume_at = time.monotonic() + retry_epoch - time.time()
                print(f'[PUBLIC API] retaining prior run cooldown until epoch {retry_epoch:.0f}', flush=True)
        self._cooldown_path = Path(cooldown_path) if cooldown_path else None
        if self._cooldown_path and self._cooldown_path.exists():
            lines = self._cooldown_path.read_text(encoding='ascii').splitlines()
            retry_epoch = float(lines[0]) if lines else 0.0
            if math.isfinite(retry_epoch):
                self._resume_at = max(self._resume_at, time.monotonic() + retry_epoch - time.time())
            if len(lines) > 1 and lines[1].strip() == '1':
                self._succeeded = max(self._succeeded, 1)
            if len(lines) > 2 and (carried_interval := float(lines[2])) > self._min_interval:
                self._min_interval = carried_interval
            self._blocked = len(lines) > 3 and lines[3] == '1'

    @property
    def has_succeeded(self):
        with self._lock:
            return self._succeeded > 0

    def save_cooldown(self):
        """Carry request spacing and server cooldown across phases of this run."""
        counts = json.dumps({
            'started': self._started_requests,
            'recorded_statuses': dict(self._reported_responses),
            'ip_blocks': self._ip_blocks,
            'interval_seconds': self._min_interval,
        })
        print('[PUBLIC API COUNTS] ' + counts, flush=True)
        summary = os.environ.get('GITHUB_STEP_SUMMARY')
        if summary:
            phase = os.environ.get('SCRAPE_MODE', 'public')
            with open(summary, 'a', encoding='utf-8') as stream:
                stream.write(f'- Public API {phase} counts: `{counts}`\n')
        clock_now = time.monotonic()
        epoch_now = time.time()
        if self._cooldown_path:
            remaining = max(0.0, max(self._resume_at, self._next_request_at) - clock_now)
            text = (f'{epoch_now + remaining}\n{int(self._succeeded > 0)}\n'
                    f'{self._min_interval}\n{int(self._blocked)}')
            self._cooldown_path.write_text(text, encoding='ascii')
        if self._server_cooldown_path and self._server_delayed and self._resume_at > clock_now:
            temporary = self._server_cooldown_path.with_suffix('.tmp')
            temporary.write_text(str(epoch_now + self._resume_at - clock_now), encoding='ascii')
            temporary.replace(self._server_cooldown_path)

    def before_request(self, deadline):
        while True:
            with self._lock:
                if self._blocked:
                    raise PublicLookupDeferred('public API paused after an access denial')
                now = time.monotonic()
                wait = max(self._resume_at, self._next_request_at) - now
                remaining = deadline - now
                if remaining <= max(0.0, wait):
                    raise PublicLookupDeferred('public API deadline or cooldown')
                if wait <= 0:
                    self._started_requests += 1
                    self._next_request_at = now + self._min_interval
                    return min(15, remaining)
            time.sleep(wait)

    def record_response(self, status, retry_after=None, *, ip_blocked=False):
        with self._lock:
            self._reported_responses[status] += 1
            if ip_blocked:
                self._ip_blocks += 1
            if status == 200:
                self._denied = 0
                self._succeeded += 1
            elif status in {401, 403}:
                self._denied += 1
            else:
                self._denied = 0
            if (status == 403 and not ip_blocked and self.retry_access_denial and self._succeeded > 0
                    and self._denied == 1 and not self._blocked):
                self._server_delayed = True
                self._resume_at = max(self._resume_at, time.monotonic() + max(900.0, retry_after or 0))
                print('[PUBLIC API] HTTP 403 in recovery; waiting before one retry', flush=True)
            elif ((ip_blocked or status == 401 or self._denied >= 4 or (self._denied and self._succeeded))
                  and not self._blocked):
                self._blocked = True
                self._server_delayed = True
                self._resume_at = max(self._resume_at, time.monotonic() + max(900.0, retry_after or 0))
                reason = f' ({PUBLIC_IP_BLOCK})' if ip_blocked else ''
                print(f'[PUBLIC API] pausing this run after HTTP {status}{reason}', flush=True)
            if status == 429 or (status == 503 and retry_after is not None):
                self._server_delayed = True
                self._min_interval = max(self._min_interval, 4.5)
                delay = max(5, retry_after if retry_after is not None else 30)
                until = time.monotonic() + delay
                if until > self._resume_at:
                    self._resume_at = until
                    print(f'[PUBLIC API] HTTP {status}; shared cooldown {delay:g}s', flush=True)


def public_error_label(error):
    reason = f' ({PUBLIC_IP_BLOCK})' if getattr(error, 'ip_blocked', False) else ''
    if isinstance(error, urllib.error.HTTPError):
        return f'HTTP {error.code}{reason}'
    if isinstance(error, PublicLookupDeferred):
        if error.status_code is not None:
            return f'deferred:HTTP {error.status_code}{reason}'
        return 'budget'
    return type(error).__name__


def fetch_public_ticket(code, *, policy=None, deadline=float('inf')):
    """Same unauthenticated public action used by the site's ticket page.

    A null ticket is absence of public evidence, NOT evidence of withdrawal.
    Authentication failures and malformed replies must propagate as errors.
    """
    request = urllib.request.Request(
        PUBLIC_QUERY_URL,
        data=json.dumps({'action': 'get_ticket_by_share_code_public',
                         'payload': {'shareCode': code}}).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'},
    )
    policy = PublicRequestPolicy() if policy is None else policy
    retrying_denial = False
    for attempt in range(3):
        try:
            timeout = policy.before_request(deadline)
        except PublicLookupDeferred as error:
            if retrying_denial:
                raise PublicLookupDeferred('HTTP 403 cooldown exceeds remaining budget',
                                           status_code=403) from error
            raise
        retrying_denial = False
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or payload.get('success') is not True or 'ticket' not in payload:
                raise ValueError('Invalid public ticket response')
            ticket = payload['ticket']
            if ticket is not None and not isinstance(ticket, dict):
                raise ValueError('Invalid public ticket object')
            if ticket is not None and all(
                    isinstance(ticket.get(key), str) and ticket[key].strip()
                    for key in ('shareCode', 'eventId', 'status')):
                policy.record_response(200)
            return ticket
        except urllib.error.HTTPError as error:
            status = error.code
            retry_after = retry_after_seconds(error.headers.get('Retry-After') if error.headers else None)
            error.ip_blocked = public_ip_blocked(error)
            error.close()
            policy.record_response(status, retry_after, ip_blocked=error.ip_blocked)
            # A longer recovery run can wait out one access denial after
            # proven progress. The shared cooldown also covers throttling.
            if status == 403 and policy.has_succeeded:
                if policy.retry_access_denial and attempt == 0 and not error.ip_blocked:
                    retrying_denial = True
                    continue
                raise PublicLookupDeferred(
                    f'HTTP {status} after successful lookups', status_code=status, ip_blocked=error.ip_blocked
                ) from error
            if status not in RETRYABLE_PUBLIC_HTTP or attempt >= 1:
                if (status == 429 or (status == 503 and retry_after is not None)) and policy.has_succeeded:
                    raise PublicLookupDeferred(
                        f'repeated HTTP {status} after successful lookups', status_code=status
                    ) from error
                raise
            if status == 429 or (status == 503 and retry_after is not None):
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            if attempt == 2:
                raise
        delay = 0.75 * (attempt + 1)
        if time.monotonic() + delay >= deadline:
            raise PublicLookupDeferred('public API retry would exceed deadline')
        time.sleep(delay)


def apply_public_evidence(master, old_code, ticket, firestore_event_id, now):
    row = master[old_code]
    if row.get('state_checked_at', '') > now:
        return 'stale'
    if ticket is None:
        row['observation_state'] = 'absent_unknown'
        row['state_checked_at'] = now
        row['public_ticket_status'] = ''
        return 'not_found'
    code = str(ticket.get('shareCode') or '').strip()
    if not code or ticket.get('eventId') != firestore_event_id:
        raise ValueError('Public ticket identity/event mismatch')
    raw_status = ticket.get('status')
    if not isinstance(raw_status, str) or not raw_status.strip():
        raise ValueError('Missing public ticket status')
    status = raw_status
    if status == 'completed':
        status = 'sold'
    if status not in {'active', 'sold', 'cancelled', 'expired', 'inactive', 'paused'}:
        status = 'unclassified'
    if status == 'active' and ticket.get('isSold') is True:
        raise ValueError('Conflicting public ticket state')
    content = public_updates(ticket, now)
    if code != old_code:
        target = master.get(code)
        if target is not None and target.get('observation_state') == 'alias':
            raise ValueError('Public response points to a superseded alias')
        # The event API can lag a public shareCode rotation even while the
        # listing is active. The direct public response proves the new code
        # and event identity; keep the old code as an alias instead of losing
        # the verified content or inventing a terminal state.
        if target is None:
            target = row.copy()
            target.update(ticket_id=code, first_observed_at=now, last_observed_at=now,
                          first_observed_source='public_alias_observed',
                          absence_first_observed_at=now)
            master[code] = target
        if target is None or target.get('event_id') != row.get('event_id'):
            raise ValueError('Canonical public code is not in the event snapshot')
        row['canonical_ticket_id'] = code
        row['observation_state'] = 'alias'
        row['state_checked_at'] = now
        target['canonical_ticket_id'] = code
        dates = [d for d in [target.get('identity_first_observed_at'),
                            target.get('first_observed_at'),
                            row.get('identity_first_observed_at'),
                            row.get('first_observed_at')] if d]
        target['identity_first_observed_at'] = min(dates) if dates else now
        target['first_observed_source'] = 'public_alias_observed'
        # Original rows and their first/last sightings remain intact.
        for field in ['seller_name', 'seller_rating', 'order_num', 'ticket_tags']:
            if not target.get(field) and row.get(field):
                target[field] = row[field]
        row = target
    # A cached response may predate a later API observation in another artist's
    # copy of the same event. Identity linkage remains valid, but stale state
    # and price must never replace more recent evidence.
    if row.get('state_checked_at', '') > now:
        return 'alias' if code != old_code else 'stale'
    row['state_checked_at'] = now
    row['status_source'] = 'public_detail'
    row['public_ticket_status'] = raw_status
    row.update(content)
    if status == 'active':
        # Keep the API sighting timestamp when it already confirmed this code;
        # state_checked_at independently records the later public lookup.
        if ((code != old_code and row.get('observation_state') != 'active')
                or not row.get('last_observed_at')
                or (row.get('observation_state') != 'active' and not row.get('absence_first_observed_at'))):
            row['last_observed_at'] = now
        row['observation_state'] = 'active'
    elif status == 'sold':
        if row.get('status') != 'sold':
            row['sold_at'] = now
            row['sold_at_source'] = 'transition_observed'
        row['status'] = 'sold'
        row['observation_state'] = 'sold_confirmed'
        row['last_observed_at'] = now
    elif status == 'cancelled':
        row['status'] = 'deleted'
        row['observation_state'] = 'deleted_confirmed'
        row['last_observed_at'] = now
    elif status in {'inactive', 'paused', 'unclassified'}:
        # Publicly unavailable is not proof of a sale or permanent withdrawal.
        # Keep the last historical status and sighting; retry on later polls.
        row['observation_state'] = 'public_unclassified' if status == 'unclassified' else status
    else:
        row['observation_state'] = 'expired'
    on_request = ticket.get('isPriceOnRequest') is True
    if 'isPriceOnRequest' in ticket:
        row['is_price_on_request'] = str(on_request)
        if on_request:
            row['price_source'] = 'on_request'
        elif row.get('price_source') == 'on_request':
            row['price_source'] = 'unknown'
    value = ticket.get('pricePerTicket')
    if value is not None:
        try:
            price = float(value)
        except (ValueError, TypeError):
            price = float('nan')
        if 0 <= price < float('inf'):
            if price > 0 and not on_request and row.get('is_price_on_request') == 'True':
                row['is_price_on_request'] = 'False'
            row['price'] = int(price) if price.is_integer() else price
            row['price_source'] = ('on_request' if row.get('is_price_on_request') == 'True'
                                   else 'public_detail' if price > 0 else 'unknown')
    return 'alias' if code != old_code else status


def prepare_public_listings(master, prior, active_codes, now, *, recheck_after=timedelta(0)):
    """Mark missing listings and rank new absences and unverified rows first."""
    counts = {'checked': 0, 'alias': 0, 'sold': 0, 'cancelled': 0,
              'not_found': 0, 'active': 0, 'expired': 0, 'inactive': 0, 'paused': 0,
              'unclassified': 0, 'stale': 0, 'failed': 0,
              'pending': 0, 'historical_unresolved': 0, 'full_from_alias': 0}
    now_dt = datetime.fromisoformat(now)
    cutoff = now_dt - timedelta(hours=36)
    candidates = []
    for code, previous in prior.items():
        row = master.get(code)
        if row is None or code in active_codes or row.get('status') != 'listing':
            continue
        if row.get('observation_state') in {'alias', 'expired'}:
            continue
        if row.get('observation_state') == 'absent_unknown':
            first_absence = row.get('absence_first_observed_at', '')
            last_check = row.get('state_checked_at', '')
            if (first_absence >= cutoff.isoformat(sep=' ')
                    and last_check <= (now_dt - timedelta(hours=6)).isoformat(sep=' ')):
                candidates.append((code, False))
            continue
        newly_missing = (not row.get('absence_first_observed_at')
                         and previous.get('observation_state') in {None, '', 'active'})
        if not row.get('absence_first_observed_at'):
            row['absence_first_observed_at'] = now
        previously_unavailable = previous.get('observation_state') in {'inactive', 'paused', 'public_unclassified'}
        previously_unlisted_active = (
            not newly_missing
            and previous.get('observation_state') == 'active'
            and previous.get('status_source') == 'public_detail'
            and bool(previous.get('absence_first_observed_at'))
        )
        if not previously_unavailable and not previously_unlisted_active:
            row['observation_state'] = 'absent_unverified'
        try:
            last_seen = previous.get('last_observed_at', '')
            if previously_unavailable or previously_unlisted_active:
                # A recently verified stopped or unlisted listing remains worth refreshing;
                # do not silently abandon it 36h after its last active sighting.
                last_seen = max(last_seen, previous.get('state_checked_at', ''))
            recent = datetime.fromisoformat(last_seen) >= cutoff
        except ValueError:
            recent = False
        if not recent:
            counts['historical_unresolved'] += 1
            continue
        if (previously_unavailable or previously_unlisted_active) and recheck_after > timedelta(0):
            try:
                checked_dt = datetime.fromisoformat(previous.get('state_checked_at', ''))
                if now_dt - checked_dt < recheck_after:
                    continue
            except ValueError:
                pass
        candidates.append((code, newly_missing))
    def priority(candidate):
        code, newly_missing = candidate
        previous = prior[code]
        if newly_missing:
            return 0, ''
        if master[code].get('observation_state') == 'absent_unverified':
            # Give the oldest unresolved listings a chance before they age out.
            return 1, previous.get('last_observed_at', '')
        return 2, previous.get('state_checked_at', '')

    candidates.sort(key=lambda item: prior[item[0]].get('last_observed_at', ''), reverse=True)
    candidates.sort(key=priority)
    counts['pending'] = len(candidates)
    return counts, candidates


def query_public_listings(codes, cache, budget, deadline, policy):
    """Share the deadline, rate policy and response cache across listings."""
    def query(code):
        if time.monotonic() >= deadline:
            return code, None, 'budget', ''
        if code in cache:
            return code, *cache[code]
        try:
            value = (fetch_public_ticket(code) if policy is None else
                     fetch_public_ticket(code, policy=policy, deadline=deadline))
            checked_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as error:
            label = public_error_label(error)
            if not is_deferred_public_error(label):
                time.sleep(0.5)
            return code, None, label, ''
        time.sleep(0.5)
        return code, value, None, checked_at
    cached = [c for c in codes if c in cache]
    fresh = [c for c in codes if c not in cache]
    selected = cached + fresh[:budget[0]]
    budget[0] -= min(len(fresh), budget[0])
    # Recovery may retry a denied request after cooldown. Keep that path
    # serial so there is only one in-flight request when it pauses.
    workers = 1 if policy is not None and policy.retry_access_denial else 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {code: (ticket, error, checked_at)
                for code, ticket, error, checked_at in pool.map(query, selected)
                if error != 'budget'}


def apply_public_listing_check(master, code, event_id, response, cache, counts):
    ticket, error, checked_at = response
    if str(error or '').startswith('deferred:'):
        label = error.removeprefix('deferred:')
        errors = counts.setdefault('request_errors', {})
        errors[label] = errors.get(label, 0) + 1
        return
    counts['pending'] -= 1
    if error:
        counts['failed'] += 1
        errors = counts.setdefault('request_errors', {})
        errors[error] = errors.get(error, 0) + 1
        return
    try:
        result = apply_public_evidence(master, code, ticket, event_id, checked_at)
    except ValueError:
        counts['failed'] += 1
        return
    counts['checked'] += 1
    counts[result] += 1
    if result == 'alias' and isinstance(ticket.get('description'), str):
        target = master[ticket['shareCode']]
        if target.get('description_checked_at') == checked_at:
            counts['full_from_alias'] += 1
    if result != 'stale':
        remember_verified(cache, ticket, error, checked_at)


def reconcile_public_listings(master, prior, active_codes, event_id, now,
                              cache, budget, deadline, policy=None):
    """Reconcile one event immediately for callers outside the inventory batch."""
    counts, candidates = prepare_public_listings(master, prior, active_codes, now)
    responses = query_public_listings([code for code, _ in candidates], cache, budget, deadline, policy)
    for code, response in responses.items():
        apply_public_listing_check(master, code, event_id, response, cache, counts)
    flatten_canonical_links(master)
    return counts


def flatten_canonical_links(master):
    """Flatten only links that have a same-event root; preserve every row."""
    for code, row in master.items():
        target = row.get('canonical_ticket_id', code)
        seen = {code}
        while (target in master and target not in seen
               and master[target].get('event_id') == row.get('event_id')):
            seen.add(target)
            next_code = master[target].get('canonical_ticket_id') or target
            if next_code == target:
                row['canonical_ticket_id'] = target
                break
            target = next_code
