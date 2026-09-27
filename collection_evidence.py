"""Public listing evidence; never identify listings by matching descriptions."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

PUBLIC_QUERY_URL = 'https://asia-northeast1-ticketen-prod.cloudfunctions.net/ticket_queries_get'
EVIDENCE_COLUMNS = [
    'canonical_ticket_id', 'identity_first_observed_at', 'observation_state',
    'state_checked_at', 'absence_first_observed_at', 'status_source',
    'is_price_on_request', 'price_source',
]


def fetch_public_ticket(code):
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
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    if payload.get('success') is not True or 'ticket' not in payload:
        raise ValueError('Invalid public ticket response')
    ticket = payload['ticket']
    if ticket is not None and not isinstance(ticket, dict):
        raise ValueError('Invalid public ticket object')
    return ticket


def apply_public_evidence(master, old_code, ticket, firestore_event_id, now):
    row = master[old_code]
    if ticket is None:
        row['observation_state'] = 'absent_unknown'
        row['state_checked_at'] = now
        return 'not_found'
    code = str(ticket.get('shareCode') or '').strip()
    if not code or ticket.get('eventId') != firestore_event_id:
        raise ValueError('Public ticket identity/event mismatch')
    status = ticket.get('status')
    if status == 'completed':
        status = 'sold'
    if status not in {'active', 'sold', 'cancelled', 'expired', 'inactive'}:
        raise ValueError('Unknown public ticket status')
    if status == 'active' and ticket.get('isSold') is True:
        raise ValueError('Conflicting public ticket state')
    if code != old_code:
        target = master.get(code)
        # Public terminal confirmation may name a canonical code omitted from
        # the event sold list. Retain that explicitly verified lifecycle too.
        if target is None and status in {'sold', 'cancelled'}:
            target = row.copy()
            target.update(ticket_id=code, first_observed_at=now,
                          first_observed_source='public_alias_observed')
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
    row['state_checked_at'] = now
    row['status_source'] = 'public_detail'
    if status == 'active':
        # Keep the API sighting timestamp when it already confirmed this code;
        # state_checked_at independently records the later public lookup.
        if row.get('observation_state') != 'active' or not row.get('last_observed_at'):
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
    elif status == 'inactive':
        # Publicly unavailable is not proof of a sale or permanent withdrawal.
        # Keep the last historical status and sighting; retry on later polls.
        row['observation_state'] = 'inactive'
    else:
        row['observation_state'] = 'expired'
    if 'isPriceOnRequest' in ticket:
        row['is_price_on_request'] = str(ticket['isPriceOnRequest'] is True)
    value = ticket.get('pricePerTicket')
    if value is not None:
        try:
            price = float(value)
        except (ValueError, TypeError):
            price = float('nan')
        if 0 <= price < float('inf'):
            row['price'] = int(price) if price.is_integer() else price
            row['price_source'] = ('on_request' if ticket.get('isPriceOnRequest') is True
                                   else 'public_detail' if price > 0 else 'unknown')
    return 'alias' if code != old_code else status


def reconcile_public_listings(master, prior, active_codes, event_id, now,
                              cache, budget, deadline):
    """Bounded, resumable checks prioritizing shareCodes before they expire.

    Old historical rows are retained as unknown; text similarity never links
    them. Retry transport failures on a later poll, not as a false deletion.
    """
    counts = {'checked': 0, 'alias': 0, 'sold': 0, 'cancelled': 0,
              'not_found': 0, 'active': 0, 'expired': 0, 'inactive': 0, 'failed': 0,
              'pending': 0, 'historical_unresolved': 0}
    cutoff = datetime.fromisoformat(now) - timedelta(hours=36)
    candidates = []
    for code, previous in prior.items():
        row = master.get(code)
        if row is None or code in active_codes or row.get('status') != 'listing':
            continue
        if row.get('observation_state') in {'alias', 'expired', 'absent_unknown'}:
            continue
        row.setdefault('absence_first_observed_at', now)
        if not row.get('absence_first_observed_at'):
            row['absence_first_observed_at'] = now
        row['observation_state'] = 'absent_unverified'
        try:
            recent = datetime.fromisoformat(previous.get('last_observed_at', '')) >= cutoff
        except ValueError:
            recent = False
        if not recent:
            counts['historical_unresolved'] += 1
            continue
        candidates.append(code)
    candidates.sort(key=lambda c: prior[c].get('last_observed_at', ''), reverse=True)
    def query(code):
        if time.monotonic() >= deadline:
            return code, None, 'budget'
        if code not in cache:
            try:
                cache[code] = (fetch_public_ticket(code), None)
            except Exception as error:
                cache[code] = (None, type(error).__name__)
            time.sleep(0.5)  # At most two concurrent public requests.
        value, error = cache[code]
        return code, value, error
    cached = [c for c in candidates if c in cache]
    fresh = [c for c in candidates if c not in cache]
    selected = cached + fresh[:budget[0]]
    budget[0] -= min(len(fresh), budget[0])
    counts['pending'] += len(candidates) - len(selected)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for code, ticket, error in pool.map(query, selected):
            if error:
                counts['pending' if error == 'budget' else 'failed'] += 1
                continue
            try:
                checked_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result = apply_public_evidence(master, code, ticket, event_id, checked_at)
            except ValueError:
                counts['failed'] += 1
                continue
            counts['checked'] += 1
            counts[result] += 1
    # Persist direct canonical links even after several proved code rotations.
    for code, row in master.items():
        target = row.get('canonical_ticket_id', code)
        seen = {code}
        while target in master and target not in seen:
            seen.add(target)
            next_code = master[target].get('canonical_ticket_id', target)
            if next_code == target:
                break
            target = next_code
        if target in master and target != code and master[target].get('event_id') == row.get('event_id'):
            row['canonical_ticket_id'] = target
    return counts
