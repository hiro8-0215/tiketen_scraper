"""Evidence rules shared by audit and frozen snapshot validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def read_snapshot(folder):
    parts = [pd.read_csv(p, dtype=str).fillna('').assign(source_file=p.name)
             for p in sorted(Path(folder).glob('*_master.csv'))]
    if not parts:
        raise ValueError(f'No master files: {folder}')
    f = pd.concat(parts, ignore_index=True)
    created = f.get('created_at_unix', pd.Series('', index=f.index)).str.replace(r'\.0$', '', regex=True)
    f['logical_id'] = 'ticket:' + f.ticket_id
    stable = created.ne('') & f.event_id.ne('')
    f.loc[stable, 'logical_id'] = f.loc[stable, 'event_id'] + '|' + created[stable]
    for col in ['last_observed_at', 'first_observed_at', 'sold_at']:
        f[col] = pd.to_datetime(f[col], errors='coerce')
    f['priority'] = f.status.map({'deleted':0, 'listing':1, 'sold':2})
    return f.sort_values(['last_observed_at','priority'], na_position='first').drop_duplicates('logical_id', keep='last').reset_index(drop=True)


def snapshot_end(frame):
    return frame.last_observed_at.max()


def coverage_records(folders):
    records = []
    for folder in folders:
        for path in Path(folder).glob('observation_*.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def covered(records, event, start, end, max_gap_hours=2):
    """Only explicit successful full event polls certify a negative window."""
    times = sorted(set(pd.Timestamp(r['observed_at']) for r in records
        if r.get('event_id') == event and r.get('complete') is True))
    if not times:
        return False
    times = pd.DatetimeIndex(times)
    left = times[times <= start]
    right = times[times >= end]
    if not len(left) or not len(right):
        return False
    chain = times[(times >= left[-1]) & (times <= right[0])]
    return bool((np.diff(chain.asi8) <= pd.Timedelta(hours=max_gap_hours).value).all())


def demand_label(initial, later, start, horizon):
    deadline = start + pd.Timedelta(days=int(horizon))
    if later is None or pd.isna(later.last_observed_at):
        return -1
    # Listing confirmed at/after the deadline; no persistence past last sighting.
    if later.status == 'listing' and later.last_observed_at >= deadline:
        return 0
    if later.status == 'sold' and later.get('sold_at_source', '') == 'transition_observed':
        when = later.sold_at
        if pd.notna(when) and start < when <= deadline:
            return 1
    if later.status == 'deleted' and start < later.last_observed_at <= deadline:
        return 2
    # A later outcome does not prove that the listing survived until deadline.
    return -1


def audit(folders):
    output = []
    for folder in folders:
        f = read_snapshot(folder)
        end = snapshot_end(f)
        events = []
        for event, part in f.groupby('event_id'):
            active = part[part.status.eq('listing')]
            events.append({'event_id':event, 'last_observed':str(part.last_observed_at.max()),
                'listing':len(active), 'stale_listing_over_24h':int((active.last_observed_at < end-pd.Timedelta(days=1)).sum()),
                'sold':int(part.status.eq('sold').sum())})
        output.append({'folder':str(folder), 'end':str(end), 'tickets':len(f), 'events':events})
    return output
