"""Use a saved input snapshot, then independently join later outcome evidence."""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from observation import read_snapshot, snapshot_end, demand_label, covered, coverage_records

BASE = Path(__file__).resolve().parents[1]


def execute(args):
    folder = BASE / (args.kind + ('_state_model' if args.kind == 'demand' else '_arrival_model'))
    sys.path.insert(0, str(folder))
    import data_loader
    if args.kind == 'demand':
        data_loader.FAIR_PRICE_CACHE = args.prices
    from features import add_market_features
    from modeling import enforce_monotonic_horizons
    tickets = data_loader.load_tickets(args.source)
    raw = read_snapshot(args.source)
    later = read_snapshot(args.target).set_index('logical_id')
    moment = snapshot_end(raw)
    payload = joblib.load(folder / 'artifacts' / ('demand_state.joblib' if args.kind == 'demand' else 'alternative_arrival.joblib'))
    if moment <= pd.Timestamp(payload['observation_cutoff']):
        raise ValueError('Prediction snapshot overlaps model training')
    # No fabricated daily landmarks. Include only listings confirmed in the
    # current acquisition window. Freshness is explicit and reported.
    fresh = moment - pd.Timedelta(hours=2)
    active = tickets[tickets.status.eq('listing') & tickets.last_observed_at.ge(fresh)
                     & (tickets.performance_at.isna() | tickets.performance_at.gt(moment))].copy()
    active['landmark_at'] = moment
    active['days_since_listing'] = (moment-active.first_observed_at).dt.total_seconds()/86400
    active['days_until_event'] = (active.performance_at-moment).dt.total_seconds()/86400
    parts = []
    if not active.empty:
        frame = add_market_features(active, tickets)
        identifiers = raw.set_index('ticket_id').logical_id.to_dict()
        records = coverage_records([args.source, args.target])
        known = set(raw.logical_id)
        arrivals = later[~later.index.isin(known)].copy()
        if args.kind == 'alternative':
            target_tickets = data_loader.load_tickets(args.target)
            candidate_ids = set(arrivals.ticket_id)
            candidates = target_tickets[target_tickets.ticket_id.isin(candidate_ids)]
            arrival_by_ticket = arrivals.set_index('ticket_id')
        for h in sorted(map(int, payload['selected_features'])):
            selected = payload['selected_features'][str(h)]
            x = frame[selected['numeric']+selected['categorical']]
            out = frame[[c for c in ['ticket_id','event_id','price','fair_price','market_price_median'] if c in frame]].copy()
            out['landmark_at'] = moment
            out['horizon_days'] = h
            out['fold'] = -1
            if args.kind == 'demand':
                from modeling import aligned_probabilities, apply_temperature
                out[['p_active','p_sold','p_deleted']] = apply_temperature(aligned_probabilities(payload['models'][str(h)], x), payload['temperatures'][str(h)])
                labels = []
                for _, row in frame.iterrows():
                    key = identifiers.get(row.ticket_id)
                    final = later.loc[key] if key in later.index else None
                    label = demand_label(row, final, moment, h)
                    if pd.notna(row.performance_at) and row.performance_at < moment+pd.Timedelta(days=h):
                        label = -1
                    labels.append(label)
                out['true_state'] = labels
            else:
                from modeling import predict_positive_probability, calibrate
                from config import MIN_SAVINGS_YEN, MIN_SAVINGS_PCT, SEMANTIC_COMPARABLE_FIELDS
                out['p_alternative'] = calibrate(payload['calibrators'][str(h)], predict_positive_probability(payload['models'][str(h)], x))
                # Candidate semantics come from the target snapshot; outcomes
                # never enter the source features. Unknown compatibility is
                # not treated as proven comparability in this strict evaluator.
                labels, savings = [], []
                deadline = moment + pd.Timedelta(days=h)
                for _, row in frame.iterrows():
                    valid = candidates.event_id.eq(row.event_id)
                    for col in ['quantity','ticket_type','name_type'] + list(SEMANTIC_COMPARABLE_FIELDS):
                        valid &= candidates[col].fillna('__missing__').astype(str).eq(str(row[col]))
                    # Actual first sighting <= deadline, and API creation after
                    # source time, prevents old listings being counted as new.
                    c = candidates[valid].copy()
                    # Old archives recorded API creation as first observation.
                    # Only a provenance-marked scrape timestamp supports a
                    # dated positive arrival on a 1- or 3-day horizon.
                    provenance = arrival_by_ticket.get('first_observed_source', pd.Series('', index=arrival_by_ticket.index))
                    reliable = c.ticket_id.map(provenance).eq('scrape_observed')
                    observed = c.ticket_id.map(arrival_by_ticket.first_observed_at)
                    created = c.ticket_id.map(arrival_by_ticket.created_at_unix)
                    created = pd.to_datetime(pd.to_numeric(created, errors='coerce'), unit='ms', utc=True).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
                    c = c[reliable & created.gt(moment) & observed.gt(moment) & observed.le(deadline) & c.price.le(row.price-max(MIN_SAVINGS_YEN, row.price*MIN_SAVINGS_PCT))]
                    complete = covered(records, row.event_id, moment, deadline)
                    label = 1 if len(c) else (0 if complete else -1)
                    if pd.notna(row.performance_at) and row.performance_at < deadline:
                        label = -1
                    labels.append(label)
                    savings.append(max(0.,row.price-c.price.min()) if len(c) else 0.)
                out['true_alternative'] = labels
                out[f'potential_savings_{h}d'] = savings
            parts.append(out)
    result = enforce_monotonic_horizons(pd.concat(parts,ignore_index=True)) if parts else pd.DataFrame()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    args.output.with_suffix('.audit.json').write_text(json.dumps({'source':str(args.source),'target':str(args.target),
        'prediction_time':str(moment),'freshness_hours':2,'fresh_tickets':len(active),'prediction_rows':len(result),
        'manual_metadata':'current; historical versions unavailable; conditional retrospective evaluation'},indent=2),encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['demand','alternative'],required=True)
    for name in ['source','target','prices','output']:
        p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())
