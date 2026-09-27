"""Read-only audit of downloaded collector outputs against a preserved seed."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_master(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['ticket_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate IDs within {path.name}')
    return {r['ticket_id']: r for r in rows}


def audit(before_dir, after_dir):
    errors, totals, states = [], Counter(), Counter()
    for source in sorted(before_dir.glob('*_master.csv')):
        destination = after_dir / source.name
        if not destination.exists():
            errors.append(f'Missing master: {source.name}')
            continue
        before, after = read_master(source), read_master(destination)
        totals['before_rows'] += len(before)
        totals['after_rows'] += len(after)
        lost = set(before) - set(after)
        totals['lost_original_ids'] += len(lost)
        if lost:
            errors.append(f'{source.name}: lost {len(lost)} original IDs')
        for code, row in after.items():
            states[row.get('observation_state') or 'legacy_unclassified'] += 1
            original = before.get(code)
            if original:
                first = original.get('first_observed_at')
                if first and row.get('first_observed_at') != first:
                    errors.append(f'{source.name}/{code}: original first sighting changed')
                if original.get('status') != row.get('status'):
                    totals['status_transitions'] += 1
                    if row.get('status_source') not in {'public_detail', 'event_api', 'complete_api_absence'}:
                        errors.append(f'{source.name}/{code}: transition without source')
            else:
                totals['added_rows'] += 1
            canonical = row.get('canonical_ticket_id') or code
            if canonical != code:
                totals['alias_rows'] += 1
                target = after.get(canonical)
                if not target or target.get('event_id') != row.get('event_id'):
                    errors.append(f'{source.name}/{code}: invalid canonical target')
                elif (target.get('canonical_ticket_id') or canonical) != canonical:
                    errors.append(f'{source.name}/{code}: canonical chain not flattened')
                if row.get('observation_state') != 'alias':
                    errors.append(f'{source.name}/{code}: linked row not marked alias')
            if row.get('is_price_on_request') == 'True':
                totals['price_on_request_rows'] += 1
            if row.get('observation_state') in {'absent_unknown', 'absent_unverified'}:
                if row.get('status') != 'listing':
                    errors.append(f'{source.name}/{code}: unknown absence assigned terminal state')
    latest = {}
    for path in sorted(after_dir.glob('observation_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                key = (item['performer'], item['event_id'])
                if item.get('observed_at', '') >= latest.get(key, {}).get('observed_at', ''):
                    latest[key] = item
    checks = Counter()
    later_state_updates = 0
    for (performer, event), observation in latest.items():
        if 'public_status_checks' not in observation:
            continue
        checks.update(observation['public_status_checks'])
        rows = read_master(after_dir / f'{performer}_master.csv')
        if observation.get('api_fetch_complete'):
            api_ids = observation.get('api_active_ids')
            if api_ids is not None:
                if len(api_ids) != len(set(api_ids)) or len(api_ids) != observation['api_active_count']:
                    errors.append(f'{performer}/{event}: inconsistent API ID inventory')
                for code in api_ids:
                    row = rows.get(code)
                    if not row or row.get('event_id') != event:
                        errors.append(f'{performer}/{event}/{code}: API ID not saved')
                        continue
                    if row.get('observation_state') == 'active':
                        continue
                    confirmed_later = (
                        row.get('state_checked_at', '') >= observation['observed_at']
                        and row.get('observation_state') in {
                            'alias', 'sold_confirmed', 'deleted_confirmed', 'inactive', 'expired'}
                        and (row.get('status_source') == 'public_detail'
                             or row.get('observation_state') == 'alias'))
                    if confirmed_later:
                        later_state_updates += 1
                    else:
                        errors.append(f'{performer}/{event}/{code}: API state lost without later evidence')
                continue
            active = sum(r.get('event_id') == event and r.get('observation_state') == 'active'
                         and r.get('last_observed_at') == observation['observed_at']
                         for r in rows.values())
            expected = observation['api_active_count']
            if active != expected:
                errors.append(f'{performer}/{event}: API active={expected}, saved current={active}')
    ledger_records = 0
    for path in after_dir.glob('ticket_changes_*.jsonl'):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                json.loads(line)
                ledger_records += 1
    return {'ok': not errors, 'totals': dict(totals), 'observation_states': dict(states),
            'latest_public_checks': dict(checks), 'change_ledger_records': ledger_records,
            'api_ids_with_later_confirmed_state_updates': later_state_updates,
            'errors': errors,
            'limitations': 'Unknown historical absences and anonymous API sales remain unlabelled; this audit does not prove all sales were individually recovered.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit(args.before, args.after)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(output, encoding='utf-8')
    print(output)
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
