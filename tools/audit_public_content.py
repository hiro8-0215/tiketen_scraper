"""Audit full-content provenance without changing masters; optionally sample live pages."""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collection_content import content_refresh_reason
from collection_evidence import fetch_public_ticket
from scraper import sanitize_unicode


DEFERRED_CONTENT_REASONS = {
    'unverified_full_description', 'expired_full_description', 'public_description_preview_changed',
    'public_description_older_than_list',
}


def audit(folder, live_limit=0, *, rolling=False, inventory_first=False):
    errors, active, samples, latest, warnings = [], {}, [], {}, []
    backlog = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    absence_cutoff = (now - timedelta(hours=36)).isoformat(sep=' ')
    for path in folder.glob('*_master.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            if row.get('observation_state') in {'absent_unverified', 'absent_unknown'}:
                first_absence = row.get('absence_first_observed_at', '')
                if first_absence >= absence_cutoff:
                    findings = errors if rolling else backlog
                    findings.append({'id': row['ticket_id'], 'reason': 'recent_absence_without_outcome'})
            if row.get('observation_state') != 'active':
                continue
            code = row['ticket_id']
            if rolling or inventory_first:
                reason = content_refresh_reason(row, now, require_latest=True)
                if reason:
                    findings = backlog if inventory_first and reason in DEFERRED_CONTENT_REASONS else errors
                    findings.append({'id': code, 'reason': reason})
            else:
                if row.get('description_source') != 'public_detail' or row.get('description_is_full') != 'True':
                    errors.append({'id':code,'reason':'unverified_full_description'})
                if not row.get('content_checked_at') or row['content_checked_at'] < row.get('last_observed_at',''):
                    errors.append({'id':code,'reason':'content_older_than_api_observation'})
            if not row.get('public_event_id'):
                errors.append({'id':code,'reason':'missing_public_event_id'})
            active[(row['event_id'],code)] = row
    for path in sorted(folder.glob('content_observation_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            lines = list(stream)
        for line in lines:
            item = json.loads(line)
            latest[item['performer']] = item
    for performer, item in latest.items():
        if (item.get('failed') or item.get('pending')
                or (item.get('null') and item.get('scope') == 'observed_active')):
            findings = warnings if rolling or inventory_first else errors
            if item.get('failed'):
                findings = errors
            findings.append({'performer':performer,'reason':'incomplete_public_refresh',
                           'failed':item.get('failed'),'pending':item.get('pending'),'null':item.get('null')})
        elif item.get('null'):
            warnings.append({'performer':performer,'reason':'legacy_refresh_included_unobserved_rows',
                             'null':item['null'],
                             'note':'Historical nulls remain unconfirmed; every active row is still checked above.'})
    polls = {}
    for path in sorted(folder.glob('observation_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                polls[(item.get('performer'),item.get('event_id'))] = item
    for (performer,event), item in polls.items():
        checks = item.get('public_status_checks', {})
        if unresolved_sold := item.get('unresolved_sold_count'):
            warnings.append({'performer': performer, 'event': event,
                             'reason': 'anonymous_sold_rows', 'count': unresolved_sold})
        if checks.get('failed') or checks.get('pending'):
            findings = backlog if inventory_first and not checks.get('failed') else errors
            findings.append({'performer':performer,'event':event,
                             'reason':'incomplete_public_state_refresh',
                             'failed':checks.get('failed', 0),'pending':checks.get('pending', 0)})
    if not active:
        errors.append({'reason':'no_active_rows_to_verify'})
    # Evenly distributed across sorted event/ID population; deterministic.
    ordered = sorted(active)
    selected = [ordered[i * len(ordered) // min(live_limit,len(ordered))]
                for i in range(min(live_limit,len(ordered)))] if live_limit and ordered else []
    fields = {'price':'pricePerTicket','quantity':'quantity','raw_description':'description',
              'perf_date':'eventDate','perf_time':'eventStartTime','venue':'venue',
              'ticket_type':'ticketType','name_type':'nameGender','delivery_method':'deliveryMethod'}
    for key in selected:
        row = active[key]
        result = {'event':key[0],'id':key[1], 'differences':[]}
        try:
            ticket = fetch_public_ticket(key[1])
            if ticket is None:
                result['now_unavailable'] = True
            else:
                result['public_status'] = ticket.get('status')
                if ticket.get('eventId') != row['public_event_id'] or ticket.get('shareCode') != row.get('canonical_ticket_id',row['ticket_id']):
                    errors.append({'id':key[1],'reason':'live_identity_mismatch'})
                for field, public_key in fields.items():
                    if public_key not in ticket:
                        continue
                    public = ticket[public_key]
                    equal = (float(row.get(field) or 0) == float(public or 0)
                             if field in {'price','quantity'} else
                             row.get(field,'') == ('' if public is None else sanitize_unicode(public)))
                    if not equal:
                        result['differences'].append({'field':field,'saved':row.get(field), 'public':public})
        except Exception as exc:
            errors.append({'id':key[1],'reason':'live_query_failed','error':type(exc).__name__})
        samples.append(result)
        time.sleep(0.5)
    return {'ok':not errors,'public_coverage_complete':not errors and not backlog,
            'unique_active_rows':len(active),'live_samples':samples,
            'latest_content_refresh':latest,'errors':errors,'warnings':warnings,'backlog':backlog,
            'limitations':'Live values can change after collection. Historical null tickets cannot be reconstructed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--live-limit',type=int,default=0)
    parser.add_argument('--report',type=Path)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument('--rolling', action='store_true',
                        help='Require verified active descriptions within the rolling refresh policy.')
    policy.add_argument('--inventory-first', action='store_true',
                        help='Record deferred lookups separately from inventory errors.')
    args=parser.parse_args()
    report=audit(args.data_dir,max(0,args.live_limit),rolling=args.rolling,
                 inventory_first=args.inventory_first)
    if args.report:
        args.report.write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding='utf-8')
    backlog_counts = Counter()
    for item in report['backlog']:
        backlog_counts[item['reason']] += item.get('pending', 1)
    error_counts = Counter(item['reason'] for item in report['errors'])
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a', encoding='utf-8') as summary:
            summary.write('### Public ticket evidence\n\n')
            summary.write(f"- Active listing rows: {report['unique_active_rows']:,}\n")
            summary.write(f"- Known listing states and active content fully checked: {report['public_coverage_complete']}\n")
            anonymous = sum(item['count'] for item in report['warnings'] if item['reason'] == 'anonymous_sold_rows')
            if anonymous:
                summary.write(f"- Sold rows without a listing identifier: {anonymous:,}\n")
            for reason, count in sorted(backlog_counts.items()):
                summary.write(f'- Deferred {reason}: {count:,}\n')
            for reason, count in sorted(error_counts.items()):
                summary.write(f'- Quality errors {reason}: {count:,}\n')
    print(json.dumps({'ok':report['ok'],'public_coverage_complete':report['public_coverage_complete'],
                      'unique_active_rows':report['unique_active_rows'],'backlog_counts':dict(backlog_counts),
                      'error_counts':dict(error_counts),
                      'live_samples':len(report['live_samples']),'errors':report['errors'][:20],
                      'warnings':report['warnings']},ensure_ascii=True))
    raise SystemExit(0 if report['ok'] else 1)


if __name__=='__main__':
    main()
