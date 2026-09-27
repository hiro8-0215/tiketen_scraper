"""Audit full-content provenance without changing masters; optionally sample live pages."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collection_evidence import fetch_public_ticket
from scraper import sanitize_unicode


def audit(folder, live_limit=0):
    errors, active, samples, latest, warnings = [], {}, [], {}, []
    for path in folder.glob('*_master.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            if row.get('observation_state') != 'active':
                continue
            code = row['ticket_id']
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
            errors.append({'performer':performer,'reason':'incomplete_public_refresh',
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
        if checks.get('failed') or checks.get('pending'):
            errors.append({'performer':performer,'event':event,
                           'reason':'incomplete_public_state_refresh',
                           'failed':checks.get('failed'),'pending':checks.get('pending')})
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
    return {'ok':not errors,'unique_active_rows':len(active),'live_samples':samples,
            'latest_content_refresh':latest,'errors':errors,'warnings':warnings,
            'limitations':'Live values can change after collection. Historical null tickets cannot be reconstructed.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--live-limit',type=int,default=0)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    report=audit(args.data_dir,max(0,args.live_limit))
    if args.report:
        args.report.write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding='utf-8')
    print(json.dumps({'ok':report['ok'],'unique_active_rows':report['unique_active_rows'],
                      'live_samples':len(report['live_samples']),'errors':report['errors'][:20],
                      'warnings':report['warnings']},ensure_ascii=True))
    raise SystemExit(0 if report['ok'] else 1)


if __name__=='__main__':
    main()
