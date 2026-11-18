import csv
import json
import importlib.util
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from pathlib import Path

spec=importlib.util.spec_from_file_location('audit_public_content',Path(__file__).resolve().parents[1]/'tools/audit_public_content.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContentAuditTest(unittest.TestCase):
    def write(self, folder, **extra):
        row={'ticket_id':'code','event_id':'event','observation_state':'active',
             'description_source':'public_detail','description_is_full':'True',
             'content_checked_at':'2026-09-27 02:00:00','last_observed_at':'2026-09-27 01:00:00',
             'public_event_id':'fire',**extra}
        with (folder/'group_master.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)

    def test_verified_and_empty_full_text_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder,raw_description='')
            self.assertTrue(module.audit(folder)['ok'])

    def test_rolling_audit_uses_actual_coverage_and_verified_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            check = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
            self.write(folder, raw_description='full text', api_description='full...',
                       description_checked_at=check.isoformat(sep=' ', timespec='seconds'),
                       last_observed_at=(check - timedelta(minutes=30)).isoformat(sep=' ', timespec='seconds'))
            (folder/'content_observation_20260927.jsonl').write_text(json.dumps({
                'performer':'group', 'scope':'observed_active', 'pending':1}))
            passing = module.audit(folder, rolling=True)
            self.assertTrue(passing['ok'])
            self.assertEqual(passing['warnings'][0]['reason'], 'incomplete_public_refresh')
            self.write(folder, description_source='event_api_preview')
            self.assertFalse(module.audit(folder, rolling=True)['ok'])

    def test_preview_and_stale_content_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder,description_source='event_api_preview',content_checked_at='2026-09-27 00:00:00')
            report=module.audit(folder)
            self.assertFalse(report['ok'])
            self.assertEqual(len(report['errors']),2)

    def test_empty_saved_time_matches_public_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder,perf_time='')
            with patch.object(module,'fetch_public_ticket',return_value={
                    'shareCode':'code','eventId':'fire','eventStartTime':None}), patch.object(module.time,'sleep'):
                report=module.audit(folder,1)
            self.assertTrue(report['ok'])
            self.assertEqual(report['live_samples'][0]['differences'],[])

    def test_null_observed_active_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder)
            (folder/'content_observation_20260927.jsonl').write_text(json.dumps({
                'performer':'group','scope':'observed_active','null':1}))
            self.assertFalse(module.audit(folder)['ok'])

    def test_legacy_null_warning_does_not_excuse_stale_active_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder)
            (folder/'content_observation_20260927.jsonl').write_text(json.dumps({
                'performer':'group','null':1}))
            report=module.audit(folder)
            self.assertTrue(report['ok'])
            self.assertEqual(len(report['warnings']),1)
            self.write(folder,content_checked_at='2026-09-27 00:00:00')
            self.assertFalse(module.audit(folder)['ok'])

    def test_inventory_run_reports_budget_backlog_and_retains_hard_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, description_is_full='False', description_source='event_api_preview')
            log = folder/'observation_20260927.jsonl'
            event = {'performer':'group', 'event_id':'event'}
            log.write_text(json.dumps({**event, 'public_status_checks':{'pending':42,'failed':0}}))
            report = module.audit(folder, inventory_first=True)
            self.assertTrue(report['ok'])
            self.assertFalse(report['public_coverage_complete'])
            self.assertEqual([item['reason'] for item in report['backlog']],
                             ['unverified_full_description','incomplete_public_state_refresh'])
            self.assertEqual(report['backlog'][1]['pending'], 42)
            self.assertFalse(module.audit(folder, rolling=True)['ok'])
            log.write_text(json.dumps({**event, 'public_status_checks':{'pending':42,'failed':1}}))
            report = module.audit(folder, inventory_first=True)
            self.assertFalse(report['ok'])
            self.assertEqual(report['errors'][0]['failed'], 1)

    def test_failed_stopped_listing_query_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder)
            (folder/'observation_20260927.jsonl').write_text(json.dumps({
                'performer':'group','event_id':'event','public_status_checks':{'failed':1}}))
            self.assertFalse(module.audit(folder)['ok'])

    def test_recent_absence_with_null_reply_cannot_pass_complete_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            check = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            self.write(folder, description_checked_at=check, absence_first_observed_at='')
            path = folder / 'group_master.csv'
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                active = next(reader)
            with path.open('a', encoding='utf-8', newline='') as stream:
                csv.DictWriter(stream, fieldnames=list(active)).writerow({
                    **active, 'ticket_id': 'missing', 'observation_state': 'absent_unknown',
                    'absence_first_observed_at': check,
                })
            report = module.audit(folder, inventory_first=True)
            self.assertTrue(report['ok'])
            self.assertFalse(report['public_coverage_complete'])
            self.assertEqual(report['backlog'][0]['reason'], 'recent_absence_without_outcome')
            self.assertFalse(module.audit(folder, rolling=True)['ok'])

    def test_inventory_audit_requires_current_api_ids_in_saved_master(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            when = now.strftime('%Y-%m-%d %H:%M:%S')
            self.write(folder, description_checked_at=when, last_observed_at=when)
            poll = {'performer': 'group', 'event_id': 'event',
                    'observed_at': when, 'observed_at_utc': now.isoformat(),
                    'api_fetch_complete': True, 'api_active_count': 1,
                    'api_active_ids': ['code']}
            log = folder / f"observation_{now:%Y%m%d}.jsonl"
            log.write_text(json.dumps(poll) + '\n', encoding='utf-8')
            offset = timezone(timedelta(hours=-12 if now.hour < 12 else 14))
            local_start = now.astimezone(offset)
            self.assertNotEqual(local_start.date(), now.date())
            report = module.audit(folder, inventory_first=True, inventory_since=local_start)
            self.assertTrue(report['ok'])
            self.assertEqual(report['verified_inventory_rows'], 1)
            poll['api_active_ids'] = ['missing']
            log.write_text(json.dumps(poll) + '\n', encoding='utf-8')
            report = module.audit(folder, inventory_first=True, inventory_since=now)
            self.assertFalse(report['ok'])
            self.assertEqual(report['errors'][0]['reason'], 'api_inventory_not_preserved')

    def test_only_proven_provider_ip_blocks_allow_drive_backup_of_verified_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            now = datetime.now(timezone.utc).replace(microsecond=0)
            when = now.strftime('%Y-%m-%d %H:%M:%S')
            self.write(folder, description_checked_at=when, last_observed_at=when)
            log = folder / f"observation_{now:%Y%m%d}.jsonl"
            poll = {'performer': 'group', 'event_id': 'event', 'observed_at': when,
                    'observed_at_utc': now.isoformat(), 'api_fetch_complete': True,
                    'api_active_count': 1, 'api_active_ids': ['code'],
                    'public_status_checks': {'failed': 1, 'ip_blocked_failures': 1}}
            details = folder / f"content_observation_{now:%Y%m%d}.jsonl"
            details.write_text(json.dumps({'performer': 'group', 'failed': 1,
                                           'ip_blocked_failures': 1}), encoding='utf-8')
            for change, eligible in [({}, True), ({'api_fetch_complete': False}, False),
                                     ({'api_active_ids': ['missing']}, False),
                                     ({'public_status_checks': {'failed': 2, 'ip_blocked_failures': 1}}, False)]:
                with self.subTest(change=change):
                    log.write_text(json.dumps({**poll, **change}), encoding='utf-8')
                    report = module.audit(folder, inventory_first=True, inventory_since=now)
                    self.assertFalse(report['ok'])
                    self.assertFalse(report['public_coverage_complete'])
                    self.assertEqual(report['drive_inventory_verified'], eligible)
            log.write_text(json.dumps(poll), encoding='utf-8')
            self.assertFalse(module.audit(folder, rolling=True, inventory_since=now)['drive_inventory_verified'])
