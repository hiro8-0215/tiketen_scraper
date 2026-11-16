import csv
import json
import importlib.util
import tempfile
import unittest
from datetime import datetime, timedelta
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
            check = datetime.now() - timedelta(hours=1)
            self.write(folder, raw_description='full text', api_description='full...',
                       description_checked_at=check.isoformat(sep=' ', timespec='seconds'),
                       last_observed_at=(check + timedelta(minutes=30)).isoformat(sep=' ', timespec='seconds'))
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
