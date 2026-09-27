import csv
import json
import importlib.util
import tempfile
import unittest
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

    def test_preview_and_stale_content_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder,description_source='event_api_preview',content_checked_at='2026-09-27 00:00:00')
            report=module.audit(folder)
            self.assertFalse(report['ok'])
            self.assertEqual(len(report['errors']),2)

    def test_failed_stopped_listing_query_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.write(folder)
            (folder/'observation_20260927.jsonl').write_text(json.dumps({
                'performer':'group','event_id':'event','public_status_checks':{'failed':1}}))
            self.assertFalse(module.audit(folder)['ok'])
