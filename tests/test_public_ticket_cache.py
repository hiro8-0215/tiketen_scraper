import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import collection_evidence as evidence
import public_ticket_cache as cache_io
import scraper


class PublicTicketCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checked = '2026-09-27 02:00:00'
        self.ticket = {'shareCode': 'new', 'eventId': 'fire', 'status': 'active',
                       'description': 'full text', 'quantity': 2}

    def row(self):
        return {'ticket_id': 'new', 'event_id': 'event', 'public_event_id': 'fire',
                'status': 'listing', 'observation_state': 'active',
                'last_observed_at': '2026-09-27 01:00:00',
                'state_checked_at': '2026-09-27 01:00:00'}

    def test_verified_rotation_reply_is_reused_across_process_phases(self):
        old = self.row()
        old.update(ticket_id='old', observation_state='active')
        rows = {'old': old, 'new': self.row()}
        cache = {}
        with (patch.object(evidence, 'fetch_public_ticket', return_value=self.ticket) as request,
              patch.object(evidence.time, 'sleep')):
            report = evidence.reconcile_public_listings(rows, {'old': old.copy()}, {'new'},
                        'fire', '2026-09-27 02:00:00', cache, [10], float('inf'))
        self.assertEqual(report['alias'], 1)
        request.assert_called_once_with('old')
        with patch.dict(os.environ, {'PUBLIC_TICKET_CACHE_PATH': str(self.root / 'cache.json')}):
            cache_io.save_run_cache(cache)
            loaded = cache_io.load_run_cache()
        self.assertIn('new', loaded)
        with (patch.object(scraper, 'DATA_DIR', str(self.root)),
              patch.object(scraper, 'fetch_public_ticket') as repeated,
              patch.object(scraper.time, 'sleep') as sleep):
            pending = scraper.enrich_ticket_details('group', rows, ['new'], loaded)
        self.assertFalse(pending)
        repeated.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(rows['new']['raw_description'], 'full text')
        self.assertEqual(rows['new']['description_is_full'], 'True')

    def test_old_wrong_event_and_null_cached_replies_are_refetched(self):
        entries = [(self.ticket, None, '2026-09-27 00:00:00'),
                   ({**self.ticket, 'eventId': 'wrong'}, None, self.checked),
                   (None, None, self.checked)]
        for entry in entries:
            with self.subTest(entry=entry):
                rows = {'new': self.row()}
                with (patch.object(scraper, 'DATA_DIR', str(self.root)),
                      patch.object(scraper, 'fetch_public_ticket', return_value=self.ticket) as request,
                      patch.object(scraper.time, 'sleep')):
                    scraper.enrich_ticket_details('group', rows, ['new'], {'new': entry})
                request.assert_called_once_with('new')
                self.assertEqual(rows['new']['description_is_full'], 'True')

    def test_failed_identity_does_not_seed_the_canonical_cache(self):
        row = self.row()
        row.update(ticket_id='old')
        cache = {}
        with (patch.object(evidence, 'fetch_public_ticket', return_value={**self.ticket, 'eventId': 'wrong'}),
              patch.object(evidence.time, 'sleep')):
            result = evidence.reconcile_public_listings({'old': row}, {'old': row.copy()}, set(),
                        'fire', '2026-09-27 02:00:00', cache, [10], float('inf'))
        self.assertEqual(result['failed'], 1)
        self.assertNotIn('new', cache)

    def test_failures_are_not_persisted_or_reused_without_explicit_run_path(self):
        path = self.root / 'cache.json'
        with patch.dict(os.environ, {'PUBLIC_TICKET_CACHE_PATH': str(path)}):
            cache_io.save_run_cache({'good': (self.ticket, None, self.checked),
                                     'failed': (None, 'TimeoutError', '')})
            self.assertEqual(set(cache_io.load_run_cache()), {'good'})
        with patch.dict(os.environ, {'PUBLIC_TICKET_CACHE_PATH': ''}):
            self.assertEqual(cache_io.load_run_cache(), {})
        self.assertEqual(set(json.loads(path.read_text())), {'good'})


if __name__ == '__main__':
    unittest.main()
