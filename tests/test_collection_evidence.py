import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import collection_evidence as evidence
import scraper


class CollectionEvidenceTest(unittest.TestCase):
    def rows(self):
        return {'old': {'ticket_id': 'old', 'event_id': 'event', 'status': 'listing',
                        'first_observed_at': '2026-09-25 01:00:00',
                        'last_observed_at': '2026-09-26 23:00:00', 'price': 12000},
                'new': {'ticket_id': 'new', 'event_id': 'event', 'status': 'listing',
                        'first_observed_at': '2026-09-27 01:00:00', 'price': 13000}}

    def test_public_alias_preserves_both_ids_and_first_observation(self):
        rows = self.rows()
        result = evidence.apply_public_evidence(rows, 'old',
            {'shareCode': 'new', 'eventId': 'firestore', 'status': 'active'},
            'firestore', '2026-09-27 02:00:00')
        self.assertEqual(result, 'alias')
        self.assertEqual(set(rows), {'old', 'new'})
        self.assertEqual(rows['old']['canonical_ticket_id'], 'new')
        self.assertEqual(rows['old']['last_observed_at'], '2026-09-26 23:00:00')
        self.assertEqual(rows['new']['identity_first_observed_at'], '2026-09-25 01:00:00')
        self.assertEqual(rows['new']['first_observed_at'], '2026-09-27 01:00:00')

    def test_null_public_ticket_does_not_fabricate_sale_or_deletion(self):
        rows=self.rows()
        evidence.apply_public_evidence(rows, 'old', None, 'firestore', '2026-09-27 02:00:00')
        self.assertEqual(rows['old']['status'], 'listing')
        self.assertEqual(rows['old']['observation_state'], 'absent_unknown')
        self.assertNotIn('sold_at', rows['old'])

    def test_public_sold_is_confirmed_transition(self):
        rows=self.rows()
        evidence.apply_public_evidence(rows, 'old',
            {'shareCode':'old','eventId':'firestore','status':'sold'},
            'firestore','2026-09-27 02:00:00')
        self.assertEqual(rows['old']['status'],'sold')
        self.assertEqual(rows['old']['sold_at_source'],'transition_observed')

    def test_public_sold_canonical_code_missing_from_api_is_retained(self):
        rows=self.rows()
        evidence.apply_public_evidence(rows,'old',
            {'shareCode':'sold-code','eventId':'firestore','status':'sold'},
            'firestore','2026-09-27 02:00:00')
        self.assertIn('old',rows)
        self.assertEqual(rows['sold-code']['status'],'sold')
        self.assertEqual(rows['old']['canonical_ticket_id'],'sold-code')

    def test_mismatched_event_cannot_link(self):
        rows=self.rows()
        with self.assertRaises(ValueError):
            evidence.apply_public_evidence(rows,'old',
                {'shareCode':'new','eventId':'other','status':'active'},
                'firestore','2026-09-27 02:00:00')
        self.assertNotIn('canonical_ticket_id',rows['old'])

    def test_inactive_is_not_a_sale_or_confirmed_withdrawal(self):
        rows = self.rows()
        evidence.apply_public_evidence(rows, 'old',
            {'shareCode':'old','eventId':'firestore','status':'inactive'},
            'firestore','2026-09-27 02:00:00')
        self.assertEqual(rows['old']['status'], 'listing')
        self.assertEqual(rows['old']['observation_state'], 'inactive')
        self.assertEqual(rows['old']['last_observed_at'], '2026-09-26 23:00:00')
        self.assertNotIn('sold_at', rows['old'])

    def test_confirmed_sale_keeps_explicit_public_price_not_stale_listing_price(self):
        rows = self.rows()
        evidence.apply_public_evidence(rows, 'old',
            {'shareCode':'old','eventId':'firestore','status':'sold',
             'pricePerTicket':15000,'isPriceOnRequest':False},
            'firestore','2026-09-27 02:00:00')
        self.assertEqual(rows['old']['price'], 15000)
        self.assertEqual(rows['old']['price_source'], 'public_detail')

    def test_inactive_alias_missing_from_active_api_retains_both_rows(self):
        rows = self.rows()
        evidence.apply_public_evidence(rows, 'old',
            {'shareCode':'inactive-code','eventId':'firestore','status':'inactive'},
            'firestore','2026-09-27 02:00:00')
        self.assertEqual(rows['old']['observation_state'], 'alias')
        self.assertEqual(rows['inactive-code']['observation_state'], 'inactive')
        self.assertEqual(rows['inactive-code']['status'], 'listing')
        self.assertEqual(rows['inactive-code']['identity_first_observed_at'], '2026-09-25 01:00:00')
        self.assertNotIn('sold_at', rows['inactive-code'])

    def test_paused_and_unknown_public_statuses_never_fabricate_terminal_labels(self):
        for status, expected in [('paused','paused'), ('new-site-state','public_unclassified')]:
            with self.subTest(status=status):
                rows = self.rows()
                evidence.apply_public_evidence(rows, 'old',
                    {'shareCode':'unlisted-code','eventId':'firestore','status':status},
                    'firestore','2026-09-27 02:00:00')
                self.assertEqual(rows['unlisted-code']['observation_state'], expected)
                self.assertEqual(rows['unlisted-code']['public_ticket_status'], status)
                self.assertEqual(rows['unlisted-code']['status'], 'listing')
                self.assertNotIn('sold_at', rows['unlisted-code'])

    def test_transport_failure_stays_retryable_and_does_not_update_last_seen(self):
        rows=self.rows()
        with patch.object(evidence,'fetch_public_ticket',side_effect=TimeoutError), patch.object(evidence.time,'sleep'):
            counts=evidence.reconcile_public_listings(rows,{'old':rows['old'].copy()},
                {'new'},'firestore','2026-09-27 02:00:00',{},[1],time.monotonic()+60)
        self.assertEqual(counts['failed'],1)
        self.assertEqual(rows['old']['observation_state'],'absent_unverified')
        self.assertEqual(rows['old']['last_observed_at'],'2026-09-26 23:00:00')

    def test_zero_budget_makes_no_requests(self):
        rows=self.rows()
        with patch.object(evidence,'fetch_public_ticket') as request:
            counts=evidence.reconcile_public_listings(rows,{'old':rows['old'].copy()},
                {'new'},'firestore','2026-09-27 02:00:00',{},[0],time.monotonic()+60)
        request.assert_not_called()
        self.assertEqual(counts['pending'],1)

    def test_change_ledger_does_not_duplicate_unchanged_polls(self):
        rows=self.rows()
        with tempfile.TemporaryDirectory() as folder, patch.object(scraper,'DATA_DIR',folder):
            scraper.save_ticket_changes('artist',{},rows,'2026-09-27 02:00:00','utc')
            before={k:r.copy() for k,r in rows.items()}
            rows['old']['state_checked_at']='2026-09-27 03:00:00'
            scraper.save_ticket_changes('artist',before,rows,'2026-09-27 03:00:00','utc')
            path=next(Path(folder).glob('ticket_changes_*.jsonl'))
            records=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records),2)

    def test_public_active_cannot_be_overwritten_by_absence_inference(self):
        rows=self.rows()
        evidence.apply_public_evidence(rows,'old',
            {'shareCode':'old','eventId':'firestore','status':'active'},
            'firestore','2026-09-27 02:00:00')
        scraper.mark_confirmed_absences_deleted(rows,{'event':{'new'}},'2026-09-27 02:00:00')
        self.assertEqual(rows['old']['status'],'listing')

    def test_cached_state_cannot_replace_newer_api_state_or_price(self):
        rows = self.rows()
        rows['new'].update(observation_state='active', status_source='event_api',
                           state_checked_at='2026-09-27 02:05:00')
        cache = {'old': ({'shareCode':'new','eventId':'firestore','status':'cancelled',
                          'pricePerTicket':12000}, None, '2026-09-27 02:00:00')}
        counts = evidence.reconcile_public_listings(rows, {'old':rows['old'].copy()},
            {'new'}, 'firestore','2026-09-27 02:10:00',cache,[0],time.monotonic()+60)
        self.assertEqual(counts['alias'], 1)
        self.assertEqual(rows['old']['state_checked_at'], '2026-09-27 02:00:00')
        self.assertEqual(rows['new']['status'], 'listing')
        self.assertEqual(rows['new']['observation_state'], 'active')
        self.assertEqual(rows['new']['price'], 13000)

    def test_stale_null_cannot_replace_a_newer_known_state(self):
        rows = self.rows()
        rows['old'].update(observation_state='active',
                           state_checked_at='2026-09-27 02:05:00')
        result = evidence.apply_public_evidence(rows, 'old', None,
                                                'firestore','2026-09-27 02:00:00')
        self.assertEqual(result, 'stale')
        self.assertEqual(rows['old']['observation_state'], 'active')

if __name__=='__main__':
    unittest.main()
