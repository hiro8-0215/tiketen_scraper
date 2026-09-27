import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collection_content import apply_api_preview, public_updates
import collection_evidence as evidence
import scraper


class ContentTest(unittest.TestCase):
    def test_preview_does_not_overwrite_full_text_over_two_polls(self):
        row = {'raw_description':'complete conditions and seat information',
               'description_source':'public_detail','description_is_full':'True'}
        for preview in ['complete...', 'changed preview...']:
            apply_api_preview(row, {'description':preview})
            self.assertEqual(row['raw_description'], 'complete conditions and seat information')
            self.assertEqual(row['api_description'], preview)
            self.assertEqual(row['description_is_full'], 'True')

    def test_legacy_full_text_is_preserved(self):
        row = {'raw_description':'legacy browser full text','details_fetched':'True'}
        apply_api_preview(row, {'description':'legacy...'})
        self.assertEqual(row['raw_description'], 'legacy browser full text')

    def test_unverified_preview_can_be_updated(self):
        row = {}
        apply_api_preview(row, {'description':'old...'})
        apply_api_preview(row, {'description':'new...'})
        self.assertEqual(row['raw_description'],'new...')
        self.assertEqual(row['description_is_full'],'False')

    def test_inactive_updates_quantity_and_full_text_without_fake_sale(self):
        row = {'ticket_id':'code','event_id':'event','status':'listing',
               'quantity':3,'raw_description':'preview...',
               'last_observed_at':'2026-09-27 01:00:00'}
        evidence.apply_public_evidence({'code':row}, 'code',
            {'shareCode':'code','eventId':'fire','status':'inactive','quantity':4,
             'description':'full seat conditions', 'venue':'venue',
             'seatType':{'isSpecified':True,'value':'A block'}},
            'fire','2026-09-27 02:00:00')
        self.assertEqual(row['quantity'],4)
        self.assertEqual(row['raw_description'],'full seat conditions')
        self.assertEqual(row['description_is_full'],'True')
        self.assertEqual(row['status'],'listing')
        self.assertEqual(row['last_observed_at'],'2026-09-27 01:00:00')

    def test_stale_response_cannot_replace_content(self):
        row = {'ticket_id':'code','event_id':'event','status':'listing',
               'quantity':4, 'raw_description':'latest full',
               'state_checked_at':'2026-09-27 03:00:00'}
        result = evidence.apply_public_evidence({'code':row}, 'code',
            {'shareCode':'code','eventId':'fire','status':'active','quantity':2,'description':'old'},
            'fire','2026-09-27 02:00:00')
        self.assertEqual(result,'stale')
        self.assertEqual(row['quantity'],4)
        self.assertEqual(row['raw_description'],'latest full')

    def test_missing_and_invalid_values_do_not_fabricate_full_content(self):
        self.assertNotIn('description_is_full',public_updates({},'now'))
        for value in [0, -1, 1.5, True, float('nan')]:
            with self.assertRaises(ValueError):
                public_updates({'quantity':value},'now')
        self.assertEqual(public_updates({'description':''},'now')['raw_description'],'')

    def test_refresh_includes_previously_fetched_and_retains_on_null(self):
        rows = {'code':{'ticket_id':'code','event_id':'event','public_event_id':'fire',
                       'status':'listing','observation_state':'active','details_fetched':'True',
                       'raw_description':'keep full'}}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(scraper,'DATA_DIR',temp), patch.object(scraper,'fetch_public_ticket',return_value=None), patch.object(scraper.time,'sleep'):
                scraper.enrich_ticket_details('group',rows,['code'])
            self.assertEqual(rows['code']['raw_description'],'keep full')
            self.assertTrue(list(Path(temp).glob('content_observation_*.jsonl')))

    def test_empty_public_description_is_full_not_a_failure(self):
        updates=public_updates({'description':'','quantity':2},'2026-09-27 02:00:00')
        self.assertEqual(updates['description_is_full'],'True')

    def test_recently_confirmed_inactive_is_not_abandoned_after_36_hours(self):
        row = {'ticket_id':'code','event_id':'event','status':'listing',
               'last_observed_at':'2026-09-01 00:00:00','state_checked_at':'2026-09-27 01:00:00',
               'observation_state':'inactive'}
        with patch.object(evidence,'fetch_public_ticket',return_value={
                'shareCode':'code','eventId':'fire','status':'inactive','quantity':4,
                'description':'full'}), patch.object(evidence.time,'sleep'):
            report=evidence.reconcile_public_listings({'code':row},{'code':row.copy()},set(),
                'fire','2026-09-27 02:00:00',{},[10],float('inf'))
        self.assertEqual(report['checked'],1)
        self.assertEqual(row['quantity'],4)


if __name__ == '__main__':
    unittest.main()
