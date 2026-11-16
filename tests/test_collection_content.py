import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from collection_content import apply_api_preview, apply_event_content, content_refresh_reason, public_updates
import collection_evidence as evidence
import scraper


class ContentTest(unittest.TestCase):
    def test_event_metadata_does_not_certify_or_overwrite_full_text(self):
        row = public_updates({'description': 'full text', 'seatType': 'orchestra'}, '2026-11-16 09:00:00')
        apply_event_content(row, {
            'description': 'full...', 'eventStartTime': '19:30', 'seatType': None,
            'tags': [{'label': 'QR'}, '同行'],
        })
        self.assertEqual(row['raw_description'], 'full text')
        self.assertEqual(row['description_checked_at'], '2026-11-16 09:00:00')
        self.assertEqual(row['ticket_tags'], 'QR / 同行')
        self.assertEqual(row['seat_type'], '"orchestra"')
        self.assertEqual(row['perf_time'], '19:30')

    def test_only_verified_recent_matching_text_is_reusable(self):
        row = public_updates({'description': 'starts here\nrest of full text'}, '2026-11-16 09:00:00')
        row.update(api_description='starts here...', last_observed_at='2026-11-16 10:00:00')
        now = datetime(2026, 11, 16, 11)
        self.assertIsNone(content_refresh_reason(row, now))
        self.assertEqual(content_refresh_reason(row, datetime(2026, 11, 17, 9)),
                         'expired_full_description')
        row['api_description'] = 'starts here'
        self.assertEqual(content_refresh_reason(row, now), 'public_description_preview_changed')
        row['api_description'] = 'changed...'
        self.assertEqual(content_refresh_reason(row, now), 'public_description_preview_changed')
        self.assertEqual(content_refresh_reason(row, datetime(2026, 11, 17, 9)),
                         'public_description_preview_changed')
        row['description_is_full'] = 'False'
        self.assertEqual(content_refresh_reason(row, now), 'unverified_full_description')

    def test_global_priority_saves_new_and_changed_before_old_group_rechecks(self):
        now = datetime(2026, 11, 16, 21)
        active = {'event_id':'event', 'public_event_id':'fire', 'status':'listing',
                  'observation_state':'active', 'last_observed_at':'2026-11-16 20:55:00'}
        old = {**active, 'ticket_id':'old',
               **public_updates({'description':'earlier verified'}, '2026-11-15 12:00:00')}
        changed = {**active, 'ticket_id':'changed', 'api_description':'updated conditions...',
                   **public_updates({'description':'previous conditions'}, '2026-11-15 12:00:00')}
        new = {**active, 'ticket_id':'new', 'raw_description':'preview...',
               'description_source':'event_api_preview'}
        checked = '2026-11-16 21:00:00'
        responses = {code: ({'shareCode':code, 'eventId':'fire', 'status':'active',
                             'description':text}, None, checked)
                     for code, text in [('changed','updated conditions in full'),
                                        ('new','full new description')]}
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(scraper, 'DATA_DIR', temp),
                  patch.object(scraper, 'query_public_listings', return_value=responses) as lookup,
                  patch.object(scraper, 'is_time_remaining', return_value=False),
                  patch.object(scraper, 'save_snapshots')):
                scraper.save_master('early', {'old':old})
                scraper.save_master('late', {'changed':changed})
                scraper.save_master('new-group', {'new':new})
                scraper.refresh_priority_content([{'name':'early'}, {'name':'late'}, {'name':'new-group'}],
                                                 now, {}, None)
                self.assertEqual(lookup.call_args.args[0], ['changed','new','old'])
                saved_old = scraper.load_master('early')['old']
                self.assertEqual(saved_old['description_checked_at'], '2026-11-15 12:00:00')
                self.assertEqual(saved_old['raw_description'], 'earlier verified')
                self.assertEqual(scraper.load_master('late')['changed']['raw_description'],
                                 'updated conditions in full')
                self.assertEqual(scraper.load_master('new-group')['new']['description_checked_at'], checked)

    def test_preview_does_not_overwrite_full_text_over_two_polls(self):
        row = {'raw_description':'complete conditions and seat information',
               'description_source':'public_detail','description_is_full':'True'}
        for preview in ['complete...', 'changed preview...']:
            apply_api_preview(row, {'description':preview})
            self.assertEqual(row['raw_description'], 'complete conditions and seat information')
            self.assertEqual(row['api_description'], preview)
            self.assertEqual(row['description_is_full'], 'True')

    def test_verified_empty_full_text_is_preserved(self):
        row = public_updates({'description':''}, 'now')
        apply_api_preview(row, {'description':'preview...'})
        self.assertEqual(row['raw_description'], '')
        self.assertEqual(row['description_is_full'], 'True')
        self.assertEqual(row['description_source'], 'public_detail')
        self.assertEqual(row['api_description'], 'preview...')

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
