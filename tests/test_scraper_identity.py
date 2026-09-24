import sys
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scraper


class ScraperIdentityTest(unittest.TestCase):
    def test_future_events_remain_in_poll_set_after_performer_link_disappears(self):
        master = {
            'future': {'status': 'listing', 'event_id': 'future-event',
                       'perf_date': '2026-12-01'},
            'past': {'status': 'listing', 'event_id': 'past-event',
                     'perf_date': '2026-08-01'},
        }
        self.assertEqual(
            scraper._events_to_poll(['current-event'], master,
                                    datetime(2026, 9, 24)),
            ['current-event', 'future-event'],
        )

    def test_missing_created_at_never_forms_a_shared_identity(self):
        self.assertIsNone(scraper._listing_identity_key({
            'event_id': 'event', 'created_at_unix': ''
        }))
        self.assertIsNone(scraper._ticket_match_key({
            'event_id': 'event', 'created_at_unix': '', 'price': '10000'
        }))
        self.assertIsNone(scraper._listing_identity_key({
            'event_id': 'event', 'created_at_unix': 'None'
        }))

    def test_missing_created_at_cannot_rekey_another_active_ticket(self):
        old = {'ticket_id': 'old', 'event_id': 'event',
               'created_at_unix': '', 'status': 'listing'}
        master = {'old': old}
        row, changed = scraper._rekey_active_listing(
            master, {'old': old}, {}, 'new', None
        )
        self.assertIsNone(row)
        self.assertFalse(changed)
        self.assertEqual(list(master), ['old'])

    def test_textual_none_created_at_does_not_collapse_master(self):
        rows = {
            code: {'ticket_id': code, 'event_id': 'event',
                   'created_at_unix': 'None', 'status': 'listing'}
            for code in ['a', 'b']
        }
        result, removed = scraper.canonicalize_master(rows)
        self.assertEqual(removed, 0)
        self.assertEqual(set(result), {'a', 'b'})

    def test_missing_created_at_does_not_hide_disappearance(self):
        old = {
            str(index): {'ticket_id': str(index), 'event_id': 'event',
                         'created_at_unix': '', 'status': 'listing',
                         'perf_date': '2026-12-01', 'price': '10000'}
            for index in range(20)
        }
        with self.assertRaises(scraper.ScrapeIntegrityError):
            scraper.validate_event_snapshot(
                'event', [{'status': 'active', 'shareCode': 'new',
                           'createdAt': None, 'pricePerTicket': '10000'}],
                old, old, datetime(2026, 9, 24),
            )

    def test_identity_uncertain_event_collects_current_active_without_deleting_old(self):
        old = {
            str(index): {'ticket_id': str(index), 'event_id': 'event',
                         'created_at_unix': '', 'status': 'listing',
                         'perf_date': '2026-12-01', 'price': '10000',
                         'first_observed_at': '2026-09-01 00:00:00',
                         'last_observed_at': '2026-09-01 00:00:00'}
            for index in range(20)
        }
        api = [{'status': 'active', 'shareCode': 'new',
                'createdAt': None, 'pricePerTicket': 10000,
                'eventDate': '2026-12-01'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'targets.json').write_text('["group"]', encoding='utf-8')
            with (patch.object(scraper, 'DATA_DIR', str(path)),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.object(scraper, 'get_events', return_value=['event']),
                  patch.object(scraper, 'get_event_id_from_slug', return_value='id'),
                  patch.object(scraper, 'fetch_all_tickets', return_value=api),
                  patch.object(scraper, 'load_master', return_value=old),
                  patch.object(scraper, 'save_snapshots'),
                  patch('builtins.print')):
                scraper.main()
            with (path / 'group_master.csv').open(encoding='utf-8-sig') as stream:
                import csv
                saved = {row['ticket_id']: row for row in csv.DictReader(stream)}
            self.assertEqual(len(saved), 21)
            self.assertEqual(saved['new']['status'], 'listing')
            self.assertTrue(all(saved[str(index)]['status'] == 'listing'
                                for index in range(20)))
            log = next(path.glob('observation_*.jsonl'))
            record = json.loads(log.read_text(encoding='utf-8').strip())
            self.assertTrue(record['identity_uncertain'])
            self.assertFalse(record['complete'])
            self.assertFalse(record['absence_classification_complete'])

    def test_sold_without_identifier_cannot_match_by_price(self):
        row = {'ticket_id': 'active', 'event_id': 'event',
               'created_at_unix': '', 'price': '10000'}
        self.assertIsNone(scraper._sold_match(
            {'status': 'sold', 'createdAt': None,
             'pricePerTicket': '10000'},
            'event', {'active': row}, {}, {},
        ))
        self.assertIs(scraper._sold_match(
            {'status': 'sold', 'shareCode': 'active',
             'createdAt': None, 'pricePerTicket': '10000'},
            'event', {'active': row}, {}, {},
        ), row)

    def test_api_pass_keeps_distinct_listings_without_created_at(self):
        responses = [
            [{'status': 'active', 'shareCode': code,
              'createdAt': None, 'pricePerTicket': 10000,
              'eventDate': '2026-12-01'} for code in ['a', 'b']],
            [{'status': 'active', 'shareCode': 'a',
              'createdAt': None, 'pricePerTicket': 10000,
              'eventDate': '2026-12-01'},
             {'status': 'sold', 'shareCode': 'b',
              'createdAt': None, 'pricePerTicket': 10000,
              'eventDate': '2026-12-01'}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'targets.json').write_text('["group"]', encoding='utf-8')
            with (patch.object(scraper, 'DATA_DIR', str(path)),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.object(scraper, 'get_events', return_value=['event']),
                  patch.object(scraper, 'get_event_id_from_slug', return_value='id'),
                  patch.object(scraper, 'fetch_all_tickets', side_effect=responses),
                  patch.object(scraper, 'save_snapshots'),
                  patch('builtins.print')):
                scraper.main()
                scraper.main()
            with (path / 'group_master.csv').open(encoding='utf-8-sig') as stream:
                import csv
                saved = {row['ticket_id']: row for row in csv.DictReader(stream)}
            self.assertEqual(set(saved), {'a', 'b'})
            self.assertEqual(saved['a']['status'], 'listing')
            self.assertEqual(saved['b']['status'], 'sold')
            self.assertEqual(saved['b']['sold_at_source'], 'transition_observed')
            log = list(path.glob('observation_*.jsonl'))
            self.assertEqual(len(log), 1)
            records = [json.loads(line) for line in log[0].read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records[-1]['missing_created_at_count'], 2)
            self.assertEqual(records[-1]['unresolved_sold_count'], 0)

    def test_anonymous_sold_does_not_mark_another_listing_deleted(self):
        responses = [
            [{'status': 'active', 'shareCode': code,
              'createdAt': None, 'pricePerTicket': 10000,
              'eventDate': '2026-12-01'} for code in ['a', 'b']],
            [{'status': 'active', 'shareCode': 'a',
              'createdAt': None, 'pricePerTicket': 10000,
              'eventDate': '2026-12-01'},
             {'status': 'sold', 'createdAt': None,
              'pricePerTicket': 10000, 'eventDate': '2026-12-01'}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'targets.json').write_text('["group"]', encoding='utf-8')
            with (patch.object(scraper, 'DATA_DIR', str(path)),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.object(scraper, 'get_events', return_value=['event']),
                  patch.object(scraper, 'get_event_id_from_slug', return_value='id'),
                  patch.object(scraper, 'fetch_all_tickets', side_effect=responses),
                  patch.object(scraper, 'save_snapshots'),
                  patch('builtins.print')):
                scraper.main()
                scraper.main()
            with (path / 'group_master.csv').open(encoding='utf-8-sig') as stream:
                import csv
                saved = {row['ticket_id']: row for row in csv.DictReader(stream)}
            self.assertEqual(saved['b']['status'], 'listing')
            self.assertEqual(len(saved), 2)
            log = next(path.glob('observation_*.jsonl'))
            latest = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
            self.assertEqual(latest['unresolved_sold_count'], 1)
            self.assertFalse(latest['absence_classification_complete'])
            inventory = path / scraper.ANONYMOUS_SOLD_INVENTORY
            record = json.loads(inventory.read_text(encoding='utf-8').strip())
            self.assertEqual(record['identity_source'], 'anonymous_sold_api')
            self.assertIsNone(record['sold_at'])
            self.assertEqual(record['ticket']['pricePerTicket'], 10000)

    def test_anonymous_sold_inventory_preserves_multiplicity_without_duplication(self):
        sold = {'status': 'sold', 'pricePerTicket': 12000,
                'description': 'same\u2028listing', 'eventDate': '2026-12-01'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(scraper, 'DATA_DIR', directory):
                scraper.save_anonymous_sold_inventory(
                    {'event': [sold, sold]}, '2026-09-24 01:00:00'
                )
                scraper.save_anonymous_sold_inventory(
                    {'event': [sold]}, '2026-09-24 02:00:00'
                )
            lines = (Path(directory) / scraper.ANONYMOUS_SOLD_INVENTORY).read_text(
                encoding='utf-8'
            ).splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record['max_observed_count'], 2)
            self.assertEqual(record['first_observed_at'], '2026-09-24 01:00:00')
            self.assertEqual(record['last_observed_at'], '2026-09-24 02:00:00')

    def test_market_snapshot_separates_stale_from_current_listings(self):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rows = {
            'fresh': {'ticket_id': 'fresh', 'event_id': 'event',
                      'perf_date': '2026-12-01', 'perf_time': '18:00',
                      'venue': 'venue', 'status': 'listing', 'price': '12000',
                      'first_observed_at': now, 'last_observed_at': now},
            'stale': {'ticket_id': 'stale', 'event_id': 'event',
                      'perf_date': '2026-12-01', 'perf_time': '18:00',
                      'venue': 'venue', 'status': 'listing', 'price': '9000',
                      'first_observed_at': now,
                      'last_observed_at': '2026-01-01 00:00:00'},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (patch.object(scraper, 'SNAPSHOT_DIR', str(root / 'snapshot')),
                  patch.object(scraper, 'MARKET_DIR', str(root / 'market'))):
                scraper.save_snapshots('group', rows)
            import csv
            with next((root / 'market').glob('*.csv')).open(
                encoding='utf-8-sig', newline=''
            ) as stream:
                record = next(csv.DictReader(stream))
            self.assertEqual(record['active_tickets'], '2')
            self.assertEqual(record['current_active_tickets'], '1')
            self.assertEqual(record['current_avg_price'], '12000.0')

    def test_existing_rotations_are_collapsed_and_details_are_preserved(self):
        old = {
            "ticket_id": "old", "event_id": "event", "created_at_unix": "1",
            "status": "deleted", "last_observed_at": "2026-08-28 02:00:00",
            "seller_name": "seller", "details_fetched": "True",
        }
        current = {
            "ticket_id": "current", "event_id": "event", "created_at_unix": "1",
            "status": "listing", "last_observed_at": "2026-08-28 02:00:00",
            "seller_name": "", "details_fetched": "False",
        }
        result, removed = scraper.canonicalize_master({"old": old, "current": current})
        self.assertEqual(removed, 1)
        self.assertEqual(list(result), ["current"])
        self.assertEqual(result["current"]["status"], "listing")
        self.assertEqual(result["current"]["seller_name"], "seller")
        self.assertEqual(result["current"]["details_fetched"], "True")

    def test_rotated_share_code_is_not_an_unexplained_disappearance(self):
        old = {
            "ticket_id": "old-code", "event_id": "event",
            "created_at_unix": "123", "price": "10000",
            "status": "listing", "perf_date": "2026-09-01",
        }
        tickets = [{
            "status": "active", "shareCode": "new-code",
            "createdAt": "123", "pricePerTicket": "12000",
        }]
        active = scraper.validate_event_snapshot(
            "event", tickets, {"old-code": old}, {"old-code": old},
            datetime(2026, 8, 28),
        )
        self.assertEqual(active, {"new-code"})

    def test_duplicate_created_at_in_api_is_rejected_before_updates(self):
        tickets = [
            {'status': 'active', 'shareCode': code,
             'createdAt': '123', 'pricePerTicket': 10000}
            for code in ['a', 'b']
        ]
        with self.assertRaises(scraper.ScrapeIntegrityError):
            scraper.validate_event_snapshot(
                'event', tickets, {}, {}, datetime(2026, 9, 24)
            )

    def test_rekey_preserves_row_and_removes_old_master_key(self):
        row = {
            "ticket_id": "old-code", "event_id": "event",
            "created_at_unix": "123", "status": "listing",
            "seller_name": "preserved seller",
        }
        master = {"old-code": row}
        by_share = {"old-code": row}
        identity = scraper._listing_identity_key(row)
        by_identity = {identity: row}

        result, changed = scraper._rekey_active_listing(
            master, by_share, by_identity, "new-code", identity
        )

        self.assertTrue(changed)
        self.assertIs(result, row)
        self.assertNotIn("old-code", master)
        self.assertIs(master["new-code"], row)
        self.assertEqual(row["seller_name"], "preserved seller")

    def test_new_share_code_after_sale_does_not_erase_sold_history(self):
        sold = {'ticket_id': 'old', 'event_id': 'event',
                'created_at_unix': '123', 'status': 'sold'}
        master = {'old': sold}
        identity = scraper._listing_identity_key(sold)
        row, changed = scraper._rekey_active_listing(
            master, {'old': sold}, {identity: sold}, 'new', identity
        )
        self.assertIsNone(row)
        self.assertFalse(changed)
        self.assertEqual(master['old']['status'], 'sold')

    def test_same_share_code_can_be_reactivated(self):
        sold = {'ticket_id': 'same', 'event_id': 'event',
                'created_at_unix': '123', 'status': 'sold'}
        row, changed = scraper._rekey_active_listing(
            {'same': sold}, {'same': sold}, {}, 'same', None
        )
        self.assertIs(row, sold)
        self.assertFalse(changed)
        self.assertEqual(scraper.validate_event_snapshot(
            'event', [{'status': 'active', 'shareCode': 'same'}],
            {}, {'same': sold}, datetime(2026, 9, 24),
        ), {'same'})

    def test_missing_api_created_at_preserves_existing_identity(self):
        response = [{
            'status': 'active', 'shareCode': 'same',
            'createdAt': None, 'pricePerTicket': 10000,
            'eventDate': '2026-12-01',
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'targets.json').write_text('["group"]', encoding='utf-8')
            existing = {
                'ticket_id': 'same', 'event_id': 'event',
                'created_at_unix': '123', 'status': 'listing',
                'perf_date': '2026-12-01', 'price': '10000',
            }
            with (patch.object(scraper, 'DATA_DIR', str(path)),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.object(scraper, 'get_events', return_value=['event']),
                  patch.object(scraper, 'get_event_id_from_slug', return_value='id'),
                  patch.object(scraper, 'fetch_all_tickets', return_value=response),
                  patch.object(scraper, 'load_master', return_value={'same': existing}),
                  patch.object(scraper, 'save_snapshots'),
                  patch('builtins.print')):
                scraper.main()
            self.assertEqual(existing['created_at_unix'], '123')


if __name__ == "__main__":
    unittest.main()
