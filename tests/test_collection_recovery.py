import io
import json
import tempfile
import threading
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import collection_evidence as evidence
import scraper
from tools import check_collection_coverage


def http_error(code, **headers):
    return urllib.error.HTTPError('https://example.invalid', code, 'test', headers, None)


def response(**payload):
    return io.BytesIO(json.dumps(payload).encode())


class PublicRequestRecoveryTest(unittest.TestCase):
    def test_permanent_http_errors_get_one_attempt_and_log_the_status(self):
        for status in (401, 403, 404):
            with self.subTest(status=status), patch.object(
                    evidence.urllib.request, 'urlopen', side_effect=http_error(status)) as request:
                with self.assertRaises(urllib.error.HTTPError) as failure:
                    evidence.fetch_public_ticket('code')
                request.assert_called_once()
                self.assertEqual(evidence.public_error_label(failure.exception), f'HTTP {status}')

    def test_429_cooldown_is_shared_and_can_defer_another_listing(self):
        policy = evidence.PublicRequestPolicy()
        clock = [100.0]

        def sleep(seconds):
            clock[0] += seconds

        with (patch.object(evidence.time, 'monotonic', side_effect=lambda: clock[0]),
              patch.object(evidence.time, 'sleep', side_effect=sleep) as wait,
              patch.object(evidence.urllib.request, 'urlopen', side_effect=[
                  http_error(429, **{'Retry-After': '45'}),
                  response(success=True, ticket={'shareCode': 'recovered'})
              ]) as request):
            with self.assertRaises(evidence.PublicLookupDeferred):
                evidence.fetch_public_ticket('rate-limited', policy=policy, deadline=140)
            with self.assertRaises(evidence.PublicLookupDeferred):
                evidence.fetch_public_ticket('no-room-left', policy=policy, deadline=140)
            self.assertEqual(request.call_count, 1)
            wait.assert_not_called()
            ticket = evidence.fetch_public_ticket('recovered', policy=policy, deadline=150)
            self.assertEqual(ticket['shareCode'], 'recovered')
            wait.assert_called_once_with(45)
            self.assertEqual(request.call_args.kwargs['timeout'], 5)

    def test_workers_share_spacing_and_deadline(self):
        clock = [100.0]

        def sleep(seconds):
            clock[0] += seconds

        policy = evidence.PublicRequestPolicy(min_interval=2.5)
        with (patch.object(evidence.time, 'monotonic', side_effect=lambda: clock[0]),
              patch.object(evidence.time, 'sleep', side_effect=sleep) as wait):
            policy.before_request(110)
            with self.assertRaises(evidence.PublicLookupDeferred):
                policy.before_request(102)
            policy.before_request(110)
            wait.assert_called_once_with(2.5)
            self.assertEqual(clock[0], 102.5)

    def test_normal_pass_allows_two_workers_with_a_half_second_wait(self):
        both_workers = threading.Barrier(2, timeout=3)

        def lookup(code, **kwargs):
            both_workers.wait()
            return {'shareCode': code}

        with (patch.object(evidence, 'fetch_public_ticket', side_effect=lookup),
              patch.object(evidence.time, 'sleep') as pause):
            results = evidence.query_public_listings(
                ['one', 'two'], {}, [2], float('inf'), evidence.PublicRequestPolicy())
        self.assertEqual([results[code][0]['shareCode'] for code in ['one', 'two']], ['one', 'two'])
        self.assertEqual([call.args[0] for call in pause.call_args_list], [0.5, 0.5])

    def test_single_throttle_can_recover_and_cooldown_survives_phase_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cooldown'
            clock = [100.0]

            def sleep(seconds):
                clock[0] += seconds

            with (patch.object(evidence.time, 'monotonic', side_effect=lambda: clock[0]),
                  patch.object(evidence.time, 'time', side_effect=lambda: clock[0] + 10_000),
                  patch.object(evidence.time, 'sleep', side_effect=sleep) as wait,
                  patch.object(evidence.urllib.request, 'urlopen', side_effect=[
                      http_error(429, **{'Retry-After': '30'}),
                      http_error(429, **{'Retry-After': '10'}),
                      response(success=True, ticket={'shareCode': 'code'}),
                  ]) as request):
                policy = evidence.PublicRequestPolicy(path)
                with self.assertRaises(evidence.PublicLookupDeferred):
                    evidence.fetch_public_ticket('previous', policy=policy, deadline=120)
                policy.save_cooldown()
                policy = evidence.PublicRequestPolicy(path)
                ticket = evidence.fetch_public_ticket('code', policy=policy, deadline=180)
                self.assertEqual(ticket['shareCode'], 'code')
                self.assertEqual(request.call_count, 3)
                self.assertEqual([call.args[0] for call in wait.call_args_list], [30, 10])
                with self.assertRaises(evidence.PublicLookupDeferred):
                    policy.before_request(clock[0] + 4)

    def test_repeated_access_denials_stop_other_requests(self):
        policy = evidence.PublicRequestPolicy()
        with patch.object(evidence.urllib.request, 'urlopen',
                          side_effect=[http_error(403) for _ in range(4)]) as request:
            for _ in range(4):
                with self.assertRaises(urllib.error.HTTPError):
                    evidence.fetch_public_ticket('code', policy=policy)
            with self.assertRaises(evidence.PublicLookupDeferred):
                evidence.fetch_public_ticket('another', policy=policy)
            self.assertEqual(request.call_count, 4)

    def test_mid_run_access_denial_after_success_defers_and_carries_across_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            cooldown = Path(directory) / 'cooldown'
            policy = evidence.PublicRequestPolicy(cooldown)
            rows = {
                'ok': {'ticket_id': 'ok', 'event_id': 'e1', 'status': 'listing',
                       'last_observed_at': '2026-11-16 10:00:00'},
                'throttled': {'ticket_id': 'throttled', 'event_id': 'e1', 'status': 'listing',
                              'last_observed_at': '2026-11-16 09:00:00'},
            }
            prior = {k: v.copy() for k, v in rows.items()}
            with (patch.object(evidence.urllib.request, 'urlopen', side_effect=[
                      response(success=True, ticket={'shareCode': 'ok', 'eventId': 'fire', 'status': 'sold'}),
                      http_error(403),
                  ]) as request,
                  patch.object(evidence.time, 'sleep')):
                counts = evidence.reconcile_public_listings(
                    rows, prior, set(), 'fire', '2026-11-16 11:00:00',
                    {}, [2], float('inf'), policy=policy,
                )
            self.assertEqual(request.call_count, 2)
            self.assertEqual(counts['checked'], 1)
            self.assertEqual(counts['sold'], 1)
            self.assertEqual(counts['failed'], 0)
            self.assertEqual(counts['pending'], 1)
            self.assertEqual(counts.get('request_errors'), {'HTTP 403': 1})
            policy.save_cooldown()
            next_phase = evidence.PublicRequestPolicy(cooldown)
            self.assertTrue(next_phase.has_succeeded)
            with self.assertRaises(evidence.PublicLookupDeferred):
                next_phase.before_request(evidence.time.monotonic() + 480)

    def test_invalid_json_shape_is_not_a_successful_null_ticket(self):
        policy = evidence.PublicRequestPolicy()
        with patch.object(evidence.urllib.request, 'urlopen', side_effect=[
                response(success=False, ticket=None),
                response(success=True, ticket=None),
                http_error(403),
        ]):
            with self.assertRaises(ValueError):
                evidence.fetch_public_ticket('code', policy=policy)
            self.assertIsNone(evidence.fetch_public_ticket('code', policy=policy))
            with self.assertRaises(urllib.error.HTTPError):
                evidence.fetch_public_ticket('code', policy=policy)
        self.assertFalse(policy.has_succeeded)

    def test_authentication_error_is_hard_after_valid_response(self):
        policy = evidence.PublicRequestPolicy()
        with patch.object(evidence.urllib.request, 'urlopen', side_effect=[
                response(success=True, ticket={'shareCode': 'ok', 'eventId': 'fire', 'status': 'active'}),
                http_error(401),
        ]):
            evidence.fetch_public_ticket('ok', policy=policy)
            with self.assertRaises(urllib.error.HTTPError):
                evidence.fetch_public_ticket('denied', policy=policy)

    def test_recovery_retries_one_mid_run_denial_after_cooldown(self):
        policy = evidence.PublicRequestPolicy(retry_access_denial=True)
        clock = [100.0]
        ticket = {'shareCode': 'code', 'eventId': 'fire', 'status': 'active'}

        def sleep(seconds):
            clock[0] += seconds

        with (patch.object(evidence.time, 'monotonic', side_effect=lambda: clock[0]),
              patch.object(evidence.time, 'sleep', side_effect=sleep) as wait,
              patch.object(evidence.urllib.request, 'urlopen', side_effect=[
                  response(success=True, ticket=ticket), http_error(403),
                  response(success=True, ticket=ticket),
              ]) as request):
            evidence.fetch_public_ticket('already-valid', policy=policy, deadline=1100)
            self.assertEqual(evidence.fetch_public_ticket('code', policy=policy, deadline=1100), ticket)
            self.assertEqual(request.call_count, 3)
            wait.assert_called_once_with(900)

    def test_server_cooldown_survives_queued_run_without_inheriting_success(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / 'current'
            durable = Path(directory) / 'durable'
            clock = [100.0]
            with (patch.object(evidence.time, 'monotonic', side_effect=lambda: clock[0]),
                  patch.object(evidence.time, 'time', side_effect=lambda: clock[0] + 10_000)):
                policy = evidence.PublicRequestPolicy(stage, server_cooldown_path=durable)
                policy.record_response(200)
                policy.record_response(403)
                policy.save_cooldown()
                following_phase = evidence.PublicRequestPolicy(stage, server_cooldown_path=durable)
                with self.assertRaises(evidence.PublicLookupDeferred):
                    following_phase.before_request(2000)
                following_run = evidence.PublicRequestPolicy(server_cooldown_path=durable)
                self.assertFalse(following_run.has_succeeded)
                with self.assertRaises(evidence.PublicLookupDeferred):
                    following_run.before_request(999)
                clock[0] = 1000
                self.assertIsInstance(following_run.before_request(1100), (int, float))

    def test_historical_unverified_rows_are_reported_outside_the_recent_lookup_budget(self):
        now = '2026-11-17 12:00:00'
        rows = {
            'old': {'status': 'listing', 'observation_state': 'absent_unverified',
                    'last_observed_at': '2026-11-15 12:00:00'},
            **{f'new-{i}': {'status': 'listing', 'observation_state': 'active',
                            'last_observed_at': '2026-11-17 11:00:00'} for i in range(22)}
        }
        counts, candidates = evidence.prepare_public_listings(
            rows, {code: row.copy() for code, row in rows.items()}, set(), now)
        self.assertEqual(counts['historical_unresolved'], 1)
        self.assertEqual({code for code, _ in candidates}, rows.keys() - {'old'})

    def test_public_lookup_deadline_reserves_time_for_workflow_finalization(self):
        with (patch.dict(scraper.os.environ, {'PUBLIC_RUN_DEADLINE_EPOCH': '1200'}),
              patch.object(scraper.time, 'time', return_value=1000),
              patch.object(scraper.time, 'monotonic', return_value=500)):
            self.assertEqual(scraper.public_deadline_after(12), 700)
            self.assertEqual(scraper.public_deadline_after(1), 560)

    def test_valid_same_event_alias_chain_is_flattened_without_merging_rows(self):
        rows = {
            'a': {'event_id': 'e1', 'canonical_ticket_id': 'b', 'price': '10'},
            'b': {'event_id': 'e1', 'canonical_ticket_id': 'c', 'price': '20'},
            'c': {'event_id': 'e1', 'canonical_ticket_id': 'c', 'price': '30'},
            'other': {'event_id': 'e2', 'canonical_ticket_id': 'a', 'price': '40'},
            'cycle1': {'event_id': 'e1', 'canonical_ticket_id': 'cycle2', 'price': '50'},
            'cycle2': {'event_id': 'e1', 'canonical_ticket_id': 'cycle1', 'price': '60'},
        }
        evidence.flatten_canonical_links(rows)
        self.assertEqual(rows['a']['canonical_ticket_id'], 'c')
        self.assertEqual(rows['other']['canonical_ticket_id'], 'a')
        self.assertEqual(rows['cycle1']['canonical_ticket_id'], 'cycle2')
        self.assertEqual(rows['cycle2']['canonical_ticket_id'], 'cycle1')
        self.assertEqual([row['price'] for row in rows.values()], ['10', '20', '30', '40', '50', '60'])


class CoverageRecoveryTest(unittest.TestCase):
    def test_event_discovery_accepts_same_site_absolute_and_relative_links(self):
        page = '''
            <a href="/events/relative">Relative</a>
            <a href="https://ticketen.jp/events/absolute">Absolute</a>
            <a href="/events/absolute/?round=1#list">Duplicate</a>
            <a href="//ticketen.jp/events/other">Same site</a>
            <a href="https://other.example/events/foreign">Foreign</a>
            <a href="https://ticketen.jp.example/events/fake">Fake host</a>
            <a href="https://[bad">Malformed</a>
            <a href="/events/relative/day/1">Schedule</a>
        '''
        with patch.object(scraper, 'fetch_html', return_value=page):
            self.assertEqual(scraper.get_events('group'), ['relative', 'absolute', 'other'])

    def run_empty_event(self, active_response):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'targets.json').write_text('["group"]', encoding='utf-8')
            path = folder / 'coverage.json'
            now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            old_ticket = {
                'ticket_id': 'old', 'canonical_ticket_id': 'old', 'event_id': 'event',
                'status': 'listing', 'observation_state': 'active', 'last_observed_at': now,
                'raw_description': 'retain this', 'price': '2000',
            }
            with (patch.object(scraper, 'DATA_DIR', directory),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.dict(scraper.os.environ, {'COLLECTION_COVERAGE_PATH': str(path),
                                                   'PUBLIC_TICKET_CACHE_PATH': ''}),
                  patch.object(scraper, 'get_events', return_value=['event']),
                  patch.object(scraper, 'load_run_cache') as old_cache,
                  patch.object(scraper, 'get_event_id_from_slug', return_value='fire-id'),
                  patch.object(scraper, 'fetch_all_tickets', side_effect=[[], active_response]) as fetch,
                  patch.object(evidence, 'fetch_public_ticket', return_value=None),
                  patch.object(evidence.time, 'sleep'),
                  patch.object(scraper, 'save_snapshots')):
                scraper.save_master('group', {'old': old_ticket})
                scraper.main()
                saved = scraper.load_master('group')['old']
            old_cache.assert_not_called()
            self.assertEqual(fetch.call_args_list[1].kwargs, {'active_only': True})
            observation = json.loads(next(folder.glob('observation_*.jsonl')).read_text(encoding='utf-8'))
            return saved, observation, json.loads(path.read_text(encoding='utf-8'))

    def test_confirmed_empty_inventory_counts_as_fetched_but_not_as_sale(self):
        saved, observation, coverage = self.run_empty_event([])
        self.assertEqual(coverage, {'expected_events': 1, 'fetched_events': 1, 'issues': []})
        self.assertTrue(observation['api_fetch_complete'])
        self.assertTrue(observation['empty_inventory_confirmed'])
        self.assertFalse(observation['absence_classification_complete'])
        self.assertEqual(saved['observation_state'], 'absent_unknown')
        self.assertEqual(saved['status'], 'listing')
        self.assertEqual(saved['raw_description'], 'retain this')

    def test_inconsistent_empty_inventory_keeps_stored_state_and_flags_coverage(self):
        saved, observation, coverage = self.run_empty_event([{'status': 'active'}])
        self.assertFalse(observation['api_fetch_complete'])
        self.assertEqual(coverage['fetched_events'], 0)
        self.assertEqual(coverage['issues'][0]['reason'], 'empty_inventory_unverified')
        self.assertEqual(saved['observation_state'], 'active')

    def test_url_alias_retains_stored_event_and_fetches_inventory_once(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'targets.json').write_text('["group"]', encoding='utf-8')
            coverage = folder / 'coverage.json'
            old = {
                'ticket_id': 'ticket', 'event_id': 'old-url', 'public_event_id': 'fire-id',
                'status': 'listing', 'observation_state': 'active', 'perf_date': '2099-02-01',
                'first_observed_at': '2026-11-16 08:00:00',
                'last_observed_at': '2026-11-16 08:00:00',
            }
            ticket = {
                'status': 'active', 'eventId': 'fire-id', 'shareCode': 'ticket',
                'pricePerTicket': 2500, 'quantity': 1,
            }
            with (patch.object(scraper, 'DATA_DIR', directory),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.dict(scraper.os.environ, {'COLLECTION_COVERAGE_PATH': str(coverage),
                                                   'PUBLIC_TICKET_CACHE_PATH': ''}),
                  patch.object(scraper, 'get_events', return_value=['new-url', 'old-url']),
                  patch.object(scraper, 'get_event_id_from_slug', return_value='fire-id'),
                  patch.object(scraper, 'fetch_all_tickets', return_value=[ticket]) as fetch,
                  patch.object(scraper, 'save_snapshots')):
                scraper.save_master('group', {'ticket': old})
                scraper.main()
                saved = scraper.load_master('group')
            fetch.assert_called_once_with('fire-id')
            self.assertEqual(set(saved), {'ticket'})
            self.assertEqual(saved['ticket']['event_id'], 'old-url')
            self.assertEqual(saved['ticket']['first_observed_at'], old['first_observed_at'])
            self.assertEqual(saved['ticket']['price'], '2500')
            self.assertEqual(json.loads(coverage.read_text(encoding='utf-8')),
                             {'expected_events':1, 'fetched_events':1, 'issues':[]})

    def test_different_public_event_identity_still_rejects_share_code_collision(self):
        event = {'event_id':'other-url','public_event_id':'other-fire'}
        ticket = {'status':'active','shareCode':'ticket','eventId':'current-fire'}
        with self.assertRaisesRegex(scraper.ScrapeIntegrityError, 'belongs to another event'):
            scraper.validate_event_snapshot('current-url', [ticket], {}, {'ticket':event},
                                            datetime.now(), public_event_id='current-fire')

    def test_false_success_or_empty_listing_without_zero_total_is_rejected(self):
        for success, total in [(False, 0), (True, 3), (True, None)]:
            with self.subTest(success=success, total=total), patch.object(
                    scraper.urllib.request, 'urlopen',
                    return_value=response(success=success, tickets=[], hasMore=False, totalCount=total)):
                with self.assertRaises(scraper.ScrapeIntegrityError):
                    scraper.fetch_all_tickets('event', active_only=True)

    def test_api_prioritizes_new_absences_in_later_performer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'targets.json').write_text('["early","late"]', encoding='utf-8')
            seen_at = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            common = {'event_id':'event', 'public_event_id':'fire-id', 'status':'listing',
                      'last_observed_at':seen_at, 'perf_date':'2099-01-01'}
            old = {**common, 'ticket_id':'old', 'observation_state':'absent_unverified',
                   'absence_first_observed_at':seen_at}
            sale = {**common, 'ticket_id':'sale', 'observation_state':'active'}
            cancelled = {**common, 'ticket_id':'cancelled', 'observation_state':'active'}

            def lookup(code, **kwargs):
                return {'shareCode':code, 'eventId':'fire-id',
                        'status':{'sale':'sold', 'cancelled':'cancelled'}[code]}

            with (patch.object(scraper, 'DATA_DIR', directory),
                  patch.object(scraper, 'SCRAPE_MODE', 'api'),
                  patch.dict(scraper.os.environ, {'PUBLIC_STATUS_CHECK_LIMIT':'2',
                                                   'PUBLIC_STATUS_RUNTIME_MINUTES':'5',
                                                   'PUBLIC_TICKET_CACHE_PATH':'',
                                                   'PUBLIC_API_COOLDOWN_PATH':''}),
                  patch.object(scraper, 'get_events', return_value=['event']),
                  patch.object(scraper, 'get_event_id_from_slug', return_value='fire-id'),
                  patch.object(scraper, 'fetch_all_tickets', return_value=[
                      {'status':'active', 'shareCode':'still-live', 'eventId':'fire-id',
                       'pricePerTicket':3000, 'quantity':1}]),
                  patch.object(evidence, 'fetch_public_ticket', side_effect=lookup) as request,
                  patch.object(evidence.time, 'sleep'),
                  patch.object(scraper, 'save_snapshots')):
                scraper.save_master('early', {'old':old})
                scraper.save_master('late', {'sale':sale, 'cancelled':cancelled})
                scraper.main()
                early, late = scraper.load_master('early'), scraper.load_master('late')
            self.assertCountEqual([call.args[0] for call in request.call_args_list], ['sale','cancelled'])
            self.assertEqual(early['old']['observation_state'], 'absent_unverified')
            self.assertEqual(late['sale']['observation_state'], 'sold_confirmed')
            self.assertEqual(late['cancelled']['observation_state'], 'deleted_confirmed')
            polls = [json.loads(line) for line in next(folder.glob('observation_*.jsonl')).read_text(
                encoding='utf-8').splitlines()]
            counts = {r['performer']:r['public_status_checks'] for r in polls}
            self.assertEqual(counts['early']['pending'], 1)
            self.assertEqual(counts['late']['newly_missing'], 2)
            self.assertEqual(counts['late']['newly_missing_pending'], 0)

    def test_prior_unavailable_evidence_survives_a_deferred_recheck(self):
        checked_at = '2026-11-16 10:10:00'
        row = {'status':'listing','observation_state':'inactive','last_observed_at':checked_at,
               'state_checked_at':checked_at,'absence_first_observed_at':checked_at}
        counts, candidates = evidence.prepare_public_listings({'code':row}, {'code':row.copy()},
                                                               set(), '2026-11-16 11:00:00')
        self.assertEqual(candidates, [('code', False)])
        self.assertEqual(counts['pending'], 1)
        self.assertEqual(row['observation_state'], 'inactive')
        self.assertEqual(row['state_checked_at'], checked_at)

    def test_unverified_absences_precede_verified_inactive_and_respect_cooldown(self):
        rows = {
            'recent_inactive': {
                'status': 'listing', 'observation_state': 'inactive',
                'last_observed_at': '2026-11-16 10:50:00',
                'state_checked_at': '2026-11-16 10:55:00',
                'absence_first_observed_at': '2026-11-16 10:55:00',
            },
            'due_inactive_newer_check': {
                'status': 'listing', 'observation_state': 'inactive',
                'last_observed_at': '2026-11-16 04:00:00',
                'state_checked_at': '2026-11-16 04:30:00',
                'absence_first_observed_at': '2026-11-16 04:30:00',
            },
            'due_inactive_older_check': {
                'status': 'listing', 'observation_state': 'inactive',
                'last_observed_at': '2026-11-16 03:00:00',
                'state_checked_at': '2026-11-16 03:30:00',
                'absence_first_observed_at': '2026-11-16 03:30:00',
            },
            'unverified': {
                'status': 'listing', 'observation_state': 'absent_unverified',
                'last_observed_at': '2026-11-16 02:00:00',
                'state_checked_at': '2026-11-16 02:00:00',
                'absence_first_observed_at': '2026-11-16 02:30:00',
            },
            'newly_missing': {
                'status': 'listing', 'observation_state': 'active',
                'last_observed_at': '2026-11-16 10:00:00',
                'state_checked_at': '2026-11-16 10:00:00',
                'absence_first_observed_at': '',
            },
        }
        prior = {k: v.copy() for k, v in rows.items()}
        counts, candidates = evidence.prepare_public_listings(
            rows, prior, set(), '2026-11-16 11:00:00', recheck_after=timedelta(hours=6)
        )
        self.assertEqual(
            [code for code, _ in candidates],
            ['newly_missing', 'unverified', 'due_inactive_older_check', 'due_inactive_newer_check'],
        )
        self.assertEqual(counts['pending'], 4)
        self.assertEqual(rows['recent_inactive']['observation_state'], 'inactive')

    def test_state_priority_counts_real_attempts_and_ignores_pending_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'observation_20261116.jsonl'
            items = [
                {'performer': 'group', 'event_id': 'attempted', 'public_status_checks': {'checked': 1},
                 'observed_at_utc': '2026-11-16T01:20:00+00:00'},
                {'performer': 'group', 'event_id': 'attempted', 'public_status_checks': {'pending': 5},
                 'observed_at_utc': '2026-11-16T01:21:00+00:00'},
                {'performer': 'another', 'event_id': 'only_pending', 'public_status_checks': {'pending': 8},
                 'observed_at_utc': '2026-11-16T01:22:00+00:00'},
            ]
            path.write_text(''.join(json.dumps(item) + '\n' for item in reversed(items)), encoding='utf-8')
            with patch.object(scraper, 'DATA_DIR', directory):
                self.assertEqual(scraper.public_state_attempts(),
                                 {'group': {'attempted': '2026-11-16T01:20:00+00:00'}})

    def test_unvisited_group_precedes_recently_attempted_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'content_observation_20261116.jsonl'
            items = [
                {'performer': 'attempted', 'scope': 'observed_active', 'failed': 15,
                 'observed_at_utc': '2026-11-16T01:20:00+00:00'},
                {'performer': 'only_pending', 'scope': 'observed_active', 'pending': 5,
                 'observed_at_utc': '2026-11-16T01:21:00+00:00'},
            ]
            path.write_text(''.join(json.dumps(item) + '\n' for item in items), encoding='utf-8')
            with patch.object(scraper, 'DATA_DIR', directory):
                ordered = scraper.detail_refresh_order([{'name': 'attempted'}, {'name': 'only_pending'}])
            self.assertEqual(ordered[0]['name'], 'only_pending')

    def test_incomplete_coverage_fails_after_the_save_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'coverage.json'
            path.write_text(json.dumps({'expected_events': 2, 'fetched_events': 1,
                                        'issues': [{'reason': 'event_poll_failed'}]}), encoding='utf-8')
            with (patch('sys.argv', ['check_collection_coverage.py', str(path)]),
                  patch.dict(check_collection_coverage.os.environ, {'GITHUB_STEP_SUMMARY': ''}),
                  patch('builtins.print') as printed,
                  self.assertRaises(SystemExit) as failure):
                check_collection_coverage.main()
            self.assertEqual(failure.exception.code, 1)
            self.assertTrue(any('::warning title=Incomplete event coverage::' in str(call.args[0])
                                for call in printed.call_args_list if call.args))
