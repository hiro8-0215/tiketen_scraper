import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ticket_change_log as ledger
import scraper


class TicketChangeLogTest(unittest.TestCase):
    def test_migration_preserves_every_byte_order_and_record(self):
        # Include UTF-8, escaped Unicode, CRLF, and repeated records.
        original = ('{"text":"日本語🎫"}\r\n' + '{"text":"\\u65e5"}\n') * 5
        raw = original.encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'ticket_changes_20260930.jsonl'
            source.write_bytes(raw)
            report = ledger.migrate_legacy_log(source, max_bytes=70)
            parts = ledger.change_log_paths(directory)
            self.assertFalse(source.exists())
            self.assertGreater(len(parts), 1)
            self.assertTrue(all(p.stat().st_size <= 70 for p in parts))
            self.assertEqual(b''.join(p.read_bytes() for p in parts), raw)
            self.assertEqual(report['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(report['records'], 10)
            self.assertEqual(ledger.migrate_legacy_logs(directory, max_bytes=70), [])

    def test_rotation_repeated_append_and_schema_preservation(self):
        records = [{'schema_version': 'ticket_change_v1', 'ticket_id': str(i),
                    'before': {'price': 100, 'raw_description': '日本語'},
                    'after': {'price': 200, 'raw_description': '全文🎫'}} for i in range(7)]
        line_size = len((json.dumps(records[0], ensure_ascii=True) + '\n').encode('utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            ledger.append_changes(directory, '20261001', records[:3], max_bytes=line_size * 2)
            ledger.append_changes(directory, '20261001', records[3:], max_bytes=line_size * 2)
            parts = ledger.change_log_paths(directory)
            self.assertEqual(len(parts), 4)
            self.assertTrue(all(p.stat().st_size <= line_size * 2 for p in parts))
            actual = [json.loads(line) for p in parts for line in p.read_bytes().splitlines()]
            self.assertEqual(actual, records)
            ledger.append_changes(directory, '20261002', records[:1], max_bytes=line_size * 2)
            self.assertEqual(len(ledger.change_log_paths(directory, '20261002')), 1)

    def test_interrupted_publication_can_resume_without_loss_or_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'ticket_changes_20260930.jsonl'
            raw = b'{"id":1}\n{"id":2}\n{"id":3}\n'
            source.write_bytes(raw)
            replace = ledger.os.replace
            count = 0

            def interrupt(src, dest):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError('simulated interruption')
                return replace(src, dest)

            with patch.object(ledger.os, 'replace', side_effect=interrupt), self.assertRaises(OSError):
                ledger.migrate_legacy_log(source, max_bytes=10)
            self.assertEqual(source.read_bytes(), raw)
            ledger.migrate_legacy_log(source, max_bytes=10)
            self.assertEqual(b''.join(p.read_bytes() for p in ledger.change_log_paths(directory)), raw)
            self.assertFalse(any(p.is_dir() for p in Path(directory).iterdir()))

    def test_conflicting_part_never_overwrites_original_or_existing_part(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'ticket_changes_20260930.jsonl'
            part = Path(directory) / 'ticket_changes_20260930_001.jsonl'
            source.write_bytes(b'{"id":1}\n')
            part.write_bytes(b'{"id":9}\n')
            with self.assertRaises(RuntimeError):
                ledger.migrate_legacy_log(source, max_bytes=30)
            self.assertEqual(source.read_bytes(), b'{"id":1}\n')
            self.assertEqual(part.read_bytes(), b'{"id":9}\n')

    def test_oversized_record_and_truncated_source_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'ticket_changes_20260930.jsonl'
            for raw in [b'{"description":"very long text"}\n', b'{"id":1}']:
                source.write_bytes(raw)
                with self.assertRaises(ValueError):
                    ledger.migrate_legacy_log(source, max_bytes=10)
                self.assertEqual(source.read_bytes(), raw)
                self.assertEqual(ledger.change_log_paths(directory), [source])
            source.unlink()
            with self.assertRaises(ValueError):
                ledger.append_changes(directory, '20260930', [{'text': 'long'}], max_bytes=10)
            self.assertEqual(ledger.change_log_paths(directory), [])

    def test_reader_sorts_parts_numerically_and_accepts_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            names = ['ticket_changes_20260930_1000.jsonl', 'ticket_changes_20260930_999.jsonl',
                     'ticket_changes_20260929.jsonl', 'ticket_changes_20260930_001.jsonl']
            for name in names:
                (Path(directory) / name).touch()
            self.assertEqual([p.name for p in ledger.change_log_paths(directory)],
                             [names[2], names[3], names[1], names[0]])

    def test_scraper_appends_after_migrating_a_legacy_log(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'ticket_changes_20260930.jsonl'
            legacy.write_bytes(b'{"historical":true}\n')
            rows = {'code': {'event_id': 'event', 'status': 'listing', 'price': 12000}}
            with patch.object(scraper, 'DATA_DIR', directory), patch.object(scraper, 'datetime') as clock:
                clock.now.return_value.strftime.return_value = '20260930'
                scraper.save_ticket_changes('artist', {}, rows, 'observed', 'utc')
            parts = ledger.change_log_paths(directory)
            records = [json.loads(line) for p in parts for line in p.read_bytes().splitlines()]
            self.assertEqual(records[0], {'historical': True})
            self.assertEqual(records[1]['schema_version'], 'ticket_change_v1')
            self.assertEqual(records[1]['after']['price'], 12000)
            self.assertFalse(legacy.exists())


if __name__ == '__main__':
    unittest.main()
