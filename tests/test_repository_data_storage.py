import gzip
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import repository_data_storage as storage


class RepositoryDataStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_gzip_round_trip_is_byte_exact_and_deterministic(self):
        source = self.root / 'group_master.csv'
        raw = ('\ufeffid,description\r\n1,"日本語\r\n複数行"\r\n2,last').encode('utf-8')
        source.write_bytes(raw)
        archive = storage.pack_csv(source, threshold=1)
        first = archive.read_bytes()
        self.assertFalse(source.exists())
        storage.restore_data(self.root)
        self.assertEqual(source.read_bytes(), raw)
        storage.pack_csv(source, threshold=10**9)  # Once packed, retain its representation.
        self.assertEqual(archive.read_bytes(), first)

    def test_small_csv_stays_in_original_form(self):
        source = self.root / 'group_master.csv'
        source.write_bytes(b'id\n1\n')
        self.assertEqual(storage.pack_data(self.root), [])
        self.assertEqual(source.read_bytes(), b'id\n1\n')

    def test_corrupt_gzip_does_not_publish_partial_csv(self):
        archive = self.root / 'group_master.csv.gz'
        archive.write_bytes(gzip.compress(b'original')[:-5])
        with self.assertRaises((EOFError, OSError)):
            storage.restore_data(self.root)
        self.assertFalse(archive.with_suffix('').exists())
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_restore_refuses_to_overwrite_edits_and_masters_only_skips_snapshots(self):
        source = self.root / 'group_master.csv'
        source.write_bytes(b'original')
        storage.pack_csv(source, threshold=1)
        source.write_bytes(b'my edits')
        with self.assertRaisesRegex(RuntimeError, 'refusing to overwrite'):
            storage.restore_data(self.root)
        self.assertEqual(source.read_bytes(), b'my edits')
        snapshot = self.root / 'snapshot'
        snapshot.mkdir()
        (snapshot / 'group.csv.gz').write_bytes(gzip.compress(b'snapshot'))
        storage.restore_data(self.root, masters_only=True, skip_existing=True)
        self.assertFalse((snapshot / 'group.csv').exists())

    def test_uncompressible_or_failed_write_keeps_source_and_old_archive(self):
        source = self.root / 'group_master.csv'
        source.write_bytes(b'original')
        archive = source.with_suffix('.csv.gz')
        archive.write_bytes(b'previous good archive')
        with patch.object(storage, 'MAX_GIT_BYTES', 1), self.assertRaises(RuntimeError):
            storage.pack_csv(source, threshold=1)
        self.assertEqual(source.read_bytes(), b'original')
        self.assertEqual(archive.read_bytes(), b'previous good archive')
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_sparse_commit_preserves_absent_historical_logs_and_saves_csv_gzip(self):
        def git(*args, input=None):
            return subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', 'commit.gpgsign=false',
                                   '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', *args],
                                  cwd=self.root, input=input, check=True, capture_output=True).stdout
        git('init')
        data = self.root / 'data'
        data.mkdir()
        (data / 'ticket_changes_20200101_001.jsonl').write_bytes(b'{"old":true}\n')
        master = data / 'group_master.csv'
        master.write_bytes(b'id,value\n1,original\n')
        git('add', '.')
        git('commit', '-m', 'initial')
        old_hash = git('rev-parse', 'HEAD:data/ticket_changes_20200101_001.jsonl')
        git('sparse-checkout', 'set', '--no-cone', '/data/*_master.csv', '/data/*_master.csv.gz')
        self.assertFalse((data / 'ticket_changes_20200101_001.jsonl').exists())
        master.write_bytes(b'id,value\n1,updated\n')
        storage.pack_data(data, threshold=1)
        git('add', '--sparse', 'data/')
        git('commit', '-m', 'compressed update')
        self.assertEqual(git('rev-parse', 'HEAD:data/ticket_changes_20200101_001.jsonl'), old_hash)
        tree = git('ls-tree', '-r', '--name-only', 'HEAD').decode()
        self.assertIn('data/group_master.csv.gz', tree)
        self.assertNotIn('data/group_master.csv\n', tree)
        storage.restore_data(data, masters_only=True)
        self.assertEqual(master.read_bytes(), b'id,value\n1,updated\n')
        # Regenerate a smaller CSV while its tracked gzip is outside sparse paths.
        master.unlink()
        git('sparse-checkout', 'set', '--no-cone', '/data/*_master.csv')
        self.assertFalse((data / 'group_master.csv.gz').exists())
        data.mkdir(exist_ok=True)
        master.write_bytes(b'id\n1\n')
        storage.pack_data(data, threshold=10**9)
        self.assertFalse(master.exists())
        self.assertEqual(gzip.decompress((data / 'group_master.csv.gz').read_bytes()), b'id\n1\n')


if __name__ == '__main__':
    unittest.main()
