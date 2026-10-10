import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import drive_backup_parts as backup
import upload_to_gdrive as uploader


class DriveBackupPartsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'snow-man_master.csv'
        self.raw = ('\ufeffticket_id,description\r\n1,"日本語\r\n複数行"\r\n'
                    '2,"quotes ""inside"""\r\n3,last').encode('utf-8')
        self.source.write_bytes(self.raw)

    def test_multiline_bom_crlf_and_last_line_round_trip_exactly(self):
        parts, manifest = backup.split_csv(self.source, self.root / 'parts', limit=64)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(path.stat().st_size <= 64 for path in parts))
        restored = backup.restore_manifest(manifest, self.root / 'restored')
        self.assertEqual(restored.read_bytes(), self.raw)
        self.assertEqual(self.source.read_bytes(), self.raw)
        for part in parts:
            with part.open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(next(csv.reader(stream)), ['ticket_id', 'description'])

    def test_single_record_too_large_fails_without_changing_source(self):
        with self.assertRaisesRegex(ValueError, 'one CSV record'):
            backup.split_csv(self.source, self.root / 'parts', limit=30)
        self.assertEqual(self.source.read_bytes(), self.raw)

    def test_missing_corrupt_and_reordered_parts_do_not_replace_output(self):
        for failure in ['missing', 'corrupt', 'order']:
            with self.subTest(failure=failure):
                parts, manifest = backup.split_csv(self.source, self.root / failure, limit=64)
                output = self.root / 'restored'
                output.mkdir(exist_ok=True)
                target = output / self.source.name
                target.write_bytes(b'previous-good-data')
                if failure == 'missing':
                    parts[-1].unlink()
                elif failure == 'corrupt':
                    parts[-1].write_bytes(parts[-1].read_bytes().replace(b'last', b'bad!'))
                else:
                    text = manifest.read_text()
                    manifest.write_text(text.replace('snow-man.part001', 'snow-man.part999'))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    backup.restore_manifest(manifest, output)
                self.assertEqual(target.read_bytes(), b'previous-good-data')
                self.assertEqual(list(output.glob('.restore-*')), [])

    def test_global_manifest_restores_all_original_names_and_ignores_stale_files(self):
        small = self.root / 'group_master.csv'
        small.write_bytes(b'id,value\n1,small\n')
        staging = self.root / 'parts'
        data, manifests = backup.prepare_uploads([small, self.source], staging, upload_limit=64)
        self.assertEqual([p.name for p in manifests], [backup.BACKUP_MANIFEST])
        for path in data:
            if path.parent != staging:
                (staging / path.name).write_bytes(path.read_bytes())
        (staging / self.source.name).write_bytes(b'stale unsplit copy')
        (staging / 'snow-man.part999_master.csv').write_bytes(b'obsolete')
        restored = backup.restore_directory(staging, self.root / 'restored')
        self.assertEqual({p.name for p in restored}, {small.name, self.source.name})
        self.assertEqual((self.root / 'restored' / self.source.name).read_bytes(), self.raw)
        self.assertEqual((self.root / 'restored' / small.name).read_bytes(), small.read_bytes())

    def test_shrinking_source_replaces_split_manifest_with_original_entry(self):
        staging = self.root / 'parts'
        backup.prepare_uploads([self.source], staging, upload_limit=64)
        self.source.write_bytes(b'id,value\n1,smaller\n')
        backup.prepare_uploads([self.source], staging, upload_limit=64)
        (staging / self.source.name).write_bytes(self.source.read_bytes())
        restored = backup.restore_directory(staging, self.root / 'restored')
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].read_bytes(), self.source.read_bytes())

    def test_incomplete_snapshot_does_not_replace_any_existing_output(self):
        small = self.root / 'a_master.csv'
        small.write_bytes(b'id\n1\n')
        staging = self.root / 'parts'
        data, _ = backup.prepare_uploads([small, self.source], staging, upload_limit=64)
        (staging / small.name).write_bytes(small.read_bytes())
        data[-1].unlink()
        output = self.root / 'restored'
        output.mkdir()
        (output / small.name).write_bytes(b'previous')
        with self.assertRaises(FileNotFoundError):
            backup.restore_directory(staging, output)
        self.assertEqual((output / small.name).read_bytes(), b'previous')
        self.assertFalse((output / self.source.name).exists())

    def test_parts_without_manifest_are_rejected_even_with_stale_original(self):
        staging = self.root / 'parts'
        parts, manifest = backup.split_csv(self.source, staging, limit=64)
        manifest.unlink()
        (staging / self.source.name).write_bytes(b'stale')
        with self.assertRaisesRegex(ValueError, 'without a completed manifest'):
            backup.restore_directory(staging, self.root / 'restored')

    def test_legacy_unsplit_folder_still_restores(self):
        restored = backup.restore_directory(self.root, self.root / 'restored')
        self.assertEqual(restored[0].read_bytes(), self.raw)

    def test_path_traversal_in_manifest_is_rejected(self):
        _, manifest = backup.split_csv(self.source, self.root / 'parts', limit=64)
        manifest.write_text(manifest.read_text().replace('snow-man.part001_master.csv', '../evil_master.csv'))
        with self.assertRaisesRegex(ValueError, 'Invalid part filename'):
            backup.restore_manifest(manifest, self.root / 'restored')

    def test_manifest_upload_is_last_and_shares_budget_and_staging_is_cleaned(self):
        attempted = []
        def upload(url, token, path, folder, deadline=None):
            attempted.append((path, deadline))
            self.assertTrue(path.exists())
        with (patch.dict(uploader.os.environ, {
                  'GDRIVE_WEBAPP_URL': 'https://example.invalid', 'GDRIVE_UPLOAD_TOKEN': 'test',
                  'GDRIVE_SOURCE_DIR': str(self.root), 'GDRIVE_INCLUDE_JSONL': ''}),
              patch.object(uploader, 'MAX_FILE_BYTES', 64),
              patch.object(uploader, 'upload_file', side_effect=upload), patch('builtins.print')):
            uploader.main()
        self.assertEqual(attempted[-1][0].name, backup.BACKUP_MANIFEST)
        self.assertEqual(len({deadline for _, deadline in attempted}), 1)
        self.assertTrue(all(not path.exists() for path, _ in attempted))
        self.assertEqual(self.source.read_bytes(), self.raw)

    def test_manifest_is_not_published_after_part_failure(self):
        attempted = []
        def upload(url, token, path, folder, deadline=None):
            attempted.append(path.name)
            raise RuntimeError('permanent rejection')
        with (patch.dict(uploader.os.environ, {
                  'GDRIVE_WEBAPP_URL': 'https://example.invalid', 'GDRIVE_UPLOAD_TOKEN': 'test',
                  'GDRIVE_SOURCE_DIR': str(self.root), 'GDRIVE_INCLUDE_JSONL': ''}),
              patch.object(uploader, 'MAX_FILE_BYTES', 64),
              patch.object(uploader, 'upload_file', side_effect=upload), patch('builtins.print'),
              self.assertRaisesRegex(RuntimeError, 'permanent rejection')):
            uploader.main()
        self.assertNotIn(backup.BACKUP_MANIFEST, attempted)


if __name__ == '__main__':
    unittest.main()
