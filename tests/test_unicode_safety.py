import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scraper


class UnicodeSafetyTest(unittest.TestCase):
    def test_valid_surrogate_pair_is_preserved_as_unicode(self):
        self.assertEqual(scraper.sanitize_unicode("seat \ud83c\udfab"), "seat 🎫")

    def test_lone_surrogate_is_replaced_and_master_is_writable(self):
        row = {
            "ticket_id": "ticket-1",
            "raw_description": "broken \ud83c text",
            "status": "listing",
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(scraper, "DATA_DIR", directory):
                scraper.save_master("group", {"ticket-1": row})
            with open(
                Path(directory) / "group_master.csv",
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                saved = next(csv.DictReader(handle))

        self.assertEqual(saved["raw_description"], "broken \ufffd text")

    def test_failed_write_does_not_replace_existing_master(self):
        with tempfile.TemporaryDirectory() as directory:
            master_path = Path(directory) / "group_master.csv"
            master_path.write_text("existing-data", encoding="utf-8")
            writer = MagicMock()
            writer.writerow.side_effect = OSError("simulated disk error")
            with (
                patch.object(scraper, "DATA_DIR", directory),
                patch.object(scraper.csv, "DictWriter", return_value=writer),
                self.assertRaises(OSError),
            ):
                scraper.save_master("group", {"ticket-1": {"ticket_id": "ticket-1"}})

            self.assertEqual(master_path.read_text(encoding="utf-8"), "existing-data")
            self.assertFalse(Path(str(master_path) + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
