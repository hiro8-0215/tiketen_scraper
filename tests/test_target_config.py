import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scraper


class TargetConfigTest(unittest.TestCase):
    def test_legacy_string_target_remains_supported(self):
        self.assertEqual(
            scraper.normalize_targets(["snow-man"]),
            [{"name": "snow-man", "source": "snow-man"}],
        )

    def test_source_id_can_change_without_changing_master_name(self):
        self.assertEqual(
            scraper.normalize_targets([
                {"name": "travis-japan", "source": "current-ticketen-id"}
            ]),
            [{"name": "travis-japan", "source": "current-ticketen-id"}],
        )

    def test_duplicate_output_names_are_rejected(self):
        with self.assertRaises(ValueError):
            scraper.normalize_targets([
                "b-and-zai",
                {"name": "b-and-zai", "source": "new-id"},
            ])

    def test_api_pass_records_valid_zero_event_target(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "targets.json").write_text(
                json.dumps(["empty-group"]), encoding="utf-8"
            )
            with (
                patch.object(scraper, "DATA_DIR", directory),
                patch.object(scraper, "SCRAPE_MODE", "api"),
                patch.object(
                    scraper,
                    "get_events",
                    side_effect=scraper.NoEventsFound("valid page, no events"),
                ),
                patch.object(scraper, "save_snapshots"),
                patch("builtins.print"),
            ):
                scraper.main()

            master = Path(directory, "empty-group_master.csv")
            self.assertTrue(master.exists())
            self.assertEqual(len(master.read_text(encoding="utf-8-sig").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
