import sys
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scraper


class TargetConfigTest(unittest.TestCase):
    def test_performer_308_redirect_stays_on_ticketen_host(self):
        redirect = urllib.error.HTTPError(
            'https://ticketen.jp/performers/old', 308,
            'Permanent Redirect', {'Location': '/performers/new'}, None,
        )
        response = MagicMock()
        response.read.return_value = b'<html>ok</html>'
        with patch.object(scraper.urllib.request, 'urlopen',
                          side_effect=[redirect, response]) as fetch:
            self.assertEqual(
                scraper.fetch_html('https://ticketen.jp/performers/old'),
                '<html>ok</html>',
            )
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args.args[0].full_url,
                         'https://ticketen.jp/performers/new')

    def test_performer_308_to_other_host_is_rejected(self):
        redirect = urllib.error.HTTPError(
            'https://ticketen.jp/performers/old', 308,
            'Permanent Redirect', {'Location': 'https://other.example/path'}, None,
        )
        with patch.object(scraper.urllib.request, 'urlopen',
                          side_effect=redirect):
            with self.assertRaises(scraper.ScrapeIntegrityError):
                scraper.fetch_html('https://ticketen.jp/performers/old')

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
