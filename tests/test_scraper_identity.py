import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scraper


class ScraperIdentityTest(unittest.TestCase):
    def test_existing_rotations_are_collapsed_and_details_are_preserved(self):
        old = {"ticket_id": "old", "event_id": "event", "created_at_unix": "1", "status": "deleted", "last_observed_at": "2026-08-28 02:00:00", "seller_name": "seller", "details_fetched": "True"}
        current = {"ticket_id": "current", "event_id": "event", "created_at_unix": "1", "status": "listing", "last_observed_at": "2026-08-28 02:00:00", "seller_name": "", "details_fetched": "False"}
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


if __name__ == "__main__":
    unittest.main()
