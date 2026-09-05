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

    def test_missing_created_at_has_no_logical_identity(self):
        row = {
            "ticket_id": "new-code", "event_id": "event",
            "created_at_unix": "", "status": "listing",
        }
        self.assertIsNone(scraper._listing_identity_key(row))
        self.assertIsNone(scraper._ticket_match_key(row))

    def test_missing_identity_does_not_reuse_an_unrelated_sold_row(self):
        sold = {
            "ticket_id": "sold-code", "event_id": "event",
            "created_at_unix": "", "status": "sold",
        }
        master = {"sold-code": sold}
        result, changed = scraper._rekey_active_listing(
            master, {"sold-code": sold}, {None: sold},
            "new-active-code", None,
        )
        self.assertIsNone(result)
        self.assertFalse(changed)
        self.assertEqual(master["sold-code"]["status"], "sold")

    def test_different_share_code_does_not_rekey_sold_identity(self):
        sold = {
            "ticket_id": "sold-code", "event_id": "event",
            "created_at_unix": "123", "status": "sold",
        }
        identity = scraper._listing_identity_key(sold)
        master = {"sold-code": sold}
        result, changed = scraper._rekey_active_listing(
            master, {"sold-code": sold}, {identity: sold},
            "new-active-code", identity,
        )
        self.assertIsNone(result)
        self.assertFalse(changed)
        self.assertIn("sold-code", master)

    def test_same_share_code_can_be_reactivated(self):
        sold = {
            "ticket_id": "same-code", "event_id": "event",
            "created_at_unix": "123", "status": "sold",
        }
        identity = scraper._listing_identity_key(sold)
        result, changed = scraper._rekey_active_listing(
            {"same-code": sold}, {"same-code": sold}, {identity: sold},
            "same-code", None,
        )
        self.assertIs(result, sold)
        self.assertFalse(changed)

        active = scraper.validate_event_snapshot(
            "event",
            [{"status": "active", "shareCode": "same-code"}],
            {}, {"same-code": sold}, datetime(2026, 9, 5),
        )
        self.assertEqual(active, {"same-code"})


if __name__ == "__main__":
    unittest.main()
