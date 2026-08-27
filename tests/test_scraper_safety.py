import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scraper


def stored_listing(ticket_id, event_id="future-event", perf_date="2026-12-01"):
    return {
        "ticket_id": ticket_id,
        "event_id": event_id,
        "status": "listing",
        "perf_date": perf_date,
        "created_at_unix": ticket_id,
        "price": "10000",
        "last_observed_at": "2026-08-01 00:00:00",
    }


def active_response(ticket_id):
    return {
        "status": "active",
        "shareCode": ticket_id,
        "createdAt": ticket_id,
        "pricePerTicket": "10000",
    }


class ScraperSafetyTest(unittest.TestCase):
    def test_sold_match_key_is_scoped_to_event(self):
        row = stored_listing("a", event_id="event-a")
        self.assertNotEqual(
            scraper._ticket_match_key(row),
            scraper._ticket_match_key(row, "event-b"),
        )

    def test_sold_id_does_not_collide_across_events_or_prices(self):
        first = scraper._sold_ticket_id("event-a", "123", "10000")
        self.assertNotEqual(
            first, scraper._sold_ticket_id("event-b", "123", "10000")
        )
        self.assertNotEqual(
            first, scraper._sold_ticket_id("event-a", "123", "12000")
        )

    def test_empty_response_preserves_existing_listings(self):
        master = {"a": stored_listing("a")}
        with self.assertRaises(scraper.ScrapeIntegrityError):
            scraper.validate_event_snapshot(
                "future-event", [], master, master, datetime(2026, 8, 27)
            )

    def test_unknown_api_status_is_rejected(self):
        with self.assertRaises(scraper.ScrapeIntegrityError):
            scraper.validate_event_snapshot(
                "future-event",
                [{"status": "new-active-state"}],
                {},
                {},
                datetime(2026, 8, 27),
            )

    def test_mass_unexplained_disappearance_is_rejected(self):
        prior = {str(i): stored_listing(str(i)) for i in range(100)}
        tickets = [active_response(str(i)) for i in range(10)]
        with self.assertRaises(scraper.ScrapeIntegrityError):
            scraper.validate_event_snapshot(
                "future-event", tickets, prior, prior, datetime(2026, 8, 27)
            )

    def test_past_event_can_close_without_active_tickets(self):
        prior = {
            "a": stored_listing("a", perf_date="2026-07-01"),
        }
        tickets = [{
            "status": "sold", "createdAt": "unrelated", "pricePerTicket": "1"
        }]
        result = scraper.validate_event_snapshot(
            "future-event", tickets, prior, prior, datetime(2026, 8, 27)
        )
        self.assertEqual(result, set())

    def test_only_validated_events_can_mark_absences_deleted(self):
        master = {
            "keep": stored_listing("keep", event_id="failed-event"),
            "active": stored_listing("active", event_id="valid-event"),
            "gone": stored_listing("gone", event_id="valid-event"),
        }
        changed = scraper.mark_confirmed_absences_deleted(
            master, {"valid-event": {"active"}}, "2026-08-27 00:00:00"
        )
        self.assertEqual(changed, 1)
        self.assertEqual(master["keep"]["status"], "listing")
        self.assertEqual(master["active"]["status"], "listing")
        self.assertEqual(master["gone"]["status"], "deleted")

    @patch("scraper.urllib.request.urlopen", side_effect=OSError("network down"))
    def test_network_error_is_not_converted_to_empty_success(self, _):
        with self.assertRaises(scraper.ScrapeIntegrityError):
            scraper.fetch_all_tickets("event-id")


if __name__ == "__main__":
    unittest.main()
