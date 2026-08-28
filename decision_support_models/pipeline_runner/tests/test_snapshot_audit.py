import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snapshot_audit import audit_snapshot, validate_snapshot


FIELDS = [
    "ticket_id", "event_id", "created_at_unix", "status",
    "last_observed_at", "sold_at",
]


def write_master(snapshot: Path, rows: list[dict]) -> None:
    snapshot.mkdir(parents=True, exist_ok=True)
    with (snapshot / "group_master.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class SnapshotAuditTest(unittest.TestCase):
    def test_current_snapshot_with_listing_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_27"
            write_master(snapshot, [
                {
                    "ticket_id": "a",
                    "status": "listing",
                    "last_observed_at": "2026-08-27 09:00:00",
                },
                {
                    "ticket_id": "b",
                    "status": "sold",
                    "last_observed_at": "2026-08-27 08:00:00",
                },
            ])
            self.assertTrue(validate_snapshot(snapshot)["ok"])

    def test_terminal_wipe_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_26"
            write_master(snapshot, [
                {
                    "ticket_id": str(index),
                    "status": "deleted",
                    "last_observed_at": "2026-07-28 01:31:00",
                }
                for index in range(100)
            ])
            with self.assertRaisesRegex(RuntimeError, "zero listing"):
                validate_snapshot(snapshot)

    def test_historical_override_is_explicit_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_26"
            write_master(snapshot, [{
                "ticket_id": "a",
                "status": "deleted",
                "last_observed_at": "2026-07-28 01:31:00",
            }])
            report = validate_snapshot(snapshot, allow_historical=True)
            self.assertTrue(report["ok"])
            self.assertTrue(report["historical_override_used"])
            self.assertGreater(len(report["historical_issues"]), 0)

    def test_old_deletion_spike_is_still_blocked_after_new_listing(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_27"
            rows = [
                {
                    "ticket_id": str(index),
                    "status": "deleted",
                    "last_observed_at": "2026-07-28 01:31:00",
                }
                for index in range(100)
            ]
            rows.append({
                "ticket_id": "new-active",
                "status": "listing",
                "last_observed_at": "2026-08-27 09:00:00",
            })
            write_master(snapshot, rows)
            with self.assertRaisesRegex(RuntimeError, "one timestamp"):
                validate_snapshot(snapshot)

    def test_structural_error_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_27"
            write_master(snapshot, [{
                "ticket_id": "a",
                "status": "mystery",
                "last_observed_at": "2026-08-27 09:00:00",
            }])
            with self.assertRaisesRegex(RuntimeError, "unknown statuses"):
                validate_snapshot(snapshot, allow_historical=True)

    def test_duplicate_conflict_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_27"
            write_master(snapshot, [{
                "ticket_id": "same",
                "status": "deleted",
                "last_observed_at": "2026-08-27 09:00:00",
            }])
            with (snapshot / "other_master.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({
                    "ticket_id": "same",
                    "status": "sold",
                    "last_observed_at": "2026-08-27 09:00:00",
                })
            report = audit_snapshot(snapshot)
            self.assertEqual(report["conflicting_duplicate_status_ids"], 1)

    def test_rotated_share_code_prefers_direct_listing_over_inferred_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_28"
            write_master(snapshot, [
                {"ticket_id": "old-code", "event_id": "event", "created_at_unix": "123", "status": "deleted", "last_observed_at": "2026-08-28 02:00:00"},
                {"ticket_id": "new-code", "event_id": "event", "created_at_unix": "123", "status": "listing", "last_observed_at": "2026-08-28 02:00:00"},
            ])
            report = validate_snapshot(snapshot)
            self.assertEqual(report["canonical_rows"], 1)
            self.assertEqual(report["status_counts"], {"listing": 1})

    def test_bootstrap_sale_time_spike_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / "data_8_28"
            rows = [{"ticket_id": f"sold-{index}", "event_id": "event", "created_at_unix": str(index), "status": "sold", "last_observed_at": "2026-08-28 02:00:00", "sold_at": "2026-08-27 07:17:28"} for index in range(100)]
            rows.append({"ticket_id": "active", "event_id": "event", "created_at_unix": "active", "status": "listing", "last_observed_at": "2026-08-28 02:00:00"})
            write_master(snapshot, rows)
            with self.assertRaisesRegex(RuntimeError, "bootstrap sale times"):
                validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
