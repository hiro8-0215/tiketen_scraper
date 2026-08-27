import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import feature_columns, feature_profiles
from modeling import (
    enforce_monotonic_horizons,
    normalize_probability_rows,
    probability_metrics,
    temporal_group_splits,
)
from timeline import add_end_times, build_landmarks
from train import _select_profile


class TimelineTest(unittest.TestCase):
    def _tickets(self):
        return pd.DataFrame({
            "ticket_id": ["sold", "deleted", "active"],
            "event_id": ["e", "e", "e"],
            "first_observed_at": pd.to_datetime(["2026-01-01"] * 3),
            "last_observed_at": pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-10"]),
            "sold_at": pd.to_datetime(["2026-01-03", None, None]),
            "status": ["sold", "deleted", "listing"],
            "performance_at": pd.to_datetime(["2026-02-01"] * 3),
            "price": [10000, 12000, 9000],
            "duplicate_group": ["a", "b", "c"],
        })

    def test_competing_labels_are_distinct(self):
        frame = build_landmarks(self._tickets(), horizons=(3,), cutoff=pd.Timestamp("2026-01-10"))
        first = frame[frame["landmark_at"].eq(pd.Timestamp("2026-01-01"))].set_index("ticket_id")
        self.assertEqual(first.loc["sold", "state_3d"], 1)
        self.assertEqual(first.loc["deleted", "state_3d"], 2)
        self.assertEqual(first.loc["active", "state_3d"], 0)

    def test_impossible_end_before_first_is_quarantined(self):
        tickets = self._tickets()
        invalid = tickets.iloc[[0]].copy()
        invalid["ticket_id"] = "invalid-sold"
        invalid["first_observed_at"] = pd.Timestamp("2026-01-03")
        invalid["sold_at"] = pd.Timestamp("2026-01-02")
        tickets = pd.concat([tickets, invalid], ignore_index=True)

        cleaned, _ = add_end_times(tickets, cutoff=pd.Timestamp("2026-01-10"))

        self.assertNotIn("invalid-sold", set(cleaned.ticket_id))
        self.assertEqual(cleaned.attrs["excluded_temporal_anomalies"], 1)
        self.assertEqual(
            cleaned.attrs["excluded_temporal_anomaly_ticket_ids"], ["invalid-sold"]
        )

    def test_future_columns_never_become_features(self):
        frame = build_landmarks(self._tickets(), horizons=(3,), cutoff=pd.Timestamp("2026-01-10"))
        numeric, categorical = feature_columns(frame)
        selected = set(numeric + categorical)
        self.assertNotIn("outcome_at", selected)
        self.assertNotIn("state_3d", selected)
        self.assertNotIn("last_observed_at", selected)

    def test_validation_is_strictly_after_purged_training(self):
        frame = pd.DataFrame({
            "duplicate_group": [f"g{i}" for i in range(10) for _ in range(4)],
            "landmark_at": pd.to_datetime([
                f"2026-01-{i + day + 1:02d}" for i in range(10) for day in range(4)
            ]),
        })
        for _, training, validation in temporal_group_splits(frame, horizon_days=1, n_splits=3):
            self.assertLess(
                (frame.loc[training, "landmark_at"] + pd.Timedelta(days=1)).max(),
                frame.loc[validation, "landmark_at"].min(),
            )

    def test_first_fold_warmup_expands_until_all_demand_classes_exist(self):
        frame = pd.DataFrame({
            "duplicate_group": [f"g{i}" for i in range(15)],
            "landmark_at": [
                pd.Timestamp("2026-01-01") + pd.Timedelta(days=3 * i)
                for i in range(15)
            ],
            "state_1d": [0, 0, 0, 1, 2] + [0] * 10,
        })
        splits = list(temporal_group_splits(
            frame, horizon_days=1, n_splits=4, target="state_1d"
        ))

        self.assertEqual(len(splits), 4)
        for _, training, _ in splits:
            self.assertEqual(set(frame.loc[training, "state_1d"]), {0, 1, 2})
        self.assertEqual(
            frame.loc[splits[0][2], "duplicate_group"].iloc[0], "g5"
        )
        self.assertEqual(
            frame.loc[splits[1][2], "duplicate_group"].iloc[0], "g6"
        )

    def test_first_fold_requires_configured_rows_per_class(self):
        frame = pd.DataFrame({
            "duplicate_group": [f"g{i}" for i in range(25)],
            "landmark_at": [
                pd.Timestamp("2026-01-01") + pd.Timedelta(days=3 * i)
                for i in range(25)
            ],
            "state_1d": [0] * 5 + [1, 1, 2, 2] + [0] * 16,
        })
        splits = list(temporal_group_splits(
            frame,
            horizon_days=1,
            n_splits=4,
            target="state_1d",
            min_class_count=2,
        ))

        self.assertEqual(frame.loc[splits[0][2], "duplicate_group"].iloc[0], "g9")
        self.assertEqual(frame.loc[splits[1][2], "duplicate_group"].iloc[0], "g10")
        self.assertGreaterEqual(
            frame.loc[splits[0][1], "state_1d"].value_counts().min(), 2
        )

    def test_semantics_are_a_separate_ablation_profile(self):
        frame = build_landmarks(self._tickets(), horizons=(3,), cutoff=pd.Timestamp("2026-01-10"))
        frame["semantic_seat_level"] = ["不明"] * (len(frame) - 1) + ["アリーナ"]
        frame["semantic_is_random"] = [0.0] * (len(frame) - 1) + [1.0]
        profiles = feature_profiles(frame)
        self.assertNotIn("semantic_seat_level", profiles["tabular"][1])
        self.assertIn("semantic_seat_level", profiles["semantic"][1])

    def test_semantic_profile_requires_real_improvement(self):
        base = {"log_loss": .50, "multiclass_brier": .20}
        improved = {"log_loss": .49, "multiclass_brier": .20}
        self.assertEqual(_select_profile({"tabular": base, "semantic": improved}), "semantic")
        self.assertEqual(_select_profile({"tabular": improved, "semantic": base}), "tabular")

    def test_competing_probabilities_are_monotonic_by_horizon(self):
        frame = pd.DataFrame({
            "ticket_id": ["x"] * 3, "landmark_at": pd.to_datetime(["2026-01-01"] * 3),
            "horizon_days": [1, 3, 7], "p_active": [.4, .6, .5],
            "p_sold": [.4, .2, .3], "p_deleted": [.2, .2, .2],
        })
        result = enforce_monotonic_horizons(frame).sort_values("horizon_days")
        self.assertTrue(result.p_sold.is_monotonic_increasing)
        self.assertTrue(result.p_deleted.is_monotonic_increasing)
        self.assertTrue(((result.p_active + result.p_sold + result.p_deleted) - 1).abs().max() < 1e-9)

    def test_metrics_tolerate_floating_point_probability_boundary_error(self):
        epsilon = 2.220446049250313e-16
        probabilities = normalize_probability_rows(
            [[-epsilon, 0.4, 0.6 + epsilon], [0.2, 0.3, 0.5]]
        )
        report = probability_metrics(
            pd.Series([2, 0]).to_numpy(int), probabilities
        )

        self.assertGreaterEqual(probabilities.min(), 0.0)
        self.assertAlmostEqual(probabilities[0].sum(), 1.0)
        self.assertEqual(report["rows"], 2)


if __name__ == "__main__":
    unittest.main()
