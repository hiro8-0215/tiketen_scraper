import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import feature_columns, feature_profiles
from modeling import enforce_monotonic_horizons, temporal_group_splits
from timeline import build_landmarks, prepare_end_times
from train import _select_profile


class AlternativeTimelineTest(unittest.TestCase):
    def setUp(self):
        self.tickets = pd.DataFrame({
            "ticket_id": ["current", "cheap", "different_quantity", "wrong_seat"],
            "event_id": ["e"] * 4, "quantity": ["2", "2", "1", "2"],
            "ticket_type": ["a"] * 4, "name_type": ["x"] * 4,
            "first_observed_at": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-03", "2026-01-02"]),
            "last_observed_at": pd.to_datetime(["2026-01-10"] * 4),
            "sold_at": pd.to_datetime([None] * 4), "status": ["listing"] * 4,
            "performance_at": pd.to_datetime(["2026-02-01"] * 4),
            "price": [20000, 17000, 10000, 12000], "duplicate_group": ["a", "b", "c", "d"],
            "semantic_seat_level": ["アリーナ", "アリーナ", "アリーナ", "スタンド"],
            "semantic_row_position": ["不明"] * 4,
            "semantic_visibility": ["通常"] * 4,
        })

    def test_only_comparable_cheaper_arrival_is_positive(self):
        frame = build_landmarks(self.tickets, horizons=(3,), cutoff=pd.Timestamp("2026-01-10"))
        row = frame[(frame.ticket_id == "current") & (frame.landmark_at == pd.Timestamp("2026-01-01"))].iloc[0]
        self.assertEqual(row["alternative_3d"], 1)
        self.assertEqual(row["future_best_price_3d"], 17000)

    def test_outcomes_are_forbidden_features(self):
        frame = build_landmarks(self.tickets, horizons=(3,), cutoff=pd.Timestamp("2026-01-10"))
        numeric, categorical = feature_columns(frame)
        self.assertNotIn("future_best_price_3d", numeric + categorical)
        self.assertNotIn("alternative_3d", numeric + categorical)

    def test_performance_before_first_observation_is_quarantined(self):
        tickets = self.tickets.copy()
        tickets.loc[0, "performance_at"] = pd.Timestamp("2025-12-31")
        cleaned = prepare_end_times(tickets)
        self.assertNotIn("current", set(cleaned.ticket_id))
        self.assertEqual(cleaned.attrs["excluded_temporal_anomalies"], 1)

    def test_post_performance_unlabelled_rows_are_not_built(self):
        tickets = self.tickets.iloc[[0]].copy()
        tickets.loc[:, "performance_at"] = pd.Timestamp("2026-01-04")
        tickets.loc[:, "status"] = "sold"
        tickets.loc[:, "sold_at"] = pd.Timestamp("2026-01-08")
        frame = build_landmarks(
            tickets, horizons=(1,), cutoff=pd.Timestamp("2026-01-10")
        )
        self.assertEqual(frame.landmark_at.max(), pd.Timestamp("2026-01-03"))

    def test_validation_is_strictly_after_purged_training(self):
        frame = pd.DataFrame({
            "duplicate_group": [f"g{i}" for i in range(10) for _ in range(4)],
            "landmark_at": pd.to_datetime([
                f"2026-01-{i + day + 1:02d}" for i in range(10) for day in range(4)
            ]),
        })
        for _, training, validation in temporal_group_splits(frame, horizon=1, n_splits=3):
            self.assertLess(
                (frame.loc[training, "landmark_at"] + pd.Timedelta(days=1)).max(),
                frame.loc[validation, "landmark_at"].min(),
            )

    def test_first_fold_warmup_expands_until_both_classes_exist(self):
        frame = pd.DataFrame({
            "duplicate_group": [f"g{i}" for i in range(15)],
            "landmark_at": [
                pd.Timestamp("2026-01-01") + pd.Timedelta(days=3 * i)
                for i in range(15)
            ],
            "alternative_1d": [0, 0, 0, 1] + [0] * 11,
        })
        splits = list(temporal_group_splits(
            frame, horizon=1, n_splits=4, target="alternative_1d"
        ))

        self.assertEqual(len(splits), 4)
        for _, training, _ in splits:
            self.assertEqual(set(frame.loc[training, "alternative_1d"]), {0, 1})
        self.assertEqual(
            frame.loc[splits[0][2], "duplicate_group"].iloc[0], "g4"
        )
        self.assertEqual(
            frame.loc[splits[1][2], "duplicate_group"].iloc[0], "g6"
        )

    def test_first_fold_requires_configured_rows_per_class(self):
        frame = pd.DataFrame({
            "duplicate_group": [f"g{i}" for i in range(20)],
            "landmark_at": [
                pd.Timestamp("2026-01-01") + pd.Timedelta(days=3 * i)
                for i in range(20)
            ],
            "alternative_1d": [0] * 4 + [1, 1] + [0] * 14,
        })
        splits = list(temporal_group_splits(
            frame,
            horizon=1,
            n_splits=4,
            target="alternative_1d",
            min_class_count=2,
        ))

        self.assertEqual(frame.loc[splits[0][2], "duplicate_group"].iloc[0], "g6")
        self.assertEqual(frame.loc[splits[1][2], "duplicate_group"].iloc[0], "g8")
        self.assertGreaterEqual(
            frame.loc[splits[0][1], "alternative_1d"].value_counts().min(), 2
        )

    def test_semantics_are_a_separate_ablation_profile(self):
        frame = build_landmarks(self.tickets, horizons=(3,), cutoff=pd.Timestamp("2026-01-10"))
        frame["semantic_seat_level"] = ["不明"] * (len(frame) - 1) + ["アリーナ"]
        frame["semantic_is_random"] = [0.0] * (len(frame) - 1) + [1.0]
        profiles = feature_profiles(frame)
        self.assertNotIn("semantic_seat_level", profiles["tabular"][1])
        self.assertIn("semantic_seat_level", profiles["semantic"][1])

    def test_semantic_profile_requires_probability_improvement(self):
        base = {"log_loss": .50, "brier": .20, "pr_auc": .60}
        improved = {"log_loss": .49, "brier": .20, "pr_auc": .60}
        self.assertEqual(_select_profile({"tabular": base, "semantic": improved}), "semantic")
        self.assertEqual(_select_profile({"tabular": improved, "semantic": base}), "tabular")

    def test_arrival_probability_is_monotonic_by_horizon(self):
        frame = pd.DataFrame({
            "ticket_id": ["x"] * 3, "landmark_at": pd.to_datetime(["2026-01-01"] * 3),
            "horizon_days": [1, 3, 7], "p_alternative": [.4, .2, .3],
        })
        result = enforce_monotonic_horizons(frame).sort_values("horizon_days")
        self.assertTrue(result.p_alternative.is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
