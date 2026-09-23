import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision import apply_policy, regret
from decision import summarize
from train_policy import _chronological_partitions_by_horizon, _mean_regret


class DecisionTest(unittest.TestCase):
    def test_each_horizon_uses_its_own_chronological_boundary(self):
        rows = []
        for horizon, start in ((1, "2026-01-01"), (7, "2026-02-01")):
            for offset in range(10):
                rows.append({
                    "horizon_days": horizon,
                    "landmark_at": pd.Timestamp(start) + pd.Timedelta(hours=offset),
                })
        partitions = _chronological_partitions_by_horizon(pd.DataFrame(rows))

        self.assertEqual(set(partitions), {1, 7})
        for training, validation in partitions.values():
            self.assertFalse(training.empty)
            self.assertFalse(validation.empty)
            self.assertLess(training.landmark_at.max(), validation.landmark_at.min())

    def test_high_disappearance_and_discount_can_trigger_buy(self):
        frame = pd.DataFrame({"discount_ratio": [.20], "disappearance_probability": [.8], "p_alternative": [.1]})
        result = apply_policy(frame, {"disappear_weight": .5, "alternative_weight": .25, "buy_threshold": .1, "wait_threshold": .5})
        self.assertEqual(result.action.iloc[0], "buy_now")

    def test_deleted_has_smaller_wait_penalty_than_sold(self):
        base = pd.DataFrame({"action": ["wait", "wait"], "potential_savings": [0, 0], "true_state": [1, 2], "true_alternative": [0, 0]})
        values = regret(base, {"sold_miss_penalty": 10000, "deleted_multiplier": .3})
        self.assertEqual(values[0], 10000)
        self.assertEqual(values[1], 3000)

    def test_fast_grid_loss_matches_policy_summary(self):
        frame = pd.DataFrame({
            "discount_ratio": [.2, -.1, .05, .0],
            "disappearance_probability": [.8, .2, .5, .1],
            "p_alternative": [.1, .8, .4, .2],
            "potential_savings": [3000, 0, 1500, 500],
            "true_state": [1, 0, 2, 1],
            "true_alternative": [0, 1, 0, 1],
        })
        policy = {
            "disappear_weight": .5, "alternative_weight": .25,
            "buy_threshold": .1, "wait_threshold": .5,
        }
        profile = {"sold_miss_penalty": 10000, "deleted_multiplier": .3}
        expected = summarize(apply_policy(frame, policy), profile)["mean_regret_yen"]
        self.assertEqual(_mean_regret(frame, policy, profile), expected)


if __name__ == "__main__":
    unittest.main()
