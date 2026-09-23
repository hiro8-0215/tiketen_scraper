import unittest

import pandas as pd

from data_loader import _bootstrap_sale_time_spikes


class BootstrapSpikeTest(unittest.TestCase):
    def test_large_single_transition_timestamp_is_quarantined(self):
        rows = 100
        frame = pd.DataFrame({
            "status": ["sold"] * rows + ["listing"],
            "sold_at_source": ["transition_observed"] * rows + [""],
            "sold_at": pd.to_datetime(["2026-09-04 19:01:10"] * rows + [None]),
        })
        spikes = _bootstrap_sale_time_spikes(frame)
        self.assertEqual(list(spikes), [pd.Timestamp("2026-09-04 19:01:10")])

    def test_normal_distributed_transitions_are_retained(self):
        frame = pd.DataFrame({
            "status": ["sold"] * 60 + ["listing"] * 40,
            "sold_at_source": ["transition_observed"] * 60 + [""] * 40,
            "sold_at": pd.to_datetime(
                [f"2026-09-{1 + index // 10:02d} {index % 10:02d}:00:00" for index in range(60)]
                + [None] * 40
            ),
        })
        self.assertTrue(_bootstrap_sale_time_spikes(frame).empty)


if __name__ == "__main__":
    unittest.main()
