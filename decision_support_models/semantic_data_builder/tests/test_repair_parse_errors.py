import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair_parse_errors import recover_frame
from schema import unknown


class ParseErrorRepairTest(unittest.TestCase):
    def test_only_parse_error_rows_are_replaced(self):
        good = {"text_hash": "good", **unknown("qwen_semantic_compact")}
        bad = {"text_hash": "bad", **unknown("parse_error")}
        frame = pd.DataFrame([good, bad])
        failures = {
            "bad": {
                "text_hash": "bad",
                "first_response": "[0,0,1,0,0,3,0,0]",
                "retry_response": "[0,0,1,0,0,3,0,0]",
            }
        }

        repaired, report = recover_frame(frame, failures)

        self.assertEqual(report["recovered"], 1)
        self.assertEqual(report["unresolved"], 0)
        indexed = repaired.set_index("text_hash")
        self.assertEqual(indexed.at["good", "semantic_source"], "qwen_semantic_compact")
        self.assertEqual(indexed.at["bad", "semantic_is_fc_early"], 1)
        self.assertEqual(indexed.at["bad", "semantic_is_random"], 1)

    def test_invalid_log_response_stays_parse_error(self):
        frame = pd.DataFrame([{"text_hash": "bad", **unknown("parse_error")}])
        repaired, report = recover_frame(
            frame,
            {"bad": {"first_response": "[9]", "retry_response": "broken"}},
        )
        self.assertEqual(report["recovered"], 0)
        self.assertEqual(report["unresolved"], 1)
        self.assertEqual(repaired.iloc[0].semantic_source, "parse_error")


if __name__ == "__main__":
    unittest.main()
