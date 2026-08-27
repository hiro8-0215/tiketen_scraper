import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import parse_response, text_hash, unknown


class SemanticSchemaTest(unittest.TestCase):
    def test_hash_is_stable_after_outer_whitespace(self):
        self.assertEqual(text_hash("  同行です  "), text_hash("同行です"))

    def test_price_key_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_response('{"price_estimate": 10000}')

    def test_compact_response_decodes_all_fields(self):
        value = parse_response("[1,2,1,5,3,4,2,1,0]")
        self.assertEqual(value["semantic_seat_level"], "アリーナ")
        self.assertEqual(value["semantic_row_position"], "前方")
        self.assertEqual(value["semantic_name_status"], "名義変更可")
        self.assertEqual(value["semantic_distribution_type"], "番手選択")
        self.assertEqual(value["semantic_is_fc_early"], 1)
        self.assertEqual(value["semantic_is_random"], 0)

    def test_compact_response_accepts_full_width_digits(self):
        value = parse_response("回答:［０，０，０，０，０，０，０，０，１］".replace("［", "[").replace("］", "]"))
        self.assertEqual(value["semantic_is_random"], 1)

    def test_compact_response_accepts_common_separators(self):
        value = parse_response("[0 0 0 0 0 3 0 0 1]")
        self.assertEqual(value["semantic_distribution_type"], "ランダム")
        self.assertEqual(value["semantic_is_random"], 1)

    def test_eight_field_response_recovers_redundant_flags(self):
        value = parse_response("[0,0,1,0,0,3,0,0]")
        self.assertEqual(value["semantic_winning_route"], "FC初期")
        self.assertEqual(value["semantic_distribution_type"], "ランダム")
        self.assertEqual(value["semantic_is_fc_early"], 1)
        self.assertEqual(value["semantic_is_random"], 1)
        self.assertEqual(value["semantic_source"], "qwen_semantic_compact_recovered8")

    def test_eight_field_response_rejects_invalid_category(self):
        with self.assertRaises(ValueError):
            parse_response("[0,0,1,9,0,3,0,0]")

    def test_invalid_compact_response_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_response("[9,0,0]")

    def test_unknown_has_complete_schema(self):
        value = unknown()
        self.assertEqual(value["semantic_seat_level"], "不明")
        self.assertEqual(value["semantic_is_random"], 0)


if __name__ == "__main__":
    unittest.main()
