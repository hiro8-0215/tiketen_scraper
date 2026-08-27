import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader
from config import SEMANTIC_CATEGORICAL_FEATURES, SEMANTIC_NUMERIC_FEATURES, SEMANTIC_SCHEMA_VERSION


class SemanticCoverageTest(unittest.TestCase):
    def test_partial_coverage_is_rejected(self):
        record = {
            "text_hash": data_loader._semantic_hash("one"),
            "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
            **{column: "不明" for column in SEMANTIC_CATEGORICAL_FEATURES},
            **{column: 0 for column in SEMANTIC_NUMERIC_FEATURES},
        }
        frame = pd.DataFrame({"raw_description": ["one", "two"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            semantic_path, manifest_path = root / "semantic.csv", root / "manifest.json"
            pd.DataFrame([record]).to_csv(semantic_path, index=False)
            manifest_path.write_text(json.dumps({
                "complete": True, "schema_version": SEMANTIC_SCHEMA_VERSION,
                "parse_errors": 0, "unique_descriptions": 2,
            }), encoding="utf-8")
            with patch.object(data_loader, "SEMANTIC_FEATURES_FILE", semantic_path), patch.object(data_loader, "SEMANTIC_MANIFEST_FILE", manifest_path):
                with self.assertRaises(ValueError):
                    data_loader.attach_complete_semantics(frame)


if __name__ == "__main__":
    unittest.main()
