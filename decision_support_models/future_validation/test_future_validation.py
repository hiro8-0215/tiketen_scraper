import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run


class FutureValidationTests(unittest.TestCase):
    def test_snapshot_audit_detects_new_rows_and_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root / "old", root / "new"
            old.mkdir(); new.mkdir()
            columns = ["ticket_id", "event_id", "created_at_unix", "status"]
            pd.DataFrame([
                ["old-a", "event-a", "1", "listing"],
                ["old-b", "event-b", "2", "sold"],
            ], columns=columns).to_csv(old / "x_master.csv", index=False)
            pd.DataFrame([
                ["rotated-a", "event-a", "1", "deleted"],
                ["old-b", "event-b", "2", "sold"],
                ["new-c", "event-c", "3", "listing"],
            ], columns=columns).to_csv(new / "x_master.csv", index=False)
            result = run.snapshot_audit(old, new)
        self.assertEqual(result["new_ticket_rows"], 1)
        self.assertEqual(result["transitions"]["listing->deleted"], 1)
        self.assertEqual(result["new_sold_id_count"], 0)
        self.assertTrue(result["sold_collection_requires_review"])

    def test_launch_json_contains_complete_future_validation(self):
        launch = json.loads((run.ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8"))
        names = {item["name"] for item in launch["configurations"]}
        self.assertTrue(any("[25 3モデル検証][1 完全実行]" in name for name in names))

    def test_frozen_artifacts_exist(self):
        missing = [str(path) for path in run.MODELS if not path.exists()]
        self.assertEqual(missing, [])

    def test_runner_persists_completion_before_opening_viewer(self):
        source = (run.HERE / "run.py").read_text(encoding="utf-8")
        completed_save = source.index("save(output/'manifest.json',manifest)", source.index("manifest['status']='checked'"))
        viewer_call = source.index("viewer = subprocess.run")
        self.assertLess(completed_save, viewer_call)


if __name__ == "__main__":
    unittest.main()
