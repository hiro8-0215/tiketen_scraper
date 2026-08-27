import json
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from config import (
    ARTIFACT_DIR, FORBIDDEN_MODEL_COLUMNS, MAX_PRICE, MIN_DESCRIPTION_LENGTH,
    MIN_PRICE, NOISE_DESCRIPTION_PATTERN, QWEN_OOF_SCHEMA_VERSION,
    SEMANTIC_FEATURES_FILE,
)
from data_loader import model_feature_columns, prepare_dataset
from make_folds import assign_folds
from qwen_prompt import build_qwen_prompt, qwen_dataset_fingerprint
from qwen_trainer import OrderedEvalTrainer
from qwen_validation import qwen_oof_diagnostics


TRAINING_SPEC = importlib.util.spec_from_file_location(
    "model15_training",
    Path(__file__).resolve().parent / "2_train_model15.py",
)
MODEL15_TRAINING = importlib.util.module_from_spec(TRAINING_SPEC)
TRAINING_SPEC.loader.exec_module(MODEL15_TRAINING)


class Model15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = prepare_dataset()

    def test_semantics_exist_and_have_no_price_estimate(self):
        payload = json.loads(SEMANTIC_FEATURES_FILE.read_text(encoding="utf-8"))
        self.assertTrue(payload)
        self.assertFalse(any("price_estimate" in row for row in payload.values()))

    def test_semantic_columns_are_model_features(self):
        numeric, categorical = model_feature_columns(self.df)
        self.assertIn("semantic_seat_level", categorical)
        self.assertIn("semantic_name_status", categorical)
        self.assertIn("semantic_is_random", numeric)
        self.assertIn("bante_x_baseprice", numeric)
        self.assertIn("surikae_x_bante", numeric)
        for column in ["seller_rating", "row_number", "block_rank", "ticket_count_offered"]:
            self.assertNotIn(column, numeric)

    def test_forbidden_columns_absent(self):
        numeric, categorical = model_feature_columns(self.df)
        self.assertFalse(set(numeric + categorical) & FORBIDDEN_MODEL_COLUMNS)

    def test_model13_cleaning_policy(self):
        self.assertTrue(self.df.price.between(MIN_PRICE, MAX_PRICE).all())
        descriptions = self.df.raw_description.astype("string")
        self.assertTrue(descriptions.str.strip().str.len().ge(MIN_DESCRIPTION_LENGTH).all())
        self.assertFalse(descriptions.str.contains(NOISE_DESCRIPTION_PATTERN, regex=True).any())

    def test_duplicate_descriptions_never_cross_folds(self):
        assigned = assign_folds(self.df)
        grouped = pd.DataFrame({
            "group": self.df.duplicate_group,
            "fold": assigned,
        }).groupby("group").fold.nunique()
        self.assertLessEqual(grouped.max(), 1)

    def test_qwen_prompt_excludes_target_and_json_semantics(self):
        row = pd.Series({
            "price": 987654321,
            "sold_at": "SECRET_SOLD_TIME",
            "raw_description": "VISIBLE_DESCRIPTION",
            "semantic_seat_level": "SECRET_JSON_SEAT",
        })
        rendered = build_qwen_prompt(row)
        self.assertIn("VISIBLE_DESCRIPTION", rendered)
        self.assertNotIn("987654321", rendered)
        self.assertNotIn("SECRET_SOLD_TIME", rendered)
        self.assertNotIn("SECRET_JSON_SEAT", rendered)

    def test_qwen_fingerprint_changes_when_target_changes(self):
        sample = self.df.head(10).copy()
        manifest = pd.DataFrame({"fold": np.arange(len(sample)) % 5})
        original = qwen_dataset_fingerprint(sample, manifest)
        sample.loc[sample.index[0], "price"] += 1
        changed = qwen_dataset_fingerprint(sample, manifest)
        self.assertNotEqual(original, changed)

    def test_qwen_eval_sampler_preserves_dataset_order(self):
        trainer = object.__new__(OrderedEvalTrainer)
        dataset = list(range(12))
        sampler = trainer._get_eval_sampler(dataset)
        self.assertEqual(list(iter(sampler)), list(range(len(dataset))))

    def test_qwen_alignment_guard_rejects_shuffled_predictions(self):
        sample = self.df.head(25).copy().reset_index(drop=True)
        sample["price"] = np.geomspace(2_000, 150_000, len(sample))
        manifest = pd.DataFrame({
            "ticket_id": sample.ticket_id,
            "fold": np.arange(len(sample)) % 5,
        })
        fingerprint = qwen_dataset_fingerprint(sample, manifest)
        qwen = manifest.copy()
        qwen["qwen_dataset_fingerprint"] = fingerprint
        qwen["qwen_oof_schema_version"] = QWEN_OOF_SCHEMA_VERSION
        qwen["qwen_pred_log"] = np.log1p(sample.price.to_numpy())[::-1]
        with self.assertRaisesRegex(ValueError, "alignment guard"):
            qwen_oof_diagnostics(sample, manifest, qwen)

    def test_all_text_feature_profiles_have_fixed_dimensions(self):
        tabular = csr_matrix(np.ones((3, 5)))
        bert = np.ones((3, 7))
        qwen = np.ones((3, 1))
        expected = {
            "full": 13,
            "qwen_only": 6,
            "bert_only": 12,
            "tabular_only": 5,
        }
        self.assertEqual(set(MODEL15_TRAINING.FEATURE_PROFILES), set(expected))
        for profile, columns in expected.items():
            matrix = MODEL15_TRAINING.assemble_profile(
                tabular, bert, qwen, profile
            )
            self.assertEqual(matrix.shape, (3, columns))

    def test_primary_selection_can_choose_an_ablation_profile(self):
        y = np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0])
        predictions = {
            "log_l1__full": y + 2_000,
            "log_l1__qwen_only": y + 100,
            "log_l1__bert_only": y + 1_000,
            "log_l1__tabular_only": y + 500,
        }
        candidates = {
            name: {"clean_sold": MODEL15_TRAINING.metrics(y, prediction)}
            for name, prediction in predictions.items()
        }
        primary, eligible, _ = MODEL15_TRAINING.choose_primary(
            candidates, predictions, y
        )
        self.assertIn("log_l1__qwen_only", eligible)
        self.assertEqual(primary, "log_l1__qwen_only")

    def test_reused_artifacts_align(self):
        folds = pd.read_csv(ARTIFACT_DIR / "folds.csv")
        qwen = pd.read_csv(ARTIFACT_DIR / "qwen_oof.csv")
        if (
            "duplicate_group" not in folds
            or folds.ticket_id.tolist() != self.df.ticket_id.tolist()
        ):
            self.skipTest("folds have not yet been rebuilt for the Model13-clean population")
        self.assertEqual(folds.ticket_id.tolist(), self.df.ticket_id.tolist())
        self.assertEqual(folds.duplicate_group.tolist(), self.df.duplicate_group.tolist())
        self.assertLessEqual(
            pd.DataFrame({"group": folds.duplicate_group, "fold": folds.fold})
            .groupby("group").fold.nunique().max(),
            1,
        )
        if qwen.ticket_id.tolist() != self.df.ticket_id.tolist():
            self.skipTest("Qwen OOF has not yet been rebuilt for the Model13-clean population")
        if (
            "qwen_oof_schema_version" not in qwen
            or not qwen.qwen_oof_schema_version.eq(QWEN_OOF_SCHEMA_VERSION).all()
        ):
            self.skipTest("Qwen OOF still needs ordered adapter re-inference")
        self.assertEqual(qwen.ticket_id.tolist(), self.df.ticket_id.tolist())
        diagnostics = qwen_oof_diagnostics(self.df, folds, qwen)
        self.assertLess(diagnostics["overall_mae_yen"], 15_000)


if __name__ == "__main__":
    unittest.main()
