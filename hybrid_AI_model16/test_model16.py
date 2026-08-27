import unittest

import numpy as np
import pandas as pd

from config import (
    BASE_EXPERTS,
    BERT_PCA_DIMS,
    FORBIDDEN_MODEL_COLUMNS,
    N_INNER_FOLDS,
    SEED,
    TARGET,
)
from data_loader import catboost_frame, model_feature_columns, prepare_dataset
from modeling import (
    assign_inner_folds,
    blend_predictions,
    mape_training_weight,
    optimize_global_weights,
)


class Model16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = prepare_dataset()
        cls.numeric, cls.categorical = model_feature_columns(cls.df)

    def test_same_clean_sold_population(self):
        self.assertEqual(len(self.df), 7313)
        self.assertTrue(self.df[TARGET].between(2_000, 150_000).all())

    def test_no_forbidden_or_target_features(self):
        features = set(self.numeric + self.categorical)
        self.assertFalse(features & FORBIDDEN_MODEL_COLUMNS)
        self.assertNotIn(TARGET, features)

    def test_engineered_listing_time_features_exist(self):
        for column in [
            "delivery_channel", "delivery_timing", "event_prior_available",
            "prior_median_to_base_price", "log_event_prior_sold_count",
            "seat_rule_category", "perf_day_of_week_category",
            "perf_day_sin", "perf_day_cos", "perf_hour_sin", "perf_hour_cos",
        ]:
            self.assertIn(column, self.numeric + self.categorical)

    def test_numeric_features_have_no_infinity(self):
        values = self.df[self.numeric].apply(pd.to_numeric, errors="coerce")
        self.assertFalse(np.isinf(values.to_numpy(dtype=float)).any())

    def test_constant_features_are_removed(self):
        for column in self.numeric + self.categorical:
            self.assertGreater(self.df[column].nunique(dropna=False), 1)

    def test_catboost_categories_are_strings(self):
        frame = catboost_frame(self.df.head(50), self.numeric, self.categorical)
        for column in self.categorical:
            self.assertTrue(frame[column].map(lambda value: isinstance(value, str)).all())

    def test_inner_duplicate_groups_never_cross(self):
        folds = assign_inner_folds(self.df, N_INNER_FOLDS, SEED, TARGET)
        maximum = pd.DataFrame(
            {"group": self.df.duplicate_group, "fold": folds}
        ).groupby("group").fold.nunique().max()
        self.assertLessEqual(maximum, 1)

    def test_mape_weight_is_global_and_inverse_price(self):
        y = np.array([5_000.0, 10_000.0, 50_000.0])
        weight = mape_training_weight(y)
        self.assertGreater(weight[0], weight[1])
        self.assertGreater(weight[1], weight[2])
        self.assertAlmostEqual(float(weight.mean()), 1.0)

    def test_global_blend_has_one_constant_simplex_vector(self):
        y = np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0])
        matrix = np.column_stack([
            y + 500, y - 500, y * 1.02, y * 0.98,
        ])
        weights = optimize_global_weights(y, matrix)
        self.assertEqual(len(weights), len(BASE_EXPERTS))
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)
        self.assertEqual(blend_predictions(matrix, weights).shape, y.shape)

    def test_bert_dimensions_are_low_capacity(self):
        self.assertEqual(tuple(sorted(BERT_PCA_DIMS)), BERT_PCA_DIMS)
        self.assertLess(max(BERT_PCA_DIMS), 768)

    def test_qwen_is_not_an_expert(self):
        self.assertFalse(any("qwen" in name.lower() for name in BASE_EXPERTS))


if __name__ == "__main__":
    unittest.main()
