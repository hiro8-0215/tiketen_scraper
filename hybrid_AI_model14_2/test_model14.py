import unittest
import numpy as np
from config import FORBIDDEN_MODEL_COLUMNS
from data_loader import model_feature_columns, prepare_dataset
from make_folds import assign_folds


class Model14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = prepare_dataset()
        cls.folds = assign_folds(cls.df)

    def test_sold_positive_prices_only(self):
        self.assertTrue(self.df.status.eq("sold").all())
        self.assertTrue(self.df.price.gt(0).all())

    def test_ticket_ids_are_unique(self):
        self.assertFalse(self.df.ticket_id.duplicated().any())

    def test_exact_descriptions_never_cross_folds(self):
        counts = self.df.assign(fold=self.folds).groupby("duplicate_group").fold.nunique()
        self.assertEqual(int(counts.max()), 1)

    def test_no_forbidden_model_columns(self):
        numeric, categorical = model_feature_columns(self.df)
        self.assertFalse(set(numeric + categorical) & FORBIDDEN_MODEL_COLUMNS)

    def test_prior_sales_are_strictly_asof(self):
        self.assertTrue(np.isfinite(self.df.event_prior_sold_count).all())
        self.assertTrue(self.df.event_prior_sold_count.ge(0).all())


if __name__ == "__main__":
    unittest.main()

