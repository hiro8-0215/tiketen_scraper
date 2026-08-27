"""Trainer helpers that keep evaluation and prediction rows deterministic."""
from __future__ import annotations

import numpy as np
from torch.utils.data import SequentialSampler
from transformers import Trainer


class OrderedEvalTrainer(Trainer):
    """Group training batches by length, but never reorder eval/predict rows.

    Transformers 5.13 applies ``train_sampling_strategy=group_by_length`` to
    its evaluation sampler too. ``Trainer.predict`` then returns predictions in
    sampler order, while Model15 stores them in dataset order. A sequential
    sampler is therefore mandatory for a row-aligned OOF feature.
    """

    def _get_eval_sampler(self, eval_dataset):
        if eval_dataset is None:
            return None
        return SequentialSampler(eval_dataset)


def qwen_regression_metrics(eval_prediction):
    predictions, labels = eval_prediction
    pred_log = np.asarray(predictions, dtype=float).reshape(-1)
    true_log = np.asarray(labels, dtype=float).reshape(-1)
    pred_yen = np.maximum(0.0, np.expm1(pred_log))
    true_yen = np.maximum(0.0, np.expm1(true_log))
    return {
        "mae_yen": float(np.mean(np.abs(pred_yen - true_yen))),
        "log_mse": float(np.mean((pred_log - true_log) ** 2)),
    }
