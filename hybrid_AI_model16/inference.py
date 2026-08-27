"""Inference helpers for a completed Model16 payload."""
from __future__ import annotations

import joblib
import numpy as np

from config import ARTIFACT_DIR, PIPELINE_VERSION, RAW_PRICE_SCALE, WEIGHT_EPSILON
from data_loader import catboost_frame


def predict_payload(payload, df, bert_embeddings=None):
    if payload.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("Model16 payload version is stale")
    order = payload["expert_order"]
    weights = np.asarray(payload["weights"], float)
    if (
        len(order) != len(weights)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("Invalid Model16 global weights")
    predictions = []
    experts = payload["experts"]
    for index, name in enumerate(order):
        if weights[index] < WEIGHT_EPSILON:
            predictions.append(np.zeros(len(df), float))
            continue
        expert = experts[name]
        if name in {"lgbm_log_mae", "lgbm_raw_mape"}:
            matrix = expert["preprocessor"].transform(df)
            native = expert["model"].booster_.predict(matrix)
            prediction = (
                np.maximum(0, np.expm1(native))
                if name == "lgbm_log_mae"
                else np.maximum(0, native * RAW_PRICE_SCALE)
            )
        elif name == "catboost_raw_mae":
            frame = catboost_frame(df, payload["numeric"], payload["categorical"])
            prediction = np.maximum(
                0, expert["model"].predict(frame) * RAW_PRICE_SCALE
            )
        elif name == "bert_ridge":
            if bert_embeddings is None:
                raise ValueError("This Model16 requires BERT embeddings")
            else:
                transformed = expert["pca"].transform(bert_embeddings)
                transformed = expert["scaler"].transform(transformed)
                prediction = np.maximum(
                    0, np.expm1(expert["model"].predict(transformed))
                )
        else:
            raise ValueError(f"Unknown Model16 expert: {name}")
        predictions.append(prediction)
    return np.maximum(0, np.column_stack(predictions) @ weights)


def load_and_predict(df, bert_embeddings=None, model_path=None):
    path = model_path or ARTIFACT_DIR / "model16.joblib"
    payload = joblib.load(path)
    return predict_payload(payload, df, bert_embeddings)
