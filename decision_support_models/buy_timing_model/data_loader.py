"""Load and align data-file outputs; no imports from other model folders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ALTERNATIVE_OOF, DEMAND_OOF


DEMAND_REQUIRED = {
    "ticket_id", "landmark_at", "horizon_days", "true_state",
    "p_active", "p_sold", "p_deleted", "price",
}
ALTERNATIVE_REQUIRED = {
    "ticket_id", "landmark_at", "horizon_days", "true_alternative", "p_alternative",
}


def load_oof(demand_path: Path = DEMAND_OOF, alternative_path: Path = ALTERNATIVE_OOF) -> pd.DataFrame:
    demand = pd.read_csv(demand_path, parse_dates=["landmark_at"])
    alternative = pd.read_csv(alternative_path, parse_dates=["landmark_at"])
    missing_demand = DEMAND_REQUIRED - set(demand)
    missing_alternative = ALTERNATIVE_REQUIRED - set(alternative)
    if missing_demand or missing_alternative:
        raise ValueError(f"Missing OOF columns: demand={sorted(missing_demand)}, alternative={sorted(missing_alternative)}")
    if "fold" not in demand or "fold" not in alternative:
        raise ValueError("OOF inputs must include temporal fold identifiers")
    demand_probability = demand[["p_active", "p_sold", "p_deleted"]].apply(
        pd.to_numeric, errors="coerce"
    )
    alternative_probability = pd.to_numeric(
        alternative["p_alternative"], errors="coerce"
    )
    if (
        not np.isfinite(demand_probability.to_numpy(float)).all()
        or not demand_probability.ge(0).all().all()
        or not demand_probability.le(1).all().all()
        or not np.allclose(demand_probability.sum(axis=1), 1.0, atol=1e-7)
    ):
        raise ValueError("Demand OOF probabilities are invalid or do not sum to one")
    if (
        not np.isfinite(alternative_probability.to_numpy(float)).all()
        or not alternative_probability.between(0, 1, inclusive="both").all()
    ):
        raise ValueError("Alternative OOF probabilities are invalid")
    if not demand["true_state"].isin([0, 1, 2]).all():
        raise ValueError("Demand OOF contains an invalid true_state")
    if not alternative["true_alternative"].isin([0, 1]).all():
        raise ValueError("Alternative OOF contains an invalid target")
    demand = demand.rename(columns={"fold": "demand_fold"})
    alternative = alternative.rename(columns={"fold": "alternative_fold"})
    keys = ["ticket_id", "landmark_at", "horizon_days"]
    alternative_columns = keys + [name for name in alternative if name not in keys and name not in demand]
    frame = demand.merge(alternative[alternative_columns], on=keys, how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("Demand and alternative OOF inputs have no aligned rows")
    # Alternative OOF keeps a horizon-specific audit column; normalize it.
    frame["potential_savings"] = 0.0
    for horizon in frame.horizon_days.unique():
        source = f"potential_savings_{int(horizon)}d"
        if source in frame:
            mask = frame.horizon_days.eq(horizon)
            frame.loc[mask, "potential_savings"] = pd.to_numeric(frame.loc[mask, source], errors="coerce").fillna(0)
    reference_candidates = [name for name in ("fair_price", "market_prior_sold_median", "market_price_median") if name in frame]
    frame["reference_price"] = np.nan
    for column in reference_candidates:
        values = pd.to_numeric(frame[column], errors="coerce").where(lambda item: item.gt(0))
        frame["reference_price"] = frame["reference_price"].fillna(values)
    frame["reference_price"] = frame["reference_price"].fillna(pd.to_numeric(frame.price, errors="coerce"))
    if frame["reference_price"].isna().any() or (frame["reference_price"] <= 0).any():
        raise ValueError("Cannot establish a valid reference price")
    frame["discount_ratio"] = ((frame.reference_price - frame.price) / frame.reference_price).clip(-2, 1)
    frame["disappearance_probability"] = (frame.p_sold + frame.p_deleted).clip(0, 1)
    return frame.sort_values(["landmark_at", "ticket_id", "horizon_days"]).reset_index(drop=True)
