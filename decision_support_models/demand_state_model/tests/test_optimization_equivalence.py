import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import TRAILING_DAYS, _count_between, add_market_features
from timeline import STATIC_KEEP, _landmarks, add_end_times, build_landmarks


def legacy_landmarks(tickets, horizons, cutoff):
    prepared, cutoff = add_end_times(tickets, cutoff)
    static = [column for column in prepared.columns if column in STATIC_KEEP]
    rows = []
    for row in prepared.itertuples(index=False):
        values = row._asdict()
        for moment in _landmarks(values["first_observed_at"], values["known_until"]):
            record = {column: values[column] for column in static}
            record["landmark_at"] = moment
            record["days_since_listing"] = (
                moment - values["first_observed_at"]
            ).total_seconds() / 86400
            performance = values.get("performance_at")
            record["days_until_event"] = (
                (performance - moment).total_seconds() / 86400
                if pd.notna(performance) else np.nan
            )
            for horizon in horizons:
                deadline = moment + pd.Timedelta(days=horizon)
                label = -1
                if pd.notna(values.get("outcome_at")) and values["outcome_at"] <= deadline:
                    label = 1 if values["status"] == "sold" else 2
                elif values["known_until"] >= deadline:
                    label = 0
                record[f"state_{horizon}d"] = label
            rows.append(record)
    result = pd.DataFrame(rows)
    result["landmark_date"] = result["landmark_at"].dt.floor("D")
    return result.sort_values(["landmark_at", "ticket_id"]).reset_index(drop=True)


def legacy_market(landmarks, tickets):
    result, tickets = landmarks.copy(), tickets.copy()
    state_rows, cheaper_lookup = [], {}
    tickets["end_at"] = tickets["outcome_at"]
    for event_id, event_landmarks in result.groupby(
        "event_id", dropna=False, observed=True
    ):
        source = tickets[tickets["event_id"].eq(event_id)].copy()
        first = source.first_observed_at.dropna().sort_values().to_numpy("datetime64[ns]")
        sold = source.sold_at.dropna().sort_values().to_numpy("datetime64[ns]")
        deleted = source.loc[
            source.status.eq("deleted"), "last_observed_at"
        ].dropna().sort_values().to_numpy("datetime64[ns]")
        sold_source = source[source.sold_at.notna()].sort_values("sold_at")
        for moment in sorted(event_landmarks.landmark_at.unique()):
            moment = pd.Timestamp(moment)
            active = source.loc[
                source.first_observed_at.le(moment)
                & (source.end_at.isna() | source.end_at.gt(moment))
            ]
            prices = pd.to_numeric(active.price, errors="coerce").dropna().to_numpy(float)
            prior = pd.to_numeric(
                sold_source.loc[sold_source.sold_at.lt(moment), "price"],
                errors="coerce",
            ).dropna()
            record = {
                "event_id": event_id,
                "landmark_at": moment,
                "market_active_count": len(active),
                "market_price_min": float(np.min(prices)) if len(prices) else np.nan,
                "market_price_median": float(np.median(prices)) if len(prices) else np.nan,
                "market_price_q25": float(np.quantile(prices, .25)) if len(prices) else np.nan,
                "market_price_q75": float(np.quantile(prices, .75)) if len(prices) else np.nan,
                "market_prior_sold_count": len(prior),
                "market_prior_sold_median": float(prior.median()) if len(prior) else np.nan,
            }
            for days in TRAILING_DAYS:
                begin = moment - pd.Timedelta(days=days)
                record[f"market_new_{days}d"] = _count_between(first, begin, moment)
                record[f"market_sold_{days}d"] = _count_between(sold, begin, moment)
                record[f"market_deleted_{days}d"] = _count_between(deleted, begin, moment)
            state_rows.append(record)
            cheaper_lookup[(event_id, moment)] = np.sort(prices)
    market = pd.DataFrame(state_rows).drop_duplicates(["event_id", "landmark_at"])
    result = result.merge(
        market, on=["event_id", "landmark_at"], how="left", validate="many_to_one"
    )
    result["price_to_market_median"] = result.price / result.market_price_median.replace(0, np.nan)
    result["price_to_prior_sold_median"] = result.price / result.market_prior_sold_median.replace(0, np.nan)
    cheaper = []
    for row in result[["event_id", "landmark_at", "price"]].itertuples(index=False):
        values = cheaper_lookup.get((row.event_id, row.landmark_at), np.array([]))
        price = float(row.price) if pd.notna(row.price) else np.nan
        cheaper.append(
            int(np.searchsorted(values, price, side="left")) if np.isfinite(price) else 0
        )
    result["market_cheaper_active_count"] = cheaper
    for days in TRAILING_DAYS:
        result[f"market_sell_through_{days}d"] = result[f"market_sold_{days}d"] / np.maximum(
            result.market_active_count + result[f"market_sold_{days}d"], 1
        )
        result[f"market_disappearance_{days}d"] = (
            result[f"market_sold_{days}d"] + result[f"market_deleted_{days}d"]
        ) / np.maximum(
            result.market_active_count
            + result[f"market_sold_{days}d"]
            + result[f"market_deleted_{days}d"],
            1,
        )
    return result


class OptimizationEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.cutoff = pd.Timestamp("2026-01-10 12:00")
        self.tickets = pd.DataFrame({
            "ticket_id": ["s1", "d1", "a1", "s2", "d2"],
            "event_id": ["e1", "e1", "e1", "e2", "e2"],
            "first_observed_at": pd.to_datetime([
                "2026-01-01 10:00", "2026-01-02 12:00", "2026-01-03 08:00",
                "2026-01-01 09:30", "2026-01-04 09:30",
            ]),
            "last_observed_at": pd.to_datetime([
                "2026-01-05 10:00", "2026-01-07 12:00", "2026-01-10 12:00",
                "2026-01-08 09:30", "2026-01-09 09:30",
            ]),
            "sold_at": pd.to_datetime([
                "2026-01-05 10:00", None, None, "2026-01-08 09:30", None,
            ]),
            "status": ["sold", "deleted", "listing", "sold", "deleted"],
            "performance_at": pd.to_datetime(["2026-02-01"] * 5),
            "price": [10000.0, 12000.0, 9000.0, 15000.0, 13000.0],
            "duplicate_group": ["g1", "g2", "g3", "g4", "g5"],
        })

    def test_vectorized_landmarks_match_legacy_values(self):
        expected = legacy_landmarks(self.tickets, (1, 3, 7), self.cutoff)
        actual = build_landmarks(self.tickets, (1, 3, 7), self.cutoff)
        assert_frame_equal(actual, expected, check_dtype=False, check_categorical=False)

    def test_vectorized_market_features_match_legacy_values(self):
        prepared, _ = add_end_times(self.tickets, self.cutoff)
        landmarks = build_landmarks(prepared, (1, 3, 7), self.cutoff)
        expected = legacy_market(landmarks, prepared)
        actual = add_market_features(landmarks, prepared)
        columns = [
            column for column in actual
            if column.startswith("market_") or column.startswith("price_to_")
        ]
        assert_frame_equal(
            actual[columns], expected[columns], check_dtype=False,
            check_categorical=False, rtol=1e-12, atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
