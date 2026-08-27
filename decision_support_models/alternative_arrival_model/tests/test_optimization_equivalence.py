import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MIN_SAVINGS_PCT, MIN_SAVINGS_YEN, SEMANTIC_COMPARABLE_FIELDS
from features import add_market_features
from timeline import COMPARABLE, STATIC_KEEP, _times, build_landmarks, prepare_end_times


def legacy_landmarks(tickets, horizons, cutoff):
    prepared = prepare_end_times(tickets)
    cutoff = pd.Timestamp(cutoff)
    static = [column for column in prepared if column in STATIC_KEEP]
    groups = {}
    for key, value in prepared.groupby(list(COMPARABLE), dropna=False, observed=True):
        ordered = value.sort_values("first_observed_at")
        groups[key] = (
            ordered.first_observed_at.to_numpy("datetime64[ns]"),
            pd.to_numeric(ordered.price, errors="coerce").to_numpy(float),
            {
                column: ordered[column].fillna("不明").astype(str).to_numpy()
                for column in SEMANTIC_COMPARABLE_FIELDS
            },
        )
    rows = []
    for row in prepared.itertuples(index=False):
        values = row._asdict()
        end = values["outcome_at"] if pd.notna(values["outcome_at"]) else min(
            cutoff,
            values["performance_at"]
            if pd.notna(values["performance_at"]) else cutoff,
        )
        candidate_times, candidate_prices, candidate_semantics = groups[
            tuple(values[column] for column in COMPARABLE)
        ]
        for moment in _times(values["first_observed_at"], end):
            record = {column: values[column] for column in static}
            record["landmark_at"] = moment
            record["days_since_listing"] = (
                moment - values["first_observed_at"]
            ).total_seconds() / 86400
            record["days_until_event"] = (
                (values["performance_at"] - moment).total_seconds() / 86400
                if pd.notna(values["performance_at"]) else np.nan
            )
            price = float(values["price"])
            threshold = price - max(MIN_SAVINGS_YEN, price * MIN_SAVINGS_PCT)
            for horizon in horizons:
                deadline = moment + pd.Timedelta(days=horizon)
                observed = deadline <= cutoff and (
                    pd.isna(values["performance_at"])
                    or deadline <= values["performance_at"]
                )
                left = np.searchsorted(candidate_times, np.datetime64(moment), side="right")
                right = np.searchsorted(candidate_times, np.datetime64(deadline), side="right")
                prices = candidate_prices[left:right]
                times = candidate_times[left:right]
                cheaper = np.isfinite(prices) & (prices <= threshold)
                for column in SEMANTIC_COMPARABLE_FIELDS:
                    current = str(values.get(column, "不明"))
                    candidates = candidate_semantics[column][left:right]
                    if current != "不明":
                        cheaper &= (candidates == "不明") | (candidates == current)
                found = bool(cheaper.any())
                best = float(prices[cheaper].min()) if found else np.nan
                first = (
                    pd.Timestamp(times[np.flatnonzero(cheaper)[0]]) if found else pd.NaT
                )
                record[f"alternative_{horizon}d"] = int(found) if observed else -1
                record[f"future_best_price_{horizon}d"] = best
                record[f"alternative_first_at_{horizon}d"] = first
                record[f"potential_savings_{horizon}d"] = (
                    max(0.0, price - best) if found else 0.0
                )
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["landmark_at", "ticket_id"]
    ).reset_index(drop=True)


def legacy_market(landmarks, tickets):
    result, tickets = landmarks.copy(), tickets.copy()
    states = []
    for event_id, moments in result.groupby(
        "event_id", dropna=False, observed=True
    ):
        source = tickets[tickets.event_id.eq(event_id)]
        for moment in sorted(moments.landmark_at.unique()):
            moment = pd.Timestamp(moment)
            active = source[
                source.first_observed_at.le(moment)
                & (source.outcome_at.isna() | source.outcome_at.gt(moment))
            ]
            prices = pd.to_numeric(active.price, errors="coerce").dropna()
            record = {
                "event_id": event_id,
                "landmark_at": moment,
                "market_active_count": len(active),
                "market_price_min": prices.min() if len(prices) else np.nan,
                "market_price_median": prices.median() if len(prices) else np.nan,
                "market_price_q25": prices.quantile(.25) if len(prices) else np.nan,
                "market_price_q75": prices.quantile(.75) if len(prices) else np.nan,
            }
            for days in (1, 3, 7):
                start = moment - pd.Timedelta(days=days)
                record[f"market_new_{days}d"] = int(
                    source.first_observed_at.between(
                        start, moment, inclusive="left"
                    ).sum()
                )
                record[f"market_sold_{days}d"] = int(
                    source.sold_at.between(start, moment, inclusive="left").sum()
                )
                deleted = source.loc[source.status.eq("deleted"), "last_observed_at"]
                record[f"market_deleted_{days}d"] = int(
                    deleted.between(start, moment, inclusive="left").sum()
                )
            states.append(record)
    market = pd.DataFrame(states).drop_duplicates(["event_id", "landmark_at"])
    result = result.merge(
        market, on=["event_id", "landmark_at"], how="left", validate="many_to_one"
    )
    result["price_to_market_median"] = result.price / result.market_price_median.replace(0, np.nan)
    result["market_price_spread"] = result.market_price_q75 - result.market_price_q25
    return result


class OptimizationEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.cutoff = pd.Timestamp("2026-01-10 12:00")
        self.tickets = pd.DataFrame({
            "ticket_id": ["current", "cheap", "wrong_seat", "other", "sold"],
            "event_id": ["e1", "e1", "e1", "e2", "e1"],
            "quantity": [2, 2, 2, 1, 2],
            "ticket_type": ["a", "a", "a", "b", "a"],
            "name_type": ["x", "x", "x", "y", "x"],
            "first_observed_at": pd.to_datetime([
                "2026-01-01 10:00", "2026-01-03 11:00", "2026-01-02 09:00",
                "2026-01-04 08:00", "2026-01-05 10:00",
            ]),
            "last_observed_at": pd.to_datetime([
                "2026-01-10 12:00", "2026-01-10 12:00", "2026-01-10 12:00",
                "2026-01-10 12:00", "2026-01-08 10:00",
            ]),
            "sold_at": pd.to_datetime([None, None, None, None, "2026-01-08 10:00"]),
            "status": ["listing", "listing", "listing", "listing", "sold"],
            "performance_at": pd.to_datetime(["2026-02-01"] * 5),
            "price": [20000.0, 17000.0, 12000.0, 9000.0, 16000.0],
            "duplicate_group": ["g1", "g2", "g3", "g4", "g5"],
            "semantic_seat_level": [
                "アリーナ", "アリーナ", "スタンド", "アリーナ", "不明"
            ],
            "semantic_row_position": ["前方", "前方", "前方", "不明", "不明"],
            "semantic_visibility": ["通常"] * 5,
        })

    def test_vectorized_labels_match_legacy_values(self):
        expected = legacy_landmarks(self.tickets, (1, 3, 7), self.cutoff)
        actual = build_landmarks(self.tickets, (1, 3, 7), self.cutoff)
        assert_frame_equal(actual, expected, check_dtype=False, check_categorical=False)

    def test_vectorized_market_features_match_legacy_values(self):
        prepared = prepare_end_times(self.tickets)
        landmarks = build_landmarks(self.tickets, (1, 3, 7), self.cutoff)
        expected = legacy_market(landmarks, prepared)
        actual = add_market_features(landmarks, prepared)
        columns = [
            column for column in actual
            if column.startswith("market_") or column == "price_to_market_median"
        ]
        assert_frame_equal(
            actual[columns], expected[columns], check_dtype=False,
            check_categorical=False, rtol=1e-12, atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
