"""
Tests for twin_timeline.py (time dimension of the twin; save/load)
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset
from src.twin_timeline import timeline, first_alerts, snapshot, compare_snapshots


def _portfolio(seed=0, n=3 * 252):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    hist = pd.DataFrame({
        "SPY": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n))),
        "AAPL": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.018, n))),
        "BND": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, n))),
    }, index=idx)
    assets = [
        Asset("SPY", 300, 100, hist["SPY"].iloc[-1], sector="Equity ETF", asset_class="Equity"),
        Asset("AAPL", 200, 100, hist["AAPL"].iloc[-1], sector="Technology", asset_class="Equity"),
        Asset("BND", 300, 100, hist["BND"].iloc[-1], sector="Bond ETF", asset_class="Fixed Income"),
    ]
    return Portfolio(assets, hist, risk_profile="flexible")


class TestTimeline:
    def test_monthly_points_after_one_year(self):
        p = _portfolio()
        tl = timeline(p)
        assert len(tl) >= 20
        assert tl["date"].min() >= p.historical_prices.index[0] + pd.Timedelta(days=365)
        assert tl["resilience"].between(0, 100).all()

    def test_no_look_ahead(self):
        # Changing prices after a date must not change the timeline up to that date
        p = _portfolio()
        tl1 = timeline(p)
        cut = tl1["date"].iloc[len(tl1) // 2]
        altered = _portfolio()
        h = altered.historical_prices
        h.loc[h.index > cut] *= 0.5
        tl2 = timeline(altered)
        early1 = tl1[tl1["date"] <= cut].reset_index(drop=True)
        early2 = tl2[tl2["date"] <= cut].reset_index(drop=True)
        pd.testing.assert_series_equal(early1["resilience"], early2["resilience"])
        pd.testing.assert_series_equal(early1["n_alert"], early2["n_alert"])

    def test_last_point_matches_live_analysis(self):
        # Expanding window: the latest month end uses the same data as the live twin
        from src.resilience import compute_resilience
        p = _portfolio()
        tl = timeline(p)
        assert tl["date"].iloc[-1] == p.historical_prices.index[-1]
        assert tl["resilience"].iloc[-1] == pytest.approx(compute_resilience(p)["score"])

    def test_first_alerts_one_row_per_title(self):
        fa = first_alerts(timeline(_portfolio()))
        assert fa["alert"].is_unique


class TestSnapshot:
    def test_json_roundtrip_and_no_change(self):
        p = _portfolio()
        s = json.loads(json.dumps(snapshot(p, goal={"target": 1e6, "years": 10, "monthly": 500})))
        assert s["goal"]["years"] == 10
        assert {h["ticker"] for h in s["holdings"]} == {"SPY", "AAPL", "BND"}
        metrics, holdings = compare_snapshots(s, s)
        assert (metrics["Μεταβολή"].dropna() == 0).all()
        assert holdings.empty

    def test_weight_changes_reported(self):
        p = _portfolio()
        old = snapshot(p)
        p.assets[2].quantity *= 3  # buy more bonds
        new = snapshot(p)
        _, holdings = compare_snapshots(old, new)
        bnd = holdings[holdings["Ticker"] == "BND"].iloc[0]
        assert bnd["Μεταβολή (μ.)"] > 0


def test_forward_drawdown_is_hindsight_and_non_positive():
    tl = timeline(_portfolio())
    fd = tl["next_drawdown_pct"].dropna()
    assert len(fd) > 0 and (fd <= 0).all()
    # The last points are too close to the end of the data to have 6 months after
    assert np.isnan(tl["next_drawdown_pct"].iloc[-1])
