"""
Unit tests for resilience.py (composite Resilience Index)
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset
from src.resilience import compute_resilience, recovery_stats, resilience_band, WEIGHTS


def _idx(n):
    end = pd.offsets.BDay().rollback(pd.Timestamp.today().normalize())
    return pd.date_range(end=end, periods=n, freq="B")


def _gbm(seed, mu, sigma, n=750):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(mu / 252, sigma / np.sqrt(252), n)))


def _portfolio(spec, profile="flexible"):
    """spec: list of (ticker, value, sector, asset_class, mu, sigma)."""
    idx = _idx(750)
    hist = pd.DataFrame({t: _gbm(i, mu, s) for i, (t, _, _, _, mu, s) in enumerate(spec)},
                        index=idx)
    assets = [Asset(ticker=t, quantity=v / hist[t].iloc[-1], entry_price=hist[t].iloc[0],
                    current_price=hist[t].iloc[-1], sector=sec, asset_class=ac)
              for t, v, sec, ac, _, _ in spec]
    return Portfolio(assets, hist, risk_profile=profile)


CONCENTRATED = [
    ("NVDA", 80_000, "Technology", "Equity", 0.20, 0.50),
    ("AAPL", 20_000, "Technology", "Equity", 0.12, 0.30),
]
DIVERSIFIED = [
    ("SPY", 25_000, "Equity ETF", "Equity", 0.08, 0.17),
    ("VXUS", 15_000, "International ETF", "International Equity", 0.06, 0.16),
    ("BND", 35_000, "Bond ETF", "Fixed Income", 0.03, 0.05),
    ("TLT", 10_000, "Bond ETF", "Fixed Income", 0.03, 0.12),
    ("GLD", 10_000, "Commodities ETF", "Commodity", 0.05, 0.15),
    ("JNJ", 5_000, "Healthcare", "Equity", 0.06, 0.18),
]


class TestComputeResilience:
    def test_score_and_components_in_range(self):
        r = compute_resilience(_portfolio(DIVERSIFIED))
        assert 0 <= r["score"] <= 100
        assert set(r["components"]) == set(WEIGHTS)
        for v in r["components"].values():
            assert 0.0 <= v <= 1.0

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_diversified_more_resilient_than_concentrated(self):
        div = compute_resilience(_portfolio(DIVERSIFIED))
        conc = compute_resilience(_portfolio(CONCENTRATED, profile="dynamic"))
        assert div["score"] > conc["score"] + 15
        assert div["components"]["concentration"] > conc["components"]["concentration"]
        assert div["components"]["stress"] > conc["components"]["stress"]

    def test_sequential_shock_worse_than_first_shock_alone(self):
        r = compute_resilience(_portfolio(DIVERSIFIED))
        p = _portfolio(DIVERSIFIED)
        from src.scenario_engine import STANDARD_SCENARIOS
        first_only = 1 - p.apply_scenario(STANDARD_SCENARIOS["market_down_20"]).total_value / p.total_value
        assert r["raw"]["sequential_loss_pct"] / 100 > first_only

    def test_score_is_weighted_sum_of_components(self):
        r = compute_resilience(_portfolio(DIVERSIFIED))
        expected = 100 * sum(WEIGHTS[k] * v for k, v in r["components"].items())
        assert r["score"] == pytest.approx(expected)


class TestRecoveryStats:
    def test_recovered_drawdown(self):
        # +0 for 50 days, −20% over 20 days, back up over 40 days, flat afterwards
        r = np.r_[np.zeros(50), np.full(20, np.log(0.8) / 20),
                  np.full(40, -np.log(0.8) / 40), np.zeros(50)]
        s = recovery_stats(pd.Series(r, index=_idx(len(r))))
        assert s["recovered"]
        assert s["recovery_days"] == pytest.approx(40, abs=1)
        assert s["max_drawdown"] == pytest.approx(-0.2, abs=1e-6)

    def test_unrecovered_drawdown_is_penalised(self):
        r = np.r_[np.zeros(50), np.full(20, np.log(0.8) / 20), np.zeros(30)]
        s = recovery_stats(pd.Series(r, index=_idx(len(r))))
        assert not s["recovered"]
        assert s["recovery_days"] == pytest.approx(2 * 30, abs=1)


def test_band_labels():
    assert resilience_band(80) == "Υψηλή"
    assert resilience_band(55) == "Μέτρια"
    assert resilience_band(35) == "Χαμηλή"
    assert resilience_band(10) == "Πολύ χαμηλή"
