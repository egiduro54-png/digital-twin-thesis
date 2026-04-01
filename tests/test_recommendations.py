"""
Unit tests for recommendations.py
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset
from src.risk_monitor import RiskMonitor
from src.recommendations import RecommendationEngine, PRIORITY_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(seed, n=500, sigma=0.20):
    rng = np.random.default_rng(seed)
    lr = rng.normal(0.0003, sigma / np.sqrt(252), n)
    p = 100 * np.exp(np.cumsum(lr))
    return pd.Series(p, index=pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B"))


def _asset(ticker, qty, price, sector="Technology", asset_class="Equity"):
    return Asset(ticker=ticker, quantity=qty, entry_price=price * 0.9,
                 current_price=price, sector=sector, asset_class=asset_class)


def _make_portfolio(assets, risk_profile="moderate"):
    tickers = [a.ticker for a in assets]
    hist = pd.DataFrame(
        {t: _prices(i, sigma=0.05 if "BND" in t else 0.20).values
         for i, t in enumerate(tickers)},
        index=_prices(0).index,
    )
    return Portfolio(assets, hist, risk_profile=risk_profile)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecommendationEngine:
    def test_generate_recs_returns_list(self):
        assets = [
            _asset("AAPL", 900, 100),  # 90% concentration
            _asset("BND", 100, 100, "Fixed Income", "Fixed Income"),
        ]
        p = _make_portfolio(assets)
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()
        assert isinstance(recs, list)

    def test_concentrated_portfolio_has_recs(self):
        assets = [
            _asset("AAPL", 700, 100, "Technology"),
            _asset("MSFT", 100, 100, "Technology"),
            _asset("BND", 200, 100, "Fixed Income", "Fixed Income"),
        ]
        p = _make_portfolio(assets)
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()
        assert len(recs) > 0

    def test_recs_have_required_fields(self):
        assets = [
            _asset("AAPL", 800, 100),
            _asset("BND", 200, 100, "Fixed Income", "Fixed Income"),
        ]
        p = _make_portfolio(assets)
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()

        for rec in recs:
            assert "priority" in rec, f"Missing 'priority' in {rec}"
            assert "title" in rec
            assert "action" in rec
            assert "why" in rec
            assert "confidence" in rec

    def test_recs_sorted_by_priority(self):
        assets = [
            _asset("AAPL", 600, 100),
            _asset("MSFT", 200, 100),
            _asset("BND", 200, 100, "Fixed Income", "Fixed Income"),
        ]
        p = _make_portfolio(assets)
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()

        priorities = [PRIORITY_ORDER.get(r["priority"], 99) for r in recs]
        assert priorities == sorted(priorities), "Recommendations not sorted by priority"

    def test_recs_ids_are_sequential(self):
        assets = [
            _asset("AAPL", 800, 100),
            _asset("BND", 200, 100, "Fixed Income", "Fixed Income"),
        ]
        p = _make_portfolio(assets)
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()
        ids = [r["id"] for r in recs]
        assert ids == list(range(1, len(recs) + 1))

    def test_healthy_portfolio_has_no_critical_recs(self):
        # Well-balanced portfolio shouldn't have critical recommendations
        assets = [
            _asset("SPY",  200, 100, "Equity ETF", "Equity"),
            _asset("VXUS", 80, 55, "International ETF", "International Equity"),
            _asset("BND",  600, 100, "Fixed Income", "Fixed Income"),
            _asset("GLD",  50, 185, "Commodity", "Commodity"),
        ]
        p = _make_portfolio(assets, risk_profile="conservative")
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()
        critical = [r for r in recs if r["priority"] == "critical"]
        assert len(critical) == 0
