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
    return pd.Series(p, index=pd.date_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n, freq="B"))


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
            _asset("BND",  600, 100, "Bond ETF", "Fixed Income"),
            _asset("GLD",  50, 185, "Commodity", "Commodity"),
        ]
        p = _make_portfolio(assets, risk_profile="conservative")
        monitor = RiskMonitor(p)
        engine = RecommendationEngine(p, monitor, use_optimizer=False)
        recs = engine.generate_recommendations()
        critical = [r for r in recs if r["priority"] == "critical"]
        assert len(critical) == 0


class TestCoordinatedPlan:
    """All trade-producing steps edit one shared plan, so they must not overshoot."""

    def _overweight_equity(self, profile="moderate"):
        assets = [
            _asset("SPY", 500, 100, "Equity ETF", "Equity"),
            _asset("AAPL", 150, 100, "Technology", "Equity"),
            _asset("MSFT", 150, 100, "Technology", "Equity"),
            _asset("JPM", 100, 100, "Financials", "Equity"),
            _asset("BND", 100, 100, "Bond ETF", "Fixed Income"),
        ]
        return _make_portfolio(assets, risk_profile=profile)

    def _apply(self, p, trades):
        w = {a.ticker: a.current_value for a in p.assets}
        for t in trades:
            sign = 1 if t["action"] == "BUY" else -1
            w[t["ticker"]] = w.get(t["ticker"], 0.0) + sign * t["value"]
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def test_trades_reach_profile_without_overshoot(self):
        p = self._overweight_equity()
        engine = RecommendationEngine(p, RiskMonitor(p))
        recs = engine.generate_recommendations()
        all_trades = [t for r in recs for t in r.get("trades", [])]
        w = self._apply(p, all_trades)
        bonds = w.get("BND", 0) + w.get("AGG", 0) + w.get("TLT", 0) + w.get("VCIT", 0)
        target_fi = p.TARGET_ALLOCATION["moderate"]["fixed_income"]
        # Within the drift caution band (5%) of the profile target, not beyond it
        assert abs(bonds - target_fi) <= 0.05 + 0.01

    def test_step_trades_sum_to_plan_trades(self):
        p = self._overweight_equity()
        engine = RecommendationEngine(p, RiskMonitor(p))
        recs = engine.generate_recommendations()
        net = {}
        for t in (t for r in recs for t in r.get("trades", [])):
            net[t["ticker"]] = net.get(t["ticker"], 0) + (1 if t["action"] == "BUY" else -1) * t["value"]
        plan = {t["ticker"]: (1 if t["action"] == "BUY" else -1) * t["value"]
                for t in engine.plan_trades()}
        for tk, v in plan.items():
            assert net.get(tk, 0) == pytest.approx(v, abs=0.01 * p.total_value)

    def test_buys_are_funded_by_sells(self):
        p = self._overweight_equity()
        recs = RecommendationEngine(p, RiskMonitor(p)).generate_recommendations()
        trades = [t for r in recs for t in r.get("trades", [])]
        buys = sum(t["value"] for t in trades if t["action"] == "BUY")
        sells = sum(t["value"] for t in trades if t["action"] == "SELL")
        assert buys <= sells * 1.01 + 1


class TestCashClass:
    """The strategies' third class is cash; alternatives get only a small slice of it."""

    def _with_cash_prices(self, assets, profile):
        p = _make_portfolio(assets, risk_profile=profile)
        p.historical_prices["BIL"] = 100 * np.exp(np.linspace(0, 0.04, len(p.historical_prices)))
        return p

    def test_missing_third_class_is_filled_with_cash_not_gold(self):
        p = self._with_cash_prices([
            _asset("SPY", 800, 100, "Equity ETF", "Equity"),
            _asset("BND", 200, 100, "Bond ETF", "Fixed Income"),
        ], "defensive")
        engine = RecommendationEngine(p, RiskMonitor(p))
        engine.generate_recommendations()
        w = engine.plan.w
        assert w.get("GLD", 0.0) == 0.0
        assert w.get("BIL", 0.0) == pytest.approx(p.TARGET_ALLOCATION["defensive"]["other"], abs=0.02)

    def test_gold_is_capped_and_the_rest_moves_to_cash(self):
        p = self._with_cash_prices([
            _asset("SPY", 300, 100, "Equity ETF", "Equity"),
            _asset("BND", 300, 100, "Bond ETF", "Fixed Income"),
            _asset("GLD", 400, 100, "Commodities ETF", "Commodity"),
        ], "liquidity_plus")
        engine = RecommendationEngine(p, RiskMonitor(p))
        engine.generate_recommendations()
        w = engine.plan.w
        assert w["GLD"] <= RecommendationEngine.ALT_CAP + 1e-6
        third = p.TARGET_ALLOCATION["liquidity_plus"]["other"]
        assert w["GLD"] + w.get("BIL", 0.0) == pytest.approx(third, abs=0.02)
