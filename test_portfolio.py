"""
Unit tests for portfolio.py

Run with: python -m pytest tests/test_portfolio.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_asset(ticker="AAPL", quantity=100, entry_price=150.0, current_price=180.0,
               sector="Technology", asset_class="Equity"):
    return Asset(
        ticker=ticker,
        quantity=quantity,
        entry_price=entry_price,
        current_price=current_price,
        sector=sector,
        asset_class=asset_class,
    )


def make_price_series(n=500, seed=42, annual_return=0.08, annual_vol=0.20):
    """Generate synthetic daily price series."""
    rng = np.random.default_rng(seed)
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    log_returns = rng.normal(daily_return, daily_vol, n)
    prices = 100 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    return pd.Series(prices, index=dates)


@pytest.fixture
def simple_portfolio():
    """A two-asset portfolio with synthetic history."""
    aapl = make_asset("AAPL", 100, 150.0, 180.0, "Technology")
    bnd = make_asset("BND", 500, 80.0, 82.0, "Fixed Income", "Fixed Income")

    prices_aapl = make_price_series(seed=1)
    prices_bnd = make_price_series(seed=2, annual_return=0.03, annual_vol=0.05)

    history = pd.DataFrame({"AAPL": prices_aapl.values, "BND": prices_bnd.values},
                           index=prices_aapl.index)

    return Portfolio(
        assets=[aapl, bnd],
        historical_prices=history,
        risk_profile="moderate",
        name="Test Portfolio",
    )


# ---------------------------------------------------------------------------
# Asset Tests
# ---------------------------------------------------------------------------

class TestAsset:
    def test_current_value(self):
        a = make_asset(quantity=100, current_price=200.0)
        assert a.current_value == 20_000.0

    def test_cost_basis(self):
        a = make_asset(quantity=100, entry_price=150.0)
        assert a.cost_basis == 15_000.0

    def test_unrealized_pnl(self):
        a = make_asset(quantity=100, entry_price=150.0, current_price=180.0)
        assert a.unrealized_pnl == pytest.approx(3_000.0)

    def test_unrealized_pnl_pct(self):
        a = make_asset(quantity=100, entry_price=100.0, current_price=120.0)
        assert a.unrealized_pnl_pct == pytest.approx(20.0)

    def test_to_dict_keys(self):
        a = make_asset()
        d = a.to_dict()
        for key in ("ticker", "current_value", "cost_basis", "unrealized_pnl",
                    "sector", "asset_class"):
            assert key in d


# ---------------------------------------------------------------------------
# Portfolio Basics
# ---------------------------------------------------------------------------

class TestPortfolioBasics:
    def test_total_value(self, simple_portfolio):
        p = simple_portfolio
        expected = 100 * 180.0 + 500 * 82.0
        assert p.total_value == pytest.approx(expected)

    def test_weights_sum_to_one(self, simple_portfolio):
        weights = simple_portfolio.get_weights()
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_dict_keys(self, simple_portfolio):
        wd = simple_portfolio.get_weights_dict()
        assert "AAPL" in wd and "BND" in wd

    def test_tickers(self, simple_portfolio):
        assert simple_portfolio.tickers == ["AAPL", "BND"]

    def test_num_holdings(self, simple_portfolio):
        assert len(simple_portfolio.assets) == 2


# ---------------------------------------------------------------------------
# Metric Calculations
# ---------------------------------------------------------------------------

class TestMetricCalculations:
    def test_volatility_positive(self, simple_portfolio):
        vol = simple_portfolio.calculate_volatility()
        assert not np.isnan(vol)
        assert vol > 0

    def test_volatility_annualised_range(self, simple_portfolio):
        # Expect typical range 5–50% for a mixed portfolio
        vol = simple_portfolio.calculate_volatility()
        assert 0.01 < vol < 1.0

    def test_sharpe_ratio_finite(self, simple_portfolio):
        sharpe = simple_portfolio.calculate_sharpe_ratio()
        assert not np.isnan(sharpe)

    def test_max_drawdown_negative(self, simple_portfolio):
        dd = simple_portfolio.calculate_max_drawdown()
        assert dd <= 0.0

    def test_var_negative(self, simple_portfolio):
        var = simple_portfolio.calculate_var(0.95)
        assert var < 0.0

    def test_diversification_ratio_above_one(self, simple_portfolio):
        # AAPL (high vol) + BND (low vol) should give ratio > 1
        dr = simple_portfolio.calculate_diversification_ratio()
        assert dr > 1.0

    def test_get_metrics_keys(self, simple_portfolio):
        m = simple_portfolio.get_metrics()
        for key in ("total_value", "sharpe_ratio", "volatility_annual_pct",
                    "max_drawdown_pct", "beta"):
            assert key in m

    def test_metrics_cached(self, simple_portfolio):
        # Second call returns same object
        m1 = simple_portfolio.get_metrics()
        m2 = simple_portfolio.get_metrics()
        assert m1 is m2


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TestComposition:
    def test_by_asset_class_present(self, simple_portfolio):
        comp = simple_portfolio.get_composition()
        assert "by_asset_class" in comp

    def test_concentration_keys(self, simple_portfolio):
        comp = simple_portfolio.get_composition()
        conc = comp["concentration"]
        for key in ("largest_single_asset_pct", "top_5_assets_pct",
                    "herfindahl_index", "num_holdings"):
            assert key in conc

    def test_herfindahl_range(self, simple_portfolio):
        comp = simple_portfolio.get_composition()
        hhi = comp["concentration"]["herfindahl_index"]
        assert 0 < hhi <= 1.0


# ---------------------------------------------------------------------------
# Risk Profile Alignment
# ---------------------------------------------------------------------------

class TestRiskProfileAlignment:
    def test_alignment_keys(self, simple_portfolio):
        align = simple_portfolio.get_risk_profile_alignment()
        for k in ("risk_profile", "current_equity_pct", "current_fixed_income_pct",
                  "equity_drift_pct", "max_drift"):
            assert k in align

    def test_conservative_target(self):
        assets = [make_asset("BND", 1000, 80, 82, "Fixed Income", "Fixed Income")]
        hist = pd.DataFrame({"BND": make_price_series(seed=10).values},
                            index=make_price_series(seed=10).index)
        p = Portfolio(assets, hist, risk_profile="conservative")
        align = p.get_risk_profile_alignment()
        assert align["risk_profile"] == "conservative"
        assert align["target_equity_pct"] == 30.0


# ---------------------------------------------------------------------------
# Scenario Application
# ---------------------------------------------------------------------------

class TestScenarioApplication:
    def test_market_down_reduces_equity_value(self, simple_portfolio):
        scenario = {
            "name": "Market Down 20%",
            "market_change": -0.20,
            "rate_change": 0.0,
            "volatility_multiplier": 1.0,
            "sector_overrides": {},
        }
        original_aapl_value = simple_portfolio.assets[0].current_value
        scenario_p = simple_portfolio.apply_scenario(scenario)
        scenario_aapl_value = scenario_p.assets[0].current_value
        # AAPL (equity) should drop ~20%
        assert scenario_aapl_value < original_aapl_value

    def test_scenario_returns_new_portfolio(self, simple_portfolio):
        scenario = {"name": "Test", "market_change": -0.10,
                    "rate_change": 0.0, "volatility_multiplier": 1.0, "sector_overrides": {}}
        s_p = simple_portfolio.apply_scenario(scenario)
        assert s_p is not simple_portfolio

    def test_scenario_marked_as_scenario(self, simple_portfolio):
        scenario = {"name": "Test", "market_change": 0.0,
                    "rate_change": 0.0, "volatility_multiplier": 1.0, "sector_overrides": {}}
        s_p = simple_portfolio.apply_scenario(scenario)
        assert s_p.is_scenario is True

    def test_bonds_less_sensitive_than_equities(self, simple_portfolio):
        scenario = {
            "name": "Market Down 30%",
            "market_change": -0.30,
            "rate_change": 0.0,
            "volatility_multiplier": 1.0,
            "sector_overrides": {},
        }
        s_p = simple_portfolio.apply_scenario(scenario)
        aapl_change = (s_p.assets[0].current_price / simple_portfolio.assets[0].current_price) - 1
        bnd_change = (s_p.assets[1].current_price / simple_portfolio.assets[1].current_price) - 1
        # Bonds should drop less than equity
        assert abs(bnd_change) < abs(aapl_change)
