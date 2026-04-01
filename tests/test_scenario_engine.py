"""
Unit tests for scenario_engine.py
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset
from src.scenario_engine import ScenarioEngine, STANDARD_SCENARIOS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _prices(seed, n=500, sigma=0.20):
    rng = np.random.default_rng(seed)
    lr = rng.normal(0.0003, sigma / np.sqrt(252), n)
    p = 100 * np.exp(np.cumsum(lr))
    return pd.Series(p, index=pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B"))


def _asset(ticker, qty, price, sector="Technology", asset_class="Equity"):
    return Asset(ticker=ticker, quantity=qty, entry_price=price * 0.9,
                 current_price=price, sector=sector, asset_class=asset_class)


@pytest.fixture
def mixed_portfolio():
    assets = [
        _asset("AAPL", 100, 180, "Technology", "Equity"),
        _asset("BND",  500, 82,  "Fixed Income", "Fixed Income"),
    ]
    hist = pd.DataFrame({
        "AAPL": _prices(1).values,
        "BND": _prices(2, sigma=0.05).values,
    }, index=_prices(1).index)
    return Portfolio(assets, hist, risk_profile="moderate")


@pytest.fixture
def engine(mixed_portfolio):
    return ScenarioEngine(mixed_portfolio)


# ---------------------------------------------------------------------------
# Scenario Listing
# ---------------------------------------------------------------------------

class TestScenarioListing:
    def test_list_scenarios_not_empty(self, engine):
        scenarios = engine.list_scenarios()
        assert len(scenarios) > 0

    def test_list_scenarios_has_required_fields(self, engine):
        for s in engine.list_scenarios():
            assert "id" in s
            assert "name" in s
            assert "category" in s

    def test_standard_scenarios_available(self, engine):
        ids = {s["id"] for s in engine.list_scenarios()}
        for key in ("market_down_20", "financial_crisis_2008", "inflation_spike"):
            assert key in ids


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_market_down_reduces_portfolio_value(self, engine, mixed_portfolio):
        scenario_p = engine.simulate_scenario("market_down_20")
        assert scenario_p.total_value < mixed_portfolio.total_value

    def test_market_up_increases_portfolio_value(self, engine, mixed_portfolio):
        scenario_p = engine.simulate_scenario("market_up_20")
        assert scenario_p.total_value > mixed_portfolio.total_value

    def test_scenario_portfolio_is_different_object(self, engine, mixed_portfolio):
        s = engine.simulate_scenario("market_down_10")
        assert s is not mixed_portfolio

    def test_unknown_scenario_raises(self, engine):
        with pytest.raises(KeyError):
            engine.simulate_scenario("not_a_real_scenario")


# ---------------------------------------------------------------------------
# Custom Scenarios
# ---------------------------------------------------------------------------

class TestCustomScenarios:
    def test_create_custom_scenario_returns_dict(self, engine):
        s = engine.create_custom_scenario(
            name="Custom Test",
            market_change=-0.15,
            rate_change=0.005,
            volatility_multiplier=1.2,
        )
        assert "name" in s
        assert s["market_change"] == -0.15

    def test_custom_scenario_registered(self, engine):
        engine.create_custom_scenario("Custom Reg Test", market_change=-0.05)
        ids = {s["id"] for s in engine.list_scenarios()}
        assert "custom_reg_test" in ids

    def test_simulate_custom(self, engine, mixed_portfolio):
        custom = engine.create_custom_scenario(
            name="Big Crash Custom",
            market_change=-0.40,
        )
        s_p = engine.simulate_custom(custom)
        assert s_p.total_value < mixed_portfolio.total_value


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_compare_metrics_structure(self, engine):
        result = engine.compare_portfolio_metrics("market_down_20")
        assert "metrics" in result
        assert "summary" in result
        assert "scenario_params" in result

    def test_compare_metrics_total_value_changes(self, engine, mixed_portfolio):
        result = engine.compare_portfolio_metrics("market_down_20")
        assert result["summary"]["current_total_value"] == pytest.approx(
            mixed_portfolio.total_value, rel=0.01
        )
        assert result["summary"]["scenario_total_value"] < mixed_portfolio.total_value

    def test_compare_multiple_returns_base(self, engine):
        result = engine.compare_multiple_scenarios(["market_down_10", "market_down_20"])
        assert "base" in result

    def test_asset_impact_sorted_by_loss(self, engine):
        impacts = engine.get_asset_impact("market_down_20")
        changes = [i["dollar_change"] for i in impacts]
        # Should be sorted ascending (biggest loss first)
        assert changes == sorted(changes)


# ---------------------------------------------------------------------------
# Worst Scenario
# ---------------------------------------------------------------------------

class TestWorstScenario:
    def test_find_worst_scenario(self, engine):
        worst = engine.find_worst_scenario(
            ["market_down_10", "market_down_20", "market_down_30"]
        )
        assert worst["scenario_id"] == "market_down_30"
        assert worst["portfolio_loss"] < 0
