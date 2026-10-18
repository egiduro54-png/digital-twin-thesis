"""
Tests for the asset-specific scenario model in Portfolio.apply_scenario:
per-asset beta, per-fund bond duration and return-based volatility scaling.
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset, BOND_DURATIONS


def _idx(n):
    end = pd.offsets.BDay().rollback(pd.Timestamp.today().normalize())
    return pd.date_range(end=end, periods=n, freq="B")


def _market_and_assets(n=750, seed=0):
    """SPY plus a high-beta (≈2) and a low-beta (≈0.3) stock built from it."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0004, 0.01, n)
    hi = 2.0 * m + rng.normal(0, 0.005, n)
    lo = 0.3 * m + rng.normal(0, 0.005, n)
    idx = _idx(n)
    to_px = lambda r: 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({"SPY": to_px(m), "HI": to_px(hi), "LO": to_px(lo)}, index=idx)


def _asset(t, price, ac="Equity", sector="Technology"):
    return Asset(ticker=t, quantity=100, entry_price=price, current_price=price,
                 sector=sector, asset_class=ac)


SCEN = {"market_change": -0.20, "rate_change": 0.0, "volatility_multiplier": 1.0,
        "sector_overrides": {}}


class TestBeta:
    def test_betas_are_blume_adjusted_estimates(self):
        hist = _market_and_assets()
        p = Portfolio([_asset("HI", 100), _asset("LO", 100), _asset("SPY", 100, "Equity ETF")], hist)
        b = p.asset_betas()
        # raw ≈ 2.0 → 0.67·2 + 0.33 ≈ 1.67; raw ≈ 0.3 → ≈ 0.53
        assert b["HI"] == pytest.approx(1.67, abs=0.1)
        assert b["LO"] == pytest.approx(0.53, abs=0.1)
        assert b["SPY"] == pytest.approx(1.0, abs=1e-6)

    def test_high_beta_falls_more(self):
        hist = _market_and_assets()
        p = Portfolio([_asset("HI", 100), _asset("LO", 100), _asset("SPY", 100, "Equity ETF")], hist)
        s = p.apply_scenario(SCEN)
        px = {a.ticker: a.current_price for a in s.assets}
        assert px["HI"] < px["SPY"] < px["LO"]
        assert px["SPY"] == pytest.approx(80.0, abs=0.01)

    def test_no_market_data_defaults_to_beta_one(self):
        hist = _market_and_assets().drop(columns="SPY")
        p = Portfolio([_asset("HI", 100)], hist)
        s = p.apply_scenario(SCEN)
        assert s.assets[0].current_price == pytest.approx(80.0)


class TestBondDuration:
    def test_long_duration_fund_moves_more(self):
        hist = _market_and_assets().rename(columns={"HI": "TLT", "LO": "BND"})
        p = Portfolio([_asset("TLT", 100, "Fixed Income", "Bond ETF"),
                       _asset("BND", 100, "Fixed Income", "Bond ETF")], hist)
        s = p.apply_scenario({**SCEN, "market_change": 0.0, "rate_change": 0.01})
        px = {a.ticker: a.current_price for a in s.assets}
        assert px["TLT"] == pytest.approx(100 * (1 - BOND_DURATIONS["TLT"] * 0.01))
        assert px["BND"] == pytest.approx(100 * (1 - BOND_DURATIONS["BND"] * 0.01))
        assert px["TLT"] < px["BND"]


class TestVolatilityScaling:
    def _trending(self):
        # Strong uptrend: early prices far below the mean, the case where
        # price-level scaling used to produce negative prices
        idx = _idx(750)
        rng = np.random.default_rng(1)
        r = rng.normal(0.003, 0.02, 750)
        return pd.DataFrame({"SPY": 100 * np.exp(np.cumsum(r))}, index=idx)

    def test_prices_stay_positive_at_high_multiplier(self):
        p = Portfolio([_asset("SPY", 100, "Equity ETF")], self._trending())
        s = p.apply_scenario({**SCEN, "volatility_multiplier": 4.0})
        assert (s.historical_prices["SPY"].dropna() > 0).all()

    def test_return_std_scales_by_multiplier(self):
        hist = self._trending()
        p = Portfolio([_asset("SPY", 100, "Equity ETF")], hist)
        s = p.apply_scenario({**SCEN, "volatility_multiplier": 3.0})
        r0 = np.log(hist["SPY"]).diff().dropna()
        r1 = np.log(s.historical_prices["SPY"]).diff().dropna()
        assert r1.std() / r0.std() == pytest.approx(3.0, rel=1e-6)
        assert r1.mean() == pytest.approx(r0.mean(), rel=1e-6)

    def test_scenario_sharpe_has_no_artifact(self):
        p = Portfolio([_asset("SPY", 100, "Equity ETF")], self._trending())
        live = p.calculate_sharpe_ratio()
        crisis = p.apply_scenario({**SCEN, "volatility_multiplier": 4.0}).calculate_sharpe_ratio()
        assert np.isfinite(crisis)
        assert crisis < live
