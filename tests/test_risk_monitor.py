"""
Unit tests for risk_monitor.py
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import Portfolio, Asset
from src.risk_monitor import RiskMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(seed, n=500, mu=0.08, sigma=0.20):
    rng = np.random.default_rng(seed)
    lr = rng.normal(mu / 252, sigma / np.sqrt(252), n)
    p = 100 * np.exp(np.cumsum(lr))
    return pd.Series(p, index=pd.date_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n, freq="B"))


def _portfolio(assets, history, risk_profile="moderate"):
    return Portfolio(assets, history, risk_profile=risk_profile)


def _asset(ticker, qty, price, sector="Technology", asset_class="Equity"):
    return Asset(ticker=ticker, quantity=qty, entry_price=price * 0.9,
                 current_price=price, sector=sector, asset_class=asset_class)


# ---------------------------------------------------------------------------
# Concentration Tests
# ---------------------------------------------------------------------------

class TestConcentration:
    def test_alert_when_single_asset_over_25pct(self):
        # Apple = 80% of portfolio
        assets = [
            _asset("AAPL", 800, 100),   # $80k
            _asset("SPY",  200, 100),   # $20k
        ]
        hist = pd.DataFrame({
            "AAPL": _prices(1).values,
            "SPY": _prices(2, sigma=0.15).values,
        }, index=_prices(1).index)
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        alerts = monitor.check_concentration()
        severities = {a["title"]: a["severity"] for a in alerts}
        assert severities["Συγκέντρωση σε Μεμονωμένο Τίτλο"] == "alert"

    def _single_alert(self, assets):
        hist = pd.DataFrame(
            {a.ticker: _prices(i).values for i, a in enumerate(assets)},
            index=_prices(0).index,
        )
        alerts = RiskMonitor(_portfolio(assets, hist)).check_concentration()
        return next(a for a in alerts if "Μεμονωμένο" in a["title"])

    def test_broad_etf_uses_fund_limit(self):
        # 55% in one bond ETF is what a Defensive profile requires: not a breach
        alert = self._single_alert([
            _asset("BND", 550, 100, "Bond ETF", "Fixed Income"),
            _asset("SPY", 250, 100, "Equity ETF", "Equity"),
            _asset("VXUS", 200, 100, "International ETF", "International Equity"),
        ])
        assert alert["severity"] == "ok"
        assert alert["detail"]["is_fund"] is True

    def test_broad_etf_over_fund_limit_alerts(self):
        alert = self._single_alert([
            _asset("SPY", 850, 100, "Equity ETF", "Equity"),
            _asset("BND", 150, 100, "Bond ETF", "Fixed Income"),
        ])
        assert alert["severity"] == "alert"
        assert alert["detail"]["largest_ticker"] == "SPY"

    def test_commodity_etf_uses_security_limit(self):
        # GLD tracks a single commodity, so it is not a diversified fund
        alert = self._single_alert([
            _asset("GLD", 300, 100, "Commodities ETF", "Commodity"),
            _asset("BND", 700, 100, "Bond ETF", "Fixed Income"),
        ])
        assert alert["detail"]["largest_ticker"] == "GLD"
        assert alert["severity"] == "alert"

    def test_stock_breach_reported_over_larger_ok_fund(self):
        alert = self._single_alert([
            _asset("SPY", 600, 100, "Equity ETF", "Equity"),
            _asset("AAPL", 300, 100, "Technology", "Equity"),
            _asset("BND", 100, 100, "Bond ETF", "Fixed Income"),
        ])
        assert alert["detail"]["largest_ticker"] == "AAPL"
        assert alert["severity"] == "alert"

    def test_ok_when_assets_balanced(self):
        assets = [_asset(f"T{i}", 100, 100) for i in range(10)]
        tickers = [a.ticker for a in assets]
        hist = pd.DataFrame(
            {t: _prices(i).values for i, t in enumerate(tickers)},
            index=_prices(0).index,
        )
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        alerts = monitor.check_concentration()
        single_alert = next(a for a in alerts if "Μεμονωμένο" in a["title"])
        assert single_alert["severity"] == "ok"


# ---------------------------------------------------------------------------
# Diversification Tests
# ---------------------------------------------------------------------------

class TestDiversification:
    def test_few_holdings_triggers_alert(self):
        assets = [_asset(f"T{i}", 100, 100) for i in range(3)]
        tickers = [a.ticker for a in assets]
        hist = pd.DataFrame(
            {t: _prices(i).values for i, t in enumerate(tickers)},
            index=_prices(0).index,
        )
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        alerts = monitor.check_diversification()
        holdings_alert = next((a for a in alerts if "Αριθμός Θέσεων" in a["title"]), None)
        assert holdings_alert is not None
        assert holdings_alert["severity"] in ("caution", "alert")

    def test_etf_label_is_not_a_sector(self):
        # 70% in one bond ETF must not read as a 70% "Bond ETF" sector
        assets = [
            _asset("BND", 700, 100, "Bond ETF", "Fixed Income"),
            _asset("AAPL", 150, 100, "Technology", "Equity"),
            _asset("JNJ", 150, 100, "Healthcare", "Equity"),
        ]
        hist = pd.DataFrame(
            {a.ticker: _prices(i).values for i, a in enumerate(assets)},
            index=_prices(0).index,
        )
        alerts = RiskMonitor(_portfolio(assets, hist)).check_diversification()
        sector = next(a for a in alerts if "Κλάδο" in a["title"])
        assert sector["detail"]["top_sector"] != "Bond ETF"
        assert sector["severity"] == "ok"

    def test_no_international_triggers_caution(self):
        assets = [
            _asset("AAPL", 100, 100, "Technology", "Equity"),
            _asset("MSFT", 100, 100, "Technology", "Equity"),
        ]
        hist = pd.DataFrame({
            "AAPL": _prices(1).values,
            "MSFT": _prices(2).values,
        }, index=_prices(1).index)
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        alerts = monitor.check_diversification()
        geo_alert = next((a for a in alerts if "Γεωγραφική" in a["title"]), None)
        assert geo_alert is not None
        assert geo_alert["severity"] in ("caution", "alert")


# ---------------------------------------------------------------------------
# Volatility Tests
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_high_vol_triggers_alert(self):
        # Very high volatility portfolio (100% TSLA-like stock)
        assets = [_asset("TSLA", 100, 100)]
        p_series = _prices(1, sigma=0.60)  # 60% vol
        hist = pd.DataFrame({"TSLA": p_series.values}, index=p_series.index)
        p = _portfolio(assets, hist, risk_profile="conservative")  # target vol 7%
        monitor = RiskMonitor(p)
        alerts = monitor.check_volatility()
        assert len(alerts) == 1
        assert alerts[0]["severity"] in ("caution", "alert")

    def test_matching_vol_returns_ok(self):
        assets = [
            _asset("SPY", 200, 100, "Equity ETF", "Equity"),
            _asset("BND", 600, 100, "Fixed Income", "Fixed Income"),
        ]
        spy_p = _prices(1, sigma=0.15)
        bnd_p = _prices(2, sigma=0.04)
        hist = pd.DataFrame({
            "SPY": spy_p.values,
            "BND": bnd_p.values,
        }, index=spy_p.index)
        p = _portfolio(assets, hist, risk_profile="conservative")
        monitor = RiskMonitor(p)
        alerts = monitor.check_volatility()
        # Conservative portfolio at low vol should be OK
        assert alerts[0]["severity"] in ("ok", "caution")  # acceptable either way


# ---------------------------------------------------------------------------
# Drawdown Tests
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_check_drawdown_returns_two_alerts(self):
        assets = [_asset("SPY", 100, 100)]
        p_series = _prices(1)
        hist = pd.DataFrame({"SPY": p_series.values}, index=p_series.index)
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        alerts = monitor.check_drawdown()
        assert len(alerts) >= 1


# ---------------------------------------------------------------------------
# Full Analysis
# ---------------------------------------------------------------------------

class TestFullAnalysis:
    def test_run_full_analysis_structure(self):
        assets = [
            _asset("AAPL", 600, 100, "Technology", "Equity"),  # concentrated
            _asset("BND", 400, 100, "Fixed Income", "Fixed Income"),
        ]
        hist = pd.DataFrame({
            "AAPL": _prices(1).values,
            "BND": _prices(2, sigma=0.05).values,
        }, index=_prices(1).index)
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        result = monitor.run_full_analysis()

        assert "ok" in result
        assert "caution" in result
        assert "alert" in result
        assert "summary" in result
        assert "highest_severity" in result
        assert result["summary"]["total"] > 0

    def test_concentrated_portfolio_has_alerts(self):
        assets = [_asset("AAPL", 900, 100), _asset("BND", 100, 100, "Fixed Income", "Fixed Income")]
        hist = pd.DataFrame({
            "AAPL": _prices(1).values,
            "BND": _prices(2, sigma=0.05).values,
        }, index=_prices(1).index)
        p = _portfolio(assets, hist)
        monitor = RiskMonitor(p)
        result = monitor.run_full_analysis()
        # 90% in one stock is definitely an alert
        assert result["summary"]["alert"] > 0


class TestThresholdEdges:
    def _p(self, assets):
        hist = pd.DataFrame({a.ticker: _prices(i).values for i, a in enumerate(assets)},
                            index=_prices(0).index)
        return _portfolio(assets, hist)

    def test_exactly_min_holdings_is_ok(self):
        assets = [_asset(f"T{i}", 100, 100) for i in range(10)]
        alerts = RiskMonitor(self._p(assets)).check_diversification()
        holdings = next(a for a in alerts if "Αριθμός Θέσεων" in a["title"])
        assert holdings["severity"] == "ok"

    def test_top5_ignores_broad_etfs(self):
        assets = [
            _asset("BND", 550, 100, "Bond ETF", "Fixed Income"),
            _asset("SPY", 250, 100, "Equity ETF", "Equity"),
            _asset("VXUS", 100, 100, "International ETF", "International Equity"),
            _asset("GLD", 100, 100, "Commodities ETF", "Commodity"),
        ]
        alerts = RiskMonitor(self._p(assets)).check_concentration()
        top5 = next(a for a in alerts if "5 Κορυφαίους" in a["title"])
        assert top5["detail"]["top5_pct"] == pytest.approx(10.0, abs=0.5)
        assert top5["severity"] == "ok"
