"""
portfolio.py

The Digital Twin of an investor's portfolio.

Responsibilities:
  - Store all asset positions and metadata
  - Calculate financial metrics (volatility, Sharpe, beta, drawdown, etc.)
  - Support scenario-based clones (apply_scenario returns a modified copy)
  - Expose structured data for the risk monitor and recommendation engine
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pypfopt import risk_models, expected_returns

logger = logging.getLogger(__name__)

# Annualisation factor for daily data
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # 4% – approximate current US risk-free rate


@dataclass
class Asset:
    """A single position in the portfolio."""
    ticker: str
    quantity: float
    entry_price: float
    current_price: float
    name: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    asset_class: str = "Equity"
    country: str = "Unknown"

    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.current_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.current_value / self.cost_basis - 1) * 100

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "name": self.name or self.ticker,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "current_price": round(self.current_price, 2),
            "current_value": round(self.current_value, 2),
            "cost_basis": round(self.cost_basis, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 2),
            "sector": self.sector,
            "asset_class": self.asset_class,
            "country": self.country,
        }


class Portfolio:
    """
    Digital Twin of an investor's portfolio.

    Parameters
    ----------
    assets : list[Asset]
        Positions in the portfolio.
    historical_prices : pd.DataFrame
        Adjusted close prices (Date × Ticker) used for volatility/correlation.
    risk_profile : str
        One of 'conservative', 'moderate', 'aggressive'.
    name : str
        Human-readable label (shown in dashboard).
    is_scenario : bool
        True when this object represents a scenario simulation (not live).
    """

    # Target volatility — source: Eurobank GR Advisory slide 10 (5yr historic volatility)
    TARGET_VOLATILITY = {
        "liquidity_plus": 0.02,
        "defensive":      0.04,
        "flexible":       0.07,
        "growth":         0.10,
        "dynamic":        0.14,
        # legacy aliases
        "conservative": 0.04,
        "moderate":     0.07,
        "aggressive":   0.14,
    }

    # Strategy weights — source: Eurobank GR Advisory slide 9
    TARGET_ALLOCATION = {
        "liquidity_plus": {"equity": 0.05, "fixed_income": 0.60, "other": 0.35},
        "defensive":      {"equity": 0.25, "fixed_income": 0.55, "other": 0.20},
        "flexible":       {"equity": 0.45, "fixed_income": 0.45, "other": 0.10},
        "growth":         {"equity": 0.65, "fixed_income": 0.30, "other": 0.05},
        "dynamic":        {"equity": 0.85, "fixed_income": 0.10, "other": 0.05},
        # legacy aliases
        "conservative": {"equity": 0.25, "fixed_income": 0.55, "other": 0.20},
        "moderate":     {"equity": 0.45, "fixed_income": 0.45, "other": 0.10},
        "aggressive":   {"equity": 0.85, "fixed_income": 0.10, "other": 0.05},
    }

    # Benchmark tickers per profile (MSCI ACWI + Bloomberg Euro Aggregate proxy)
    # Source: Eurobank GR Advisory slide 10 disclaimer
    BENCHMARK_WEIGHTS = {
        "liquidity_plus": {"ACWI": 0.05, "AGG": 0.60, "cash": 0.35},
        "defensive":      {"ACWI": 0.25, "AGG": 0.55, "cash": 0.20},
        "flexible":       {"ACWI": 0.45, "AGG": 0.45, "cash": 0.10},
        "growth":         {"ACWI": 0.65, "AGG": 0.30, "cash": 0.05},
        "dynamic":        {"ACWI": 0.85, "AGG": 0.10, "cash": 0.05},
        "conservative":   {"ACWI": 0.25, "AGG": 0.55, "cash": 0.20},
        "moderate":       {"ACWI": 0.45, "AGG": 0.45, "cash": 0.10},
        "aggressive":     {"ACWI": 0.85, "AGG": 0.10, "cash": 0.05},
    }

    def __init__(
        self,
        assets: list[Asset],
        historical_prices: pd.DataFrame,
        risk_profile: str = "moderate",
        name: str = "My Portfolio",
        is_scenario: bool = False,
    ):
        self.assets = assets
        self.historical_prices = historical_prices
        self.risk_profile = risk_profile.lower()
        self.name = name
        self.is_scenario = is_scenario

        # Cached metric computations (lazy)
        self._metrics: Optional[dict] = None
        self._returns: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Basic Portfolio Properties
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def total_value(self) -> float:
        return sum(a.current_value for a in self.assets)

    @property
    def total_cost(self) -> float:
        return sum(a.cost_basis for a in self.assets)

    @property
    def total_return_pct(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return (self.total_value / self.total_cost - 1) * 100

    def get_weights(self) -> np.ndarray:
        """Return array of portfolio weights (fraction of total value)."""
        total = self.total_value
        if total == 0:
            return np.zeros(len(self.assets))
        return np.array([a.current_value / total for a in self.assets])

    def get_weights_dict(self) -> dict[str, float]:
        weights = self.get_weights()
        return {a.ticker: float(w) for a, w in zip(self.assets, weights)}

    # ------------------------------------------------------------------
    # Returns Calculation
    # ------------------------------------------------------------------

    def get_daily_returns(self) -> pd.DataFrame:
        """Daily log-returns for holdings that have history available."""
        if self._returns is not None:
            return self._returns

        available = [t for t in self.tickers if t in self.historical_prices.columns]
        if not available:
            return pd.DataFrame()

        prices = self.historical_prices[available]
        self._returns = np.log(prices / prices.shift(1)).dropna()
        return self._returns

    def get_portfolio_daily_returns(self) -> pd.Series:
        """
        Weighted daily portfolio return series.
        Uses current weights applied to historical returns.
        """
        returns = self.get_daily_returns()
        if returns.empty:
            return pd.Series(dtype=float)

        weights = self.get_weights_dict()
        available = [t for t in returns.columns if t in weights]
        if not available:
            return pd.Series(dtype=float)

        w = np.array([weights[t] for t in available])
        # Re-normalise weights for available subset
        w = w / w.sum() if w.sum() > 0 else w
        portfolio_ret = (returns[available] * w).sum(axis=1)
        return portfolio_ret

    # ------------------------------------------------------------------
    # Core Financial Metrics
    # ------------------------------------------------------------------

    def calculate_volatility(self, years: int = 1) -> float:
        """Annualised portfolio volatility for the last `years` years."""
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty or len(daily_ret) < 20:
            return float("nan")
        cutoff = daily_ret.index[-1] - pd.DateOffset(years=years)
        subset = daily_ret[daily_ret.index >= cutoff]
        if len(subset) < 20:
            return float("nan")
        return float(subset.std() * np.sqrt(TRADING_DAYS))

    def calculate_expected_annual_return(self) -> float:
        """Annualised expected return based on historical mean daily return."""
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty:
            return float("nan")
        return float(daily_ret.mean() * TRADING_DAYS)

    def calculate_sharpe_ratio(self) -> float:
        """Sharpe ratio = (portfolio_return - risk_free_rate) / volatility."""
        ret = self.calculate_expected_annual_return()
        vol = self.calculate_volatility()
        if np.isnan(ret) or np.isnan(vol) or vol == 0:
            return float("nan")
        return (ret - RISK_FREE_RATE) / vol

    def calculate_beta(self, market_ticker: str = "SPY") -> float:
        """
        Portfolio beta relative to market (default SPY).

        Beta = Cov(portfolio, market) / Var(market)
        """
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty:
            return float("nan")

        if market_ticker not in self.historical_prices.columns:
            # If market is not in historical data, cannot compute beta
            return float("nan")

        market_prices = self.historical_prices[market_ticker]
        market_ret = np.log(market_prices / market_prices.shift(1)).dropna()

        # Align on common dates
        common_idx = daily_ret.index.intersection(market_ret.index)
        if len(common_idx) < 20:
            return float("nan")

        p = daily_ret.loc[common_idx].values
        m = market_ret.loc[common_idx].values

        cov_matrix = np.cov(p, m)
        if cov_matrix[1, 1] == 0:
            return float("nan")
        return float(cov_matrix[0, 1] / cov_matrix[1, 1])

    def calculate_max_drawdown(self) -> float:
        """
        Maximum historical drawdown of the portfolio value series.

        Returns a negative number representing the largest peak-to-trough loss.
        """
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty:
            return float("nan")

        # Build cumulative wealth index
        cum = (1 + daily_ret).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        return float(drawdown.min())

    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Historical Value-at-Risk at given confidence level.

        Returns the loss (negative number) that is not exceeded with probability
        *confidence*. E.g., VaR 95% = -0.03 means 5% of days lost more than 3%.
        """
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty or len(daily_ret) < 20:
            return float("nan")
        var_daily = float(np.percentile(daily_ret, (1 - confidence) * 100))
        # Annualise to a monthly VaR (21 trading days)
        var_monthly = var_daily * np.sqrt(21)
        return var_monthly

    def calculate_sortino_ratio(self) -> float:
        """Sortino ratio = (portfolio_return - risk_free_rate) / downside_deviation."""
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty or len(daily_ret) < 20:
            return float("nan")
        ret = self.calculate_expected_annual_return()
        if np.isnan(ret):
            return float("nan")
        downside = daily_ret[daily_ret < 0]
        if len(downside) < 5:
            return float("nan")
        downside_dev = float(downside.std() * np.sqrt(TRADING_DAYS))
        if downside_dev == 0:
            return float("nan")
        return (ret - RISK_FREE_RATE) / downside_dev

    def calculate_treynor_ratio(self) -> float:
        """Treynor ratio = (portfolio_return - risk_free_rate) / beta."""
        ret = self.calculate_expected_annual_return()
        beta = self.calculate_beta()
        if np.isnan(ret) or np.isnan(beta) or beta == 0:
            return float("nan")
        return (ret - RISK_FREE_RATE) / beta

    def calculate_information_ratio(self) -> float:
        """
        Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error

        Benchmark: blended ACWI + AGG per profile weights
        Source: Eurobank GR Advisory (MSCI ACWI + Bloomberg Euro Aggregate)
        """
        import yfinance as yf

        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty or len(daily_ret) < 20:
            return float("nan")

        bw = self.BENCHMARK_WEIGHTS.get(self.risk_profile, {})
        acwi_w = bw.get("ACWI", 0.5)
        agg_w  = bw.get("AGG",  0.5)

        def _get_prices(ticker: str) -> pd.Series:
            if ticker in self.historical_prices.columns:
                return self.historical_prices[ticker]
            try:
                raw = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw = raw["Close"]
                    if ticker in raw.columns:
                        return raw[ticker].dropna()
                    return raw.iloc[:, 0].dropna()
                return raw["Close"].dropna()
            except Exception:
                return pd.Series(dtype=float)

        bench_parts = []
        for ticker, weight in [("ACWI", acwi_w), ("AGG", agg_w)]:
            prices = _get_prices(ticker)
            if prices.empty:
                continue
            ret = np.log(prices / prices.shift(1)).dropna()
            common = daily_ret.index.intersection(ret.index)
            if len(common) > 20:
                bench_parts.append(ret.loc[common] * weight)

        if not bench_parts:
            return float("nan")

        bench_ret = bench_parts[0]
        for part in bench_parts[1:]:
            bench_ret = bench_ret.add(part, fill_value=0)

        common_idx = daily_ret.index.intersection(bench_ret.index)
        if len(common_idx) < 20:
            return float("nan")

        active_return = daily_ret.loc[common_idx] - bench_ret.loc[common_idx]
        tracking_error = float(active_return.std() * np.sqrt(TRADING_DAYS))
        if tracking_error == 0:
            return float("nan")

        ann_active = float(active_return.mean() * TRADING_DAYS)
        return ann_active / tracking_error

    def calculate_period_returns(self) -> dict:
        """Returns for standard periods: MTD, YTD, 1Y, 3Y, 5Y."""
        daily_ret = self.get_portfolio_daily_returns()
        if daily_ret.empty:
            return {}

        today = daily_ret.index[-1]
        cum = (1 + daily_ret).cumprod()

        def _ret_since(start_date):
            subset = cum[cum.index >= start_date]
            if subset.empty:
                return None
            start_val = cum[cum.index < start_date]
            if start_val.empty:
                return None
            return float((subset.iloc[-1] / start_val.iloc[-1] - 1) * 100)

        import datetime as _dt
        mtd_start = today.replace(day=1)
        ytd_start = today.replace(month=1, day=1)
        y1_start  = today - _dt.timedelta(days=365)
        y3_start  = today - _dt.timedelta(days=365*3)
        y5_start  = today - _dt.timedelta(days=365*5)

        return {
            "mtd_pct":  _ret_since(mtd_start),
            "ytd_pct":  _ret_since(ytd_start),
            "1y_pct":   _ret_since(y1_start),
            "3y_pct":   _ret_since(y3_start),
            "5y_pct":   _ret_since(y5_start),
        }

    def calculate_diversification_ratio(self) -> float:
        """
        Diversification ratio = weighted-avg individual volatilities / portfolio volatility.

        A ratio > 1 means diversification reduces risk.
        """
        returns = self.get_daily_returns()
        if returns.empty:
            return float("nan")

        weights = self.get_weights_dict()
        available = [t for t in returns.columns if t in weights]
        if not available:
            return float("nan")

        w = np.array([weights[t] for t in available])
        w = w / w.sum() if w.sum() > 0 else w
        individual_vols = returns[available].std() * np.sqrt(TRADING_DAYS)
        weighted_avg_vol = float(np.dot(w, individual_vols.values))

        port_vol = self.calculate_volatility()
        if np.isnan(port_vol) or port_vol == 0:
            return float("nan")
        return weighted_avg_vol / port_vol

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Pairwise correlation matrix of asset daily returns."""
        returns = self.get_daily_returns()
        if returns.empty:
            return pd.DataFrame()
        return returns.corr()

    # ------------------------------------------------------------------
    # Composition Analysis
    # ------------------------------------------------------------------

    def get_composition(self) -> dict:
        """
        Break down portfolio by asset class, sector, and geography.

        Returns a structured dict used by the risk monitor and dashboard.
        """
        total = self.total_value
        if total == 0:
            return {}

        by_asset_class: dict[str, float] = {}
        by_sector: dict[str, float] = {}
        by_country: dict[str, float] = {}

        for asset in self.assets:
            pct = asset.current_value / total

            ac = asset.asset_class or "Unknown"
            by_asset_class[ac] = by_asset_class.get(ac, 0.0) + pct

            sec = asset.sector or "Unknown"
            by_sector[sec] = by_sector.get(sec, 0.0) + pct

            cty = asset.country or "Unknown"
            by_country[cty] = by_country.get(cty, 0.0) + pct

        # Concentration metrics
        weights = self.get_weights()
        herfindahl = float(np.sum(weights ** 2))  # Herfindahl-Hirschman Index
        sorted_w = sorted(weights, reverse=True)
        top_5 = float(sum(sorted_w[:5]))
        largest = float(sorted_w[0]) if sorted_w else 0.0

        return {
            "by_asset_class": {k: round(v * 100, 2) for k, v in by_asset_class.items()},
            "by_sector": {k: round(v * 100, 2) for k, v in by_sector.items()},
            "by_country": {k: round(v * 100, 2) for k, v in by_country.items()},
            "concentration": {
                "largest_single_asset_pct": round(largest * 100, 2),
                "top_5_assets_pct": round(top_5 * 100, 2),
                "herfindahl_index": round(herfindahl, 4),
                "num_holdings": len(self.assets),
            },
        }

    def get_risk_profile_alignment(self) -> dict:
        """How well does current composition match the investor's risk profile?"""
        composition = self.get_composition()
        by_class = {k.lower(): v for k, v in composition.get("by_asset_class", {}).items()}

        # Sum up equity vs fixed income
        equity_pct = sum(v for k, v in by_class.items()
                         if "equity" in k or k in ("equity",))
        fi_pct = sum(v for k, v in by_class.items()
                     if "fixed income" in k or "bond" in k)
        other_pct = 100.0 - equity_pct - fi_pct

        target = self.TARGET_ALLOCATION.get(self.risk_profile, self.TARGET_ALLOCATION["moderate"])
        t_eq = target["equity"] * 100
        t_fi = target["fixed_income"] * 100

        eq_drift = equity_pct - t_eq
        fi_drift = fi_pct - t_fi

        return {
            "risk_profile": self.risk_profile,
            "current_equity_pct": round(equity_pct, 2),
            "current_fixed_income_pct": round(fi_pct, 2),
            "current_other_pct": round(other_pct, 2),
            "target_equity_pct": t_eq,
            "target_fixed_income_pct": t_fi,
            "equity_drift_pct": round(eq_drift, 2),
            "fixed_income_drift_pct": round(fi_drift, 2),
            "max_drift": round(max(abs(eq_drift), abs(fi_drift)), 2),
        }

    # ------------------------------------------------------------------
    # All Metrics (Single Aggregated View)
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """
        Compute and return all key portfolio metrics.

        Caches the result so subsequent calls are instant.
        """
        if self._metrics is not None:
            return self._metrics

        vol = self.calculate_volatility(years=1)
        vol_3y = self.calculate_volatility(years=3)
        target_vol = self.TARGET_VOLATILITY.get(self.risk_profile, 0.12)

        self._metrics = {
            "total_value": round(self.total_value, 2),
            "total_cost": round(self.total_cost, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "expected_annual_return_pct": round(
                self.calculate_expected_annual_return() * 100, 2),
            "volatility_annual_pct": round(vol * 100, 2) if not np.isnan(vol) else None,
            "volatility_3y_pct": round(vol_3y * 100, 2) if not np.isnan(vol_3y) else None,
            "target_volatility_pct": round(target_vol * 100, 2),
            "volatility_deviation_pct": round(
                (vol - target_vol) * 100, 2) if not np.isnan(vol) else None,
            "sharpe_ratio": round(self.calculate_sharpe_ratio(), 3),
            "sortino_ratio": round(self.calculate_sortino_ratio(), 3),
            "treynor_ratio": round(self.calculate_treynor_ratio(), 4),
            "information_ratio": round(self.calculate_information_ratio(), 3),
            "beta": round(self.calculate_beta(), 3),
            "max_drawdown_pct": round(self.calculate_max_drawdown() * 100, 2),
            "var_95_monthly_pct": round(self.calculate_var(0.95) * 100, 2),
            "diversification_ratio": round(self.calculate_diversification_ratio(), 3),
            "num_holdings": len(self.assets),
        }
        return self._metrics

    # ------------------------------------------------------------------
    # Scenario Support
    # ------------------------------------------------------------------

    def apply_scenario(self, scenario: dict) -> "Portfolio":
        """
        Create a modified copy of this portfolio reflecting a market scenario.

        The scenario dict contains:
          market_change       – fraction applied to equity-like assets
          rate_change         – rate change in decimal (e.g. 0.01 = +1%)
          volatility_multiplier – multiplies historical volatility
          sector_overrides    – {sector_name: additional_pct_change}

        Returns a new Portfolio instance with adjusted current prices.
        The historical_prices series is scaled accordingly so metrics
        computed on the scenario portfolio reflect the shifted regime.
        """
        market_change = scenario.get("market_change", 0.0)
        rate_change = scenario.get("rate_change", 0.0)
        vol_mult = scenario.get("volatility_multiplier", 1.0)
        sector_overrides = scenario.get("sector_overrides", {})

        # Bond duration sensitivity: price change ≈ -duration × rate_change
        # Average investment-grade bond duration ~7 years
        BOND_DURATION = 7.0
        bond_price_impact = -BOND_DURATION * rate_change

        new_assets = []
        for asset in self.assets:
            new_price = asset.current_price

            if asset.asset_class in ("Equity", "Equity ETF", "International Equity",
                                      "International ETF", "Emerging Markets ETF",
                                      "Cryptocurrency"):
                # Apply broad market change
                sector_adj = sector_overrides.get(asset.sector, 0.0)
                total_change = market_change + sector_adj
                new_price = asset.current_price * (1 + total_change)

            elif asset.asset_class in ("Fixed Income", "Bond ETF"):
                # Bond prices move inversely to rates
                new_price = asset.current_price * (1 + bond_price_impact)
                # Also apply any general market drag
                new_price *= (1 + market_change * 0.1)  # bonds partially follow

            elif asset.asset_class in ("Real Estate", "Real Estate ETF"):
                rate_adj = rate_change * -2.0  # REITs sensitive to rates
                new_price = asset.current_price * (1 + market_change * 0.8 + rate_adj)

            elif asset.asset_class in ("Commodity", "Commodities ETF"):
                # Commodities partially hedge against inflation / rate changes
                inflation_hedge = rate_change * 0.5
                new_price = asset.current_price * (1 + market_change * 0.5 + inflation_hedge)

            new_price = max(new_price, 0.0)  # prices cannot go negative

            new_asset = copy.copy(asset)
            new_asset.current_price = new_price
            new_assets.append(new_asset)

        # Scale historical prices to reflect scenario volatility shift
        scaled_history = self.historical_prices.copy()
        if vol_mult != 1.0 and not scaled_history.empty:
            # Simulate higher/lower volatility by scaling deviations from mean
            for col in scaled_history.columns:
                mean_price = scaled_history[col].mean()
                scaled_history[col] = (
                    mean_price + (scaled_history[col] - mean_price) * vol_mult
                )

        scenario_portfolio = Portfolio(
            assets=new_assets,
            historical_prices=scaled_history,
            risk_profile=self.risk_profile,
            name=f"{self.name} [{scenario.get('name', 'Scenario')}]",
            is_scenario=True,
        )
        return scenario_portfolio

    def get_allocation_changes_for_rebalance(
        self, new_weights: dict[str, float]
    ) -> list[dict]:
        """
        Show what trades are needed to move from current allocations to new_weights.

        new_weights: {ticker: target_fraction}
        Returns list of {ticker, action, current_value, target_value, delta}
        """
        total = self.total_value
        changes = []
        for asset in self.assets:
            current_w = asset.current_value / total if total > 0 else 0.0
            target_w = new_weights.get(asset.ticker, current_w)
            delta_pct = target_w - current_w
            delta_value = delta_pct * total

            if abs(delta_value) < 10:  # ignore trivial changes < $10
                continue

            changes.append({
                "ticker": asset.ticker,
                "action": "BUY" if delta_value > 0 else "SELL",
                "current_pct": round(current_w * 100, 2),
                "target_pct": round(target_w * 100, 2),
                "current_value": round(asset.current_value, 2),
                "target_value": round(target_w * total, 2),
                "delta_value": round(abs(delta_value), 2),
            })

        return sorted(changes, key=lambda x: x["delta_value"], reverse=True)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Full portfolio snapshot as a plain dict (used by dashboard)."""
        return {
            "name": self.name,
            "risk_profile": self.risk_profile,
            "is_scenario": self.is_scenario,
            "assets": [a.to_dict() for a in self.assets],
            "metrics": self.get_metrics(),
            "composition": self.get_composition(),
            "risk_profile_alignment": self.get_risk_profile_alignment(),
        }

    def __repr__(self) -> str:
        return (f"Portfolio('{self.name}', {len(self.assets)} holdings, "
                f"${self.total_value:,.0f}, {self.risk_profile})")
