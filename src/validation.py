"""
validation.py

Experimental Validation Framework — Digital Twin vs. Baseline Comparison.

Demonstrates the academic contribution by comparing:
  • Baseline (static metrics): volatility, Sharpe, historical drawdown, beta, HHI
  • Proposed (Digital Twin): RiskMonitor alerts + ScenarioEngine simulation

Methodology:
  1. 30 synthetic portfolios across 5 archetypes (concentrated, sector-concentrated,
     moderately diversified, well-diversified, conservative bond-heavy)
  2. Walk-forward split: analysis window (60%) → evaluation window (40%)
  3. Both systems score each portfolio on the analysis window only
  4. Ground truth: actual max drawdown during evaluation window
  5. Statistical comparison: Spearman ρ, Kendall τ, precision/recall for
     "fragile portfolio" detection (drawdown > threshold)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

from .portfolio import Portfolio, Asset
from .risk_monitor import RiskMonitor
from .scenario_engine import ScenarioEngine

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ticker metadata pool
# ─────────────────────────────────────────────────────────────────────────────

TICKER_META: dict[str, dict] = {
    "AAPL":  {"name": "Apple Inc.",                   "sector": "Technology",             "asset_class": "Equity",              "country": "US"},
    "MSFT":  {"name": "Microsoft Corp.",              "sector": "Technology",             "asset_class": "Equity",              "country": "US"},
    "GOOGL": {"name": "Alphabet Inc.",                "sector": "Communication Services", "asset_class": "Equity",              "country": "US"},
    "META":  {"name": "Meta Platforms Inc.",          "sector": "Communication Services", "asset_class": "Equity",              "country": "US"},
    "NVDA":  {"name": "NVIDIA Corp.",                 "sector": "Technology",             "asset_class": "Equity",              "country": "US"},
    "AMZN":  {"name": "Amazon.com Inc.",              "sector": "Consumer Discretionary", "asset_class": "Equity",              "country": "US"},
    "JPM":   {"name": "JPMorgan Chase & Co.",         "sector": "Financials",             "asset_class": "Equity",              "country": "US"},
    "BAC":   {"name": "Bank of America Corp.",        "sector": "Financials",             "asset_class": "Equity",              "country": "US"},
    "GS":    {"name": "Goldman Sachs Group Inc.",     "sector": "Financials",             "asset_class": "Equity",              "country": "US"},
    "JNJ":   {"name": "Johnson & Johnson",            "sector": "Healthcare",             "asset_class": "Equity",              "country": "US"},
    "PFE":   {"name": "Pfizer Inc.",                  "sector": "Healthcare",             "asset_class": "Equity",              "country": "US"},
    "UNH":   {"name": "UnitedHealth Group Inc.",      "sector": "Healthcare",             "asset_class": "Equity",              "country": "US"},
    "WMT":   {"name": "Walmart Inc.",                 "sector": "Consumer Staples",       "asset_class": "Equity",              "country": "US"},
    "HD":    {"name": "Home Depot Inc.",              "sector": "Consumer Discretionary", "asset_class": "Equity",              "country": "US"},
    "XOM":   {"name": "Exxon Mobil Corp.",            "sector": "Energy",                 "asset_class": "Equity",              "country": "US"},
    "CVX":   {"name": "Chevron Corp.",                "sector": "Energy",                 "asset_class": "Equity",              "country": "US"},
    "SPY":   {"name": "SPDR S&P 500 ETF Trust",      "sector": "Broad Market",           "asset_class": "Equity",              "country": "US"},
    "AGG":   {"name": "iShares Core US Agg Bond ETF","sector": "Fixed Income",           "asset_class": "Fixed Income",        "country": "US"},
    "BND":   {"name": "Vanguard Total Bond Mkt ETF", "sector": "Fixed Income",           "asset_class": "Fixed Income",        "country": "US"},
    "TLT":   {"name": "iShares 20+ Year Treasury",   "sector": "Fixed Income",           "asset_class": "Fixed Income",        "country": "US"},
    "VEA":   {"name": "Vanguard FTSE Dev Mkts ETF",  "sector": "International Equity",   "asset_class": "International Equity","country": "Developed"},
    "EEM":   {"name": "iShares MSCI Emerg Mkts ETF", "sector": "International Equity",   "asset_class": "International Equity","country": "Emerging"},
    "GLD":   {"name": "SPDR Gold Shares",             "sector": "Commodities",            "asset_class": "Commodity",           "country": "Global"},
}

# ─────────────────────────────────────────────────────────────────────────────
# 30 Portfolio Archetypes (5 groups × 6 = 30)
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_ARCHETYPES: list[dict] = [
    # ── GROUP 1: Highly Concentrated (6) ─────────────────────────────────────
    {"id": "conc_1", "label": "Single-Stock Dominant (AAPL 70%)",
     "archetype": "Συγκεντρωμένο",
     "tickers": ["AAPL", "MSFT", "GOOGL", "JPM", "WMT"],
     "weights": [0.70, 0.10, 0.08, 0.07, 0.05],
     "risk_profile": "aggressive"},

    {"id": "conc_2", "label": "Dual-Stock Concentration (MSFT+AAPL)",
     "archetype": "Συγκεντρωμένο",
     "tickers": ["MSFT", "AAPL", "NVDA", "JNJ"],
     "weights": [0.50, 0.30, 0.12, 0.08],
     "risk_profile": "aggressive"},

    {"id": "conc_3", "label": "Tech Triple Concentration",
     "archetype": "Συγκεντρωμένο",
     "tickers": ["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
     "weights": [0.30, 0.25, 0.22, 0.15, 0.08],
     "risk_profile": "aggressive"},

    {"id": "conc_4", "label": "Finance Concentrated (JPM 45%)",
     "archetype": "Συγκεντρωμένο",
     "tickers": ["JPM", "BAC", "GS", "MSFT", "XOM"],
     "weights": [0.45, 0.30, 0.15, 0.07, 0.03],
     "risk_profile": "moderate"},

    {"id": "conc_5", "label": "Energy Concentrated (XOM+CVX 85%)",
     "archetype": "Συγκεντρωμένο",
     "tickers": ["XOM", "CVX", "JPM", "AGG"],
     "weights": [0.55, 0.30, 0.10, 0.05],
     "risk_profile": "moderate"},

    {"id": "conc_6", "label": "NVDA Dominant Growth",
     "archetype": "Συγκεντρωμένο",
     "tickers": ["NVDA", "MSFT", "META", "AMZN"],
     "weights": [0.60, 0.20, 0.12, 0.08],
     "risk_profile": "aggressive"},

    # ── GROUP 2: Sector-Concentrated (6) ─────────────────────────────────────
    {"id": "sect_1", "label": "Pure Technology Sector",
     "archetype": "Κλαδικό",
     "tickers": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
     "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
     "risk_profile": "aggressive"},

    {"id": "sect_2", "label": "Pure Financials Sector",
     "archetype": "Κλαδικό",
     "tickers": ["JPM", "BAC", "GS", "WMT", "JNJ"],
     "weights": [0.40, 0.30, 0.20, 0.06, 0.04],
     "risk_profile": "moderate"},

    {"id": "sect_3", "label": "Pure Healthcare Sector",
     "archetype": "Κλαδικό",
     "tickers": ["JNJ", "UNH", "PFE", "AGG", "BND"],
     "weights": [0.40, 0.30, 0.20, 0.07, 0.03],
     "risk_profile": "moderate"},

    {"id": "sect_4", "label": "Pure Consumer Sector",
     "archetype": "Κλαδικό",
     "tickers": ["AMZN", "WMT", "HD", "MSFT"],
     "weights": [0.40, 0.30, 0.20, 0.10],
     "risk_profile": "moderate"},

    {"id": "sect_5", "label": "Pure Energy + Safe Haven",
     "archetype": "Κλαδικό",
     "tickers": ["XOM", "CVX", "GLD", "TLT"],
     "weights": [0.50, 0.30, 0.12, 0.08],
     "risk_profile": "moderate"},

    {"id": "sect_6", "label": "Big-Tech Communication",
     "archetype": "Κλαδικό",
     "tickers": ["GOOGL", "META", "AMZN", "MSFT", "NVDA"],
     "weights": [0.28, 0.25, 0.22, 0.15, 0.10],
     "risk_profile": "aggressive"},

    # ── GROUP 3: Moderately Diversified (6) ──────────────────────────────────
    {"id": "mod_1", "label": "Moderate – Tech & Finance",
     "archetype": "Μέτρια Διαφοροποίηση",
     "tickers": ["AAPL", "MSFT", "JPM", "BAC", "AGG", "BND"],
     "weights": [0.25, 0.20, 0.20, 0.15, 0.12, 0.08],
     "risk_profile": "moderate"},

    {"id": "mod_2", "label": "Moderate – Growth with Bonds",
     "archetype": "Μέτρια Διαφοροποίηση",
     "tickers": ["NVDA", "MSFT", "AMZN", "JNJ", "AGG", "GLD"],
     "weights": [0.20, 0.20, 0.20, 0.15, 0.15, 0.10],
     "risk_profile": "moderate"},

    {"id": "mod_3", "label": "Moderate – Multi-Sector",
     "archetype": "Μέτρια Διαφοροποίηση",
     "tickers": ["AAPL", "JPM", "JNJ", "XOM", "WMT", "AGG", "VEA"],
     "weights": [0.18, 0.16, 0.14, 0.14, 0.12, 0.14, 0.12],
     "risk_profile": "moderate"},

    {"id": "mod_4", "label": "Moderate – Core & Satellite",
     "archetype": "Μέτρια Διαφοροποίηση",
     "tickers": ["SPY", "AAPL", "MSFT", "JNJ", "AGG"],
     "weights": [0.40, 0.20, 0.15, 0.12, 0.13],
     "risk_profile": "moderate"},

    {"id": "mod_5", "label": "Moderate – International Mix",
     "archetype": "Μέτρια Διαφοροποίηση",
     "tickers": ["MSFT", "JNJ", "VEA", "EEM", "AGG", "GLD"],
     "weights": [0.25, 0.20, 0.20, 0.10, 0.15, 0.10],
     "risk_profile": "moderate"},

    {"id": "mod_6", "label": "Moderate – Dividend Focus",
     "archetype": "Μέτρια Διαφοροποίηση",
     "tickers": ["JNJ", "WMT", "XOM", "JPM", "CVX", "AGG"],
     "weights": [0.20, 0.18, 0.17, 0.17, 0.13, 0.15],
     "risk_profile": "moderate"},

    # ── GROUP 4: Well Diversified (6) ─────────────────────────────────────────
    {"id": "div_1", "label": "Diversified – Classic Balanced",
     "archetype": "Καλά Διαφοροποιημένο",
     "tickers": ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "WMT", "AGG", "BND", "VEA", "GLD"],
     "weights": [0.12, 0.11, 0.10, 0.10, 0.09, 0.08, 0.12, 0.10, 0.10, 0.08],
     "risk_profile": "moderate"},

    {"id": "div_2", "label": "Diversified – Global ETF Mix",
     "archetype": "Καλά Διαφοροποιημένο",
     "tickers": ["SPY", "VEA", "EEM", "AGG", "TLT", "GLD"],
     "weights": [0.35, 0.20, 0.10, 0.20, 0.10, 0.05],
     "risk_profile": "moderate"},

    {"id": "div_3", "label": "Diversified – All-Weather",
     "archetype": "Καλά Διαφοροποιημένο",
     "tickers": ["SPY", "TLT", "GLD", "EEM", "BND"],
     "weights": [0.30, 0.25, 0.15, 0.15, 0.15],
     "risk_profile": "moderate"},

    {"id": "div_4", "label": "Diversified – Risk-Parity Style",
     "archetype": "Καλά Διαφοροποιημένο",
     "tickers": ["AAPL", "JNJ", "JPM", "XOM", "WMT", "AGG", "GLD", "VEA", "EEM", "BND"],
     "weights": [0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.10, 0.10, 0.08, 0.10],
     "risk_profile": "moderate"},

    {"id": "div_5", "label": "Diversified – Multi-Asset",
     "archetype": "Καλά Διαφοροποιημένο",
     "tickers": ["MSFT", "GOOGL", "JPM", "JNJ", "CVX", "WMT", "AGG", "VEA", "GLD", "BND"],
     "weights": [0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.13, 0.10, 0.09, 0.10],
     "risk_profile": "moderate"},

    {"id": "div_6", "label": "Diversified – Defensive Balanced",
     "archetype": "Καλά Διαφοροποιημένο",
     "tickers": ["WMT", "JNJ", "MSFT", "XOM", "AGG", "GLD", "VEA", "BND"],
     "weights": [0.12, 0.12, 0.14, 0.10, 0.15, 0.10, 0.12, 0.15],
     "risk_profile": "moderate"},

    # ── GROUP 5: Bond-Heavy Conservative (6) ──────────────────────────────────
    {"id": "cons_1", "label": "Conservative – Bond Dominant",
     "archetype": "Συντηρητικό",
     "tickers": ["AGG", "BND", "TLT", "JNJ", "WMT"],
     "weights": [0.35, 0.30, 0.20, 0.10, 0.05],
     "risk_profile": "conservative"},

    {"id": "cons_2", "label": "Conservative – Income Focused",
     "archetype": "Συντηρητικό",
     "tickers": ["BND", "AGG", "JNJ", "WMT", "XOM", "GLD"],
     "weights": [0.30, 0.28, 0.15, 0.12, 0.10, 0.05],
     "risk_profile": "conservative"},

    {"id": "cons_3", "label": "Conservative – Capital Preservation",
     "archetype": "Συντηρητικό",
     "tickers": ["TLT", "AGG", "GLD", "WMT", "JNJ"],
     "weights": [0.35, 0.30, 0.15, 0.12, 0.08],
     "risk_profile": "conservative"},

    {"id": "cons_4", "label": "Conservative – Defensive Equity",
     "archetype": "Συντηρητικό",
     "tickers": ["AGG", "BND", "JNJ", "WMT", "PFE", "CVX"],
     "weights": [0.30, 0.25, 0.15, 0.15, 0.10, 0.05],
     "risk_profile": "conservative"},

    {"id": "cons_5", "label": "Conservative – Gold & Bonds",
     "archetype": "Συντηρητικό",
     "tickers": ["GLD", "TLT", "AGG", "BND", "JNJ"],
     "weights": [0.25, 0.25, 0.20, 0.20, 0.10],
     "risk_profile": "conservative"},

    {"id": "cons_6", "label": "Conservative – Ultra-Safe",
     "archetype": "Συντηρητικό",
     "tickers": ["BND", "AGG", "TLT", "GLD"],
     "weights": [0.35, 0.30, 0.25, 0.10],
     "risk_profile": "conservative"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Build Portfolio from pre-fetched price data
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_from_prices(
    archetype: dict,
    prices_df: pd.DataFrame,
    synthetic_value: float = 100_000.0,
) -> Portfolio:
    """
    Build a Portfolio from pre-fetched price data and an archetype definition.

    Entry price = first price in prices_df (simulates buying at start of window).
    Current price = last price in prices_df.
    Quantities are set so that target_value * weight = cost at entry.
    """
    tickers = archetype["tickers"]
    weights = archetype["weights"]

    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        raise ValueError(f"No price data available for tickers: {tickers}")

    avail_idx = [tickers.index(t) for t in available]
    raw_w = [weights[i] for i in avail_idx]
    total_w = sum(raw_w)
    norm_w = [w / total_w for w in raw_w]

    assets = []
    for ticker, weight in zip(available, norm_w):
        series = prices_df[ticker].dropna()
        if len(series) < 2:
            continue
        entry_price = float(series.iloc[0])
        current_price = float(series.iloc[-1])
        quantity = (synthetic_value * weight) / entry_price

        meta = TICKER_META.get(ticker, {})
        assets.append(Asset(
            ticker=ticker,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            name=meta.get("name", ticker),
            sector=meta.get("sector", "Unknown"),
            industry="",
            asset_class=meta.get("asset_class", "Equity"),
            country=meta.get("country", "US"),
        ))

    if not assets:
        raise ValueError(f"Could not build assets for archetype '{archetype['id']}'")

    return Portfolio(
        assets=assets,
        historical_prices=prices_df[[a.ticker for a in assets]],
        risk_profile=archetype.get("risk_profile", "moderate"),
        name=archetype["label"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth: Actual Drawdown During Evaluation Window
# ─────────────────────────────────────────────────────────────────────────────

def compute_actual_drawdown(
    tickers: list[str],
    weights: list[float],
    eval_prices: pd.DataFrame,
) -> float:
    """
    Compute the actual max drawdown of a portfolio during the evaluation window.
    Returns a negative percentage (e.g., -25.3 means a 25.3% max drawdown).
    """
    available = [t for t in tickers if t in eval_prices.columns]
    if not available:
        return 0.0

    avail_idx = [tickers.index(t) for t in available]
    raw_w = np.array([weights[i] for i in avail_idx], dtype=float)
    norm_w = raw_w / raw_w.sum()

    prices = eval_prices[available].ffill().bfill().dropna(how="all")
    if len(prices) < 5:
        return 0.0

    norm_prices = prices / prices.iloc[0]
    port_values = (norm_prices * norm_w).sum(axis=1)

    rolling_max = port_values.cummax()
    drawdown = (port_values - rolling_max) / rolling_max
    return round(float(drawdown.min()) * 100.0, 2)


def compute_actual_return(
    tickers: list[str],
    weights: list[float],
    eval_prices: pd.DataFrame,
) -> float:
    """Total return (%) of a portfolio during the evaluation window."""
    available = [t for t in tickers if t in eval_prices.columns]
    if not available:
        return 0.0

    avail_idx = [tickers.index(t) for t in available]
    raw_w = np.array([weights[i] for i in avail_idx], dtype=float)
    norm_w = raw_w / raw_w.sum()

    prices = eval_prices[available].ffill().bfill().dropna(how="all")
    if len(prices) < 2:
        return 0.0

    ret = ((prices.iloc[-1] / prices.iloc[0] - 1) * norm_w).sum()
    return round(float(ret) * 100.0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Risk Scorer (Static Metrics Only)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineRiskScorer:
    """
    Naive static-metrics-only risk scorer — the baseline system.

    Uses only historical statistics from the Portfolio object:
      - Annualized volatility       (higher → riskier)
      - Sharpe ratio inverse        (lower Sharpe → riskier)
      - Historical max drawdown     (larger drop → riskier)
      - Beta vs. S&P 500            (higher β → riskier)
      - Herfindahl-Hirschman Index  (higher HHI → more concentrated → riskier)

    Each factor is min-max normalized and combined into a composite [0, 100] score.
    No simulation, no scenario analysis, no rule-based alerts.
    """

    FACTOR_WEIGHTS = {
        "volatility":     0.30,
        "sharpe_inverse": 0.20,
        "drawdown":       0.25,
        "beta":           0.15,
        "hhi":            0.10,
    }

    def score(self, portfolio: Portfolio) -> dict:
        metrics = portfolio.get_metrics()
        composition = portfolio.get_composition()

        vol = metrics.get("volatility_annual_pct") or 0.0
        sharpe = metrics.get("sharpe_ratio") or 0.0
        drawdown_abs = abs(metrics.get("max_drawdown_pct") or 0.0)
        beta = max(metrics.get("beta") or 1.0, 0.0)
        hhi = composition.get("concentration", {}).get("herfindahl_index", 0.0)

        # Normalize each metric to [0, 1] using empirically chosen ranges
        vol_n      = min(vol / 45.0, 1.0)
        sharpe_n   = max(0.0, min(1.0, (2.5 - sharpe) / 5.0))
        dd_n       = min(drawdown_abs / 65.0, 1.0)
        beta_n     = min(beta / 2.5, 1.0)
        hhi_n      = min(hhi / 0.50, 1.0)

        composite = (
            self.FACTOR_WEIGHTS["volatility"]     * vol_n +
            self.FACTOR_WEIGHTS["sharpe_inverse"] * sharpe_n +
            self.FACTOR_WEIGHTS["drawdown"]       * dd_n +
            self.FACTOR_WEIGHTS["beta"]           * beta_n +
            self.FACTOR_WEIGHTS["hhi"]            * hhi_n
        ) * 100.0

        return {
            "risk_score":        round(composite, 2),
            "volatility_pct":    round(vol, 2),
            "sharpe_ratio":      round(sharpe, 3),
            "max_drawdown_pct":  round(-drawdown_abs, 2),
            "beta":              round(beta, 3),
            "hhi":               round(hhi, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Proposed Risk Scorer (Digital Twin: RiskMonitor + ScenarioEngine)
# ─────────────────────────────────────────────────────────────────────────────

class ProposedRiskScorer:
    """
    Simulation-based risk scorer — the proposed Digital Twin system.

    Combines:
      1. RiskMonitor: 6-category alert system (ok=0, caution=1, alert=2)
      2. ScenarioEngine: worst-case loss across standard stress scenarios
      3. Profile drift severity as an additional signal

    The composite score reflects both structural weaknesses (alerts) and
    forward-looking vulnerability (scenario simulation).
    """

    # Stress scenarios evaluated for each portfolio
    EVALUATION_SCENARIOS = [
        "market_down_20",
        "market_down_30",
        "financial_crisis_2008",
        "covid_crash_2020",
        "rates_up_200bps",
        "tech_crash",
        "recession",
        "inflation_spike",
    ]

    FACTOR_WEIGHTS = {
        "alert_score":   0.45,
        "scenario_loss": 0.40,
        "profile_drift": 0.15,
    }

    SEVERITY_SCORE = {"ok": 0, "caution": 1, "alert": 2}

    def score(self, portfolio: Portfolio) -> dict:
        # ── 1. Risk Monitor alerts ─────────────────────────────────────────
        monitor = RiskMonitor(portfolio)
        analysis = monitor.run_full_analysis()

        n_alerts_all = len(analysis["all"])
        raw_alert = sum(
            self.SEVERITY_SCORE.get(a["severity"], 0) for a in analysis["all"]
        )
        max_possible = n_alerts_all * 2
        alert_norm = raw_alert / max(max_possible, 1)

        # Profile drift severity [0, 1]
        drift_alerts = [
            a for a in analysis["all"]
            if "drift" in a.get("category", "").lower()
        ]
        drift_sev = max(
            (self.SEVERITY_SCORE.get(a["severity"], 0) for a in drift_alerts),
            default=0,
        )
        drift_norm = drift_sev / 2.0

        # ── 2. Scenario Engine worst-case loss ────────────────────────────
        worst_loss = 0.0
        scenario_losses: dict[str, float] = {}
        try:
            engine = ScenarioEngine(portfolio)
            for sid in self.EVALUATION_SCENARIOS:
                try:
                    comp = engine.compare_portfolio_metrics(sid)
                    loss = comp["summary"]["portfolio_change_pct"]
                    scenario_losses[sid] = loss
                    if loss < worst_loss:
                        worst_loss = loss
                except Exception:
                    pass
        except Exception:
            pass

        # Normalize: −60% or worse → 1.0, 0% → 0.0
        scenario_norm = min(abs(min(worst_loss, 0.0)) / 60.0, 1.0)

        # ── 3. Composite score ────────────────────────────────────────────
        composite = (
            self.FACTOR_WEIGHTS["alert_score"]   * alert_norm +
            self.FACTOR_WEIGHTS["scenario_loss"] * scenario_norm +
            self.FACTOR_WEIGHTS["profile_drift"] * drift_norm
        ) * 100.0

        return {
            "risk_score":             round(composite, 2),
            "n_ok":                   analysis["summary"]["ok"],
            "n_cautions":             analysis["summary"]["caution"],
            "n_alerts":               analysis["summary"]["alert"],
            "alert_score_raw":        raw_alert,
            "worst_scenario_loss_pct": round(worst_loss, 2),
            "highest_severity":       analysis["highest_severity"],
            "scenario_losses":        scenario_losses,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validation Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResults:
    """Stores the complete results of the experimental validation run."""

    n_portfolios:         int
    analysis_start:       str
    analysis_end:         str
    eval_start:           str
    eval_end:             str

    portfolio_ids:        list[str]
    portfolio_labels:     list[str]
    portfolio_archetypes: list[str]

    baseline_scores:      list[float]
    proposed_scores:      list[float]
    actual_drawdowns:     list[float]
    actual_returns:       list[float]

    baseline_details:     list[dict] = field(default_factory=list)
    proposed_details:     list[dict] = field(default_factory=list)

    metrics: dict = field(default_factory=dict)

    # ── Metric computation ──────────────────────────────────────────────────

    def compute_metrics(self, fragile_threshold: float = -15.0) -> dict:
        """
        Compute all statistical comparison metrics.

        fragile_threshold: portfolios with actual_drawdown < this are "fragile"
        """
        # Remove NaN entries
        valid = [
            (b, p, d, r)
            for b, p, d, r in zip(
                self.baseline_scores, self.proposed_scores,
                self.actual_drawdowns, self.actual_returns
            )
            if not (np.isnan(b) or np.isnan(p) or np.isnan(d))
        ]
        if len(valid) < 5:
            self.metrics = {"error": "Insufficient valid data for statistics."}
            return self.metrics

        b_arr = np.array([v[0] for v in valid])
        p_arr = np.array([v[1] for v in valid])
        d_arr = np.array([v[2] for v in valid])  # negative = loss

        # ── 1. Spearman Rank Correlation ──────────────────────────────────
        # Higher risk score should predict more negative actual drawdown
        rho_b, pval_b = stats.spearmanr(b_arr, -d_arr)
        rho_p, pval_p = stats.spearmanr(p_arr, -d_arr)

        # ── 2. Kendall's Tau ──────────────────────────────────────────────
        tau_b, ptau_b = stats.kendalltau(b_arr, -d_arr)
        tau_p, ptau_p = stats.kendalltau(p_arr, -d_arr)

        # ── 3. Fragile Portfolio Detection ───────────────────────────────
        fragile_true = np.array([dd < fragile_threshold for dd in d_arr])
        n_fragile = int(fragile_true.sum())

        if 0 < n_fragile < len(d_arr):
            k = n_fragile
            topk_b = set(np.argsort(b_arr)[::-1][:k])
            topk_p = set(np.argsort(p_arr)[::-1][:k])

            pred_b = np.array([i in topk_b for i in range(len(b_arr))])
            pred_p = np.array([i in topk_p for i in range(len(p_arr))])

            prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
                fragile_true, pred_b, average="binary", zero_division=0
            )
            prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(
                fragile_true, pred_p, average="binary", zero_division=0
            )
        else:
            prec_b = rec_b = f1_b = np.nan
            prec_p = rec_p = f1_p = np.nan

        # ── 4. MAE proxy (drawdown prediction error) ───────────────────
        # Map risk score [0,100] → predicted loss [0,60%] for comparison
        b_pred_loss = b_arr / 100.0 * 60.0
        p_pred_loss = p_arr / 100.0 * 60.0
        actual_loss = -d_arr  # positive values

        mae_b = float(np.mean(np.abs(b_pred_loss - actual_loss)))
        mae_p = float(np.mean(np.abs(p_pred_loss - actual_loss)))

        # ── 5. Improvement (proposed over baseline) ───────────────────
        improvement_spearman = (
            round(float(rho_p - rho_b), 3)
            if not (np.isnan(rho_p) or np.isnan(rho_b)) else None
        )
        improvement_f1 = (
            round(float(f1_p - f1_b), 3)
            if not (np.isnan(f1_p) or np.isnan(f1_b)) else None
        )
        improvement_mae = (
            round(float(mae_b - mae_p), 2)
            if not (np.isnan(mae_b) or np.isnan(mae_p)) else None
        )

        self.metrics = {
            "n_valid":     len(valid),
            "fragile_threshold_pct": fragile_threshold,
            "n_fragile":   n_fragile,
            "spearman": {
                "baseline_rho":   round(float(rho_b), 3),
                "baseline_pval":  round(float(pval_b), 4),
                "proposed_rho":   round(float(rho_p), 3),
                "proposed_pval":  round(float(pval_p), 4),
                "improvement":    improvement_spearman,
            },
            "kendall": {
                "baseline_tau":   round(float(tau_b), 3),
                "baseline_pval":  round(float(ptau_b), 4),
                "proposed_tau":   round(float(tau_p), 3),
                "proposed_pval":  round(float(ptau_p), 4),
            },
            "fragile_detection": {
                "baseline_precision": round(float(prec_b), 3) if not np.isnan(prec_b) else None,
                "baseline_recall":    round(float(rec_b), 3)  if not np.isnan(rec_b)  else None,
                "baseline_f1":        round(float(f1_b), 3)   if not np.isnan(f1_b)   else None,
                "proposed_precision": round(float(prec_p), 3) if not np.isnan(prec_p) else None,
                "proposed_recall":    round(float(rec_p), 3)  if not np.isnan(rec_p)  else None,
                "proposed_f1":        round(float(f1_p), 3)   if not np.isnan(f1_p)   else None,
                "f1_improvement":     improvement_f1,
            },
            "mae_proxy": {
                "baseline_mae_pct":  round(mae_b, 2),
                "proposed_mae_pct":  round(mae_p, 2),
                "improvement_pct":   improvement_mae,
            },
        }
        return self.metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Build a per-portfolio summary DataFrame for display."""
        rows = []
        for i in range(self.n_portfolios):
            b = self.baseline_details[i] if i < len(self.baseline_details) else {}
            p = self.proposed_details[i] if i < len(self.proposed_details) else {}
            rows.append({
                "ID":                         self.portfolio_ids[i],
                "Χαρτοφυλάκιο":              self.portfolio_labels[i],
                "Κατηγορία":                  self.portfolio_archetypes[i],
                "Baseline Score":             self.baseline_scores[i],
                "Proposed Score":             self.proposed_scores[i],
                "Πραγματικό Drawdown (%)":    self.actual_drawdowns[i],
                "Πραγματική Απόδοση (%)":     self.actual_returns[i],
                "Baseline: Volatility (%)":   b.get("volatility_pct"),
                "Baseline: Sharpe":           b.get("sharpe_ratio"),
                "Proposed: Alerts":           p.get("n_alerts"),
                "Proposed: Χειρ. Σενάριο (%)": p.get("worst_scenario_loss_pct"),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Validation Experiment Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ValidationExperiment:
    """
    Orchestrates the full experimental validation.

    Steps
    -----
    1. Collect all unique tickers across all 30 archetypes
    2. Fetch 5 years of historical price data in a single bulk yfinance call
    3. Split: analysis window (first 60%) / evaluation window (last 40%)
    4. Build Portfolio objects for each archetype using the analysis window
    5. Score each portfolio with BaselineRiskScorer and ProposedRiskScorer
    6. Compute ground-truth drawdowns from the evaluation window
    7. Run statistical comparisons and return ValidationResults

    Parameters
    ----------
    archetypes : list[dict] | None
        Override the default 30-portfolio set (for testing or custom experiments).
    history_years : int
        Years of historical data to fetch (default 5).
    split_fraction : float
        Fraction of data used as analysis window (default 0.60).
    progress_callback : callable | None
        Optional (step, total, message) → None for progress reporting.
    """

    def __init__(
        self,
        archetypes: list[dict] | None = None,
        history_years: int = 5,
        split_fraction: float = 0.60,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        self.archetypes = archetypes or PORTFOLIO_ARCHETYPES
        self.history_years = history_years
        self.split_fraction = split_fraction
        self._cb = progress_callback
        self._total_steps = len(self.archetypes) + 6

    def _progress(self, step: int, message: str) -> None:
        if self._cb:
            self._cb(step, self._total_steps, message)
        else:
            logger.info("[%d/%d] %s", step, self._total_steps, message)

    def _all_tickers(self) -> list[str]:
        tickers: set[str] = set()
        for arch in self.archetypes:
            tickers.update(arch["tickers"])
        return sorted(tickers)

    def fetch_data(self) -> pd.DataFrame:
        """Fetch price data for all tickers in one bulk call."""
        tickers = self._all_tickers()
        self._progress(1, f"Φόρτωση δεδομένων για {len(tickers)} tickers από Yahoo Finance…")

        end = datetime.now()
        start = end - timedelta(days=int(self.history_years * 365.25))

        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # yfinance returns MultiIndex columns when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"]
        else:
            df = raw

        # Keep only tickers with sufficient data (>= 1 year of trading days)
        df = df.dropna(thresh=min(252, len(df) // 2), axis=1)
        self._progress(2, f"Δεδομένα διαθέσιμα: {df.shape[1]} tickers, {len(df)} ημέρες")
        return df

    def split_data(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split prices at split_fraction."""
        n = len(prices)
        k = int(n * self.split_fraction)
        return prices.iloc[:k], prices.iloc[k:]

    def run(self) -> "ValidationResults":
        """Execute the full experiment and return ValidationResults."""
        # 1. Fetch & split data
        all_prices = self.fetch_data()
        analysis_prices, eval_prices = self.split_data(all_prices)

        analysis_start = str(analysis_prices.index[0].date())
        analysis_end   = str(analysis_prices.index[-1].date())
        eval_start     = str(eval_prices.index[0].date())
        eval_end       = str(eval_prices.index[-1].date())

        self._progress(3, f"Παράθυρο ανάλυσης: {analysis_start} → {analysis_end}")
        self._progress(4, f"Παράθυρο αξιολόγησης: {eval_start} → {eval_end}")

        baseline_scorer = BaselineRiskScorer()
        proposed_scorer = ProposedRiskScorer()

        portfolio_ids:        list[str]   = []
        portfolio_labels:     list[str]   = []
        portfolio_archetypes: list[str]   = []
        baseline_scores:      list[float] = []
        proposed_scores:      list[float] = []
        actual_drawdowns:     list[float] = []
        actual_returns:       list[float] = []
        baseline_details:     list[dict]  = []
        proposed_details:     list[dict]  = []

        n = len(self.archetypes)
        for i, arch in enumerate(self.archetypes):
            self._progress(5 + i, f"[{i+1}/{n}] Βαθμολόγηση: {arch['label']}")

            try:
                portfolio = build_portfolio_from_prices(arch, analysis_prices)
                b_detail  = baseline_scorer.score(portfolio)
                p_detail  = proposed_scorer.score(portfolio)
                act_dd    = compute_actual_drawdown(
                    arch["tickers"], arch["weights"], eval_prices)
                act_ret   = compute_actual_return(
                    arch["tickers"], arch["weights"], eval_prices)

                portfolio_ids.append(arch["id"])
                portfolio_labels.append(arch["label"])
                portfolio_archetypes.append(arch["archetype"])
                baseline_scores.append(b_detail["risk_score"])
                proposed_scores.append(p_detail["risk_score"])
                actual_drawdowns.append(act_dd)
                actual_returns.append(act_ret)
                baseline_details.append(b_detail)
                proposed_details.append(p_detail)

            except Exception as exc:
                logger.warning("Σφάλμα για '%s': %s", arch["id"], exc)
                portfolio_ids.append(arch["id"])
                portfolio_labels.append(arch["label"])
                portfolio_archetypes.append(arch.get("archetype", ""))
                for lst in (baseline_scores, proposed_scores,
                            actual_drawdowns, actual_returns):
                    lst.append(float("nan"))
                baseline_details.append({})
                proposed_details.append({})

        self._progress(5 + n, "Υπολογισμός στατιστικών μετρικών…")

        results = ValidationResults(
            n_portfolios=len(portfolio_ids),
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            eval_start=eval_start,
            eval_end=eval_end,
            portfolio_ids=portfolio_ids,
            portfolio_labels=portfolio_labels,
            portfolio_archetypes=portfolio_archetypes,
            baseline_scores=baseline_scores,
            proposed_scores=proposed_scores,
            actual_drawdowns=actual_drawdowns,
            actual_returns=actual_returns,
            baseline_details=baseline_details,
            proposed_details=proposed_details,
        )
        results.compute_metrics()

        self._progress(self._total_steps, "Πείραμα ολοκληρώθηκε!")
        return results
