"""
resilience.py
-------------
Composite Portfolio Resilience Index (RI), 0-100, higher = more resilient.

Six components, each mapped to [0, 1] with fixed, a-priori bounds so that the
score of one portfolio does not depend on any other portfolio:

    Component        Measures                                   Weight
    ---------------  -----------------------------------------  ------
    stress           mean loss over six stress scenarios          0.25
    sequential       loss after two consecutive shocks            0.10
                     (market −20%, then rates +2%)
    downside         historical daily CVaR 95%                    0.20
    recovery         time to recover from the max drawdown and    0.20
                     share of days spent under water
    diversification  diversification ratio Σwᵢσᵢ / σₚ             0.15
    concentration    effective number of positions (1 / HHI),     0.10
                     a broad ETF counting as several positions

The weights are fixed before any evaluation and are not tuned on results.
Every component is returned alongside the total so that each score can be
explained (and ablated in the validation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio import Portfolio, TRADING_DAYS
from .risk_monitor import is_diversified_fund
from .scenario_engine import STANDARD_SCENARIOS

WEIGHTS = {
    "stress": 0.25,
    "sequential": 0.10,
    "downside": 0.20,
    "recovery": 0.20,
    "diversification": 0.15,
    "concentration": 0.10,
}

STRESS_SCENARIOS = [
    "financial_crisis_2008", "covid_crash_2020", "recession",
    "inflation_spike", "rates_up_200bps", "tech_crash",
]
SEQUENTIAL_SHOCKS = ["market_down_20", "rates_up_200bps"]

# Normalisation bounds (value at which a component reaches 0)
MAX_STRESS_LOSS = 0.50          # mean stress loss of 50% → 0
MAX_SEQUENTIAL_LOSS = 0.50
MAX_DAILY_CVAR = 0.05           # 5% average loss on the worst 5% of days → 0
MAX_RECOVERY_DAYS = 2 * TRADING_DAYS
MAX_DIVERSIFICATION_RATIO = 2.0  # DR of 1 → 0, DR of 2 or more → 1
FULL_EFFECTIVE_N = 15.0          # 15 effective positions or more → 1
FUND_EQUIVALENT_POSITIONS = 5    # a broad ETF counts as 5 equal positions

LABELS = {
    "stress": "Αντοχή σε stress σενάρια",
    "sequential": "Αντοχή σε διαδοχικά shocks",
    "downside": "Κίνδυνος ουράς (CVaR 95%)",
    "recovery": "Ταχύτητα ανάκαμψης",
    "diversification": "Αποτελεσματικότητα διαφοροποίησης",
    "concentration": "Αποφυγή συγκέντρωσης",
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) if np.isfinite(x) else float("nan")


def _scenario_loss(portfolio: Portfolio, scenario_ids: list[str]) -> float:
    """Fractional loss after applying the scenarios in sequence (positive = loss)."""
    start = portfolio.total_value
    current = portfolio
    for sid in scenario_ids:
        current = current.apply_scenario(STANDARD_SCENARIOS[sid])
    return (start - current.total_value) / start if start > 0 else float("nan")


def recovery_stats(daily_log_returns: pd.Series) -> dict:
    """
    Days from the max-drawdown trough back to the previous peak and the share
    of days the wealth curve sits more than 5% below its running peak.
    An unrecovered drawdown counts the days elapsed since the trough, plus a
    penalty of the same length, since it is still open.
    """
    r = daily_log_returns.dropna()
    if len(r) < 20:
        return {"recovery_days": float("nan"), "underwater_frac": float("nan"),
                "recovered": False}
    wealth = np.exp(r.cumsum())
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    trough_pos = int(np.argmin(dd.values))
    peak_level = peak.iloc[trough_pos]
    after = wealth.iloc[trough_pos:]
    back = np.nonzero(after.values >= peak_level)[0]
    if len(back):
        days, recovered = int(back[0]), True
    else:
        elapsed = len(after) - 1
        days, recovered = 2 * elapsed, False
    return {
        "recovery_days": float(days),
        "underwater_frac": float((dd < -0.05).mean()),
        "recovered": recovered,
        "max_drawdown": float(dd.min()),
    }


def _effective_positions(portfolio: Portfolio) -> float:
    w = portfolio.get_weights_dict()
    hhi = 0.0
    for a in portfolio.assets:
        wi = w.get(a.ticker, 0.0)
        if is_diversified_fund(a):
            hhi += wi ** 2 / FUND_EQUIVALENT_POSITIONS
        else:
            hhi += wi ** 2
    return 1.0 / hhi if hhi > 0 else float("nan")


def _diversification_ratio(portfolio: Portfolio) -> float:
    """Σ wᵢσᵢ / σₚ, both measured over the same full history window."""
    returns = portfolio.get_daily_returns()
    if returns.empty:
        return float("nan")
    w_all = portfolio.get_weights_dict()
    cols = [t for t in returns.columns if t in w_all]
    w = np.array([w_all[t] for t in cols])
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    port_vol = float((returns[cols] @ w).std())
    if port_vol <= 0:
        return float("nan")
    return float(w @ returns[cols].std().values) / port_vol


def compute_resilience(portfolio: Portfolio) -> dict:
    """
    Returns
    -------
    {
      "score": float 0-100,
      "components": {name: 0-1 score},
      "raw": {name: underlying value},
      "weights": WEIGHTS,
      "labels": LABELS,
    }
    """
    stress_losses = {sid: _scenario_loss(portfolio, [sid]) for sid in STRESS_SCENARIOS}
    mean_stress = float(np.nanmean(list(stress_losses.values())))
    seq_loss = _scenario_loss(portfolio, SEQUENTIAL_SHOCKS)

    port_ret = portfolio.get_portfolio_daily_returns()
    simple = np.expm1(port_ret.dropna())
    if len(simple) >= 20:
        cutoff = simple.quantile(0.05)
        cvar = float(-simple[simple <= cutoff].mean())
    else:
        cvar = float("nan")
    rec = recovery_stats(port_ret)

    dr = _diversification_ratio(portfolio)
    eff_n = _effective_positions(portfolio)

    recovery_score = np.nanmean([
        1 - rec["recovery_days"] / MAX_RECOVERY_DAYS,
        1 - rec["underwater_frac"],
    ]) if np.isfinite(rec["recovery_days"]) else float("nan")

    components = {
        "stress": _clip01(1 - max(mean_stress, 0) / MAX_STRESS_LOSS),
        "sequential": _clip01(1 - max(seq_loss, 0) / MAX_SEQUENTIAL_LOSS),
        "downside": _clip01(1 - cvar / MAX_DAILY_CVAR),
        "recovery": _clip01(recovery_score),
        "diversification": _clip01((dr - 1) / (MAX_DIVERSIFICATION_RATIO - 1)),
        "concentration": _clip01((eff_n - 1) / (FULL_EFFECTIVE_N - 1)),
    }

    # Missing components (e.g. no history) are dropped and weights renormalised
    valid = {k: v for k, v in components.items() if np.isfinite(v)}
    wsum = sum(WEIGHTS[k] for k in valid)
    score = 100 * sum(WEIGHTS[k] * v for k, v in valid.items()) / wsum if wsum else float("nan")

    return {
        "score": float(score),
        "components": components,
        "raw": {
            "stress_losses_pct": {k: round(v * 100, 2) for k, v in stress_losses.items()},
            "mean_stress_loss_pct": round(mean_stress * 100, 2),
            "sequential_loss_pct": round(seq_loss * 100, 2),
            "daily_cvar95_pct": round(cvar * 100, 3) if np.isfinite(cvar) else None,
            "recovery_days": rec["recovery_days"],
            "recovered": rec["recovered"],
            "underwater_frac_pct": round(rec["underwater_frac"] * 100, 1)
            if np.isfinite(rec["underwater_frac"]) else None,
            "diversification_ratio": round(dr, 3) if np.isfinite(dr) else None,
            "effective_positions": round(eff_n, 2) if np.isfinite(eff_n) else None,
        },
        "weights": WEIGHTS,
        "labels": LABELS,
    }


def resilience_band(score: float) -> str:
    if not np.isfinite(score):
        return "Μη διαθέσιμο"
    if score >= 70:
        return "Υψηλή"
    if score >= 50:
        return "Μέτρια"
    if score >= 30:
        return "Χαμηλή"
    return "Πολύ χαμηλή"
