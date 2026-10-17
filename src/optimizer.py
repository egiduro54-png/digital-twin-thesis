"""
optimizer.py
------------
Mean-variance (Markowitz) portfolio optimisation implemented with SciPy.

Replaces the former PyPortfolioOpt dependency, which pulled in cvxpy and
contributed to the Streamlit Cloud segfault. SciPy is already a dependency
and is imported lazily inside the functions that need it.

Estimators (annualised, from daily simple returns of the price history):
    mu    = mean(r_daily) * 252
    Sigma = cov(r_daily)  * 252

Objective: maximise Sharpe = (w·mu − R_f) / sqrt(wᵀ Σ w)
Constraints:
    Σ w_i = 1,  0 ≤ w_i ≤ max_weight                 (long-only, position cap)
    Σ_{equity} w_i      ≤ target_equity + tolerance   (risk-profile cap)
    Σ_{fixed income} w_i ≥ target_fi    − tolerance   (risk-profile floor)

The profile takes precedence over the position cap: if a class has too few
holdings to meet its target under the cap, the cap is raised for that class
only. A profile constraint is skipped only when the portfolio holds no asset
of the required class (e.g. no bonds); skipped constraints are reported back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

EQUITY_CLASSES = {
    "Equity", "Equity ETF", "International Equity", "International ETF",
    "Emerging Markets ETF", "Cryptocurrency",
}
FIXED_INCOME_CLASSES = {"Fixed Income", "Bond ETF"}


def estimate_inputs(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualised expected returns and covariance matrix from a price history."""
    returns = prices.pct_change().dropna(how="any")
    mu = returns.mean().values * TRADING_DAYS
    cov = returns.cov().values * TRADING_DAYS
    return mu, cov


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                    risk_free_rate: float) -> dict:
    """Expected return, volatility and Sharpe for a weight vector (same estimator)."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}


def _feasible(w: np.ndarray, upper: np.ndarray, constraints: list[dict],
              tol: float = 1e-6) -> bool:
    if not np.all(np.isfinite(w)):
        return False
    if (w < -tol).any() or (w > upper + tol).any():
        return False
    for c in constraints:
        val = c["fun"](w)
        if c["type"] == "eq" and abs(val) > tol:
            return False
        if c["type"] == "ineq" and val < -tol:
            return False
    return True


def optimize_max_sharpe(
    prices: pd.DataFrame,
    asset_classes: dict[str, str],
    risk_free_rate: float,
    max_weight: float = 0.20,
    target_equity: float | None = None,
    target_fixed_income: float | None = None,
    tolerance: float = 0.05,
) -> dict:
    """
    Long-only max-Sharpe allocation under position and risk-profile constraints.

    Parameters
    ----------
    prices : DataFrame of historical prices, one column per ticker
    asset_classes : {ticker: asset_class}
    risk_free_rate : annual risk-free rate (fraction)
    max_weight : cap per position; raised to 1/n if the cap is infeasible
    target_equity, target_fixed_income : profile targets (fractions) or None
    tolerance : allowed deviation from the profile targets

    Returns
    -------
    dict with keys: success, weights {ticker: w}, stats, max_weight,
    constraints_applied, constraints_skipped, message
    """
    from scipy.optimize import minimize

    tickers = list(prices.columns)
    n = len(tickers)
    if n < 2:
        return {"success": False, "message": "Απαιτούνται τουλάχιστον 2 τίτλοι."}

    mu, cov = estimate_inputs(prices)
    # Ridge for numerical stability on near-singular covariance matrices
    cov = cov + np.eye(n) * 1e-8

    cap = max(max_weight, 1.0 / n + 1e-6)
    upper = np.full(n, cap)

    eq_mask = np.array([asset_classes.get(t) in EQUITY_CLASSES for t in tickers])
    fi_mask = np.array([asset_classes.get(t) in FIXED_INCOME_CLASSES for t in tickers])

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    applied, skipped = ["Σ w = 1", f"0 ≤ w ≤ {cap:.0%}"], []

    # The risk profile takes precedence over the position cap: when a class has
    # too few holdings to meet its target under the cap, the cap is raised for
    # that class only (e.g. two bond ETFs carrying a 45% fixed-income floor).
    def _raise_cap(mask, needed, label):
        k = int(mask.sum())
        if k and upper[mask].sum() < needed:
            # Only ever raise a bound; an earlier constraint may have raised it already
            upper[mask] = np.maximum(upper[mask], min(needed / k + 1e-6, 1.0))
            applied.append(f"όριο θέσης {label} → {upper[mask].max():.0%}")

    if target_equity is not None:
        eq_cap = target_equity + tolerance
        if (~eq_mask).any():
            _raise_cap(~eq_mask, 1.0 - eq_cap, "μη μετοχικών")
            constraints.append({"type": "ineq",
                                "fun": lambda w, m=eq_mask, c=eq_cap: c - w[m].sum()})
            applied.append(f"Μετοχές ≤ {eq_cap:.0%}")
        elif eq_cap < 1.0:
            skipped.append(f"Μετοχές ≤ {eq_cap:.0%} (το χαρτοφυλάκιο έχει μόνο μετοχές)")

    if target_fixed_income is not None:
        fi_floor = max(target_fixed_income - tolerance, 0.0)
        if fi_floor > 0 and fi_mask.any():
            _raise_cap(fi_mask, fi_floor, "ομολόγων")
            constraints.append({"type": "ineq",
                                "fun": lambda w, m=fi_mask, f=fi_floor: w[m].sum() - f})
            applied.append(f"Ομόλογα ≥ {fi_floor:.0%}")
        elif fi_floor > 0:
            skipped.append(f"Ομόλογα ≥ {fi_floor:.0%} (το χαρτοφυλάκιο δεν έχει ομολογιακούς τίτλους)")

    bounds = list(zip(np.zeros(n), upper))

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        return -(ret - risk_free_rate) / vol

    # Several starting points; SLSQP is local, the problem is not always convex
    starts = [np.full(n, 1.0 / n)]
    rng = np.random.default_rng(0)
    starts += [rng.dirichlet(np.ones(n)) for _ in range(4)]

    best = None
    for x0 in starts:
        x0 = np.clip(x0, 0, upper)
        x0 = x0 / x0.sum()
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
        # SLSQP often stops with "Positive directional derivative for linesearch"
        # at an already-converged point; judge the result by actual feasibility.
        if _feasible(res.x, upper, constraints) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return {"success": False,
                "message": "Ο βελτιστοποιητής δεν βρήκε εφικτή λύση με τους περιορισμούς."}

    w = np.clip(best.x, 0.0, None)
    w[w < 1e-4] = 0.0
    w = w / w.sum()

    return {
        "success": True,
        "weights": {t: float(x) for t, x in zip(tickers, w)},
        "stats": portfolio_stats(w, mu, cov, risk_free_rate),
        "max_weight": cap,
        "constraints_applied": applied,
        "constraints_skipped": skipped,
        "message": "OK",
    }


def evaluate_weights(prices: pd.DataFrame, weights: dict[str, float],
                     risk_free_rate: float) -> dict:
    """Stats of an arbitrary allocation using the same estimator as the optimiser."""
    tickers = list(prices.columns)
    mu, cov = estimate_inputs(prices)
    w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
    if w.sum() <= 0:
        return {"expected_return": float("nan"), "volatility": float("nan"),
                "sharpe": float("nan")}
    w = w / w.sum()
    return portfolio_stats(w, mu, cov, risk_free_rate)
