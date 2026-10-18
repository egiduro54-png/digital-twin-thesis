"""
twin_timeline.py
----------------
The time dimension of the Digital Twin.

timeline()
    Replays the twin at every month end of the price history using only the
    data available at that date (no look-ahead): Resilience Index, number of
    Risk Monitor alerts/cautions, volatility and drawdown. It shows how the
    portfolio's risk evolved and when the monitor would have raised an alert.
    Holdings are assumed constant (current quantities), since no transaction
    history is available.

snapshot() / compare_snapshots()
    A serialisable state of the twin (investor profile, questionnaire, goal,
    holdings and key metrics) that can be saved and loaded later, so the next
    session can show what changed since the last one.
"""

from __future__ import annotations

import copy
import datetime as dt

import numpy as np
import pandas as pd

from .portfolio import Portfolio
from .resilience import compute_resilience
from .risk_monitor import RiskMonitor

SNAPSHOT_VERSION = 1
MIN_HISTORY_DAYS = 252  # one year before the first timeline point


def _portfolio_at(portfolio: Portfolio, date: pd.Timestamp, lookback_days: int) -> Portfolio | None:
    hist = portfolio.historical_prices.loc[:date]
    if len(hist) < MIN_HISTORY_DAYS:
        return None
    hist = hist.iloc[-lookback_days:]
    last = hist.iloc[-1]
    assets = []
    for a in portfolio.assets:
        if a.ticker not in hist.columns or not np.isfinite(last[a.ticker]):
            return None
        b = copy.copy(a)
        b.current_price = float(last[a.ticker])
        assets.append(b)
    return Portfolio(assets, hist, risk_profile=portfolio.risk_profile, name=portfolio.name)


def timeline(portfolio: Portfolio, lookback_days: int = 3 * 252,
             freq: str = "ME") -> pd.DataFrame:
    """
    One row per month end: value, Resilience Index, alert/caution counts,
    1y volatility, drawdown from the running peak, and the titles of the
    alerts raised at that date.
    """
    hist = portfolio.historical_prices
    if hist.empty:
        return pd.DataFrame()
    dates = hist.resample(freq).last().index
    dates = [d for d in dates if d >= hist.index[0] + pd.Timedelta(days=365)]
    # Value of the current holdings through time (for the drawdown column)
    held = [a.ticker for a in portfolio.assets if a.ticker in hist.columns]
    qty = np.array([a.quantity for a in portfolio.assets if a.ticker in hist.columns])
    value = (hist[held].ffill() * qty).sum(axis=1)
    peak = value.cummax()

    recs = []
    for d in dates:
        p = _portfolio_at(portfolio, d, lookback_days)
        if p is None:
            continue
        try:
            ri = compute_resilience(p)
            analysis = RiskMonitor(p).run_full_analysis()
        except Exception:
            continue
        v = value.loc[:d]
        recs.append({
            "date": v.index[-1],
            "value": float(v.iloc[-1]),
            "drawdown_pct": float((v.iloc[-1] / peak.loc[:d].iloc[-1] - 1) * 100),
            "resilience": ri["score"],
            "volatility_pct": float(p.calculate_volatility() * 100),
            "n_alert": len(analysis["alert"]),
            "n_caution": len(analysis["caution"]),
            "alerts": "; ".join(a["title"] for a in analysis["alert"]),
        })
    return pd.DataFrame(recs)


def first_alerts(tl: pd.DataFrame) -> pd.DataFrame:
    """The first date each alert title appears in the timeline."""
    seen, out = set(), []
    for _, r in tl.iterrows():
        for title in filter(None, (t.strip() for t in str(r["alerts"]).split(";"))):
            if title not in seen:
                seen.add(title)
                out.append({"date": r["date"], "alert": title,
                            "drawdown_pct": r["drawdown_pct"]})
    return pd.DataFrame(out)


# ── Save / load ─────────────────────────────────────────────────────────────

def snapshot(portfolio: Portfolio, questionnaire: dict | None = None,
             goal: dict | None = None) -> dict:
    """JSON-serialisable state of the twin."""
    m = portfolio.get_metrics()
    ri = compute_resilience(portfolio)
    analysis = RiskMonitor(portfolio).run_full_analysis()
    return {
        "version": SNAPSHOT_VERSION,
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "portfolio_name": portfolio.name,
        "risk_profile": portfolio.risk_profile,
        "questionnaire": questionnaire,
        "goal": goal,
        "holdings": [{"ticker": a.ticker, "quantity": a.quantity,
                      "entry_price": a.entry_price} for a in portfolio.assets],
        "metrics": {
            "total_value": round(float(portfolio.total_value), 2),
            "volatility_pct": m.get("volatility_annual_pct"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "max_drawdown_pct": m.get("max_drawdown_pct"),
            "resilience": round(ri["score"], 1),
            "n_alert": len(analysis["alert"]),
            "n_caution": len(analysis["caution"]),
            "weights_pct": {t: round(w * 100, 2) for t, w in portfolio.get_weights_dict().items()},
        },
    }


COMPARE_LABELS = {
    "total_value": "Συνολική αξία ($)",
    "volatility_pct": "Μεταβλητότητα (%)",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown_pct": "Max drawdown (%)",
    "resilience": "Resilience Index",
    "n_alert": "Alerts",
    "n_caution": "Cautions",
}


def compare_snapshots(old: dict, new: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metric changes, holdings changes) between two snapshots."""
    om, nm = old.get("metrics", {}), new.get("metrics", {})
    metrics = pd.DataFrame([{
        "Μετρική": label,
        "Τότε": om.get(k),
        "Τώρα": nm.get(k),
        "Μεταβολή": (nm.get(k) - om.get(k))
        if isinstance(om.get(k), (int, float)) and isinstance(nm.get(k), (int, float)) else None,
    } for k, label in COMPARE_LABELS.items()])

    ow, nw = om.get("weights_pct", {}), nm.get("weights_pct", {})
    holdings = pd.DataFrame([{
        "Ticker": t, "Βάρος τότε (%)": ow.get(t, 0.0), "Βάρος τώρα (%)": nw.get(t, 0.0),
        "Μεταβολή (μ.)": round(nw.get(t, 0.0) - ow.get(t, 0.0), 2),
    } for t in sorted(set(ow) | set(nw))])
    if not holdings.empty:
        holdings = holdings[holdings["Μεταβολή (μ.)"].abs() >= 0.5]
    return metrics, holdings
