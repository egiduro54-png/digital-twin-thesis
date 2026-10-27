"""
Numbers for the four usage examples of §4.6, with market data frozen at a
given date so the text can be reproduced:

    python -m scripts.run_examples --as-of 2026-10-23

For each example: alerts, Resilience Index, key scenarios, the coordinated
plan's trades, and the same figures for the portfolio after the plan.
Writes results/examples/examples.json and prints a readable summary.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src import data_loader, utils
from src.investor import simulate_goal
from src.portfolio import Asset, Portfolio
from src.rebalance_plan import RebalancePlan
from src.recommendations import RecommendationEngine
from src.resilience import compute_resilience
from src.risk_monitor import RiskMonitor
from src.scenario_engine import ScenarioEngine

EXAMPLES = {
    "1_concentrated": ("data/examples/concentrated.csv", "flexible"),
    "2_defensive_drift": ("data/examples/defensive_drift.csv", "defensive"),
    "3_4_balanced": ("data/portfolio_moderate.csv", "flexible"),
}
SCENARIOS = ["market_down_30", "tech_crash", "financial_crisis_2008",
             "rates_up_200bps", "inflation_spike"]
CUSTOM = {"market_change": -0.15, "rate_change": 0.0075, "vol_multiplier": 1.5}


def freeze(as_of: str, run_date: str):
    """Make fetch_historical_data return history ending at `as_of`."""
    original = data_loader.fetch_historical_data
    start = (datetime.fromisoformat(run_date) - timedelta(days=5 * 365 + 30)).strftime("%Y-%m-%d")
    end = (datetime.fromisoformat(as_of) + timedelta(days=1)).strftime("%Y-%m-%d")

    def frozen(tickers, years=5, start_date=None, end_date=None):
        return original(tickers, start_date=start, end_date=end)
    utils.fetch_historical_data = frozen


def describe(p: Portfolio) -> dict:
    m = p.get_metrics()
    a = RiskMonitor(p).run_full_analysis()
    ri = compute_resilience(p)
    se = ScenarioEngine(p)
    scen = {}
    for s in SCENARIOS:
        try:
            scen[s] = round(se.compare_portfolio_metrics(s)["summary"]["portfolio_change_pct"], 1)
        except Exception as exc:
            scen[s] = str(exc)
    se.create_custom_scenario("custom", market_change=CUSTOM["market_change"],
                              rate_change=CUSTOM["rate_change"],
                              volatility_multiplier=CUSTOM["vol_multiplier"])
    custom = round(se.compare_portfolio_metrics("custom")["summary"]["portfolio_change_pct"], 1)
    worst = se.find_worst_scenario(list(se.scenarios)[:14])
    w = p.get_weights_dict()
    return {
        "value": round(p.total_value, 0),
        "volatility_pct": m.get("volatility_annual_pct"),
        "max_drawdown_pct": m.get("max_drawdown_pct"),
        "sharpe": m.get("sharpe_ratio"),
        "beta": m.get("beta"),
        "hhi": round(sum(x * x for x in w.values()), 3),
        "weights_pct": {t: round(x * 100, 1) for t, x in sorted(w.items(), key=lambda kv: -kv[1])},
        "resilience": round(ri["score"], 1),
        "ri_components": {k: round(v * 100) for k, v in ri["components"].items()},
        "alerts": [x["title"] for x in a["alert"]],
        "cautions": [x["title"] for x in a["caution"]],
        "scenarios_pct": scen,
        "custom_scenario": custom,
        "worst_scenario": worst,
    }


def after_plan(p: Portfolio, trades: list[dict]) -> Portfolio:
    assets = {a.ticker: Asset(**{**a.__dict__}) for a in p.assets}
    for t in trades:
        if t["action"] == "SELL":
            assets[t["ticker"]].quantity -= t["quantity"]
        elif t["ticker"] in assets:
            assets[t["ticker"]].quantity += t["quantity"]
        else:
            tk = t["ticker"]
            assets[tk] = Asset(ticker=tk, quantity=float(t["quantity"]), entry_price=t["price"],
                               current_price=t["price"], name=tk,
                               sector=data_loader.SECTOR_FALLBACK.get(tk, "Unknown"), industry="Unknown",
                               asset_class=data_loader.ASSET_CLASS_FALLBACK.get(tk, "Equity"), country="US")
    return Portfolio([a for a in assets.values() if a.quantity > 0], p.historical_prices,
                     risk_profile=p.risk_profile, name=p.name + " (after plan)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-10-23")
    ap.add_argument("--run-date", default="2026-10-24", help="date the original numbers were produced")
    ap.add_argument("--out", default="results/examples")
    args = ap.parse_args()
    freeze(args.as_of, args.run_date)

    out = {}
    for key, (csv, profile) in EXAMPLES.items():
        p = utils.build_portfolio(csv, risk_profile=profile, portfolio_name=key)
        eng = RecommendationEngine(p, RiskMonitor(p))
        recs = eng.generate_recommendations()
        trades = eng.plan_trades()
        q = after_plan(p, trades)
        res = {"before": describe(p), "after": describe(q),
               "plan_stats": eng.plan.stats(),
               "recommendations": [(r["priority"], r["title"]) for r in recs],
               "trades": trades,
               "sells": round(sum(t["value"] for t in trades if t["action"] == "SELL")),
               "buys": round(sum(t["value"] for t in trades if t["action"] == "BUY"))}
        if key == "3_4_balanced":
            groups = {t: RebalancePlan(p).group(t) for t in p.historical_prices.columns}
            target = round(p.total_value * 2, -3)
            res["monte_carlo"] = {
                lbl: {k: round(v, 3) if isinstance(v, float) else v
                      for k, v in simulate_goal(p.historical_prices, w, groups, p.total_value,
                                                500, 10, target).items() if k != "fan"}
                for lbl, w in [("current", p.get_weights_dict()), ("after_dt", dict(eng.plan.w))]}
            res["monte_carlo"]["target"] = target
        out[key] = res

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "examples.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
