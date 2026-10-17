"""
Unit tests for optimizer.py (SciPy Markowitz optimisation)
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import optimize_max_sharpe, evaluate_weights, estimate_inputs
from src.portfolio import Portfolio, Asset, RISK_FREE_RATE
from src.risk_monitor import RiskMonitor
from src.recommendations import RecommendationEngine


def _price_frame(specs, n=750, seed=0):
    """specs: {ticker: (annual_drift, annual_vol)} -> independent GBM price paths."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.offsets.BDay().rollback(pd.Timestamp.today().normalize()), periods=n, freq="B")
    data = {}
    for t, (mu, sigma) in specs.items():
        lr = rng.normal(mu / 252, sigma / np.sqrt(252), n)
        data[t] = 100 * np.exp(np.cumsum(lr))
    return pd.DataFrame(data, index=idx)


SPECS = {
    "EQ1": (0.12, 0.25), "EQ2": (0.10, 0.20), "EQ3": (0.08, 0.18),
    "EQ4": (0.09, 0.22), "BND": (0.03, 0.05), "AGG": (0.03, 0.06),
}
CLASSES = {"EQ1": "Equity", "EQ2": "Equity", "EQ3": "Equity", "EQ4": "Equity",
           "BND": "Bond ETF", "AGG": "Fixed Income"}


class TestOptimizeMaxSharpe:
    def test_weights_sum_to_one_and_respect_cap(self):
        prices = _price_frame(SPECS)
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE, max_weight=0.30)
        assert res["success"]
        w = np.array(list(res["weights"].values()))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all()
        assert w.max() <= 0.30 + 1e-6

    def test_profile_constraints_enforced(self):
        prices = _price_frame(SPECS)
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE, max_weight=0.40,
                                  target_equity=0.45, target_fixed_income=0.45)
        assert res["success"]
        eq = sum(w for t, w in res["weights"].items() if CLASSES[t] == "Equity")
        fi = 1.0 - eq
        assert eq <= 0.50 + 1e-6
        assert fi >= 0.40 - 1e-6
        assert not res["constraints_skipped"]

    def test_beats_equal_weight_sharpe(self):
        prices = _price_frame(SPECS)
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE, max_weight=0.40)
        equal = evaluate_weights(prices, {t: 1 / len(SPECS) for t in SPECS}, RISK_FREE_RATE)
        assert res["stats"]["sharpe"] >= equal["sharpe"] - 1e-6

    def test_infeasible_bond_floor_is_skipped(self):
        specs = {k: v for k, v in SPECS.items() if CLASSES[k] == "Equity"}
        prices = _price_frame(specs)
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE, max_weight=0.40,
                                  target_fixed_income=0.45)
        assert res["success"]
        assert any("Ομόλογα" in s for s in res["constraints_skipped"])

    def test_cap_raised_when_too_few_assets(self):
        prices = _price_frame({"EQ1": SPECS["EQ1"], "BND": SPECS["BND"]})
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE, max_weight=0.20)
        assert res["success"]
        assert res["max_weight"] >= 0.5

    def test_single_bond_meets_both_profile_constraints(self):
        # One bond must absorb the equity cap remainder (30%) AND the FI floor (25%);
        # the second check must not lower the bound the first one raised.
        specs = {k: v for k, v in SPECS.items() if k != "AGG"}
        prices = _price_frame(specs)
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE, max_weight=0.20,
                                  target_equity=0.65, target_fixed_income=0.30)
        assert res["success"]
        assert res["weights"]["BND"] >= 0.30 - 1e-6

    def test_single_asset_fails_gracefully(self):
        prices = _price_frame({"EQ1": SPECS["EQ1"]})
        res = optimize_max_sharpe(prices, CLASSES, RISK_FREE_RATE)
        assert not res["success"]


class TestEvaluateWeights:
    def test_matches_manual_computation(self):
        prices = _price_frame(SPECS)
        mu, cov = estimate_inputs(prices)
        w = np.full(len(SPECS), 1 / len(SPECS))
        stats = evaluate_weights(prices, dict(zip(prices.columns, w)), RISK_FREE_RATE)
        assert stats["expected_return"] == pytest.approx(w @ mu)
        assert stats["volatility"] == pytest.approx(np.sqrt(w @ cov @ w))


class TestOptimizationRecommendation:
    def _portfolio(self, profile="flexible"):
        prices = _price_frame(SPECS)
        qty = {"EQ1": 400, "EQ2": 300, "EQ3": 100, "EQ4": 100, "BND": 50, "AGG": 50}
        assets = [
            Asset(ticker=t, quantity=q, entry_price=90.0,
                  current_price=float(prices[t].iloc[-1]),
                  sector="Technology" if CLASSES[t] == "Equity" else "Fixed Income",
                  asset_class=CLASSES[t])
            for t, q in qty.items()
        ]
        return Portfolio(assets, prices, risk_profile=profile)

    def test_returns_trades_and_not_in_ranked_list(self):
        p = self._portfolio()
        engine = RecommendationEngine(p, RiskMonitor(p))
        opt = engine.optimization_recommendation()
        assert opt is not None
        assert opt["trades"]
        eq = sum(w for t, w in opt["optimal_weights"].items() if CLASSES[t] == "Equity")
        assert eq <= 50.0 + 1e-3
        ranked = engine.generate_recommendations()
        assert all(r.get("category") != opt["category"] for r in ranked)
