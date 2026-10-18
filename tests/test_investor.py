"""
Tests for investor.py (suitability questionnaire and goal Monte Carlo)
"""

import numpy as np
import pandas as pd
import pytest

from src.investor import QUESTIONS, score_questionnaire, simulate_goal, CMA_RETURNS


def _answers(value):
    return {q["id"]: value for q in QUESTIONS}


class TestQuestionnaire:
    def test_extremes(self):
        assert score_questionnaire(_answers(0))["profile"] == "liquidity_plus"
        assert score_questionnaire(_answers(4))["profile"] == "dynamic"

    def test_middle_is_flexible(self):
        assert score_questionnaire(_answers(2))["profile"] == "flexible"

    def test_short_horizon_caps_profile(self):
        a = _answers(4)
        a["horizon"] = 0  # needs the money within a year
        r = score_questionnaire(a)
        assert r["uncapped_profile"] == "dynamic"
        assert r["profile"] == "liquidity_plus"
        assert r["capped_by"] == "horizon"

    def test_attitude_alone_does_not_cap(self):
        a = _answers(4)
        a["drawdown_reaction"] = 0  # not a capacity question
        assert score_questionnaire(a)["capped_by"] is None


def _prices(n_months=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_months * 21, freq="B")
    eq = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx))))
    bd = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, len(idx))))
    return pd.DataFrame({"EQ": eq, "BD": bd}, index=idx)


GROUPS = {"EQ": "equity", "BD": "fixed_income"}


class TestSimulateGoal:
    def test_probability_increases_with_contributions(self):
        px = _prices()
        kw = dict(prices=px, weights={"EQ": 0.6, "BD": 0.4}, groups=GROUPS,
                  initial=10_000, years=10, target=30_000, n_paths=2000)
        low = simulate_goal(monthly_contribution=0, **kw)["probability"]
        high = simulate_goal(monthly_contribution=200, **kw)["probability"]
        assert 0 <= low < high <= 1

    def test_no_risk_no_contribution_grows_at_assumed_rate(self):
        px = _prices()
        r = simulate_goal(px, {"BD": 1.0}, GROUPS, initial=10_000, monthly_contribution=0,
                          years=10, target=1, n_paths=4000)
        assert r["median"] == pytest.approx(10_000 * (1 + CMA_RETURNS["fixed_income"]) ** 10, rel=0.05)

    def test_percentiles_ordered(self):
        r = simulate_goal(_prices(), {"EQ": 0.6, "BD": 0.4}, GROUPS, 10_000, 100, 5, 20_000,
                          n_paths=1000)
        assert r["p5"] < r["median"] < r["p95"]
        assert list(r["fan"]["year"]) == list(range(6))
