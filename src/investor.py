"""
investor.py
-----------
The investor side of the Digital Twin: a suitability questionnaire that maps
answers to one of the five risk profiles, and a Monte Carlo projection of the
probability of reaching a financial goal with a given allocation.

Questionnaire
    Seven MiFID II-style questions (capacity for loss, horizon, liquidity,
    knowledge, objective, attitude to a fall). Each answer scores 0-4; the
    total (0-28) is mapped to a profile with fixed bands. The weakest
    capacity answer caps the result: an investor who needs the money within
    a year cannot be Dynamic however high the attitude score.

Monte Carlo
    Monthly log returns are drawn from a multivariate normal with the
    covariance of the holdings' historical monthly returns and expected
    returns from long-run capital market assumptions per asset class, not
    from the recent history (a 3-year sample mean of ~20% a year would give
    absurdly optimistic goal probabilities). Contributions are added monthly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_ORDER = ["liquidity_plus", "defensive", "flexible", "growth", "dynamic"]

QUESTIONS = [
    {
        "id": "horizon",
        "text": "Σε πόσα χρόνια θα χρειαστείτε το μεγαλύτερο μέρος αυτών των χρημάτων;",
        "options": ["Σε λιγότερο από 1 έτος", "1–3 έτη", "3–5 έτη", "5–10 έτη", "Πάνω από 10 έτη"],
        "capacity": True,
    },
    {
        "id": "age",
        "text": "Σε ποια ηλικιακή ομάδα ανήκετε;",
        "options": ["Άνω των 70", "60–70", "45–59", "30–44", "Κάτω από 30"],
        "capacity": False,
    },
    {
        "id": "income",
        "text": "Πόσο σταθερό είναι το εισόδημά σας τα επόμενα χρόνια;",
        "options": ["Καθόλου σταθερό", "Λίγο σταθερό", "Μέτρια σταθερό",
                    "Σταθερό", "Πολύ σταθερό, με αποταμιεύσεις ασφαλείας"],
        "capacity": True,
    },
    {
        "id": "liquidity",
        "text": "Τι ποσοστό του χαρτοφυλακίου ίσως χρειαστεί να ρευστοποιήσετε μέσα στον επόμενο χρόνο;",
        "options": ["Πάνω από 50%", "25–50%", "10–25%", "Κάτω από 10%", "Καθόλου"],
        "capacity": True,
    },
    {
        "id": "drawdown_reaction",
        "text": "Αν το χαρτοφυλάκιό σας έπεφτε 20% σε λίγους μήνες, τι θα κάνατε;",
        "options": ["Θα πουλούσα τα πάντα", "Θα πουλούσα ένα μέρος", "Θα περίμενα ανήσυχος/η",
                    "Θα το κρατούσα ήρεμα", "Θα αγόραζα περισσότερο"],
        "capacity": False,
    },
    {
        "id": "experience",
        "text": "Ποια είναι η εμπειρία σας με επενδύσεις σε μετοχές ή ETF;",
        "options": ["Καμία", "Λίγη (καταθέσεις, ομόλογα)", "Κάποια (αμοιβαία κεφάλαια)",
                    "Αρκετή (μετοχές, ETF)", "Μεγάλη (και παράγωγα)"],
        "capacity": False,
    },
    {
        "id": "objective",
        "text": "Ποιος είναι ο κύριος στόχος σας;",
        "options": ["Διατήρηση κεφαλαίου", "Σταθερό εισόδημα", "Ισορροπία εισοδήματος και ανάπτυξης",
                    "Ανάπτυξη κεφαλαίου", "Μέγιστη ανάπτυξη, ανεκτώ μεγάλες διακυμάνσεις"],
        "capacity": False,
    },
]

# Total score (0-28) → profile
PROFILE_BANDS = [(6, "liquidity_plus"), (12, "defensive"), (18, "flexible"),
                 (23, "growth"), (28, "dynamic")]
# Lowest capacity answer (0-4) → most aggressive profile allowed
CAPACITY_CAP = {0: "liquidity_plus", 1: "defensive", 2: "flexible", 3: "growth", 4: "dynamic"}

# Long-run expected annual returns by asset class (capital market assumptions,
# in line with long-term historical averages; not sample means)
CMA_RETURNS = {"equity": 0.07, "fixed_income": 0.04, "other": 0.04}


def score_questionnaire(answers: dict[str, int]) -> dict:
    """
    answers: {question_id: option index 0-4}
    Returns {"score", "profile", "uncapped_profile", "capped_by", "max_score"}.
    """
    total = int(sum(answers.get(q["id"], 0) for q in QUESTIONS))
    uncapped = next(p for limit, p in PROFILE_BANDS if total <= limit)

    capacity = {q["id"]: answers.get(q["id"], 0) for q in QUESTIONS if q["capacity"]}
    weakest_id = min(capacity, key=capacity.get)
    cap = CAPACITY_CAP[capacity[weakest_id]]
    profile = min(uncapped, cap, key=PROFILE_ORDER.index)

    return {
        "score": total,
        "max_score": 4 * len(QUESTIONS),
        "profile": profile,
        "uncapped_profile": uncapped,
        "capped_by": weakest_id if profile != uncapped else None,
    }


def _monthly_inputs(prices: pd.DataFrame, weights: dict[str, float],
                    groups: dict[str, str]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    tickers = [t for t, w in weights.items() if w > 1e-9 and t in prices.columns]
    monthly = prices[tickers].resample("ME").last().dropna()
    lr = np.log(monthly / monthly.shift(1)).dropna()
    cov = lr.cov().values
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    # Monthly log mean so that E[simple annual return] matches the assumption
    mu = np.array([np.log1p(CMA_RETURNS[groups.get(t, "equity")]) / 12 for t in tickers])
    mu = mu - 0.5 * np.diag(cov)
    return mu, cov, tickers, w


def simulate_goal(
    prices: pd.DataFrame,
    weights: dict[str, float],
    groups: dict[str, str],
    initial: float,
    monthly_contribution: float,
    years: int,
    target: float,
    n_paths: int = 5000,
    seed: int = 0,
) -> dict:
    """
    Monte Carlo of portfolio value with monthly contributions and monthly
    rebalancing to `weights`.

    groups: {ticker: "equity" | "fixed_income" | "other"} for the return assumptions.
    Returns probability of reaching `target`, percentiles of the final value
    and a percentile fan per year for plotting.
    """
    mu, cov, tickers, w = _monthly_inputs(prices, weights, groups)
    if len(tickers) == 0 or len(mu) == 0:
        raise ValueError("Δεν υπάρχει επαρκές ιστορικό για προσομοίωση.")
    months = int(years * 12)
    rng = np.random.default_rng(seed)
    # Portfolio monthly log return per path (rebalanced each month)
    draws = rng.multivariate_normal(mu, cov, size=(n_paths, months))
    port = np.log((np.exp(draws) * w).sum(axis=2))

    values = np.empty((n_paths, months + 1))
    values[:, 0] = initial
    for m in range(months):
        values[:, m + 1] = values[:, m] * np.exp(port[:, m]) + monthly_contribution
    final = values[:, -1]
    invested = initial + monthly_contribution * months

    yearly = values[:, ::12]
    fan = pd.DataFrame({
        "year": np.arange(yearly.shape[1]),
        "p5": np.percentile(yearly, 5, axis=0),
        "p50": np.percentile(yearly, 50, axis=0),
        "p95": np.percentile(yearly, 95, axis=0),
    })
    return {
        "probability": float((final >= target).mean()),
        "median": float(np.median(final)),
        "p5": float(np.percentile(final, 5)),
        "p95": float(np.percentile(final, 95)),
        "invested": float(invested),
        # From the simulated monthly portfolio returns (rebalanced mix)
        "expected_annual_return": float((1 + np.expm1(port).mean()) ** 12 - 1),
        "annual_volatility": float(np.expm1(port).std() * np.sqrt(12)),
        "fan": fan,
    }
