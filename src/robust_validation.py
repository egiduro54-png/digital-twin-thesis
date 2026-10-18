"""
robust_validation.py

Extended (robustness) evaluation of the Digital Twin risk scorer.

The original experiment (validation.py) uses 30 hand-designed portfolios and a
single 60/40 split. This module tests whether its conclusions survive:

  1. Random portfolios  — N portfolios per window with random tickers (2–10)
     and Dirichlet weights of varying concentration, so the sample is not
     shaped by the author's archetype design.
  2. Rolling walk-forward windows — many (analysis → evaluation) windows over
     ~15 years, covering several market regimes (2015–16, 2018, 2020, 2022…).
  3. Ablation — alerts only, scenarios only, drift only, alerts+scenarios.
  4. Simple baselines — equity share, HHI, equity share + HHI, historical
     volatility, historical max drawdown, historical CVaR 95%.
  5. Paired cluster bootstrap — confidence intervals for the *difference*
     between each scorer and the static Baseline (resampling portfolios
     within each window), so "better than Baseline" is actually tested.

Primary metric: Spearman ρ between risk score and realised evaluation-window
max drawdown (averaged across windows). AUROC is reported with two fragility
definitions: fixed (drawdown < −15%, only windows where both classes exist)
and relative (worst third of drawdowns in each window — always defined).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .validation import (
    TICKER_META,
    BaselineRiskScorer,
    ProposedRiskScorer,
    build_portfolio_from_prices,
)

logger = logging.getLogger(__name__)

EQUITY_CLASSES = {
    "Equity", "Equity ETF", "International Equity", "International ETF",
    "Emerging Markets ETF", "Cryptocurrency",
}
RISK_PROFILES = ["conservative", "moderate", "aggressive"]

# Holdout universe: tickers never used while designing or tuning the system
# (the main experiment uses validation.TICKER_META). Used to check that
# results, in particular the Resilience Index weights, are not fitted to the
# main universe.
def _m(sector, asset_class="Equity"):
    return {"sector": sector, "asset_class": asset_class, "country": "US"}


HOLDOUT_META: dict[str, dict] = {
    "CSCO": _m("Technology"), "ORCL": _m("Technology"), "INTC": _m("Technology"),
    "IBM": _m("Technology"), "QCOM": _m("Technology"),
    "KO": _m("Consumer Staples"), "PEP": _m("Consumer Staples"),
    "PG": _m("Consumer Staples"), "COST": _m("Consumer Staples"),
    "MRK": _m("Healthcare"), "ABT": _m("Healthcare"), "AMGN": _m("Healthcare"),
    "C": _m("Financials"), "WFC": _m("Financials"), "MS": _m("Financials"),
    "COP": _m("Energy"), "SLB": _m("Energy"),
    "CAT": _m("Industrials"), "HON": _m("Industrials"), "MMM": _m("Industrials"),
    "DIS": _m("Communication Services"), "VZ": _m("Communication Services"),
    "MCD": _m("Consumer Discretionary"), "NKE": _m("Consumer Discretionary"),
    "LOW": _m("Consumer Discretionary"),
    "NEE": _m("Utilities"), "DUK": _m("Utilities"),
    "IWM": _m("Equity ETF"), "QQQ": _m("Equity ETF"),
    "EFA": _m("International ETF", "International Equity"),
    "IEF": _m("Bond ETF", "Fixed Income"), "SHY": _m("Bond ETF", "Fixed Income"),
    "LQD": _m("Bond ETF", "Fixed Income"),
    "VNQ": _m("Real Estate ETF", "Real Estate"),
    "SLV": _m("Commodities ETF", "Commodity"),
}
UNIVERSES = {"main": TICKER_META, "holdout": HOLDOUT_META}

# Scorers that are compared against the static Baseline
SCORER_NAMES = [
    "baseline",
    "proposed",
    "alerts_only",
    "scenarios_only",
    "drift_only",
    "alerts_scenarios",
    "equity_share",
    "hhi",
    "equity_share_hhi",
    "hist_vol",
    "hist_maxdd",
    "hist_cvar95",
    "resilience",
    "ri_stress",
    "ri_sequential",
    "ri_downside",
    "ri_recovery",
    "ri_diversification",
    "ri_concentration",
]


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio generation & windows
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_portfolios(
    n: int,
    universe: list[str],
    rng: np.random.Generator,
    min_assets: int = 2,
    max_assets: int = 10,
) -> list[dict]:
    """
    Random portfolios in the same dict format as PORTFOLIO_ARCHETYPES.

    Dirichlet concentration α ∈ {0.3, 1, 3} gives a spread from highly
    concentrated (α=0.3) to near-equal-weight (α=3) portfolios.
    """
    max_assets = min(max_assets, len(universe))
    portfolios = []
    for i in range(n):
        k = int(rng.integers(min_assets, max_assets + 1))
        tickers = list(rng.choice(universe, size=k, replace=False))
        alpha = float(rng.choice([0.3, 1.0, 3.0]))
        weights = rng.dirichlet(np.full(k, alpha))
        portfolios.append({
            "id": f"rand_{i}",
            "label": f"Random {i}",
            "archetype": "Τυχαίο",
            "tickers": tickers,
            "weights": [float(w) for w in weights],
            "risk_profile": str(rng.choice(RISK_PROFILES)),
        })
    return portfolios


def rolling_windows(
    n_days: int,
    analysis_days: int = 756,
    eval_days: int = 252,
    step_days: int = 252,
) -> list[tuple[int, int, int]]:
    """Return (analysis_start, eval_start, eval_end) iloc positions."""
    windows = []
    start = 0
    while start + analysis_days + eval_days <= n_days:
        windows.append((start, start + analysis_days,
                        start + analysis_days + eval_days))
        start += step_days
    return windows


def first_real_dates(prices: pd.DataFrame) -> pd.Series:
    """
    First date each ticker actually trades.

    fetch_historical_data back-fills missing history with the first price,
    which shows up as a flat series; treat the first price change as the
    start of real data.
    """
    changed = prices.diff().ne(0) & prices.diff().notna()
    first_change = changed.to_numpy().argmax(axis=0)
    # The day before the first price change is the first real trading day
    return pd.Series(prices.index[(first_change - 1).clip(min=0)],
                     index=prices.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Simple (non-Digital-Twin) baselines
# ─────────────────────────────────────────────────────────────────────────────

def simple_scores(arch: dict, analysis_prices: pd.DataFrame,
                  meta: dict[str, dict] | None = None) -> dict:
    """Equity share, HHI and plain historical risk measures."""
    tickers = [t for t in arch["tickers"] if t in analysis_prices.columns]
    w = np.array([arch["weights"][arch["tickers"].index(t)] for t in tickers])
    w = w / w.sum()

    equity_share = float(sum(
        wi for t, wi in zip(tickers, w)
        if (meta or TICKER_META).get(t, {}).get("asset_class", "Equity") in EQUITY_CLASSES
    ))
    hhi = float((w ** 2).sum())

    # Buy-and-hold portfolio value over the analysis window
    px = analysis_prices[tickers]
    values = (px / px.iloc[0] * w).sum(axis=1)
    rets = values.pct_change().dropna()

    hist_vol = float(rets.std() * np.sqrt(252))
    hist_maxdd = float(-(values / values.cummax() - 1).min())
    tail = rets[rets <= rets.quantile(0.05)]
    hist_cvar95 = float(-tail.mean()) if len(tail) else 0.0

    return {
        "equity_share": equity_share,
        "hhi": hhi,
        "equity_share_hhi": equity_share + hhi,
        "hist_vol": hist_vol,
        "hist_maxdd": hist_maxdd,
        "hist_cvar95": hist_cvar95,
    }


def resilience_scores(portfolio) -> dict:
    """
    Resilience Index as a risk score (100 − RI, so higher = riskier, like the
    other scorers), plus each component on its own for the ablation.
    """
    from .resilience import compute_resilience
    try:
        ri = compute_resilience(portfolio)
    except Exception as exc:  # keep the window running; record as missing
        logger.warning("Resilience failed: %s", exc)
        return {"resilience": np.nan, **{f"ri_{k}": np.nan for k in (
            "stress", "sequential", "downside", "recovery",
            "diversification", "concentration")}}
    out = {"resilience": 100.0 - ri["score"]}
    for k, v in ri["components"].items():
        out[f"ri_{k}"] = 1.0 - v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RQ3: do the rebalancing recommendations improve the next year?
# ─────────────────────────────────────────────────────────────────────────────

REBALANCE_VARIANTS = ["hold", "profile_only", "dt", "markowitz", "hold_bond_mix"]
OUTCOME_METRICS = ["maxdd", "vol", "ret", "sharpe", "cvar"]


def eval_underwater(weights: dict[str, float], eval_prices: pd.DataFrame,
                    threshold: float = 0.05) -> float:
    """Share (%) of evaluation-window days more than `threshold` below the running peak."""
    w = {t: x for t, x in weights.items() if x > 1e-9 and t in eval_prices.columns}
    total = sum(w.values())
    if total <= 0:
        return np.nan
    px = eval_prices[list(w)].ffill().bfill()
    values = (px / px.iloc[0] * np.array([w[t] / total for t in w])).sum(axis=1)
    return float(((values / values.cummax() - 1) < -threshold).mean() * 100)


def eval_stats(weights: dict[str, float], eval_prices: pd.DataFrame) -> dict:
    """
    Buy-and-hold outcome over the evaluation window of an allocation set at
    its start: max drawdown, annualised volatility and return (%), Sharpe,
    and daily CVaR 95% (%, positive = loss).
    """
    from .portfolio import RISK_FREE_RATE

    w = {t: x for t, x in weights.items() if x > 1e-9 and t in eval_prices.columns}
    total = sum(w.values())
    nan = {k: np.nan for k in OUTCOME_METRICS}
    if total <= 0:
        return nan
    px = eval_prices[list(w)].ffill().bfill()
    if len(px) < 5:
        return nan
    values = (px / px.iloc[0] * np.array([w[t] / total for t in w])).sum(axis=1)
    rets = values.pct_change().dropna()
    years = len(rets) / 252
    ret = float(values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(252))
    tail = rets[rets <= rets.quantile(0.05)]
    return {
        "maxdd": float((values / values.cummax() - 1).min() * 100),
        "vol": vol * 100,
        "ret": ret * 100,
        "sharpe": (ret - RISK_FREE_RATE) / vol if vol > 0 else np.nan,
        "cvar": float(-tail.mean() * 100) if len(tail) else np.nan,
    }


def equal_risk_bond_mix(portfolio, hold: dict[str, float],
                        target: dict[str, float]) -> dict[str, float]:
    """
    The current portfolio blended with a bond fund so that its expected
    volatility (estimated on the analysis window) equals that of `target`.
    The simplest way to take the same risk as the DT plan: if the DT does not
    beat it, its value is only "fewer equities", not its specific choices.
    """
    from .optimizer import evaluate_weights
    from .portfolio import RISK_FREE_RATE

    hist = portfolio.historical_prices
    bond = next((t for t in ("BND", "AGG", "IEF") if t in hist.columns), None)
    if bond is None:
        return hold

    def vol(w):
        cols = [t for t, x in w.items() if x > 1e-9 and t in hist.columns]
        return evaluate_weights(hist[cols].dropna(), {t: w[t] for t in cols},
                                RISK_FREE_RATE)["volatility"]

    def mix(a):
        m = {t: x * (1 - a) for t, x in hold.items()}
        m[bond] = m.get(bond, 0.0) + a
        return m

    goal = vol(target)
    if not np.isfinite(goal) or vol(hold) <= goal:
        return hold
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if vol(mix(mid)) > goal:
            lo = mid
        else:
            hi = mid
    return mix(hi)


def rebalancing_outcomes(portfolio, eval_prices: pd.DataFrame) -> dict:
    """
    Next-year outcome of four actions taken at the end of the analysis window:
      hold          keep the portfolio as it is
      profile_only  plain threshold rebalancing to the profile's class targets
                    (only when the profile-drift check fires, as a robo-advisor would)
      dt            all coordinated Digital Twin recommendations
      markowitz     the max-Sharpe alternative under the profile constraints
    """
    from .rebalance_plan import RebalancePlan
    from .recommendations import RecommendationEngine
    from .risk_monitor import RiskMonitor

    hold = portfolio.get_weights_dict()
    variants = {"hold": hold}
    try:
        monitor = RiskMonitor(portfolio)
        engine = RecommendationEngine(portfolio, monitor)
        engine.generate_recommendations()
        variants["dt"] = dict(engine.plan.w)

        analysis = monitor.run_full_analysis()
        drift = [a for a in analysis["alert"] + analysis["caution"]
                 if engine._alert_kind(a) == "drift"]
        if drift:
            plan = RebalancePlan(portfolio)
            engine._rec_profile_drift(drift[0], drift[0]["severity"], plan)
            variants["profile_only"] = dict(plan.w)
        else:
            variants["profile_only"] = hold

        opt = engine.optimization_recommendation()
        variants["markowitz"] = ({t: w / 100 for t, w in opt["optimal_weights"].items()}
                                 if opt else hold)

        variants["hold_bond_mix"] = equal_risk_bond_mix(portfolio, hold, variants["dt"])
    except Exception as exc:
        logger.warning("Rebalancing evaluation failed: %s", exc)

    out = {}
    for v in REBALANCE_VARIANTS:
        stats = eval_stats(variants[v], eval_prices) if v in variants else             {k: np.nan for k in OUTCOME_METRICS}
        for k, x in stats.items():
            out[f"{v}_{k}"] = x
    return out


def rebalancing_summary(rows: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """
    Mean next-year outcome per action, and paired differences
    (action − hold, dt − profile_only) averaged over windows, with a cluster
    bootstrap 95% CI (portfolios resampled within each window).
    """
    rng = np.random.default_rng(seed)
    groups = [df.reset_index(drop=True) for _, df in rows.groupby("window")]
    comparisons = [(v, "hold") for v in ("profile_only", "dt", "markowitz", "hold_bond_mix")] + \
                  [("dt", "hold_bond_mix")] +                   [("dt", "profile_only"), ("dt", "markowitz")]

    def stat(dfs, a, b, m):
        vals = [np.nanmean(df[f"{a}_{m}"] - df[f"{b}_{m}"]) for df in dfs]
        return float(np.nanmean(vals))

    recs = []
    for v in REBALANCE_VARIANTS:
        recs.append({"comparison": v, **{m: float(np.nanmean(
            [np.nanmean(df[f"{v}_{m}"]) for df in groups])) for m in OUTCOME_METRICS}})
    for a, b in comparisons:
        rec = {"comparison": f"{a} − {b}"}
        boots = {m: [] for m in OUTCOME_METRICS}
        for _ in range(n_boot):
            sample = [df.iloc[rng.integers(0, len(df), len(df))] for df in groups]
            for m in OUTCOME_METRICS:
                boots[m].append(stat(sample, a, b, m))
        for m in OUTCOME_METRICS:
            point = stat(groups, a, b, m)
            lo, hi = np.nanpercentile(boots[m], [2.5, 97.5])
            wins = sum(np.nanmean(df[f"{a}_{m}"] - df[f"{b}_{m}"]) > 0 for df in groups)
            rec[m] = round(point, 3)
            rec[f"{m}_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
            rec[f"{m}_windows_up"] = f"{wins}/{len(groups)}"
        recs.append(rec)
    return pd.DataFrame(recs)


def ablation_scores(p_detail: dict) -> dict:
    """Recombine ProposedRiskScorer components (weights renormalized)."""
    a = p_detail.get("alert_norm", np.nan)
    s = p_detail.get("scenario_norm", np.nan)
    d = p_detail.get("drift_norm", np.nan)
    return {
        "alerts_only": a,
        "scenarios_only": s,
        "drift_only": d,
        "alerts_scenarios": (0.45 * a + 0.40 * s) / 0.85,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fast rank-based metrics (used inside the bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def _rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(score: np.ndarray, risk: np.ndarray) -> float:
    if len(score) < 3 or np.std(score) == 0 or np.std(risk) == 0:
        return np.nan
    return float(np.corrcoef(_rank(score), _rank(risk))[0, 1])


def auroc(score: np.ndarray, fragile: np.ndarray) -> float:
    """Mann–Whitney AUROC; NaN when only one class is present."""
    pos = fragile.astype(bool)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = _rank(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _window_metric(df: pd.DataFrame, scorer: str, metric: str,
                   fixed_threshold: float) -> float:
    score = df[scorer].to_numpy(dtype=float)
    dd = df["actual_drawdown"].to_numpy(dtype=float)
    if metric == "spearman":
        return spearman(score, -dd)
    if metric == "auroc_fixed":
        return auroc(score, dd < fixed_threshold)
    if metric == "spearman_underwater":
        if "actual_underwater" not in df:
            return np.nan
        return spearman(score, df["actual_underwater"].to_numpy(dtype=float))
    if metric == "auroc_tercile":
        return auroc(score, dd <= np.quantile(dd, 1 / 3))
    raise ValueError(metric)


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

METRICS = ["spearman", "auroc_tercile", "auroc_fixed", "spearman_underwater"]


@dataclass
class RobustResults:
    rows: pd.DataFrame                      # one row per (window, portfolio)
    windows: list[dict]                     # window date ranges
    fixed_threshold: float = -15.0
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_window: pd.DataFrame = field(default_factory=pd.DataFrame)

    def compute_per_window(self) -> pd.DataFrame:
        recs = []
        for wid, df in self.rows.groupby("window"):
            rec = {"window": wid, "n": len(df),
                   "fragile_fixed_pct": float((df["actual_drawdown"] < self.fixed_threshold).mean() * 100),
                   "median_drawdown": float(df["actual_drawdown"].median())}
            for s in SCORER_NAMES:
                for m in METRICS:
                    rec[f"{s}:{m}"] = _window_metric(df, s, m, self.fixed_threshold)
            recs.append(rec)
        self.per_window = pd.DataFrame(recs)
        return self.per_window

    def bootstrap(self, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
        """
        Paired cluster bootstrap of (scorer − baseline), averaged over windows.

        Portfolios are resampled with replacement inside each window; the
        same resample is used for every scorer (paired comparison).
        """
        rng = np.random.default_rng(seed)
        if self.per_window.empty:
            self.compute_per_window()

        groups = [df.reset_index(drop=True) for _, df in self.rows.groupby("window")]
        boot = {(s, m): [] for s in SCORER_NAMES for m in METRICS}
        for _ in range(n_boot):
            per_win = {(s, m): [] for s in SCORER_NAMES for m in METRICS}
            for df in groups:
                idx = rng.integers(0, len(df), size=len(df))
                sample = df.iloc[idx]
                for s in SCORER_NAMES:
                    for m in METRICS:
                        per_win[(s, m)].append(
                            _window_metric(sample, s, m, self.fixed_threshold))
            for key, vals in per_win.items():
                boot[key].append(np.nanmean(vals) if np.any(~np.isnan(vals)) else np.nan)

        recs = []
        for s in SCORER_NAMES:
            rec = {"scorer": s}
            for m in METRICS:
                point = float(np.nanmean(self.per_window[f"{s}:{m}"]))
                base_point = float(np.nanmean(self.per_window[f"baseline:{m}"]))
                diff = np.array(boot[(s, m)]) - np.array(boot[("baseline", m)])
                diff = diff[~np.isnan(diff)]
                rec[m] = round(point, 3)
                rec[f"{m}_diff"] = round(point - base_point, 3)
                if s != "baseline" and len(diff):
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    # two-sided bootstrap p-value for H0: diff = 0
                    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
                    rec[f"{m}_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
                    rec[f"{m}_p"] = round(float(min(p, 1.0)), 3)
                # share of windows where scorer beats baseline
                wins = (self.per_window[f"{s}:{m}"] > self.per_window[f"baseline:{m}"])
                valid = self.per_window[[f"{s}:{m}", f"baseline:{m}"]].notna().all(axis=1)
                rec[f"{m}_win_rate"] = f"{int(wins[valid].sum())}/{int(valid.sum())}"
                # Same paired comparison against historical volatility alone, the
                # strongest single static measure
                if s != "hist_vol":
                    dv = np.array(boot[(s, m)]) - np.array(boot[("hist_vol", m)])
                    dv = dv[~np.isnan(dv)]
                    vol_point = float(np.nanmean(self.per_window[f"hist_vol:{m}"]))
                    rec[f"{m}_diff_vs_vol"] = round(point - vol_point, 3)
                    if len(dv):
                        lo, hi = np.percentile(dv, [2.5, 97.5])
                        rec[f"{m}_ci_vs_vol"] = f"[{lo:+.3f}, {hi:+.3f}]"
            recs.append(rec)
        self.summary = pd.DataFrame(recs)
        return self.summary


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def score_portfolio(arch: dict, analysis_prices: pd.DataFrame,
                    eval_prices: pd.DataFrame,
                    baseline: BaselineRiskScorer,
                    proposed: ProposedRiskScorer,
                    meta: dict[str, dict] | None = None) -> dict:
    portfolio = build_portfolio_from_prices(arch, analysis_prices, meta=meta)
    b = baseline.score(portfolio)
    p = proposed.score(portfolio)
    hold = portfolio.get_weights_dict()
    row = {
        "baseline": b["risk_score"],
        "proposed": p["risk_score"],
        **ablation_scores(p),
        **resilience_scores(portfolio),
        **simple_scores(arch, analysis_prices, meta),
        # Ground truth: the portfolio the investor actually holds at the end of
        # the analysis window (weights drifted with returns), held for a year.
        "actual_drawdown": eval_stats(hold, eval_prices)["maxdd"],
        # Second ground truth for the recovery dimension: share of next-year
        # days spent more than 5% below the running peak (depth × duration)
        "actual_underwater": eval_underwater(hold, eval_prices),
        **rebalancing_outcomes(portfolio, eval_prices),
        "n_assets": len(arch["tickers"]),
        "risk_profile": arch["risk_profile"],
        "tickers": ",".join(arch["tickers"]),
        "weights": ",".join(f"{w:.4f}" for w in arch["weights"]),
    }
    return row


def run_robust_validation(
    prices: pd.DataFrame,
    n_portfolios: int = 150,
    analysis_days: int = 756,
    eval_days: int = 252,
    step_days: int = 252,
    seed: int = 42,
    fixed_threshold: float = -15.0,
    progress: Callable[[str], None] | None = None,
    meta: dict[str, dict] | None = None,
) -> RobustResults:
    """
    Run the rolling-window experiment on pre-fetched prices.

    For each window, a fresh set of random portfolios is drawn from the
    tickers that already trade at the start of the analysis window.
    """
    say = progress or logger.info
    rng = np.random.default_rng(seed)
    starts = first_real_dates(prices)
    meta = meta or TICKER_META
    universe_all = [t for t in meta if t in prices.columns]

    baseline, proposed = BaselineRiskScorer(), ProposedRiskScorer()
    windows = rolling_windows(len(prices), analysis_days, eval_days, step_days)
    rows, win_info = [], []

    for wid, (a0, e0, e1) in enumerate(windows):
        analysis = prices.iloc[a0:e0]
        evaluation = prices.iloc[e0:e1]
        universe = [t for t in universe_all if starts[t] <= analysis.index[0]]
        info = {
            "window": wid,
            "analysis_start": str(analysis.index[0].date()),
            "analysis_end": str(analysis.index[-1].date()),
            "eval_start": str(evaluation.index[0].date()),
            "eval_end": str(evaluation.index[-1].date()),
            "universe_size": len(universe),
        }
        win_info.append(info)
        say(f"Window {wid + 1}/{len(windows)}: analysis {info['analysis_start']} -> "
            f"{info['analysis_end']}, eval -> {info['eval_end']} "
            f"({len(universe)} tickers)")

        for arch in generate_random_portfolios(n_portfolios, universe, rng):
            try:
                row = score_portfolio(arch, analysis, evaluation, baseline, proposed, meta)
            except Exception as exc:
                logger.warning("Window %d, %s: %s", wid, arch["id"], exc)
                continue
            rows.append({"window": wid, "portfolio": arch["id"], **row})

    results = RobustResults(rows=pd.DataFrame(rows), windows=win_info,
                            fixed_threshold=fixed_threshold)
    results.compute_per_window()
    return results


def bootstrap_original(results, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Paired bootstrap CI for (proposed − baseline) on the original 30-portfolio
    experiment (a validation.ValidationResults instance).
    """
    df = pd.DataFrame({
        "baseline": results.baseline_scores,
        "proposed": results.proposed_scores,
        "actual_drawdown": results.actual_drawdowns,
    }).dropna()
    rng = np.random.default_rng(seed)
    out = {}
    for m in ["spearman", "auroc_fixed"]:
        point_b = _window_metric(df, "baseline", m, -15.0)
        point_p = _window_metric(df, "proposed", m, -15.0)
        diffs = []
        for _ in range(n_boot):
            s = df.iloc[rng.integers(0, len(df), size=len(df))]
            d = (_window_metric(s, "proposed", m, -15.0)
                 - _window_metric(s, "baseline", m, -15.0))
            if not np.isnan(d):
                diffs.append(d)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        out[m] = {
            "baseline": round(point_b, 3),
            "proposed": round(point_p, 3),
            "diff": round(point_p - point_b, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p_value": round(float(min(p, 1.0)), 3),
        }
    return out
