"""
learned_scorer.py
-----------------
Risk scorers whose weights are learned from data instead of fixed a priori,
evaluated walk-forward on the rolling-window validation rows.

For each window k (after `min_train` windows), a model is trained only on
windows < k and scores window k, so no window is ever scored by a model that
saw its outcome. Features and target are standardised within each window:
the task is cross-sectional (which portfolios are riskier than the others in
the same year), not predicting the level of the market.

Feature sets
    dt      the Digital Twin's own components (alerts, scenario loss, profile
            drift, the six Resilience Index components)
    dt+static  the above plus the static measures (historical volatility,
            drawdown, CVaR, HHI, equity share), to test whether the DT adds
            information beyond them

Models: Ridge (linear, coefficients readable) and gradient boosting.
Explainability: permutation importance, i.e. the drop in mean Spearman ρ on
the walk-forward predictions when one feature is shuffled within each window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DT_FEATURES = [
    "alerts_only", "scenarios_only", "drift_only",
    "ri_stress", "ri_sequential", "ri_downside", "ri_recovery",
    "ri_diversification", "ri_concentration",
]
STATIC_FEATURES = ["hist_vol", "hist_maxdd", "hist_cvar95", "hhi", "equity_share"]
FEATURE_SETS = {"dt": DT_FEATURES, "dt+static": DT_FEATURES + STATIC_FEATURES}

FEATURE_LABELS = {
    "alerts_only": "Alerts RiskMonitor",
    "scenarios_only": "Απώλεια stress σεναρίων",
    "drift_only": "Απόκλιση προφίλ",
    "ri_stress": "RI: stress",
    "ri_sequential": "RI: διαδοχικά shocks",
    "ri_downside": "RI: CVaR",
    "ri_recovery": "RI: ανάκαμψη",
    "ri_diversification": "RI: διαφοροποίηση",
    "ri_concentration": "RI: συγκέντρωση",
    "hist_vol": "Ιστορική μεταβλητότητα",
    "hist_maxdd": "Ιστορικό max drawdown",
    "hist_cvar95": "Ιστορικό CVaR 95%",
    "hhi": "HHI",
    "equity_share": "Ποσοστό μετοχών",
}


def _zscore_within(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("window")[cols]
    z = (df[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z.fillna(0.0)


def _make_model(kind: str, seed: int):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0)
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                             max_iter=200, random_state=seed)
    raise ValueError(kind)


def walk_forward_scores(rows: pd.DataFrame, features: list[str], kind: str = "ridge",
                        min_train: int = 3, seed: int = 0) -> pd.Series:
    """
    Out-of-sample risk score per row (higher = riskier); NaN for the first
    `min_train` windows, which only serve as training data.
    """
    df = rows.dropna(subset=features + ["actual_drawdown"]).copy()
    X = _zscore_within(df, features)
    # Target: within-window standardised loss (−drawdown), higher = riskier
    df["_y"] = -df["actual_drawdown"]
    y = _zscore_within(df, ["_y"])["_y"]

    out = pd.Series(np.nan, index=rows.index)
    windows = sorted(df["window"].unique())
    for k in windows[min_train:]:
        train = df["window"] < k
        test = df["window"] == k
        model = _make_model(kind, seed)
        model.fit(X[train].values, y[train].values)
        out.loc[df.index[test]] = model.predict(X[test].values)
    return out


def permutation_importance(rows: pd.DataFrame, features: list[str], kind: str = "ridge",
                           min_train: int = 3, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Drop in mean per-window Spearman ρ of the walk-forward scores when each
    feature is shuffled within every window (in train and test alike).
    """
    from .robust_validation import spearman

    rng = np.random.default_rng(seed)

    def mean_rho(r: pd.DataFrame) -> float:
        s = walk_forward_scores(r, features, kind, min_train, seed)
        vals = [spearman(s[g.index].to_numpy(), -g["actual_drawdown"].to_numpy())
                for _, g in r[s.notna()].groupby("window")]
        return float(np.nanmean(vals))

    base = mean_rho(rows)
    recs = []
    for f in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = rows.copy()
            shuffled[f] = shuffled.groupby("window")[f].transform(
                lambda s: rng.permutation(s.to_numpy()))
            drops.append(base - mean_rho(shuffled))
        recs.append({"feature": f, "label": FEATURE_LABELS.get(f, f),
                     "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(recs).sort_values("importance", ascending=False).reset_index(drop=True)


def ridge_coefficients(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Coefficients of a Ridge fit on all windows (standardised units), for reading only."""
    df = rows.dropna(subset=features + ["actual_drawdown"]).copy()
    X = _zscore_within(df, features)
    df["_y"] = -df["actual_drawdown"]
    y = _zscore_within(df, ["_y"])["_y"]
    model = _make_model("ridge", 0).fit(X.values, y.values)
    return (pd.DataFrame({"feature": features,
                          "label": [FEATURE_LABELS.get(f, f) for f in features],
                          "coefficient": model.coef_})
            .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True))


def evaluate_learned(rows: pd.DataFrame, min_train: int = 3, n_boot: int = 1000,
                     seed: int = 0, fixed_threshold: float = -15.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward learned scorers vs the fixed-weight scorers on the same
    out-of-sample windows. Returns (summary, rows_with_scores); the summary has
    mean per-window Spearman / AUROC and a paired cluster-bootstrap CI of the
    difference to the static baseline.
    """
    from .robust_validation import _window_metric

    rows = rows.copy()
    learned = []
    for set_name, feats in FEATURE_SETS.items():
        for kind in ("ridge", "gbm"):
            col = f"learned_{kind}_{set_name}"
            rows[col] = walk_forward_scores(rows, feats, kind, min_train, seed)
            learned.append(col)

    test = rows[rows[learned[0]].notna()]
    scorers = ["baseline", "proposed", "resilience", "hist_vol"] + learned
    groups = [g.reset_index(drop=True) for _, g in test.groupby("window")]
    metrics = ["spearman", "auroc_fixed", "auroc_tercile"]
    if "actual_underwater" in rows:
        metrics.append("spearman_underwater")

    def per_window(dfs, s, m):
        return np.array([_window_metric(g, s, m, fixed_threshold) for g in dfs])

    rng = np.random.default_rng(seed)
    point = {(s, m): per_window(groups, s, m) for s in scorers for m in metrics}
    boots = {(s, m): [] for s in scorers for m in metrics}
    boots_vol = {(s, m): [] for s in scorers for m in metrics}
    for _ in range(n_boot):
        sample = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        base = {m: per_window(sample, "baseline", m) for m in metrics}
        vol = {m: per_window(sample, "hist_vol", m) for m in metrics}
        for s in scorers:
            for m in metrics:
                own = per_window(sample, s, m)
                boots[(s, m)].append(np.nanmean(own - base[m]))
                boots_vol[(s, m)].append(np.nanmean(own - vol[m]))

    recs = []
    for s in scorers:
        rec = {"scorer": s, "windows": len(groups)}
        for m in metrics:
            rec[m] = round(float(np.nanmean(point[(s, m)])), 3)
            diff = point[(s, m)] - point[("baseline", m)]
            rec[f"{m}_diff"] = round(float(np.nanmean(diff)), 3)
            lo, hi = np.nanpercentile(boots[(s, m)], [2.5, 97.5])
            rec[f"{m}_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
            rec[f"{m}_wins"] = f"{int(np.nansum(diff > 0))}/{int(np.sum(~np.isnan(diff)))}"
            rec[f"{m}_diff_vs_vol"] = round(float(np.nanmean(point[(s, m)] - point[("hist_vol", m)])), 3)
            lo, hi = np.nanpercentile(boots_vol[(s, m)], [2.5, 97.5])
            rec[f"{m}_ci_vs_vol"] = f"[{lo:+.3f}, {hi:+.3f}]"
        recs.append(rec)
    return pd.DataFrame(recs), rows
