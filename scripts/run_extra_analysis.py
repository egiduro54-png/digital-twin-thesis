"""
Extra analyses on the saved rolling-window rows (no new experiment):

    python -m scripts.run_extra_analysis --rows results/validation/rows.csv

Outputs next to the rows file:
    extra_per_window.csv        ranking quality per window, with the market regime
    extra_regime.csv            stress vs calm windows: mean Spearman, paired bootstrap
                                differences vs historical volatility and vs Baseline
    extra_rq3_per_window.csv    RQ3 strategy outcomes per window
    extra_auroc_thresholds.csv  AUROC sensitivity to the fragility threshold

A window is a "stress" window when the median buy-and-hold drawdown of its
portfolios in the evaluation year is below the fixed fragility threshold
(−15%), i.e. most portfolios suffered a real fall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.robust_validation import auroc, spearman

SCORERS = {
    "baseline": "Baseline",
    "proposed": "DT scorer",
    "alerts_scenarios": "DT χωρίς drift",
    "scenarios_only": "Μόνο σενάρια",
    "resilience": "Resilience Index",
    "ri_downside": "RI: CVaR/downside",
    "hist_vol": "Ιστορική μεταβλητότητα",
}
REFERENCES = ["hist_vol", "baseline"]
STRESS_THRESHOLD = -15.0
THRESHOLDS = [-10.0, -15.0, -20.0]
STRATEGIES = ["hold", "profile_only", "dt", "markowitz", "hold_bond_mix"]


def _ci(x: np.ndarray) -> str:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{lo:+.3f}, {hi:+.3f}]"


def _mean_metric(groups: list[pd.DataFrame], fn) -> dict[str, float]:
    """Mean over windows of fn(window_df, scorer) for every scorer."""
    return {s: float(np.nanmean([fn(g, s) for g in groups])) for s in SCORERS}


def _bootstrap(groups: list[pd.DataFrame], fn, n_boot: int, rng) -> dict[str, np.ndarray]:
    """Paired cluster bootstrap: resample portfolios inside each window."""
    out = {s: np.empty(n_boot) for s in SCORERS}
    for b in range(n_boot):
        sample = [g.iloc[rng.integers(0, len(g), size=len(g))] for g in groups]
        for s, v in _mean_metric(sample, fn).items():
            out[s][b] = v
    return out


def _compare(groups, fn, n_boot, rng, extra: dict) -> list[dict]:
    point = _mean_metric(groups, fn)
    boot = _bootstrap(groups, fn, n_boot, rng)
    recs = []
    for s, label in SCORERS.items():
        rec = {**extra, "scorer": s, "label": label, "value": round(point[s], 3)}
        for ref in REFERENCES:
            if s == ref:
                continue
            d = boot[s] - boot[ref]
            rec[f"diff_vs_{ref}"] = round(point[s] - point[ref], 3)
            rec[f"ci_vs_{ref}"] = _ci(d)
            rec[f"significant_vs_{ref}"] = bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)
        recs.append(rec)
    return recs


def _spearman(g: pd.DataFrame, s: str) -> float:
    return spearman(g[s].to_numpy(float), -g["actual_drawdown"].to_numpy(float))


def per_window(rows: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    dates = {w["window"]: w for w in windows}
    recs = []
    for wid, g in rows.groupby("window"):
        med = float(g["actual_drawdown"].median())
        rec = {
            "window": wid + 1,
            "eval_start": dates.get(wid, {}).get("eval_start"),
            "eval_end": dates.get(wid, {}).get("eval_end"),
            "regime": "stress" if med < STRESS_THRESHOLD else "calm",
            "median_drawdown": round(med, 1),
            "fragile_pct": round(float((g["actual_drawdown"] < STRESS_THRESHOLD).mean() * 100), 1),
        }
        for s in SCORERS:
            rec[s] = round(_spearman(g, s), 3)
        recs.append(rec)
    return pd.DataFrame(recs)


def regime(rows: pd.DataFrame, pw: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    recs = []
    for name in ["stress", "calm"]:
        ids = set(pw.loc[pw["regime"] == name, "window"] - 1)
        groups = [g for wid, g in rows.groupby("window") if wid in ids]
        if not groups:
            continue
        recs += _compare(groups, _spearman, n_boot, rng,
                         {"regime": name, "windows": len(groups)})
    return pd.DataFrame(recs)


def auroc_thresholds(rows: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    groups = [g for _, g in rows.groupby("window")]
    recs = []
    for t in THRESHOLDS:
        def fn(g, s, t=t):
            return auroc(g[s].to_numpy(float), g["actual_drawdown"].to_numpy(float) < t)
        fragile = float(np.mean([(g["actual_drawdown"] < t).mean() for g in groups]) * 100)
        recs += _compare(groups, fn, n_boot, rng,
                         {"threshold": t, "fragile_pct": round(fragile, 1)})
    return pd.DataFrame(recs)


def rq3_per_window(rows: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    cols = {f"{s}_{m}": "mean" for s in STRATEGIES for m in ["maxdd", "ret", "sharpe"]}
    out = rows.groupby("window").agg(cols).round(2).reset_index()
    out["window"] += 1
    out = pw[["window", "eval_start", "eval_end", "regime"]].merge(out, on="window")
    out["dt_minus_hold_sharpe"] = (out["dt_sharpe"] - out["hold_sharpe"]).round(2)
    out["dt_minus_bonds_sharpe"] = (out["dt_sharpe"] - out["hold_bond_mix_sharpe"]).round(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="results/validation/rows.csv")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows_path = Path(args.rows)
    out = rows_path.parent
    rows = pd.read_csv(rows_path)
    windows_file = out / "windows.json"
    windows = json.loads(windows_file.read_text()) if windows_file.exists() else []
    rng = np.random.default_rng(args.seed)

    pw = per_window(rows, windows)
    pw.to_csv(out / "extra_per_window.csv", index=False, encoding="utf-8-sig")
    regime(rows, pw, args.bootstrap, rng).to_csv(out / "extra_regime.csv", index=False, encoding="utf-8-sig")
    rq3_per_window(rows, pw).to_csv(out / "extra_rq3_per_window.csv", index=False, encoding="utf-8-sig")
    auroc_thresholds(rows, args.bootstrap, rng).to_csv(out / "extra_auroc_thresholds.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote extra_*.csv to {out}")


if __name__ == "__main__":
    main()
