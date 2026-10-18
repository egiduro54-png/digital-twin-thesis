"""
Run the extended robustness evaluation (src/robust_validation.py).

Usage (from the repo root):
    python -m scripts.run_robust_validation
    python -m scripts.run_robust_validation --n-portfolios 50 --bootstrap 200   # quick run

Outputs (in --out, default results/robust_validation/):
    prices.csv              cached price history (delete to re-download)
    rows.csv                one row per (window, portfolio) with all scores
    per_window.csv          metrics per window and scorer
    summary.csv             mean metrics, diff vs baseline, bootstrap CI, p-value
    windows.json            window date ranges
    original_bootstrap.json bootstrap CI for the original 30-portfolio experiment
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src.data_loader import fetch_historical_data
from src.robust_validation import (
    bootstrap_original,
    rebalancing_summary,
    run_robust_validation,
)
from src.validation import TICKER_META, ValidationExperiment


def load_prices(out: Path, years: int) -> pd.DataFrame:
    cache = out / "prices.csv"
    if cache.exists():
        print(f"Using cached prices: {cache}")
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    tickers = sorted(set(TICKER_META) | {"SPY"})
    print(f"Downloading {years}y of prices for {len(tickers)} tickers...")
    prices = fetch_historical_data(tickers, years=years)
    prices.to_csv(cache)
    return prices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/robust_validation")
    ap.add_argument("--years", type=int, default=15)
    ap.add_argument("--n-portfolios", type=int, default=150)
    ap.add_argument("--analysis-days", type=int, default=756)
    ap.add_argument("--eval-days", type=int, default=252)
    ap.add_argument("--step-days", type=int, default=252)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-original", action="store_true",
                    help="Skip the bootstrap of the original 30-portfolio experiment")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Original experiment + bootstrap CI ───────────────────────────────
    if not args.skip_original:
        print("\n=== Original 30-portfolio experiment (paired bootstrap) ===")
        orig = ValidationExperiment().run()
        boot = bootstrap_original(orig, n_boot=2000, seed=args.seed)
        boot["windows"] = {"analysis": [orig.analysis_start, orig.analysis_end],
                           "evaluation": [orig.eval_start, orig.eval_end]}
        (out / "original_bootstrap.json").write_text(
            json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(boot, indent=2, ensure_ascii=False))

    # ── 2. Rolling random-portfolio experiment ──────────────────────────────
    print("\n=== Rolling-window random-portfolio experiment ===")
    prices = load_prices(out, args.years)
    res = run_robust_validation(
        prices,
        n_portfolios=args.n_portfolios,
        analysis_days=args.analysis_days,
        eval_days=args.eval_days,
        step_days=args.step_days,
        seed=args.seed,
        progress=print,
    )
    res.rows.to_csv(out / "rows.csv", index=False)
    res.per_window.to_csv(out / "per_window.csv", index=False)
    (out / "windows.json").write_text(json.dumps(res.windows, indent=2), encoding="utf-8")

    # ── 3. RQ3: next-year outcome of acting on the recommendations ──────────
    print("\n=== RQ3: rebalancing outcomes (next year, buy-and-hold) ===")
    print("maxdd/ret/sharpe: higher = better; vol/cvar: lower = better")
    rq3 = rebalancing_summary(res.rows, n_boot=args.bootstrap, seed=args.seed)
    rq3.to_csv(out / "rq3_rebalancing.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(rq3.to_string(index=False))

    print(f"\nBootstrapping ({args.bootstrap} resamples)...")
    summary = res.bootstrap(n_boot=args.bootstrap, seed=args.seed)
    summary.to_csv(out / "summary.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out}/")


if __name__ == "__main__":
    main()
