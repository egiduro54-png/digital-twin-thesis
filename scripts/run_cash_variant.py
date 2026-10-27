"""
Recompute RQ3 for a saved run with the current recommendation code.

The strategies' third class is cash. Earlier runs filled it with gold; this
replays the exact portfolios of a saved run (same seed, windows, universe and
prices, plus BIL as the cash proxy) and recomputes only the RQ3 outcomes.
Risk ranking (RQ1/RQ2) does not depend on the recommendations, so it is not
recomputed.

    python -m scripts.run_cash_variant --run results/robust_validation_main_r2 \
        --universe main --out results/robust_validation_cash_main

Outputs in --out: prices.csv (with BIL), rows_rq3.csv, rq3_rebalancing.csv.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import fetch_historical_data
from src.robust_validation import (
    UNIVERSES, first_real_dates, generate_random_portfolios, rebalancing_outcomes,
    rebalancing_summary, rolling_windows,
)
from src.validation import build_portfolio_from_prices

CASH = "BIL"


def load_prices(run: Path, out: Path) -> pd.DataFrame:
    cache = out / "prices.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    prices = pd.read_csv(run / "prices.csv", index_col=0, parse_dates=True)
    if CASH not in prices.columns:
        years = (prices.index[-1] - prices.index[0]).days // 365 + 2
        cash = fetch_historical_data([CASH], years=years)[[CASH]]
        prices = prices.join(cash.reindex(prices.index).ffill().bfill())
    prices.to_csv(cache)
    return prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="saved run with prices.csv")
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-portfolios", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prices = load_prices(Path(args.run), out)
    meta = UNIVERSES[args.universe]
    # Same draws as run_robust_validation: the rng is used only to make portfolios
    rng = np.random.default_rng(args.seed)
    starts = first_real_dates(prices)
    universe_all = [t for t in meta if t in prices.columns]

    rows = []
    for wid, (a0, e0, e1) in enumerate(rolling_windows(len(prices))):
        analysis, evaluation = prices.iloc[a0:e0], prices.iloc[e0:e1]
        universe = [t for t in universe_all if starts[t] <= analysis.index[0]]
        print(f"Window {wid + 1}: {analysis.index[0].date()} -> {evaluation.index[-1].date()}",
              flush=True)
        for arch in generate_random_portfolios(args.n_portfolios, universe, rng):
            try:
                p = build_portfolio_from_prices(arch, analysis, meta=meta)
                rows.append({"window": wid, "portfolio": arch["id"],
                             **rebalancing_outcomes(p, evaluation),
                             "risk_profile": arch["risk_profile"]})
            except Exception as exc:
                print(f"  {arch['id']}: {exc}")

    rows = pd.DataFrame(rows)
    rows.to_csv(out / "rows_rq3.csv", index=False)
    rebalancing_summary(rows, n_boot=args.bootstrap, seed=args.seed).to_csv(
        out / "rq3_rebalancing.csv", index=False)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
