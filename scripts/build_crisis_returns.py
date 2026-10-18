"""
Build data/historical_crises.json: the actual peak-to-trough return of each
ticker in the historical stress windows, used to *replay* those crises in
the scenario engine instead of approximating them with factor parameters.

Run once (needs network):  python -m scripts.build_crisis_returns
The output is committed so the app needs no network access for it.

Windows are S&P 500 peak → trough:
    2008 Financial Crisis  2007-10-09 → 2009-03-09
    2020 COVID Crash       2020-02-19 → 2020-03-23

Also stored per window: the S&P 500 (SPY) return, the change in the 10-year
Treasury yield (^TNX) and the ratio of SPY's realised daily volatility in the
window to its 2010-2019 average, which become the scenario's market_change,
rate_change and volatility_multiplier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import SECTOR_FALLBACK
from src.validation import TICKER_META

WINDOWS = {
    "financial_crisis_2008": ("2007-10-09", "2009-03-09"),
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
}
CALM_PERIOD = ("2010-01-01", "2019-12-31")
OUT = Path("data/historical_crises.json")


def main():
    import yfinance as yf

    universe = set(SECTOR_FALLBACK) | set(TICKER_META)
    for csv in Path("data").glob("portfolio_*.csv"):
        universe |= set(pd.read_csv(csv)["ticker"].astype(str).str.upper())
    universe = sorted(universe | {"SPY", "^TNX"})

    px = yf.download(universe, start="2007-06-01", end="2020-06-30",
                     auto_adjust=True, progress=False)["Close"]

    spy_ret = np.log(px["SPY"]).diff().dropna()
    calm_vol = spy_ret.loc[CALM_PERIOD[0]:CALM_PERIOD[1]].std()

    out = {"_source": "Yahoo Finance adjusted close, peak-to-trough of S&P 500",
           "_calm_period": CALM_PERIOD}
    for sid, (start, end) in WINDOWS.items():
        window = px.loc[start:end]
        returns = {}
        for t in universe:
            if t == "^TNX" or t not in window.columns:
                continue
            s = window[t].dropna()
            # Only tickers that traded for the whole window
            if len(s) < 2 or s.index[0] > pd.Timestamp(start) + pd.Timedelta(days=5):
                continue
            returns[t] = round(float(s.iloc[-1] / s.iloc[0] - 1), 4)
        tnx = window["^TNX"].dropna()
        out[sid] = {
            "start": start,
            "end": end,
            "market_change": returns["SPY"],
            "rate_change": round(float(tnx.iloc[-1] - tnx.iloc[0]) / 100, 4),
            "volatility_multiplier": round(float(spy_ret.loc[start:end].std() / calm_vol), 2),
            "returns": returns,
        }
        print(f"{sid}: SPY {returns['SPY']:+.1%}, Δ10y {out[sid]['rate_change']:+.2%}, "
              f"vol×{out[sid]['volatility_multiplier']}, {len(returns)} tickers")

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
