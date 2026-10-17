"""
rebalance_plan.py
-----------------
A single, shared target allocation that the recommendation steps edit in turn.

Each recommendation used to compute its trades against the *original*
portfolio, so steps that pursued the same goal (profile drift and volatility
both moving equities into bonds) were summed in the Trade Summary and
overshot the target. Here every step edits one working weight vector and its
trades are the difference before/after that step. The steps' trades therefore
add up exactly to the move from the current portfolio to the final plan.
"""

from __future__ import annotations

import numpy as np

from .data_loader import ASSET_CLASS_FALLBACK, SECTOR_FALLBACK
from .risk_monitor import is_diversified_fund

EQUITY_CLASSES = {"Equity", "Equity ETF", "International Equity",
                  "International ETF", "Emerging Markets ETF", "Cryptocurrency"}
FIXED_INCOME_CLASSES = {"Fixed Income", "Bond ETF"}
INTERNATIONAL_CLASSES = {"International Equity", "International ETF", "Emerging Markets ETF"}

# Trades smaller than this are noise and are dropped
MIN_TRADE_VALUE = 10.0


class _Meta:
    """Minimal asset-like object for tickers that are not held yet."""

    def __init__(self, ticker: str, sector: str, asset_class: str):
        self.ticker, self.sector, self.asset_class = ticker, sector, asset_class


class RebalancePlan:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.total = portfolio.total_value
        self.w: dict[str, float] = dict(portfolio.get_weights_dict())
        self._held = {a.ticker: a for a in portfolio.assets}

    # ── metadata & prices ────────────────────────────────────────────────
    def meta(self, ticker: str):
        if ticker in self._held:
            return self._held[ticker]
        return _Meta(ticker, SECTOR_FALLBACK.get(ticker, "Unknown"),
                     ASSET_CLASS_FALLBACK.get(ticker, "Equity"))

    def group(self, ticker: str) -> str:
        ac = self.meta(ticker).asset_class
        if ac in EQUITY_CLASSES:
            return "equity"
        if ac in FIXED_INCOME_CLASSES:
            return "fixed_income"
        return "other"

    def is_international(self, ticker: str) -> bool:
        return self.meta(ticker).asset_class in INTERNATIONAL_CLASSES

    def is_fund(self, ticker: str) -> bool:
        return is_diversified_fund(self.meta(ticker))

    def price(self, ticker: str) -> float | None:
        a = self._held.get(ticker)
        if a is not None and a.current_price > 0:
            return float(a.current_price)
        hist = self.portfolio.historical_prices
        if ticker in hist.columns:
            s = hist[ticker].dropna()
            if len(s) and s.iloc[-1] > 0:
                return float(s.iloc[-1])
        return None

    def first_priced(self, candidates: list[str]) -> str | None:
        """Prefer a candidate already held, otherwise the first one with a price."""
        held = [t for t in candidates if self.w.get(t, 0) > 0]
        for t in held + candidates:
            if self.price(t) is not None:
                return t
        return None

    # ── weight views ─────────────────────────────────────────────────────
    def members(self, group: str) -> list[str]:
        return [t for t, x in self.w.items() if x > 1e-12 and self.group(t) == group]

    def share(self, tickers) -> float:
        return float(sum(self.w.get(t, 0.0) for t in tickers))

    def group_share(self, group: str) -> float:
        return self.share(self.members(group))

    def snapshot(self) -> dict[str, float]:
        return dict(self.w)

    # ── edits ────────────────────────────────────────────────────────────
    def scale_to(self, tickers: list[str], target: float):
        """Rescale the given tickers pro rata so that together they weigh `target`."""
        cur = self.share(tickers)
        if cur <= 0:
            return
        f = target / cur
        for t in tickers:
            self.w[t] = self.w.get(t, 0.0) * f

    def move(self, sources: list[str], dest: str | list[str], amount: float) -> float:
        """
        Move `amount` of weight out of `sources` (pro rata) into `dest`
        (a ticker, or a list of tickers that receive it pro rata).
        Returns the amount actually moved.
        """
        sources = [t for t in sources if self.w.get(t, 0) > 0]
        avail = self.share(sources)
        amount = min(max(amount, 0.0), avail)
        if amount <= 0:
            return 0.0
        for t in sources:
            self.w[t] -= amount * self.w[t] / avail
        dests = [dest] if isinstance(dest, str) else list(dest)
        dshare = self.share(dests)
        for t in dests:
            frac = self.w.get(t, 0.0) / dshare if dshare > 0 else 1.0 / len(dests)
            self.w[t] = self.w.get(t, 0.0) + amount * frac
        return amount

    # ── output ───────────────────────────────────────────────────────────
    def trades_since(self, before: dict[str, float]) -> list[dict]:
        trades = []
        for t in sorted(set(before) | set(self.w)):
            delta = (self.w.get(t, 0.0) - before.get(t, 0.0)) * self.total
            if abs(delta) < MIN_TRADE_VALUE:
                continue
            price = self.price(t)
            if price is None:
                continue
            qty = int(abs(delta) / price)
            if qty <= 0:
                continue
            trades.append({
                "ticker": t,
                "action": "BUY" if delta > 0 else "SELL",
                "quantity": qty,
                "price": round(price, 2),
                "value": round(qty * price, 2),
            })
        return sorted(trades, key=lambda x: (x["action"] != "SELL", -x["value"]))

    def stats(self) -> dict | None:
        """Expected return / volatility / Sharpe of the current plan weights."""
        from .optimizer import evaluate_weights
        from .portfolio import RISK_FREE_RATE

        hist = self.portfolio.historical_prices
        tickers = [t for t, x in self.w.items() if x > 1e-9 and t in hist.columns]
        if len(tickers) < 1 or self.share(tickers) < 0.95:
            return None
        prices = hist[tickers].dropna()
        if len(prices) < 60:
            return None
        return evaluate_weights(prices, {t: self.w[t] for t in tickers}, RISK_FREE_RATE)
