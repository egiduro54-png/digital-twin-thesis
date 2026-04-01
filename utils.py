"""
utils.py

Shared helper functions used across the system.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import (
    load_portfolio_csv,
    fetch_current_prices,
    fetch_historical_data,
    fetch_ticker_info,
)
from .portfolio import Portfolio, Asset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio Builder (full pipeline)
# ---------------------------------------------------------------------------

def build_portfolio(
    csv_path: str,
    risk_profile: str = "moderate",
    portfolio_name: str | None = None,
    history_years: int = 5,
) -> Portfolio:
    """
    Full pipeline: CSV → yfinance data → Portfolio object.

    Parameters
    ----------
    csv_path      : path to portfolio CSV (ticker, quantity, entry_price)
    risk_profile  : 'conservative' | 'moderate' | 'aggressive'
    portfolio_name: display name; defaults to CSV filename
    history_years : years of historical data to fetch

    Returns
    -------
    Portfolio instance ready for analysis
    """
    # 1. Load CSV
    df = load_portfolio_csv(csv_path)
    tickers = df["ticker"].tolist()

    name = portfolio_name or Path(csv_path).stem.replace("_", " ").title()
    logger.info("Building portfolio '%s' with %d holdings", name, len(tickers))

    # 2. Fetch current prices
    current_prices = fetch_current_prices(tickers)

    # 3. Fetch historical data
    historical = fetch_historical_data(tickers, years=history_years)

    # 4. Fetch ticker metadata (sector, asset class, etc.)
    info = fetch_ticker_info(tickers)

    # 5. Build Asset objects
    assets = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        price = current_prices.get(ticker, float("nan"))
        if np.isnan(price):
            logger.warning("Using entry_price for %s (no live price available)", ticker)
            price = row["entry_price"]

        meta = info.get(ticker, {})
        asset = Asset(
            ticker=ticker,
            quantity=float(row["quantity"]),
            entry_price=float(row["entry_price"]),
            current_price=float(price),
            name=meta.get("name", ticker),
            sector=meta.get("sector", "Unknown"),
            industry=meta.get("industry", "Unknown"),
            asset_class=meta.get("asset_class", "Equity"),
            country=meta.get("country", "Unknown"),
        )
        assets.append(asset)

    return Portfolio(
        assets=assets,
        historical_prices=historical,
        risk_profile=risk_profile,
        name=name,
    )


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def format_currency(value: float, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def format_pct(value: float, decimals: int = 1, show_sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def severity_color(severity: str) -> str:
    """Map severity string to a hex color for Streamlit display."""
    return {
        "ok": "#28a745",       # green
        "caution": "#ffc107",  # amber
        "alert": "#dc3545",    # red
    }.get(severity, "#6c757d")


def severity_emoji(severity: str) -> str:
    return {"ok": "✅", "caution": "⚠️", "alert": "🚨"}.get(severity, "❓")


def priority_color(priority: str) -> str:
    return {
        "critical": "#dc3545",
        "high": "#fd7e14",
        "medium": "#ffc107",
        "low": "#6c757d",
    }.get(priority, "#6c757d")


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------

def validate_portfolio_df(df: pd.DataFrame) -> list[str]:
    """
    Return a list of validation warnings for the portfolio DataFrame.
    Empty list = all good.
    """
    warnings = []
    if df.empty:
        warnings.append("Portfolio is empty.")
        return warnings

    if df["quantity"].isnull().any():
        warnings.append("Some quantities are missing.")
    if df["entry_price"].isnull().any():
        warnings.append("Some entry prices are missing.")

    if (df["quantity"] <= 0).any():
        bad = df[df["quantity"] <= 0]["ticker"].tolist()
        warnings.append(f"Non-positive quantities for: {bad}")

    if (df["entry_price"] <= 0).any():
        bad = df[df["entry_price"] <= 0]["ticker"].tolist()
        warnings.append(f"Non-positive entry prices for: {bad}")

    dupes = df[df["ticker"].duplicated()]["ticker"].tolist()
    if dupes:
        warnings.append(f"Duplicate tickers: {dupes}")

    return warnings


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
