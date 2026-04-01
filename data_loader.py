"""
data_loader.py

Handles all data ingestion:
  - Loading portfolio from CSV
  - Fetching current prices from Yahoo Finance (yfinance)
  - Fetching historical price data for volatility/correlation calculations
  - Mapping tickers to sectors and asset classes
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Default look-back window for historical analysis (5 years)
DEFAULT_HISTORY_YEARS = 5

# Sector metadata sourced from yfinance .info; fallback map for common ETFs
SECTOR_FALLBACK = {
    "SPY": "Equity ETF",
    "QQQ": "Equity ETF",
    "VTI": "Equity ETF",
    "IVV": "Equity ETF",
    "VIG": "Equity ETF",
    "VYM": "Equity ETF",
    "VXUS": "International ETF",
    "EFA": "International ETF",
    "IEMG": "Emerging Markets ETF",
    "BND": "Bond ETF",
    "TLT": "Bond ETF",
    "IEF": "Bond ETF",
    "SHY": "Bond ETF",
    "AGG": "Bond ETF",
    "LQD": "Bond ETF",
    "HYG": "Bond ETF",
    "VCIT": "Bond ETF",
    "VCSH": "Bond ETF",
    "VNQ": "Real Estate ETF",
    "IYR": "Real Estate ETF",
    "GLD": "Commodities ETF",
    "SLV": "Commodities ETF",
    "GSG": "Commodities ETF",
    "BTC-USD": "Cryptocurrency",
    "ETH-USD": "Cryptocurrency",
}

ASSET_CLASS_FALLBACK = {
    "SPY": "Equity", "QQQ": "Equity", "VTI": "Equity", "IVV": "Equity",
    "VIG": "Equity", "VYM": "Equity",
    "VXUS": "International Equity", "EFA": "International Equity",
    "IEMG": "International Equity",
    "BND": "Fixed Income", "TLT": "Fixed Income", "IEF": "Fixed Income",
    "SHY": "Fixed Income", "AGG": "Fixed Income", "LQD": "Fixed Income",
    "HYG": "Fixed Income", "VCIT": "Fixed Income", "VCSH": "Fixed Income",
    "VNQ": "Real Estate", "IYR": "Real Estate",
    "GLD": "Commodity", "SLV": "Commodity", "GSG": "Commodity",
    "BTC-USD": "Cryptocurrency", "ETH-USD": "Cryptocurrency",
}


# ---------------------------------------------------------------------------
# Portfolio CSV Loading
# ---------------------------------------------------------------------------

def load_portfolio_csv(file_path: str) -> pd.DataFrame:
    """
    Load a portfolio from a CSV file.

    Expected columns: ticker, quantity, entry_price
    Returns a DataFrame with those columns (plus validation).
    Raises ValueError for malformed data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = {"ticker", "quantity", "entry_price"}
    missing = required_columns - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {missing}")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="raise")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="raise")

    if (df["quantity"] <= 0).any():
        raise ValueError("All quantities must be positive.")
    if (df["entry_price"] <= 0).any():
        raise ValueError("All entry prices must be positive.")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Price Fetching
# ---------------------------------------------------------------------------

def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch the latest closing price for each ticker via yfinance.

    Returns a dict: {ticker: price}.
    If a ticker cannot be fetched, its price will be NaN and a warning logged.
    """
    prices = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty:
                logger.warning("No price data for %s", ticker)
                prices[ticker] = float("nan")
            else:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("Failed to fetch price for %s: %s", ticker, exc)
            prices[ticker] = float("nan")
    return prices


def fetch_historical_data(
    tickers: list[str],
    years: int = DEFAULT_HISTORY_YEARS,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for *tickers* covering *years* of history.

    Returns a DataFrame where columns are tickers and index is Date.
    Missing tickers / days are forward-filled then back-filled.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start = datetime.today() - timedelta(days=years * 365 + 30)
        start_date = start.strftime("%Y-%m-%d")

    logger.info("Downloading historical prices: %s → %s for %d tickers",
                start_date, end_date, len(tickers))

    raw = yf.download(tickers, start=start_date, end=end_date,
                      auto_adjust=True, progress=False)

    # yfinance returns multi-level columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers[:1]

    # Drop fully-missing columns with a warning
    all_nan = prices.columns[prices.isna().all()]
    if len(all_nan):
        logger.warning("No historical data for: %s", list(all_nan))
    prices = prices.drop(columns=all_nan)

    # Fill intra-series gaps (weekends / holidays)
    prices = prices.ffill().bfill()
    return prices


def calculate_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Convert price series to daily returns.

    method='log'    → log returns  (better for multi-period analysis)
    method='simple' → simple returns
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------------
# Ticker Metadata (sector, asset class, country)
# ---------------------------------------------------------------------------

def fetch_ticker_info(tickers: list[str]) -> dict[str, dict]:
    """
    Retrieve sector, industry, asset class, and country for each ticker.

    Falls back to SECTOR_FALLBACK / ASSET_CLASS_FALLBACK for common ETFs.
    Returns dict: {ticker: {sector, industry, asset_class, country, name}}
    """
    info_map = {}
    for ticker in tickers:
        meta = {
            "ticker": ticker,
            "name": ticker,
            "sector": SECTOR_FALLBACK.get(ticker, "Unknown"),
            "industry": "Unknown",
            "asset_class": ASSET_CLASS_FALLBACK.get(ticker, "Equity"),
            "country": "Unknown",
        }
        try:
            t = yf.Ticker(ticker)
            raw = t.info or {}
            if raw.get("longName"):
                meta["name"] = raw["longName"]
            if raw.get("sector"):
                meta["sector"] = raw["sector"]
            if raw.get("industry"):
                meta["industry"] = raw["industry"]
            if raw.get("country"):
                meta["country"] = raw["country"]
            # Infer asset class from quoteType
            qt = raw.get("quoteType", "").upper()
            if qt == "ETF":
                meta["asset_class"] = ASSET_CLASS_FALLBACK.get(ticker, "Equity ETF")
            elif qt == "CRYPTOCURRENCY":
                meta["asset_class"] = "Cryptocurrency"
            elif qt == "EQUITY":
                meta["asset_class"] = "Equity"
            elif qt == "MUTUALFUND":
                meta["asset_class"] = "Mutual Fund"
        except Exception as exc:
            logger.debug("Could not fetch info for %s: %s", ticker, exc)

        info_map[ticker] = meta

    return info_map


# ---------------------------------------------------------------------------
# Scenario Config
# ---------------------------------------------------------------------------

def load_scenarios_config(config_path: str) -> dict:
    """Load pre-defined scenario definitions from JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Convenience: full data load for a portfolio
# ---------------------------------------------------------------------------

def load_portfolio_with_market_data(
    csv_path: str,
    history_years: int = DEFAULT_HISTORY_YEARS,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, dict[str, dict]]:
    """
    One-shot loader: CSV → current prices + history + metadata.

    Returns:
        portfolio_df    – raw CSV as DataFrame
        current_prices  – {ticker: latest_price}
        historical_prices – DataFrame of adjusted Close prices (Date × Ticker)
        ticker_info     – {ticker: {sector, asset_class, ...}}
    """
    portfolio_df = load_portfolio_csv(csv_path)
    tickers = portfolio_df["ticker"].tolist()

    current_prices = fetch_current_prices(tickers)
    historical_prices = fetch_historical_data(tickers, years=history_years)
    ticker_info = fetch_ticker_info(tickers)

    return portfolio_df, current_prices, historical_prices, ticker_info
