"""
Unit tests for data_loader.py — tests that don't require network access.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data_loader import (
    load_portfolio_csv,
    calculate_returns,
    load_scenarios_config,
)


# ---------------------------------------------------------------------------
# load_portfolio_csv
# ---------------------------------------------------------------------------

class TestLoadPortfolioCSV:
    def _make_csv(self, content: str):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_valid_csv_loads(self):
        path = self._make_csv(
            "ticker,quantity,entry_price\n"
            "AAPL,100,150.0\n"
            "MSFT,50,300.0\n"
        )
        df = load_portfolio_csv(path)
        os.unlink(path)
        assert len(df) == 2
        assert "ticker" in df.columns
        assert df["ticker"].tolist() == ["AAPL", "MSFT"]

    def test_missing_columns_raises(self):
        path = self._make_csv("ticker,qty\nAAPL,100\n")
        with pytest.raises((ValueError, KeyError)):
            load_portfolio_csv(path)
        os.unlink(path)

    def test_ticker_uppercased(self):
        path = self._make_csv("ticker,quantity,entry_price\naapl,100,150\n")
        df = load_portfolio_csv(path)
        os.unlink(path)
        assert df["ticker"].iloc[0] == "AAPL"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_portfolio_csv("/nonexistent/path/portfolio.csv")

    def test_zero_quantity_raises(self):
        path = self._make_csv("ticker,quantity,entry_price\nAAPL,0,150\n")
        with pytest.raises(ValueError):
            load_portfolio_csv(path)
        os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_returns
# ---------------------------------------------------------------------------

class TestCalculateReturns:
    def test_simple_returns(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 105.0]})
        ret = calculate_returns(prices, method="simple")
        assert len(ret) == 2
        assert ret["A"].iloc[0] == pytest.approx(0.10, abs=0.001)

    def test_log_returns(self):
        prices = pd.DataFrame({"A": [100.0, 110.0]})
        ret = calculate_returns(prices, method="log")
        import numpy as np
        expected = float(np.log(110 / 100))
        assert ret["A"].iloc[0] == pytest.approx(expected, abs=0.001)

    def test_returns_drop_first_row_nan(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 120.0]})
        ret = calculate_returns(prices)
        assert len(ret) == 2  # first row dropped


# ---------------------------------------------------------------------------
# Scenarios Config
# ---------------------------------------------------------------------------

class TestScenariosConfig:
    def test_load_scenarios_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "scenarios_config.json"
        )
        if not os.path.exists(config_path):
            pytest.skip("scenarios_config.json not found")
        config = load_scenarios_config(config_path)
        assert "scenarios" in config
        assert len(config["scenarios"]) > 0
