"""
scenario_engine.py

Simulates market scenarios on a Portfolio (Digital Twin).

Supports:
  - Standard pre-defined scenarios (market crash, rate hike, etc.)
  - Custom user-defined scenarios
  - Comparison between multiple scenarios
  - Per-asset impact breakdown
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import Portfolio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in scenario definitions (mirrors scenarios_config.json)
# These are used as fallback when config file is unavailable.
# ---------------------------------------------------------------------------

STANDARD_SCENARIOS: dict[str, dict] = {
    "market_down_5": {
        "name": "Market Down 5%",
        "description": "Mild market correction of 5%.",
        "category": "Market Movement",
        "market_change": -0.05,
        "rate_change": 0.0,
        "volatility_multiplier": 1.1,
        "sector_overrides": {},
    },
    "market_down_10": {
        "name": "Market Down 10%",
        "description": "Moderate market pullback of 10%.",
        "category": "Market Movement",
        "market_change": -0.10,
        "rate_change": 0.0,
        "volatility_multiplier": 1.2,
        "sector_overrides": {},
    },
    "market_down_20": {
        "name": "Market Down 20%",
        "description": "Significant bear market decline of 20%.",
        "category": "Market Movement",
        "market_change": -0.20,
        "rate_change": 0.0,
        "volatility_multiplier": 1.5,
        "sector_overrides": {},
    },
    "market_down_30": {
        "name": "Market Down 30%",
        "description": "Severe bear market decline of 30%.",
        "category": "Market Movement",
        "market_change": -0.30,
        "rate_change": 0.0,
        "volatility_multiplier": 2.0,
        "sector_overrides": {},
    },
    "market_up_10": {
        "name": "Market Up 10%",
        "description": "Bull market rally of 10%.",
        "category": "Market Movement",
        "market_change": 0.10,
        "rate_change": 0.0,
        "volatility_multiplier": 0.9,
        "sector_overrides": {},
    },
    "market_up_20": {
        "name": "Market Up 20%",
        "description": "Strong bull market rally of 20%.",
        "category": "Market Movement",
        "market_change": 0.20,
        "rate_change": 0.0,
        "volatility_multiplier": 0.8,
        "sector_overrides": {},
    },
    "rates_up_100bps": {
        "name": "Interest Rates +1%",
        "description": "Central bank raises rates by 100 basis points.",
        "category": "Interest Rate",
        "market_change": -0.05,
        "rate_change": 0.01,
        "volatility_multiplier": 1.1,
        "sector_overrides": {
            "Technology": -0.08,
            "Real Estate": -0.12,
            "Utilities": -0.08,
        },
    },
    "rates_up_200bps": {
        "name": "Interest Rates +2%",
        "description": "Aggressive rate-hike cycle of 200 basis points.",
        "category": "Interest Rate",
        "market_change": -0.10,
        "rate_change": 0.02,
        "volatility_multiplier": 1.3,
        "sector_overrides": {
            "Technology": -0.15,
            "Real Estate": -0.20,
            "Utilities": -0.15,
            "Financials": 0.05,
        },
    },
    "rates_down_50bps": {
        "name": "Interest Rates -0.5%",
        "description": "Moderate rate cut of 50 basis points.",
        "category": "Interest Rate",
        "market_change": 0.03,
        "rate_change": -0.005,
        "volatility_multiplier": 0.95,
        "sector_overrides": {
            "Real Estate": 0.05,
            "Utilities": 0.04,
        },
    },
    "tech_crash": {
        "name": "Tech Sector Crash",
        "description": "Technology sector crashes 25%, dragging market down 10%.",
        "category": "Sector Shock",
        "market_change": -0.10,
        "rate_change": 0.0,
        "volatility_multiplier": 1.4,
        "sector_overrides": {
            "Technology": -0.25,
            "Communication Services": -0.15,
        },
    },
    "financial_crisis_2008": {
        "name": "2008 Financial Crisis",
        "description": "Replay of 2008: market -40%, rates up 2%, extreme volatility.",
        "category": "Stress Test",
        "market_change": -0.40,
        "rate_change": 0.02,
        "volatility_multiplier": 4.0,
        "sector_overrides": {
            "Financials": -0.55,
            "Real Estate": -0.50,
            "Technology": -0.35,
            "Energy": -0.40,
        },
    },
    "covid_crash_2020": {
        "name": "2020 COVID Crash",
        "description": "Rapid pandemic-driven crash as seen in Q1 2020.",
        "category": "Stress Test",
        "market_change": -0.30,
        "rate_change": -0.015,
        "volatility_multiplier": 3.0,
        "sector_overrides": {
            "Energy": -0.50,
            "Real Estate": -0.30,
            "Consumer Discretionary": -0.35,
            "Technology": 0.05,
            "Healthcare": 0.05,
        },
    },
    "recession": {
        "name": "Economic Recession",
        "description": "Prolonged economic downturn with rising unemployment.",
        "category": "Stress Test",
        "market_change": -0.20,
        "rate_change": -0.01,
        "volatility_multiplier": 1.8,
        "sector_overrides": {
            "Consumer Discretionary": -0.25,
            "Industrials": -0.20,
            "Financials": -0.15,
            "Consumer Staples": 0.05,
            "Healthcare": 0.02,
        },
    },
    "inflation_spike": {
        "name": "Inflation Spike",
        "description": "High inflation forces aggressive rate hikes.",
        "category": "Stress Test",
        "market_change": -0.10,
        "rate_change": 0.025,
        "volatility_multiplier": 1.5,
        "sector_overrides": {
            "Technology": -0.20,
            "Consumer Discretionary": -0.15,
            "Energy": 0.15,
            "Materials": 0.10,
            "Utilities": -0.12,
        },
    },
}


class ScenarioEngine:
    """
    Simulates market scenarios on the portfolio Digital Twin.

    Parameters
    ----------
    portfolio : Portfolio
        The base (current) portfolio to simulate against.
    custom_scenarios : dict | None
        Additional scenario definitions loaded from config file.
    """

    def __init__(self, portfolio: Portfolio, custom_scenarios: dict | None = None):
        self.portfolio = portfolio
        # Merge built-in + config-file scenarios
        self.scenarios = {**STANDARD_SCENARIOS, **(custom_scenarios or {})}

    # ------------------------------------------------------------------
    # Scenario Access
    # ------------------------------------------------------------------

    def list_scenarios(self) -> list[dict]:
        """Return scenario metadata (name, description, category) for all scenarios."""
        return [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "category": v.get("category", "Other"),
            }
            for k, v in self.scenarios.items()
        ]

    def get_scenario_definition(self, scenario_id: str) -> dict:
        if scenario_id not in self.scenarios:
            raise KeyError(f"Unknown scenario: '{scenario_id}'. "
                           f"Available: {list(self.scenarios.keys())}")
        return self.scenarios[scenario_id]

    def create_custom_scenario(
        self,
        name: str,
        market_change: float = 0.0,
        rate_change: float = 0.0,
        volatility_multiplier: float = 1.0,
        sector_overrides: dict | None = None,
        description: str = "",
    ) -> dict:
        """
        Build a custom scenario dict without saving it permanently.

        Parameters
        ----------
        market_change : fraction (e.g. -0.20 = market drops 20%)
        rate_change   : fraction (e.g. 0.01 = rates up 1%)
        volatility_multiplier : how much to scale volatility (1.0 = no change)
        sector_overrides : {sector_name: additional_fraction_change}
        """
        scenario = {
            "name": name,
            "description": description or f"Custom scenario: {name}",
            "category": "Custom",
            "market_change": market_change,
            "rate_change": rate_change,
            "volatility_multiplier": volatility_multiplier,
            "sector_overrides": sector_overrides or {},
        }
        # Register for future use in this session
        safe_id = name.lower().replace(" ", "_").replace("-", "_")[:30]
        self.scenarios[safe_id] = scenario
        return scenario

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_scenario(self, scenario_id: str) -> Portfolio:
        """
        Apply a named scenario to the portfolio and return a scenario Portfolio.
        """
        scenario_def = self.get_scenario_definition(scenario_id)
        return self.portfolio.apply_scenario(scenario_def)

    def simulate_custom(self, scenario: dict) -> Portfolio:
        """Apply a raw scenario dict (e.g., from create_custom_scenario)."""
        return self.portfolio.apply_scenario(scenario)

    # ------------------------------------------------------------------
    # Per-Asset Impact
    # ------------------------------------------------------------------

    def get_asset_impact(self, scenario_id: str) -> list[dict]:
        """
        Break down how each asset changes under the scenario.

        Returns a list of dicts sorted by absolute dollar impact.
        """
        scenario_portfolio = self.simulate_scenario(scenario_id)
        scenario_def = self.get_scenario_definition(scenario_id)

        results = []
        for orig, scen in zip(self.portfolio.assets, scenario_portfolio.assets):
            dollar_change = scen.current_value - orig.current_value
            pct_change = (
                (scen.current_price / orig.current_price - 1) * 100
                if orig.current_price > 0 else 0.0
            )
            results.append({
                "ticker": orig.ticker,
                "name": orig.name or orig.ticker,
                "sector": orig.sector,
                "asset_class": orig.asset_class,
                "current_price": round(orig.current_price, 2),
                "scenario_price": round(scen.current_price, 2),
                "current_value": round(orig.current_value, 2),
                "scenario_value": round(scen.current_value, 2),
                "dollar_change": round(dollar_change, 2),
                "pct_change": round(pct_change, 2),
            })

        return sorted(results, key=lambda x: x["dollar_change"])

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_portfolio_metrics(self, scenario_id: str) -> dict:
        """
        Side-by-side metric comparison: current portfolio vs. scenario.

        Returns a structured dict suitable for rendering a comparison table.
        """
        scenario_portfolio = self.simulate_scenario(scenario_id)
        scenario_def = self.get_scenario_definition(scenario_id)

        current_m = self.portfolio.get_metrics()
        scenario_m = scenario_portfolio.get_metrics()

        def _diff(key: str, pct: bool = False) -> dict:
            curr = current_m.get(key)
            scen = scenario_m.get(key)
            if curr is None or scen is None:
                return {"current": curr, "scenario": scen, "change": None,
                        "change_pct": None}
            change = scen - curr
            change_pct = (change / abs(curr) * 100) if curr != 0 else None
            return {
                "current": round(curr, 2),
                "scenario": round(scen, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
            }

        # Resilience: how much better than raw market change?
        market_change_pct = scenario_def.get("market_change", 0.0) * 100
        port_change_pct = (
            (scenario_portfolio.total_value / self.portfolio.total_value - 1) * 100
            if self.portfolio.total_value > 0 else 0.0
        )
        resilience_pct = port_change_pct - market_change_pct  # positive = better than market

        return {
            "scenario_name": scenario_def["name"],
            "scenario_description": scenario_def["description"],
            "scenario_params": {
                "market_change_pct": round(market_change_pct, 1),
                "rate_change_pct": round(scenario_def.get("rate_change", 0.0) * 100, 2),
                "volatility_multiplier": scenario_def.get("volatility_multiplier", 1.0),
                "sector_overrides": scenario_def.get("sector_overrides", {}),
            },
            "metrics": {
                "total_value": _diff("total_value"),
                "volatility_annual_pct": _diff("volatility_annual_pct"),
                "sharpe_ratio": _diff("sharpe_ratio"),
                "max_drawdown_pct": _diff("max_drawdown_pct"),
                "beta": _diff("beta"),
                "var_95_monthly_pct": _diff("var_95_monthly_pct"),
            },
            "summary": {
                "current_total_value": round(self.portfolio.total_value, 2),
                "scenario_total_value": round(scenario_portfolio.total_value, 2),
                "portfolio_change_pct": round(port_change_pct, 2),
                "market_change_pct": round(market_change_pct, 1),
                "resilience_vs_market_pct": round(resilience_pct, 2),
            },
        }

    def compare_multiple_scenarios(
        self, scenario_ids: list[str]
    ) -> dict[str, Any]:
        """
        Compare the current portfolio against several scenarios simultaneously.

        Returns a summary table keyed by scenario_id.
        """
        results = {
            "base": {
                "name": "Current Portfolio",
                "total_value": round(self.portfolio.total_value, 2),
                "volatility_pct": self.portfolio.get_metrics().get("volatility_annual_pct"),
                "sharpe_ratio": self.portfolio.get_metrics().get("sharpe_ratio"),
                "max_drawdown_pct": self.portfolio.get_metrics().get("max_drawdown_pct"),
                "change_pct": 0.0,
            }
        }

        for sid in scenario_ids:
            try:
                comp = self.compare_portfolio_metrics(sid)
                results[sid] = {
                    "name": comp["scenario_name"],
                    "total_value": comp["summary"]["scenario_total_value"],
                    "volatility_pct": comp["metrics"]["volatility_annual_pct"]["scenario"],
                    "sharpe_ratio": comp["metrics"]["sharpe_ratio"]["scenario"],
                    "max_drawdown_pct": comp["metrics"]["max_drawdown_pct"]["scenario"],
                    "change_pct": comp["summary"]["portfolio_change_pct"],
                    "resilience_vs_market_pct": comp["summary"]["resilience_vs_market_pct"],
                }
            except Exception as exc:
                logger.warning("Failed to simulate scenario '%s': %s", sid, exc)

        return results

    # ------------------------------------------------------------------
    # Convenience: worst-case scenario scan
    # ------------------------------------------------------------------

    def find_worst_scenario(self, scenario_ids: list[str] | None = None) -> dict:
        """
        Run all (or given) scenarios and return the one causing maximum portfolio loss.
        """
        ids = scenario_ids or list(self.scenarios.keys())
        worst_id = None
        worst_change = float("inf")

        for sid in ids:
            try:
                scen_p = self.simulate_scenario(sid)
                change = scen_p.total_value - self.portfolio.total_value
                if change < worst_change:
                    worst_change = change
                    worst_id = sid
            except Exception:
                pass

        if worst_id is None:
            return {}

        return {
            "scenario_id": worst_id,
            "scenario_name": self.scenarios[worst_id]["name"],
            "portfolio_loss": round(worst_change, 2),
            "portfolio_loss_pct": round(
                worst_change / self.portfolio.total_value * 100
                if self.portfolio.total_value > 0 else 0.0,
                2,
            ),
        }
