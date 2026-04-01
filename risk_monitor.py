"""
risk_monitor.py

Continuously monitors a Portfolio for six categories of risk:
  1. Concentration risk       – single asset or sector too large
  2. Diversification          – too few holdings, limited sector/geo spread
  3. Volatility vs. target    – actual volatility exceeds/falls short of target
  4. Drawdown risk            – historical or stress-test drawdown too severe
  5. Correlation risk         – assets move together, eroding diversification
  6. Profile drift            – portfolio composition no longer matches investor profile

Each check returns a list of Alert dicts:
  {
    'category': str,
    'severity': 'ok' | 'caution' | 'alert',
    'title': str,
    'description': str,
    'detail': dict,   <- raw numbers for dashboard display
  }
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import Portfolio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default risk thresholds (can be overridden at construction time)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    # Concentration
    "concentration_caution_pct": 20.0,      # single asset > 20% → caution
    "concentration_alert_pct": 25.0,        # single asset > 25% → alert
    "top5_caution_pct": 60.0,               # top-5 > 60% → caution
    "top5_alert_pct": 70.0,                 # top-5 > 70% → alert
    "herfindahl_caution": 0.18,
    "herfindahl_alert": 0.25,
    # Diversification
    "min_holdings_caution": 10,
    "min_holdings_alert": 5,
    "max_single_sector_caution_pct": 50.0,
    "max_single_sector_alert_pct": 65.0,
    "international_min_pct": 5.0,           # at least 5% international
    # Volatility
    "vol_deviation_caution_pct": 2.0,       # actual > target + 2% → caution
    "vol_deviation_alert_pct": 4.0,         # actual > target + 4% → alert
    "vol_under_caution_pct": -3.0,          # actual < target - 3% → caution
    # Drawdown
    "max_drawdown_caution_pct": -15.0,
    "max_drawdown_alert_pct": -25.0,
    "var_caution_pct": -10.0,
    "var_alert_pct": -15.0,
    # Correlation
    "high_corr_threshold": 0.80,            # pair is "highly correlated"
    "corr_pairs_caution_ratio": 0.30,       # > 30% pairs highly correlated → caution
    "corr_pairs_alert_ratio": 0.50,
    # Profile drift
    "drift_caution_pct": 5.0,
    "drift_alert_pct": 10.0,
}


def _severity(value: float, caution: float, alert: float,
              higher_is_worse: bool = True) -> str:
    """
    Map a numeric value to 'ok' / 'caution' / 'alert'.

    higher_is_worse=True  → alert when value > alert threshold
    higher_is_worse=False → alert when value < alert threshold
    """
    if higher_is_worse:
        if value >= alert:
            return "alert"
        if value >= caution:
            return "caution"
        return "ok"
    else:
        if value <= alert:
            return "alert"
        if value <= caution:
            return "caution"
        return "ok"


class RiskMonitor:
    """
    Runs six independent risk checks on a Portfolio.

    Usage
    -----
    monitor = RiskMonitor(portfolio, thresholds={...})  # thresholds optional
    results = monitor.run_full_analysis()
    # results = {'ok': [...], 'caution': [...], 'alert': [...], 'all': [...]}
    """

    def __init__(self, portfolio: Portfolio, thresholds: dict | None = None):
        self.portfolio = portfolio
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ------------------------------------------------------------------
    # 1. Concentration Risk
    # ------------------------------------------------------------------

    def check_concentration(self) -> list[dict]:
        alerts = []
        composition = self.portfolio.get_composition()
        conc = composition.get("concentration", {})

        largest = conc.get("largest_single_asset_pct", 0.0)
        top5 = conc.get("top_5_assets_pct", 0.0)
        hhi = conc.get("herfindahl_index", 0.0)

        # Find which ticker is the largest
        weights = self.portfolio.get_weights()
        tickers = self.portfolio.tickers
        if len(weights) > 0:
            largest_idx = int(np.argmax(weights))
            largest_ticker = tickers[largest_idx]
        else:
            largest_ticker = "N/A"

        # Single-asset concentration
        sev = _severity(
            largest,
            self.thresholds["concentration_caution_pct"],
            self.thresholds["concentration_alert_pct"],
        )
        alerts.append({
            "category": "Συγκέντρωση (Concentration)",
            "severity": sev,
            "title": "Συγκέντρωση σε Μεμονωμένο Τίτλο",
            "description": (
                f"Μεγαλύτερη θέση ({largest_ticker}): {largest:.1f}% του χαρτοφυλακίου. "
                f"Όρια: προσοχή >{self.thresholds['concentration_caution_pct']}%, "
                f"προειδοποίηση >{self.thresholds['concentration_alert_pct']}%."
            ),
            "recommendation": (
                f"Μειώστε το {largest_ticker} κάτω από "
                f"{self.thresholds['concentration_caution_pct']}% για μείωση του κινδύνου συγκέντρωσης."
                if sev != "ok" else "Η συγκέντρωση είναι εντός ασφαλών ορίων."
            ),
            "detail": {
                "largest_ticker": largest_ticker,
                "largest_pct": largest,
                "top5_pct": top5,
                "herfindahl_index": hhi,
            },
        })

        # Top-5 concentration
        sev5 = _severity(
            top5,
            self.thresholds["top5_caution_pct"],
            self.thresholds["top5_alert_pct"],
        )
        alerts.append({
            "category": "Συγκέντρωση (Concentration)",
            "severity": sev5,
            "title": "Συγκέντρωση στους 5 Κορυφαίους Τίτλους",
            "description": (
                f"Οι 5 μεγαλύτερες θέσεις καλύπτουν {top5:.1f}% του χαρτοφυλακίου. "
                f"Όριο προσοχής: >{self.thresholds['top5_caution_pct']}%."
            ),
            "recommendation": (
                "Κατανείμετε τα κεφάλαια σε περισσότερες θέσεις για μείωση εξάρτησης από τους top 5."
                if sev5 != "ok" else "Η συγκέντρωση των top-5 είναι εντός ασφαλών ορίων."
            ),
            "detail": {"top5_pct": top5},
        })

        return alerts

    # ------------------------------------------------------------------
    # 2. Diversification
    # ------------------------------------------------------------------

    def check_diversification(self) -> list[dict]:
        alerts = []
        composition = self.portfolio.get_composition()
        n = len(self.portfolio.assets)

        # Number of holdings
        sev_n = _severity(
            n,
            self.thresholds["min_holdings_caution"],
            self.thresholds["min_holdings_alert"],
            higher_is_worse=False,
        )
        alerts.append({
            "category": "Διαφοροποίηση (Diversification)",
            "severity": sev_n,
            "title": "Αριθμός Θέσεων",
            "description": (
                f"Το χαρτοφυλάκιο έχει {n} θέσεις. "
                f"Ελάχιστο συνιστώμενο: {self.thresholds['min_holdings_caution']}."
            ),
            "recommendation": (
                "Προσθέστε περισσότερες θέσεις για μείωση του κινδύνου από μεμονωμένο τίτλο."
                if sev_n != "ok" else "Το χαρτοφυλάκιο έχει επαρκή αριθμό θέσεων."
            ),
            "detail": {"num_holdings": n},
        })

        # Single-sector concentration
        by_sector = composition.get("by_sector", {})
        if by_sector:
            top_sector, top_sector_pct = max(by_sector.items(), key=lambda x: x[1])
            sev_sec = _severity(
                top_sector_pct,
                self.thresholds["max_single_sector_caution_pct"],
                self.thresholds["max_single_sector_alert_pct"],
            )
            alerts.append({
                "category": "Διαφοροποίηση (Diversification)",
                "severity": sev_sec,
                "title": "Συγκέντρωση σε Κλάδο (Sector)",
                "description": (
                    f"Ο κλάδος '{top_sector}' αντιπροσωπεύει {top_sector_pct:.1f}% του χαρτοφυλακίου. "
                    f"Όριο προσοχής: >{self.thresholds['max_single_sector_caution_pct']}%."
                ),
                "recommendation": (
                    f"Μειώστε την έκθεση στον κλάδο '{top_sector}' και διαφοροποιήστε σε άλλους κλάδους."
                    if sev_sec != "ok" else "Η κατανομή ανά κλάδο είναι ισορροπημένη."
                ),
                "detail": {"top_sector": top_sector, "top_sector_pct": top_sector_pct,
                           "all_sectors": by_sector},
            })

        # International exposure
        by_class = composition.get("by_asset_class", {})
        intl_pct = sum(v for k, v in by_class.items()
                       if "international" in k.lower() or "emerging" in k.lower())
        sev_intl = _severity(
            intl_pct,
            self.thresholds["international_min_pct"],
            0.0,
            higher_is_worse=False,
        )
        alerts.append({
            "category": "Διαφοροποίηση (Diversification)",
            "severity": sev_intl,
            "title": "Γεωγραφική Διαφοροποίηση",
            "description": (
                f"Διεθνής έκθεση: {intl_pct:.1f}%. "
                f"Ελάχιστο συνιστώμενο: {self.thresholds['international_min_pct']}%."
            ),
            "recommendation": (
                "Προσθέστε διεθνές ETF (π.χ. VXUS, EFA) για γεωγραφική διαφοροποίηση."
                if sev_intl != "ok" else "Το χαρτοφυλάκιο έχει διεθνή έκθεση."
            ),
            "detail": {"international_pct": intl_pct, "by_asset_class": by_class},
        })

        return alerts

    # ------------------------------------------------------------------
    # 3. Volatility vs. Target
    # ------------------------------------------------------------------

    def check_volatility(self) -> list[dict]:
        metrics = self.portfolio.get_metrics()
        vol = metrics.get("volatility_annual_pct")
        target = metrics.get("target_volatility_pct", 12.0)

        if vol is None:
            return [{
                "category": "Μεταβλητότητα (Volatility)",
                "severity": "ok",
                "title": "Μεταβλητότητα (Volatility)",
                "description": "Ανεπαρκή ιστορικά δεδομένα για υπολογισμό μεταβλητότητας.",
                "recommendation": "Φορτώστε τουλάχιστον 6 μήνες ιστορικών δεδομένων.",
                "detail": {},
            }]

        deviation = vol - target
        rp_label = {"conservative": "συντηρητικό", "moderate": "μέτριο", "aggressive": "επιθετικό"}.get(
            self.portfolio.risk_profile, self.portfolio.risk_profile)

        if deviation > 0:
            sev = _severity(
                deviation,
                self.thresholds["vol_deviation_caution_pct"],
                self.thresholds["vol_deviation_alert_pct"],
            )
            direction = "πάνω από τον στόχο"
        else:
            sev = _severity(
                deviation,
                self.thresholds["vol_under_caution_pct"],
                -999,
                higher_is_worse=False,
            )
            direction = "κάτω από τον στόχο"

        return [{
            "category": "Μεταβλητότητα (Volatility)",
            "severity": sev,
            "title": "Volatility έναντι Στόχου",
            "description": (
                f"Ετήσια μεταβλητότητα χαρτοφυλακίου: {vol:.1f}%. "
                f"Στόχος για '{rp_label}' προφίλ: {target:.1f}%. "
                f"Απόκλιση: {deviation:+.1f}% ({direction})."
            ),
            "recommendation": (
                "Μεταφέρετε κεφάλαια από μετοχές σε ομόλογα/σταθερά assets για μείωση μεταβλητότητας."
                if deviation > self.thresholds["vol_deviation_caution_pct"]
                else "Σκεφτείτε προσθήκη assets ανάπτυξης για αξιοποίηση του risk budget."
                if deviation < self.thresholds["vol_under_caution_pct"]
                else "Η μεταβλητότητα είναι εντός στόχου."
            ),
            "detail": {
                "current_vol_pct": vol,
                "target_vol_pct": target,
                "deviation_pct": round(deviation, 2),
            },
        }]

    # ------------------------------------------------------------------
    # 4. Drawdown Risk
    # ------------------------------------------------------------------

    def check_drawdown(self) -> list[dict]:
        metrics = self.portfolio.get_metrics()
        max_dd = metrics.get("max_drawdown_pct")     # already in pct (negative)
        var_pct = metrics.get("var_95_monthly_pct")  # monthly VaR (negative)

        alerts = []

        if max_dd is not None:
            sev_dd = _severity(
                max_dd,
                self.thresholds["max_drawdown_caution_pct"],
                self.thresholds["max_drawdown_alert_pct"],
                higher_is_worse=False,  # more negative = worse
            )
            alerts.append({
                "category": "Drawdown",
                "severity": sev_dd,
                "title": "Μέγιστη Ιστορική Απώλεια (Max Drawdown)",
                "description": (
                    f"Χειρότερη ιστορική απώλεια από κορυφή σε κατώτατο: {max_dd:.1f}%. "
                    f"Όριο προσοχής: <{self.thresholds['max_drawdown_caution_pct']}%."
                ),
                "recommendation": (
                    "Σκεφτείτε προσθήκη αμυντικών θέσεων (ομόλογα, χρυσός) για προστασία από μελλοντικές απώλειες."
                    if sev_dd != "ok" else "Η ιστορική μέγιστη απώλεια είναι εντός αποδεκτών ορίων."
                ),
                "detail": {"max_drawdown_pct": max_dd},
            })

        if var_pct is not None:
            sev_var = _severity(
                var_pct,
                self.thresholds["var_caution_pct"],
                self.thresholds["var_alert_pct"],
                higher_is_worse=False,
            )
            alerts.append({
                "category": "Drawdown",
                "severity": sev_var,
                "title": "Value-at-Risk (VaR 95%, μηνιαίο)",
                "description": (
                    f"Στο 5% των μηνών, το χαρτοφυλάκιο αναμένεται να χάσει "
                    f"τουλάχιστον {abs(var_pct):.1f}% της αξίας του."
                ),
                "recommendation": (
                    "Υψηλός κίνδυνος ουράς. Μειώστε επιθετικές μετοχικές θέσεις ή προσθέστε αντισταθμιστικές."
                    if sev_var != "ok" else "Ο κίνδυνος ουράς (tail risk) είναι εντός αποδεκτών ορίων."
                ),
                "detail": {"var_95_monthly_pct": var_pct},
            })

        return alerts

    # ------------------------------------------------------------------
    # 5. Correlation Risk
    # ------------------------------------------------------------------

    def check_correlation(self) -> list[dict]:
        corr = self.portfolio.calculate_correlation_matrix()
        if corr.empty or corr.shape[0] < 2:
            return [{
                "category": "Συσχέτιση (Correlation)",
                "severity": "ok",
                "title": "Συσχέτιση Τίτλων (Asset Correlation)",
                "description": "Απαιτούνται τουλάχιστον 2 τίτλοι με ιστορικό για υπολογισμό συσχέτισης.",
                "recommendation": "Προσθέστε περισσότερες θέσεις για ανάλυση συσχέτισης.",
                "detail": {},
            }]

        n = corr.shape[0]
        total_pairs = n * (n - 1) / 2
        high_corr_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                c = corr.iloc[i, j]
                if not np.isnan(c) and abs(c) >= self.thresholds["high_corr_threshold"]:
                    high_corr_pairs.append({
                        "asset_a": corr.index[i],
                        "asset_b": corr.columns[j],
                        "correlation": round(float(c), 3),
                    })

        ratio = len(high_corr_pairs) / total_pairs if total_pairs > 0 else 0.0

        sev = _severity(
            ratio,
            self.thresholds["corr_pairs_caution_ratio"],
            self.thresholds["corr_pairs_alert_ratio"],
        )

        avg_corr = float(np.nanmean([corr.iloc[i, j]
                                     for i in range(n) for j in range(i + 1, n)]))

        return [{
            "category": "Συσχέτιση (Correlation)",
            "severity": sev,
            "title": "Συσχέτιση Τίτλων (Asset Correlation)",
            "description": (
                f"{len(high_corr_pairs)} από {int(total_pairs)} ζεύγη τίτλων έχουν "
                f"συσχέτιση ≥ {self.thresholds['high_corr_threshold']:.2f}. "
                f"Μέση συσχέτιση ζευγών: {avg_corr:.2f}."
            ),
            "recommendation": (
                "Πολλοί τίτλοι κινούνται μαζί — το όφελος διαφοροποίησης μειώνεται. "
                "Εξετάστε τίτλους από μη συσχετισμένους κλάδους ή κατηγορίες."
                if sev != "ok"
                else "Οι συσχετίσεις των τίτλων είναι σε υγιή επίπεδα."
            ),
            "detail": {
                "high_corr_pairs": high_corr_pairs[:10],  # top 10 for display
                "high_corr_pair_count": len(high_corr_pairs),
                "total_pairs": int(total_pairs),
                "avg_correlation": round(avg_corr, 3),
            },
        }]

    # ------------------------------------------------------------------
    # 6. Profile Drift
    # ------------------------------------------------------------------

    def check_drift(self) -> list[dict]:
        alignment = self.portfolio.get_risk_profile_alignment()
        max_drift = alignment.get("max_drift", 0.0)
        eq_drift = alignment.get("equity_drift_pct", 0.0)
        fi_drift = alignment.get("fixed_income_drift_pct", 0.0)

        sev = _severity(
            max_drift,
            self.thresholds["drift_caution_pct"],
            self.thresholds["drift_alert_pct"],
        )

        rp_label = {"conservative": "Συντηρητικό", "moderate": "Μέτριο", "aggressive": "Επιθετικό"}.get(
            alignment["risk_profile"], alignment["risk_profile"])
        if eq_drift > 0:
            direction = "υπερβολικές μετοχές (πιο επιθετικό από τον στόχο)"
        elif eq_drift < 0:
            direction = "ανεπαρκείς μετοχές (πιο συντηρητικό από τον στόχο)"
        else:
            direction = "εναρμονισμένο"

        return [{
            "category": "Απόκλιση Προφίλ (Profile Drift)",
            "severity": sev,
            "title": "Εναρμόνιση με Προφίλ Κινδύνου",
            "description": (
                f"Προφίλ επενδυτή: '{rp_label}'. "
                f"Τρέχουσες μετοχές: {alignment['current_equity_pct']:.1f}% "
                f"(στόχος: {alignment['target_equity_pct']:.1f}%). "
                f"Απόκλιση: {eq_drift:+.1f}% — {direction}."
            ),
            "recommendation": (
                "Αναδιαρθρώστε το χαρτοφυλάκιο για εναρμόνιση με το στόχο προφίλ κινδύνου."
                if sev != "ok" else "Το χαρτοφυλάκιο είναι εναρμονισμένο με το προφίλ επενδυτή."
            ),
            "detail": {**alignment},
        }]

    # ------------------------------------------------------------------
    # Full Analysis
    # ------------------------------------------------------------------

    def run_full_analysis(self) -> dict[str, Any]:
        """
        Run all six risk checks.

        Returns
        -------
        {
            'all': [list of all alerts],
            'ok': [...],
            'caution': [...],
            'alert': [...],
            'summary': {'ok': n, 'caution': n, 'alert': n, 'total': n},
            'highest_severity': 'ok' | 'caution' | 'alert',
        }
        """
        all_alerts = []
        all_alerts.extend(self.check_concentration())
        all_alerts.extend(self.check_diversification())
        all_alerts.extend(self.check_volatility())
        all_alerts.extend(self.check_drawdown())
        all_alerts.extend(self.check_correlation())
        all_alerts.extend(self.check_drift())

        bucketed: dict[str, list] = {"ok": [], "caution": [], "alert": []}
        for alert in all_alerts:
            bucketed[alert["severity"]].append(alert)

        highest = "ok"
        if bucketed["caution"]:
            highest = "caution"
        if bucketed["alert"]:
            highest = "alert"

        return {
            "all": all_alerts,
            "ok": bucketed["ok"],
            "caution": bucketed["caution"],
            "alert": bucketed["alert"],
            "summary": {
                "ok": len(bucketed["ok"]),
                "caution": len(bucketed["caution"]),
                "alert": len(bucketed["alert"]),
                "total": len(all_alerts),
            },
            "highest_severity": highest,
        }
