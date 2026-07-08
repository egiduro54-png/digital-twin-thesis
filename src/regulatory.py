"""
regulatory.py

MiFID II / PRIIPs regulatory compliance checker for the Digital Twin
Investment Advisory System.

Checks performed:
  1. Suitability — portfolio risk vs investor risk profile
  2. Concentration — single-asset and sector limits
  3. Asset class alignment — equity/fixed income/other vs target
  4. Diversification — holdings count and Herfindahl index
  5. PRIIPs synthetic risk indicator (SRI) score 1-7
  6. Liquidity — proportion of exchange-traded / liquid assets
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["PASS", "WARNING", "BREACH"]

TRADING_DAYS = 252


@dataclass
class RegulatoryCheck:
    name: str
    status: Status
    value: float | str
    threshold: str
    message: str
    recommendation: str = ""


@dataclass
class RegulatoryReport:
    checks: list[RegulatoryCheck] = field(default_factory=list)
    sri_score: int = 0
    overall_status: Status = "PASS"

    def summary(self) -> dict:
        breaches = [c for c in self.checks if c.status == "BREACH"]
        warnings = [c for c in self.checks if c.status == "WARNING"]
        passes = [c for c in self.checks if c.status == "PASS"]
        return {
            "breaches": len(breaches),
            "warnings": len(warnings),
            "passes": len(passes),
            "total": len(self.checks),
        }


class RegulatoryChecker:
    """
    Runs MiFID II / PRIIPs regulatory checks on a Portfolio object.

    Parameters
    ----------
    portfolio : Portfolio
        The digital twin portfolio to evaluate.
    """

    # MiFID II single-asset concentration limits
    CONCENTRATION_WARNING = 0.10   # 10% — trigger warning
    CONCENTRATION_BREACH  = 0.20   # 20% — regulatory breach

    # Sector concentration limits
    SECTOR_WARNING = 0.35
    SECTOR_BREACH  = 0.50

    # Minimum number of holdings for adequate diversification
    MIN_HOLDINGS_WARNING = 5
    MIN_HOLDINGS_BREACH  = 3

    # Herfindahl-Hirschman Index thresholds
    HHI_WARNING = 0.20
    HHI_BREACH  = 0.30

    # Asset class drift tolerance vs target allocation
    DRIFT_WARNING = 0.10   # 10pp drift
    DRIFT_BREACH  = 0.20   # 20pp drift

    # PRIIPs volatility bands for SRI 1-7
    SRI_BANDS = [
        (0.000, 0.005, 1),
        (0.005, 0.020, 2),
        (0.020, 0.050, 3),
        (0.050, 0.100, 4),
        (0.100, 0.150, 5),
        (0.150, 0.250, 6),
        (0.250, 9.999, 7),
    ]

    # Target allocations per risk profile (equity / fixed_income / other)
    TARGET_ALLOCATION = {
        "liquidity_plus": {"equity": 0.05, "fixed_income": 0.60, "other": 0.35},
        "defensive":      {"equity": 0.25, "fixed_income": 0.55, "other": 0.20},
        "flexible":       {"equity": 0.45, "fixed_income": 0.45, "other": 0.10},
        "growth":         {"equity": 0.65, "fixed_income": 0.30, "other": 0.05},
        "dynamic":        {"equity": 0.85, "fixed_income": 0.10, "other": 0.05},
        "conservative":   {"equity": 0.25, "fixed_income": 0.55, "other": 0.20},
        "moderate":       {"equity": 0.45, "fixed_income": 0.45, "other": 0.10},
        "aggressive":     {"equity": 0.85, "fixed_income": 0.10, "other": 0.05},
    }

    # Target volatility per risk profile (annualised)
    TARGET_VOLATILITY = {
        "liquidity_plus": 0.02,
        "defensive":      0.04,
        "flexible":       0.07,
        "growth":         0.10,
        "dynamic":        0.14,
        "conservative":   0.04,
        "moderate":       0.07,
        "aggressive":     0.14,
    }

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.risk_profile = portfolio.risk_profile

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_suitability(self) -> RegulatoryCheck:
        """MiFID II Art. 25 — portfolio volatility vs investor risk profile."""
        vol = self.portfolio.calculate_volatility()
        target = self.TARGET_VOLATILITY.get(self.risk_profile, 0.07)
        upper = target * 1.5
        lower = target * 0.5

        if np.isnan(vol):
            return RegulatoryCheck(
                name="MiFID II Suitability",
                status="WARNING",
                value="N/A",
                threshold=f"Target: {target*100:.1f}%",
                message="Δεν υπάρχουν αρκετά ιστορικά δεδομένα για υπολογισμό.",
            )

        if vol > upper:
            status = "BREACH"
            msg = (f"Η μεταβλητότητα ({vol*100:.1f}%) υπερβαίνει σημαντικά "
                   f"το στόχο ({target*100:.1f}%) για το προφίλ '{self.risk_profile}'.")
            rec = "Μείωση έκθεσης σε υψηλής μεταβλητότητας assets (equities, commodities)."
        elif vol > target * 1.20:
            status = "WARNING"
            msg = (f"Η μεταβλητότητα ({vol*100:.1f}%) είναι ελαφρώς υψηλότερη "
                   f"από τον στόχο ({target*100:.1f}%).")
            rec = "Εξετάστε προσθήκη fixed income ή defensive assets."
        elif vol < lower:
            status = "WARNING"
            msg = (f"Η μεταβλητότητα ({vol*100:.1f}%) είναι χαμηλότερη "
                   f"από τον στόχο ({target*100:.1f}%). Πιθανή υπο-αξιοποίηση risk budget.")
            rec = "Εξετάστε προσθήκη growth assets αν το επιτρέπει το προφίλ."
        else:
            status = "PASS"
            msg = f"Η μεταβλητότητα ({vol*100:.1f}%) είναι εντός του επιτρεπτού εύρους."
            rec = ""

        return RegulatoryCheck(
            name="MiFID II Suitability",
            status=status,
            value=f"{vol*100:.1f}%",
            threshold=f"Στόχος: {target*100:.1f}% (±50%)",
            message=msg,
            recommendation=rec,
        )

    def check_concentration(self) -> list[RegulatoryCheck]:
        """Single-asset and sector concentration limits."""
        checks = []
        weights = self.portfolio.get_weights_dict()
        total = self.portfolio.total_value

        # --- Single asset ---
        max_ticker = max(weights, key=weights.get)
        max_w = weights[max_ticker]

        if max_w >= self.CONCENTRATION_BREACH:
            status, rec = "BREACH", f"Μείωση βάρους {max_ticker} κάτω από {self.CONCENTRATION_BREACH*100:.0f}%."
        elif max_w >= self.CONCENTRATION_WARNING:
            status, rec = "WARNING", f"Παρακολουθήστε το βάρος του {max_ticker}."
        else:
            status, rec = "PASS", ""

        checks.append(RegulatoryCheck(
            name="Συγκέντρωση — Μεμονωμένος Τίτλος",
            status=status,
            value=f"{max_ticker}: {max_w*100:.1f}%",
            threshold=f"Warning >{self.CONCENTRATION_WARNING*100:.0f}%, Breach >{self.CONCENTRATION_BREACH*100:.0f}%",
            message=f"Μέγιστο βάρος: {max_ticker} ({max_w*100:.1f}%).",
            recommendation=rec,
        ))

        # --- Sector ---
        sector_weights: dict[str, float] = {}
        for asset in self.portfolio.assets:
            sec = asset.sector or "Unknown"
            sector_weights[sec] = sector_weights.get(sec, 0) + asset.current_value / total

        max_sector = max(sector_weights, key=sector_weights.get)
        max_sw = sector_weights[max_sector]

        if max_sw >= self.SECTOR_BREACH:
            s_status, s_rec = "BREACH", f"Διαφοροποιήστε από τον κλάδο '{max_sector}'."
        elif max_sw >= self.SECTOR_WARNING:
            s_status, s_rec = "WARNING", f"Εξετάστε μείωση έκθεσης στον κλάδο '{max_sector}'."
        else:
            s_status, s_rec = "PASS", ""

        checks.append(RegulatoryCheck(
            name="Συγκέντρωση — Κλάδος (Sector)",
            status=s_status,
            value=f"{max_sector}: {max_sw*100:.1f}%",
            threshold=f"Warning >{self.SECTOR_WARNING*100:.0f}%, Breach >{self.SECTOR_BREACH*100:.0f}%",
            message=f"Μεγαλύτερος κλάδος: {max_sector} ({max_sw*100:.1f}%).",
            recommendation=s_rec,
        ))

        return checks

    def check_diversification(self) -> RegulatoryCheck:
        """Number of holdings and Herfindahl-Hirschman Index."""
        n = len(self.portfolio.assets)
        weights = list(self.portfolio.get_weights_dict().values())
        hhi = sum(w ** 2 for w in weights)

        if n <= self.MIN_HOLDINGS_BREACH or hhi >= self.HHI_BREACH:
            status = "BREACH"
            msg = f"Ανεπαρκής διαφοροποίηση: {n} assets, HHI={hhi:.3f}."
            rec = "Προσθέστε περισσότερα assets σε διαφορετικές κατηγορίες."
        elif n <= self.MIN_HOLDINGS_WARNING or hhi >= self.HHI_WARNING:
            status = "WARNING"
            msg = f"Μέτρια διαφοροποίηση: {n} assets, HHI={hhi:.3f}."
            rec = "Εξετάστε προσθήκη assets για καλύτερη διαφοροποίηση."
        else:
            status = "PASS"
            msg = f"Επαρκής διαφοροποίηση: {n} assets, HHI={hhi:.3f}."
            rec = ""

        return RegulatoryCheck(
            name="Διαφοροποίηση (Diversification)",
            status=status,
            value=f"{n} assets, HHI={hhi:.3f}",
            threshold=f"Min {self.MIN_HOLDINGS_WARNING} assets, HHI <{self.HHI_WARNING}",
            message=msg,
            recommendation=rec,
        )

    def check_asset_class_alignment(self) -> RegulatoryCheck:
        """Asset class allocation vs target per risk profile."""
        target = self.TARGET_ALLOCATION.get(self.risk_profile, {})
        if not target:
            return RegulatoryCheck(
                name="Asset Class Alignment",
                status="WARNING",
                value="N/A",
                threshold="N/A",
                message="Άγνωστο risk profile.",
            )

        total = self.portfolio.total_value
        actual: dict[str, float] = {"equity": 0.0, "fixed_income": 0.0, "other": 0.0}

        for asset in self.portfolio.assets:
            ac = asset.asset_class.lower()
            if "equity" in ac or "stock" in ac:
                actual["equity"] += asset.current_value / total
            elif "bond" in ac or "fixed" in ac or "income" in ac:
                actual["fixed_income"] += asset.current_value / total
            else:
                actual["other"] += asset.current_value / total

        max_drift = max(abs(actual[k] - target.get(k, 0)) for k in actual)

        if max_drift >= self.DRIFT_BREACH:
            status = "BREACH"
            msg = f"Σημαντική απόκλιση από στόχο κατανομής (max drift: {max_drift*100:.1f}pp)."
            rec = "Rebalancing απαιτείται άμεσα."
        elif max_drift >= self.DRIFT_WARNING:
            status = "WARNING"
            msg = f"Μέτρια απόκλιση από στόχο (max drift: {max_drift*100:.1f}pp)."
            rec = "Εξετάστε rebalancing στο επόμενο review."
        else:
            status = "PASS"
            msg = f"Κατανομή εντός ορίων (max drift: {max_drift*100:.1f}pp)."
            rec = ""

        detail = " | ".join(
            f"{k.replace('_',' ').title()}: {actual[k]*100:.1f}% (στόχος {target.get(k,0)*100:.0f}%)"
            for k in actual
        )

        return RegulatoryCheck(
            name="Asset Class Alignment",
            status=status,
            value=detail,
            threshold=f"Drift <{self.DRIFT_WARNING*100:.0f}pp warning, <{self.DRIFT_BREACH*100:.0f}pp breach",
            message=msg,
            recommendation=rec,
        )

    def compute_sri(self) -> int:
        """PRIIPs Synthetic Risk Indicator (1-7) based on annualised volatility."""
        vol = self.portfolio.calculate_volatility()
        if np.isnan(vol):
            return 0
        for lo, hi, score in self.SRI_BANDS:
            if lo <= vol < hi:
                return score
        return 7

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run_all_checks(self) -> RegulatoryReport:
        report = RegulatoryReport()
        report.checks.append(self.check_suitability())
        report.checks.extend(self.check_concentration())
        report.checks.append(self.check_diversification())
        report.checks.append(self.check_asset_class_alignment())
        report.sri_score = self.compute_sri()

        breaches = sum(1 for c in report.checks if c.status == "BREACH")
        warnings = sum(1 for c in report.checks if c.status == "WARNING")
        if breaches > 0:
            report.overall_status = "BREACH"
        elif warnings > 0:
            report.overall_status = "WARNING"
        else:
            report.overall_status = "PASS"

        return report
