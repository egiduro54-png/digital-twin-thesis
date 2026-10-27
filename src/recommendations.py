"""
recommendations.py

Generates specific, ranked, explainable rebalancing recommendations
from the risk monitor's alerts, plus an optional Markowitz alternative
(SciPy, see optimizer.py).

Each recommendation is a dict:
{
    'id': int,
    'priority': 'critical' | 'high' | 'medium' | 'low',
    'category': str,          # which risk type this addresses
    'title': str,
    'action': str,            # "Sell $X of TICKER" or "Buy $Y of TICKER"
    'trades': [               # granular trade list
        {'ticker', 'action', 'quantity', 'price', 'value'},
        ...
    ],
    'why': str,               # what risk this addresses
    'impact': dict,           # before/after metric values
    'confidence': str,        # 'high' | 'medium' | 'low'
    'alternative': str,       # another option if user disagrees
}
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import Portfolio, RISK_FREE_RATE
from .risk_monitor import RiskMonitor
from .rebalance_plan import RebalancePlan

logger = logging.getLogger(__name__)

# Priority ordering for sort
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _priority_for_severity(severity: str) -> str:
    return {"alert": "high", "caution": "medium", "ok": "low"}.get(severity, "low")


class RecommendationEngine:
    """
    Analyses risk-monitor alerts and generates ranked recommendations.

    Parameters
    ----------
    portfolio : Portfolio
    risk_monitor : RiskMonitor (already constructed with the same portfolio)
    use_optimizer : bool
        Kept for backward compatibility. The Markowitz optimisation is an
        alternative full rebalance, exposed via optimization_recommendation()
        and never mixed into generate_recommendations().
    """

    # Suggested replacements when reducing a position
    BOND_ETFS = ["BND", "AGG", "VCIT"]
    INTL_ETFS = ["VXUS", "EFA", "VEA", "IEMG"]
    BROAD_MARKET_ETFS = ["SPY", "VTI", "IVV"]
    # The strategies' third class is cash (a T-bill ETF stands in for it).
    # Alternatives (gold, REITs, commodities) live inside that class but may
    # take at most ALT_CAP of the portfolio; the rest stays in cash.
    CASH_ETFS = ["BIL"]
    ALT_CAP = 0.05

    def __init__(
        self,
        portfolio: Portfolio,
        risk_monitor: RiskMonitor,
        use_optimizer: bool = False,
    ):
        self.portfolio = portfolio
        self.risk_monitor = risk_monitor
        self.use_optimizer = use_optimizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_recommendations(self) -> list[dict]:
        """
        Run full analysis and return ranked list of recommendations.

        All trade-producing steps edit one shared RebalancePlan, in this order:
        profile drift → correlation → concentration → international exposure
        → volatility → drawdown reserve. Each step sees the result of the
        previous ones, so its trades are only what is still needed and the
        sum of all trades is exactly the move to the final plan.
        """
        analysis = self.risk_monitor.run_full_analysis()
        alerts = analysis["alert"] + analysis["caution"]
        plan = RebalancePlan(self.portfolio)
        self.plan = plan
        self._drift_applied = False

        steps = [
            ("drift", self._rec_profile_drift),
            ("correlation", self._rec_correlation),
            ("concentration", self._rec_concentration),
            ("international", self._rec_international),
            ("volatility", self._rec_volatility),
            ("drawdown", self._rec_drawdown),
            ("holdings", self._rec_holdings),
            ("sector", self._rec_sector),
        ]
        recs: list[dict] = []
        for key, generator in steps:
            step_alerts = [a for a in alerts if self._alert_kind(a) == key]
            # Earlier steps can push a position over its limit (e.g. scaling up a
            # single bond ETF), so concentration is always re-checked on the plan.
            if key == "concentration" and not step_alerts and self._drift_applied:
                step_alerts = [{"severity": "caution", "detail": {}}]
            for alert in step_alerts:
                severity = alert.get("severity", "caution")
                before = plan.snapshot()
                for r in generator(alert, severity, plan):
                    if r.get("trades") is None:
                        r["trades"] = plan.trades_since(before)
                    recs.append(r)

        recs = self._rank(recs)
        for i, r in enumerate(recs, start=1):
            r["id"] = i
        return recs

    def plan_trades(self) -> list[dict]:
        """Net trades from the current portfolio to the final plan (after generate_recommendations)."""
        plan = getattr(self, "plan", None)
        if plan is None:
            return []
        return plan.trades_since(self.portfolio.get_weights_dict())

    def get_recommendation_summary(self) -> dict:
        recs = self.generate_recommendations()
        return {
            "total": len(recs),
            "by_priority": {
                p: [r for r in recs if r["priority"] == p]
                for p in ("critical", "high", "medium", "low")
            },
            "recommendations": recs,
        }

    # ------------------------------------------------------------------
    # Alert → step routing
    # ------------------------------------------------------------------

    @staticmethod
    def _alert_kind(alert: dict) -> str | None:
        category = alert.get("category", "")
        title = alert.get("title", "")
        if "Profile Drift" in category or "Απόκλιση Προφίλ" in category:
            return "drift"
        if "Correlation" in category or "Συσχέτιση" in category:
            return "correlation"
        if "Concentration" in category or "Συγκέντρωση" in category:
            # Top-5 has no direct trade; the single-position step covers it
            return "concentration" if "Μεμονωμένο" in title or "Single" in title else None
        if "Diversification" in category or "Διαφοροποίηση" in category:
            if "Γεωγραφική" in title or "Geographic" in title:
                return "international"
            if "Αριθμός" in title or "Holdings" in title:
                return "holdings"
            if "Κλάδο" in title or "Sector" in title:
                return "sector"
            return None
        if "Volatility" in category or "Μεταβλητότητα" in category:
            return "volatility"
        if "Drawdown" in category:
            title_l = title
            return "drawdown" if ("Μέγιστη Ιστορική" in title_l or "Maximum Historical" in title_l) else None
        return None

    def _covered_note(self, plan: RebalancePlan) -> str:
        return ("Καλύπτεται από την αναδιάρθρωση για εναρμόνιση με το προφίλ κινδύνου, "
                "χωρίς πρόσθετες συναλλαγές.")

    # ------------------------------------------------------------------
    # Steps (each edits the shared plan; trades = diff of that step)
    # ------------------------------------------------------------------

    def _rec_profile_drift(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        detail = alert.get("detail", {})
        target = self.portfolio.TARGET_ALLOCATION.get(
            self.portfolio.risk_profile, self.portfolio.TARGET_ALLOCATION["moderate"])
        current_eq = detail.get("current_equity_pct", 100 * plan.group_share("equity"))
        current_fi = detail.get("current_fixed_income_pct", 100 * plan.group_share("fixed_income"))
        stats_before = plan.stats()

        added = []
        # A class the profile needs but the portfolio lacks gets a default holding,
        # seeded with an epsilon weight so scale_to has something to scale.
        defaults = {"fixed_income": self.BOND_ETFS, "other": self.CASH_ETFS, "equity": self.BROAD_MARKET_ETFS}
        for g, share in target.items():
            if share > 0 and not plan.members(g):
                t = plan.first_priced(defaults[g])
                if t is not None:
                    plan.w[t] = 1e-9
                    added.append((g, t))
        present = {g: s for g, s in target.items() if plan.members(g)}
        norm = sum(present.values())
        for g, s in present.items():
            plan.scale_to(plan.members(g), s / norm if norm > 0 else 0.0)
        # The third class is cash: alternatives keep at most ALT_CAP of it
        alts = [t for t in plan.members("other") if not plan.is_cash(t)]
        excess = plan.share(alts) - min(self.ALT_CAP, target.get("other", 0.0))
        cash = plan.first_priced(self.CASH_ETFS)
        to_cash = plan.move(alts, cash, excess) if excess > 1e-9 and cash is not None else 0.0
        self._drift_applied = True

        new_eq = 100 * plan.group_share("equity")
        new_fi = 100 * plan.group_share("fixed_income")
        stats_after = plan.stats()
        rp_label = self.portfolio.risk_profile

        why = (
            f"Το χαρτοφυλάκιο έχει {current_eq:.1f}% μετοχές και {current_fi:.1f}% ομόλογα, "
            f"ενώ το προφίλ «{rp_label}» προβλέπει {target['equity']*100:.0f}% / "
            f"{target['fixed_income']*100:.0f}%. Οι κατηγορίες επαναφέρονται στους στόχους "
            "κρατώντας τις αναλογίες των τίτλων μέσα σε κάθε κατηγορία."
        )
        if added:
            why += " Προστίθενται: " + ", ".join(t for _, t in added) + \
                   " (η κατηγορία έλειπε από το χαρτοφυλάκιο)."
        if to_cash > 0:
            why += (f" Το τρίτο σκέλος της στρατηγικής είναι μετρητά: {to_cash*100:.1f}% από "
                    f"εναλλακτικές τοποθετήσεις ({', '.join(alts)}) μεταφέρεται στο {cash}, "
                    f"ώστε αυτές να μην ξεπερνούν το {self.ALT_CAP*100:.0f}% του χαρτοφυλακίου.")
        if norm < 0.999:
            why += (" Λείπει κατηγορία-στόχος χωρίς διαθέσιμο τίτλο· το ποσοστό της "
                    "κατανεμήθηκε αναλογικά στις υπόλοιπες.")

        before_imp = {"equity_pct": round(current_eq, 1), "fixed_income_pct": round(current_fi, 1)}
        after_imp = {"equity_pct": round(new_eq, 1), "fixed_income_pct": round(new_fi, 1)}
        if stats_before and stats_after:
            before_imp["volatility_pct"] = round(stats_before["volatility"] * 100, 1)
            after_imp["volatility_pct"] = round(stats_after["volatility"] * 100, 1)

        return [{
            "priority": "critical" if severity == "alert" else "high",
            "category": "Απόκλιση Προφίλ (Profile Drift)",
            "title": "Αναδιάρθρωση για Εναρμόνιση με Προφίλ Κινδύνου",
            "action": (f"Επαναφορά κατανομής σε {new_eq:.0f}% μετοχές / {new_fi:.0f}% ομόλογα "
                       f"(από {current_eq:.0f}% / {current_fi:.0f}%)"),
            "trades": None,
            "why": why,
            "impact": {
                "before": before_imp,
                "after": after_imp,
                "improvement": f"Εναρμόνιση χαρτοφυλακίου με το προφίλ «{rp_label}».",
            },
            "confidence": "high",
            "alternative": (
                "Αν οι συνθήκες της ζωής σας έχουν αλλάξει, "
                "εξετάστε ενημέρωση του προφίλ κινδύνου αντί για αναδιάρθρωση."
            ),
        }]

    def _rec_correlation(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        pairs = alert.get("detail", {}).get("high_corr_pairs", [])
        if not pairs:
            return []
        a, b = pairs[0]["asset_a"], pairs[0]["asset_b"]
        corr = pairs[0]["correlation"]
        if plan.w.get(a, 0) <= 0 or plan.w.get(b, 0) <= 0:
            return []
        reduce = a if plan.w[a] < plan.w[b] else b
        g = plan.group(reduce)
        receivers = [t for t in plan.members(g) if t not in (a, b)]
        if not receivers:
            receivers = [t for t in plan.w if plan.w[t] > 0 and t not in (a, b)]
        if not receivers:
            return []
        moved = plan.move([reduce], receivers, plan.w[reduce] * 0.5)
        if moved <= 0:
            return []
        return [{
            "priority": _priority_for_severity(severity),
            "category": "Συσχέτιση (Correlation)",
            "title": f"Μείωση Επικαλυπτόμενης Έκθεσης ({a} / {b})",
            "action": (f"Μείωση του {reduce} κατά 50% και ανακατανομή στους υπόλοιπους "
                       "τίτλους της ίδιας κατηγορίας"),
            "trades": None,
            "why": (
                f"Τα {a} και {b} έχουν συσχέτιση {corr:.2f}. Η κατοχή και των δύο προσθέτει "
                "ελάχιστη διαφοροποίηση: όταν το ένα πέφτει, πέφτει και το άλλο."
            ),
            "impact": {
                "before": {"pair_correlation": corr},
                "after": {"description": f"Μειωμένη επικάλυψη μεταξύ {a} και {b}"},
                "improvement": "Τα κεφάλαια μεταφέρονται σε λιγότερο συσχετισμένους τίτλους.",
            },
            "confidence": "medium",
            "alternative": (
                f"Διατηρήστε και τα δύο αν έχετε διαφορετική εκτίμηση για {a} έναντι {b}, "
                "ή αντικαταστήστε το ένα με ETF άλλου κλάδου."
            ),
        }]

    def _rec_concentration(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        """Cap every position above its limit, re-evaluated on the plan so far."""
        th = self.risk_monitor.thresholds
        recs = []
        for t in sorted(plan.w, key=lambda x: -plan.w[x]):
            fund = plan.is_fund(t)
            limit = th["fund_concentration_caution_pct" if fund else "concentration_caution_pct"] / 100
            alert_lim = th["fund_concentration_alert_pct" if fund else "concentration_alert_pct"] / 100
            w0 = plan.w.get(t, 0.0)
            if w0 <= limit:
                continue
            g = plan.group(t)
            receivers = [x for x in plan.members(g) if x != t]
            if not receivers:
                receivers = [x for x in plan.w if plan.w[x] > 0 and x != t]
            if not receivers:
                continue
            before = plan.snapshot()
            plan.move([t], receivers, w0 - limit)
            kind = "διαφοροποιημένα ETF" if fund else "μεμονωμένους τίτλους"
            sev = "alert" if w0 > alert_lim else "caution"
            recs.append({
                "priority": "critical" if sev == "alert" else "high",
                "category": "Συγκέντρωση (Concentration)",
                "title": f"Μείωση Συγκέντρωσης {t}",
                "action": (f"Μείωση του {t} από {w0*100:.1f}% σε {limit*100:.0f}% και "
                           "ανακατανομή στους υπόλοιπους τίτλους της ίδιας κατηγορίας"),
                "trades": plan.trades_since(before),
                "why": (
                    f"Το {t} αποτελεί {w0*100:.1f}% του χαρτοφυλακίου (μετά τα προηγούμενα "
                    f"βήματα) — πάνω από το όριο {limit*100:.0f}% για {kind}. "
                    "Υψηλή συγκέντρωση σημαίνει ότι η πτώση ενός τίτλου έχει δυσανάλογη επίπτωση."
                ),
                "impact": {
                    "before": {"concentration_pct": round(w0 * 100, 1)},
                    "after": {"concentration_pct": round(limit * 100, 1)},
                    "improvement": f"Μειώνει το βάρος του {t} από {w0*100:.1f}% σε {limit*100:.0f}%",
                },
                "confidence": "high" if sev == "alert" else "medium",
                "alternative": (
                    f"Μειώστε το {t} στο {limit*100 + 2:.0f}% για μικρότερη προσαρμογή."
                ),
            })
        return recs

    def _rec_international(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        th = self.risk_monitor.thresholds
        intl = [t for t in plan.w if plan.w[t] > 0 and plan.is_international(t)]
        cur = plan.share(intl)
        if cur * 100 >= th["international_min_pct"]:
            return []
        target = 0.10
        etf = plan.first_priced(self.INTL_ETFS)
        us_equity = [t for t in plan.members("equity") if not plan.is_international(t)]
        if etf is None or not us_equity:
            return []
        moved = plan.move(us_equity, etf, target - cur)
        if moved <= 0:
            return []
        return [{
            "priority": _priority_for_severity(severity),
            "category": "Διαφοροποίηση (Diversification)",
            "title": "Προσθήκη Διεθνούς Έκθεσης",
            "action": (f"Μεταφορά {moved*100:.1f}% του χαρτοφυλακίου από αμερικανικές μετοχές "
                       f"σε {etf} (διεθνές ETF)"),
            "trades": None,
            "why": (
                f"Το χαρτοφυλάκιο έχει μόνο {cur*100:.1f}% διεθνή έκθεση. Η μεταφορά γίνεται "
                "μέσα στις μετοχές, ώστε να μη μεταβληθεί η κατανομή του προφίλ."
            ),
            "impact": {
                "before": {"international_pct": round(cur * 100, 1)},
                "after": {"international_pct": round((cur + moved) * 100, 1)},
                "improvement": "Μειώνει την εξάρτηση από την αμερικανική αγορά.",
            },
            "confidence": "medium",
            "alternative": "Χρησιμοποιήστε EFA (ανεπτυγμένες) ή IEMG (αναδυόμενες αγορές).",
        }]

    def _rec_volatility(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        detail = alert.get("detail", {})
        current_vol = detail.get("current_vol_pct", 0.0)
        target_vol = detail.get("target_vol_pct", 12.0)
        deviation = detail.get("deviation_pct", 0.0)

        if deviation <= 0:
            return [{
                "priority": "low",
                "category": "Μεταβλητότητα (Volatility)",
                "title": "Ανεκμετάλλευτο Risk Budget",
                "action": "Εξετάστε αύξηση της μετοχικής κατανομής για υψηλότερες αποδόσεις",
                "trades": [],
                "why": (
                    f"Η τρέχουσα μεταβλητότητα ({current_vol:.1f}%) είναι κάτω από τον στόχο "
                    f"({target_vol:.1f}%). Το risk budget δεν αξιοποιείται πλήρως."
                ),
                "impact": {
                    "before": {"volatility_pct": current_vol},
                    "after": {"volatility_pct": target_vol},
                    "improvement": "Καλύτερη μακροπρόθεσμη αναμενόμενη απόδοση για την ίδια ανοχή κινδύνου.",
                },
                "confidence": "low",
                "alternative": "Διατηρήστε την τρέχουσα κατανομή αν προτιμάτε επιπλέον περιθώριο ασφαλείας.",
            }]

        base = {
            "priority": "critical" if severity == "alert" else "high",
            "category": "Μεταβλητότητα (Volatility)",
            "title": "Μείωση Μεταβλητότητας στον Στόχο",
            "confidence": "high",
            "alternative": "Εταιρικά ομόλογα για ελαφρώς υψηλότερη απόδοση με παρόμοια μείωση μεταβλητότητας.",
        }
        stats = plan.stats()
        plan_vol = stats["volatility"] * 100 if stats else None

        # When the profile rebalancing already brings the plan within tolerance,
        # this alert needs no trades of its own.
        tol = self.risk_monitor.thresholds["vol_deviation_caution_pct"]
        if plan_vol is not None and plan_vol <= target_vol + tol:
            return [{**base,
                     "priority": "medium",
                     "action": f"Η μεταβλητότητα του σχεδίου εκτιμάται σε {plan_vol:.1f}% (στόχος {target_vol:.1f}%).",
                     "trades": [],
                     "why": (f"Η τρέχουσα μεταβλητότητα ({current_vol:.1f}%) υπερβαίνει τον στόχο "
                             f"({target_vol:.1f}%). " + (self._covered_note(plan) if self._drift_applied
                             else "Τα προηγούμενα βήματα αρκούν.")),
                     "impact": {"before": {"volatility_pct": current_vol},
                                "after": {"volatility_pct": round(plan_vol, 1)},
                                "improvement": f"Μεταβλητότητα {current_vol:.1f}% → ~{plan_vol:.1f}%"}}]

        bond = plan.first_priced(self.BOND_ETFS)
        equities = plan.members("equity")
        if bond is None or not equities:
            return []

        # Smallest equity→bond shift that reaches the target (bisection on the
        # plan's estimated volatility), capped at 30% of the portfolio. Without
        # enough history for an estimate, fall back to ~10% shift per 3% excess vol.
        start = plan.snapshot()
        # The profile allocation takes precedence: equities may only go down to
        # the profile's lower tolerance (target − drift caution), otherwise this
        # step would recreate the profile drift the previous step removed.
        target_eq = self.portfolio.TARGET_ALLOCATION.get(
            self.portfolio.risk_profile, self.portfolio.TARGET_ALLOCATION["moderate"])["equity"]
        eq_floor = max(target_eq - self.risk_monitor.thresholds["drift_caution_pct"] / 100, 0.0)
        cap = min(0.30, max(plan.share(equities) - eq_floor, 0.0))
        if cap < 0.02:  # well inside the 5% profile tolerance: not worth chasing
            return [{**base,
                     "priority": "medium",
                     "action": ("Ο στόχος μεταβλητότητας δεν επιτυγχάνεται με την κατανομή του προφίλ· "
                                "δεν προτείνεται περαιτέρω μείωση μετοχών."),
                     "trades": [],
                     "why": (
                         f"Η εκτιμώμενη μεταβλητότητα του σχεδίου ({plan_vol:.1f}%) " if plan_vol else
                         "Η μεταβλητότητα "
                     ) + (
                         f"παραμένει πάνω από τον στόχο ({target_vol:.1f}%), αλλά οι μετοχές είναι ήδη "
                         f"στο κατώτερο όριο του προφίλ ({eq_floor*100:.0f}%). Ο στόχος μεταβλητότητας "
                         "και η κατανομή-στόχος του προφίλ είναι ασυμβίβαστοι για αυτούς τους τίτλους."
                     ),
                     "impact": {"before": {"volatility_pct": current_vol},
                                "after": {"volatility_pct": round(plan_vol, 1) if plan_vol else current_vol},
                                "improvement": "Χωρίς πρόσθετη αλλαγή κατανομής."},
                     "alternative": ("Αντικαταστήστε μετοχές υψηλής μεταβλητότητας με αμυντικές "
                                     "(π.χ. VIG, USMV) ή αναθεωρήστε το προφίλ κινδύνου.")}]
        if stats is None:
            shift = min(deviation / 3.0 * 0.10, cap)
        else:
            lo, hi, best = 0.0, cap, None
            for _ in range(25):
                mid = (lo + hi) / 2
                plan.w = dict(start)
                plan.move(equities, bond, mid)
                s = plan.stats()
                if s is None:
                    break
                if s["volatility"] * 100 <= target_vol:
                    best, hi = mid, mid
                else:
                    lo = mid
            shift = best if best is not None else cap
        plan.w = dict(start)
        plan.move(equities, bond, shift)
        s = plan.stats()
        new_vol = s["volatility"] * 100 if s else None

        return [{**base,
                 "action": (f"Μεταφορά {shift*100:.1f}% του χαρτοφυλακίου από μετοχές σε {bond}"),
                 "trades": None,
                 "why": (
                     f"Η μεταβλητότητα ({current_vol:.1f}%) υπερβαίνει τον στόχο ({target_vol:.1f}%)"
                     + (f" και, μετά τα προηγούμενα βήματα, παραμένει {plan_vol:.1f}%" if plan_vol else "")
                     + ". Η μεταφορά υπολογίζεται ώστε η εκτιμώμενη μεταβλητότητα να φτάσει τον στόχο"
                     + (f", περιορισμένη ώστε οι μετοχές να μην πέσουν κάτω από {eq_floor*100:.0f}% "
                        "(όριο του προφίλ)· ο στόχος δεν επιτυγχάνεται πλήρως."
                        if new_vol is not None and new_vol > target_vol + 0.05 else ".")
                 ),
                 "impact": {"before": {"volatility_pct": round(plan_vol or current_vol, 1)},
                            "after": {"volatility_pct": round(new_vol, 1) if new_vol else target_vol},
                            "improvement": (f"Μεταβλητότητα ~{new_vol:.1f}%" if new_vol
                                            else "Μείωση μεταβλητότητας προς τον στόχο")}}]

    def _rec_drawdown(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        max_dd = alert.get("detail", {}).get("max_drawdown_pct", 0.0)
        base = {
            "priority": _priority_for_severity(severity),
            "category": "Drawdown",
            "title": "Προσθήκη Αμυντικού Αποθέματος έναντι Απωλειών",
            "confidence": "medium",
            "alternative": "Put options στο SPY ως εναλλακτική αντιστάθμιση (πιο σύνθετο).",
        }
        if self._drift_applied:
            # The profile rebalance already sets the defensive share; adding more
            # would push the plan away from the profile again.
            return [{**base,
                     "priority": "low",
                     "action": "Το αμυντικό ποσοστό ορίζεται από την αναδιάρθρωση προφίλ.",
                     "trades": [],
                     "why": (f"Η ιστορική μέγιστη απώλεια {max_dd:.1f}% είναι υψηλή. "
                             + self._covered_note(plan)),
                     "impact": {"before": {"max_drawdown_pct": max_dd},
                                "after": {"description": "Βλ. πρόταση εναρμόνισης προφίλ"},
                                "improvement": "Καμία πρόσθετη συναλλαγή."}}]

        dests = [t for t in (plan.first_priced(self.CASH_ETFS), plan.first_priced(["TLT", "IEF"])) if t]
        equities = plan.members("equity")
        if not dests or not equities:
            return []
        amount = min(0.05, 20000 / plan.total if plan.total > 0 else 0.05)
        before = plan.snapshot()
        largest = max(equities, key=lambda t: plan.w[t])
        moved = 0.0
        for d in dests:
            moved += plan.move([largest], d, amount / len(dests))
        if moved <= 0:
            return []
        return [{**base,
                 "action": (f"Μεταφορά {moved*100:.1f}% από {largest} σε αμυντικά assets "
                            f"({', '.join(dests)})"),
                 "trades": plan.trades_since(before),
                 "why": (f"Η ιστορική μέγιστη απώλεια {max_dd:.1f}% υποδηλώνει σημαντικές ζημιές "
                         "σε συνθήκες stress. Τα μετρητά διατηρούν την αξία τους και τα μακροπρόθεσμα "
                         "ομόλογα ΗΠΑ λειτουργούν ιστορικά ως αντιστάθμιση σε πτώσεις μετοχών."),
                 "impact": {"before": {"max_drawdown_pct": max_dd},
                            "after": {"defensive_added_pct": round(moved * 100, 1)},
                            "improvement": "Προσθέτει αντισταθμιστικό μαξιλάρι χρηματοδοτούμενο από μετοχές."}}]

    def _rec_holdings(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        n = alert.get("detail", {}).get("num_holdings", 0)
        return [{
            "priority": _priority_for_severity(severity),
            "category": "Διαφοροποίηση (Diversification)",
            "title": "Αύξηση Αριθμού Θέσεων",
            "action": "Προσθέστε 5-10 νέες θέσεις σε διαφορετικούς κλάδους",
            "trades": [],
            "why": (
                f"Με μόνο {n} θέσεις, ένα πρόβλημα σε μία εταιρεία επηρεάζει σημαντικά "
                "το σύνολο του χαρτοφυλακίου."
            ),
            "impact": {
                "before": {"num_holdings": n},
                "after": {"num_holdings": max(n + 5, 10)},
                "improvement": "Μειώνει τον κίνδυνο από μεμονωμένο τίτλο.",
            },
            "confidence": "medium",
            "alternative": "Ευρέα ETF (π.χ. SPY) δίνουν άμεση διαφοροποίηση με μία αγορά.",
        }]

    def _rec_sector(self, alert: dict, severity: str, plan: RebalancePlan) -> list[dict]:
        detail = alert.get("detail", {})
        top_sector = detail.get("top_sector", "")
        top_pct = detail.get("top_sector_pct", 0.0)
        return [{
            "priority": _priority_for_severity(severity),
            "category": "Διαφοροποίηση (Diversification)",
            "title": f"Μείωση Συγκέντρωσης στον Κλάδο {top_sector}",
            "action": f"Μειώστε την έκθεση στον {top_sector} από {top_pct:.0f}% κάτω από 40%",
            "trades": [],
            "why": (
                f"Ο κλάδος {top_sector} αντιπροσωπεύει {top_pct:.1f}% του χαρτοφυλακίου. "
                "Αν ο κλάδος αντιμετωπίσει αντίξοες συνθήκες, η επίπτωση θα είναι ενισχυμένη."
            ),
            "impact": {
                "before": {"top_sector_pct": round(top_pct, 1)},
                "after": {"top_sector_pct": 40.0},
                "improvement": f"Μειώνει τον κίνδυνο του κλάδου {top_sector}.",
            },
            "confidence": "medium",
            "alternative": "Στρέψτε κεφάλαια σε αμυντικούς κλάδους (Healthcare, Utilities).",
        }]

    # ------------------------------------------------------------------
    # Efficient-Frontier Optimisation Recommendation
    # ------------------------------------------------------------------

    def optimization_recommendation(self, max_weight: float = 0.20) -> dict | None:
        """
        Markowitz max-Sharpe allocation (SciPy, see src/optimizer.py) under the
        investor's risk-profile constraints.

        This is an *alternative* full rebalance, not an addition to the
        rule-based recommendations, so its trades must not be summed with them.
        """
        from .optimizer import optimize_max_sharpe, evaluate_weights

        hist = self.portfolio.historical_prices
        tickers = [t for t in self.portfolio.tickers if t in hist.columns]
        if len(tickers) < 2 or hist.empty:
            return None

        prices = hist[tickers].dropna()
        if len(prices) < 60:
            return None

        target = self.portfolio.TARGET_ALLOCATION.get(
            self.portfolio.risk_profile, self.portfolio.TARGET_ALLOCATION["moderate"])
        classes = {a.ticker: a.asset_class for a in self.portfolio.assets}

        try:
            result = optimize_max_sharpe(
                prices, classes, RISK_FREE_RATE,
                max_weight=max_weight,
                target_equity=target["equity"],
                target_fixed_income=target["fixed_income"],
            )
        except Exception as exc:
            logger.warning("Portfolio optimisation failed: %s", exc)
            return None

        if not result.get("success"):
            logger.info("Optimisation skipped: %s", result.get("message"))
            return None

        optimal = result["weights"]
        current_weights = self.portfolio.get_weights_dict()
        # Before/after measured with the same estimator so the comparison is fair
        before = evaluate_weights(prices, current_weights, RISK_FREE_RATE)
        after = result["stats"]

        changes = self.portfolio.get_allocation_changes_for_rebalance(optimal)
        trades = []
        for c in changes:
            asset = next((a for a in self.portfolio.assets if a.ticker == c["ticker"]), None)
            if asset is None or asset.current_price <= 0:
                continue
            trades.append({
                "ticker": c["ticker"],
                "action": c["action"],
                "quantity": int(c["delta_value"] / asset.current_price),
                "price": round(asset.current_price, 2),
                "value": c["delta_value"],
                "current_pct": c["current_pct"],
                "target_pct": c["target_pct"],
            })
        if not trades:
            return None

        sh_b, sh_a = before["sharpe"], after["sharpe"]
        constraints = ", ".join(result["constraints_applied"])
        skipped = result["constraints_skipped"]

        return {
            "priority": "low",
            "category": "Βελτιστοποίηση (Markowitz)",
            "title": "Εναλλακτική: Βελτιστοποίηση Efficient Frontier",
            "action": (
                f"Ανακατανομή {len(trades)} θέσεων για μεγιστοποίηση του Sharpe Ratio "
                f"υπό περιορισμούς: {constraints}."
            ),
            "trades": trades,
            "why": (
                "Κατά τη Σύγχρονη Θεωρία Χαρτοφυλακίου (Markowitz, 1952), για δεδομένο "
                "επίπεδο κινδύνου υπάρχει κατανομή με τη μέγιστη αναμενόμενη απόδοση. "
                "Η λύση υπολογίζεται με αριθμητική βελτιστοποίηση (SciPy SLSQP) επί των "
                "ιστορικών αποδόσεων και σέβεται το προφίλ κινδύνου και όριο ανά θέση."
            ),
            "impact": {
                "before": {
                    "sharpe_ratio": round(sh_b, 3),
                    "expected_return_pct": round(before["expected_return"] * 100, 2),
                    "volatility_pct": round(before["volatility"] * 100, 2),
                },
                "after": {
                    "sharpe_ratio": round(sh_a, 3),
                    "expected_return_pct": round(after["expected_return"] * 100, 2),
                    "volatility_pct": round(after["volatility"] * 100, 2),
                },
                "improvement": (
                    f"Sharpe (ίδιος εκτιμητής πριν/μετά): {sh_b:.2f} → {sh_a:.2f}"
                    + (
                        ". Το Sharpe μειώνεται επειδή η τρέχουσα κατανομή παραβιάζει το "
                        "προφίλ κινδύνου· η συμμόρφωση με τους περιορισμούς του προφίλ "
                        "έχει κόστος σε αναμενόμενη απόδοση."
                        if sh_a < sh_b else ""
                    )
                ),
            },
            "confidence": "low",
            "alternative": (
                "Οι αναμενόμενες αποδόσεις εκτιμώνται από το ιστορικό και είναι θορυβώδεις· "
                "η βέλτιστη κατανομή είναι ευαίσθητη σε αυτές. Προτιμήστε τις κανονιστικές "
                "προτάσεις αν θέλετε μικρότερες, στοχευμένες αλλαγές."
                + (f" Δεν εφαρμόστηκαν: {'; '.join(skipped)}." if skipped else "")
            ),
            "optimal_weights": {t: round(w * 100, 2) for t, w in optimal.items() if w > 0.001},
            "constraints_skipped": skipped,
        }

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, recs: list[dict]) -> list[dict]:
        """Sort recommendations by priority then confidence."""
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            recs,
            key=lambda r: (
                PRIORITY_ORDER.get(r.get("priority", "low"), 3),
                confidence_order.get(r.get("confidence", "low"), 2),
            ),
        )
