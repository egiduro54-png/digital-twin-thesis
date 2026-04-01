"""
recommendations.py

Generates specific, ranked, explainable rebalancing recommendations
from the risk monitor's alerts and (optionally) PyPortfolioOpt.

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
        When True, also generate a PyPortfolioOpt efficient-frontier recommendation.
    """

    # Suggested replacements when reducing a position
    BOND_ETFS = ["BND", "AGG", "VCIT"]
    INTL_ETFS = ["VXUS", "EFA", "IEMG"]
    BROAD_MARKET_ETFS = ["SPY", "VTI", "IVV"]

    def __init__(
        self,
        portfolio: Portfolio,
        risk_monitor: RiskMonitor,
        use_optimizer: bool = True,
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
        """
        analysis = self.risk_monitor.run_full_analysis()
        recs: list[dict] = []
        rec_id = 1

        # Process alerts (severity=alert) first, then cautions
        for alert in (analysis["alert"] + analysis["caution"]):
            generated = self._rec_for_alert(alert, rec_id)
            for r in generated:
                r["id"] = rec_id
                recs.append(r)
                rec_id += 1

        # Optionally add optimisation recommendation
        if self.use_optimizer:
            opt_rec = self._optimizer_recommendation(rec_id)
            if opt_rec:
                opt_rec["id"] = rec_id
                recs.append(opt_rec)

        recs = self._rank(recs)
        # Re-number after ranking
        for i, r in enumerate(recs, start=1):
            r["id"] = i

        return recs

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
    # Alert → Recommendation routing
    # ------------------------------------------------------------------

    def _rec_for_alert(self, alert: dict, rec_id: int) -> list[dict]:
        category = alert.get("category", "")
        severity = alert.get("severity", "caution")

        if "Concentration" in category or "Συγκέντρωση" in category:
            return self._rec_concentration(alert, severity)
        elif "Diversification" in category or "Διαφοροποίηση" in category:
            return self._rec_diversification(alert, severity)
        elif "Volatility" in category or "Μεταβλητότητα" in category:
            return self._rec_volatility(alert, severity)
        elif "Drawdown" in category:
            return self._rec_drawdown(alert, severity)
        elif "Correlation" in category or "Συσχέτιση" in category:
            return self._rec_correlation(alert, severity)
        elif "Profile Drift" in category or "Απόκλιση Προφίλ" in category:
            return self._rec_profile_drift(alert, severity)
        return []

    # ------------------------------------------------------------------
    # Recommendation generators by category
    # ------------------------------------------------------------------

    def _rec_concentration(self, alert: dict, severity: str) -> list[dict]:
        detail = alert.get("detail", {})
        largest_ticker = detail.get("largest_ticker", "")
        largest_pct = detail.get("largest_pct", 0.0)

        # Find the asset object
        asset = next((a for a in self.portfolio.assets
                      if a.ticker == largest_ticker), None)
        if asset is None:
            return []

        # Target: bring to 20%
        target_pct = 20.0
        total = self.portfolio.total_value
        excess_value = (largest_pct - target_pct) / 100.0 * total

        if excess_value < 50:
            return []

        price = asset.current_price
        shares_to_sell = int(excess_value / price) if price > 0 else 0

        trades = [{
            "ticker": largest_ticker,
            "action": "SELL",
            "quantity": shares_to_sell,
            "price": round(price, 2),
            "value": round(shares_to_sell * price, 2),
        }]

        # Show impact on concentration metric
        new_pct = (asset.current_value - excess_value) / total * 100

        return [{
            "priority": "critical" if severity == "alert" else "high",
            "category": "Συγκέντρωση (Concentration)",
            "title": f"Μείωση Συγκέντρωσης {largest_ticker}",
            "action": f"Πώληση ${excess_value:,.0f} από {largest_ticker}",
            "trades": trades,
            "why": (
                f"Το {largest_ticker} αποτελεί {largest_pct:.1f}% του χαρτοφυλακίου — "
                f"πάνω από το ασφαλές όριο του {target_pct:.0f}%. "
                "Υψηλή συγκέντρωση σημαίνει ότι η πτώση ενός τίτλου έχει δυσανάλογη επίπτωση."
            ),
            "impact": {
                "before": {"concentration_pct": round(largest_pct, 1)},
                "after": {"concentration_pct": round(new_pct, 1)},
                "improvement": f"Μειώνει το βάρος του {largest_ticker} από "
                               f"{largest_pct:.1f}% σε {new_pct:.1f}%",
            },
            "confidence": "high" if severity == "alert" else "medium",
            "alternative": (
                f"Μειώστε το {largest_ticker} στο 22% αντί του {target_pct:.0f}% "
                "για μικρότερη προσαρμογή που εξακολουθεί να βελτιώνει τον κίνδυνο."
            ),
        }]

    def _rec_diversification(self, alert: dict, severity: str) -> list[dict]:
        title = alert.get("title", "")
        detail = alert.get("detail", {})
        recs = []

        if ("Geographic" in title or "Γεωγραφική" in title) and severity != "ok":
            intl_pct = detail.get("international_pct", 0.0)
            target_intl_pct = 10.0
            total = self.portfolio.total_value
            buy_amount = (target_intl_pct - intl_pct) / 100.0 * total
            buy_amount = max(buy_amount, 0)

            if buy_amount > 50:
                suggested_etf = self.INTL_ETFS[0]
                price_estimate = 58.0
                shares = int(buy_amount / price_estimate)

                recs.append({
                    "priority": _priority_for_severity(severity),
                    "category": "Διαφοροποίηση (Diversification)",
                    "title": "Προσθήκη Διεθνούς Έκθεσης",
                    "action": f"Αγορά ${buy_amount:,.0f} σε {suggested_etf} (Διεθνές ETF)",
                    "trades": [{
                        "ticker": suggested_etf,
                        "action": "BUY",
                        "quantity": shares,
                        "price": price_estimate,
                        "value": round(shares * price_estimate, 2),
                    }],
                    "why": (
                        f"Το χαρτοφυλάκιο έχει μόνο {intl_pct:.1f}% διεθνή έκθεση. "
                        "Η συγκέντρωση αποκλειστικά σε αμερικανικές αγορές αυξάνει τον κίνδυνο μίας χώρας. "
                        "Η προσθήκη διεθνών μετοχών μειώνει τη συσχέτιση με γεγονότα της αμερικανικής αγοράς."
                    ),
                    "impact": {
                        "before": {"international_pct": round(intl_pct, 1)},
                        "after": {"international_pct": round(target_intl_pct, 1)},
                        "improvement": (
                            f"Το χαρτοφυλάκιο αποκτά {target_intl_pct:.0f}% διεθνή έκθεση, "
                            "μειώνοντας την εξάρτηση από αμερικανική αγορά."
                        ),
                    },
                    "confidence": "medium",
                    "alternative": (
                        f"Χρησιμοποιήστε EFA (ανεπτυγμένες αγορές) ή IEMG (αναδυόμενες αγορές) "
                        f"αντί για {suggested_etf}."
                    ),
                })

        if ("Number of Holdings" in title or "Αριθμός" in title) and severity != "ok":
            n = detail.get("num_holdings", 0)
            recs.append({
                "priority": _priority_for_severity(severity),
                "category": "Διαφοροποίηση (Diversification)",
                "title": "Αύξηση Αριθμού Θέσεων",
                "action": "Προσθέστε 5-10 νέες θέσεις σε διαφορετικούς κλάδους",
                "trades": [],
                "why": (
                    f"Με μόνο {n} θέσεις, ένα πρόβλημα σε μία εταιρεία "
                    "επηρεάζει σημαντικά το σύνολο του χαρτοφυλακίου. "
                    "Περισσότερες θέσεις κατανέμουν τον κίνδυνο σε ανεξάρτητες εταιρείες."
                ),
                "impact": {
                    "before": {"num_holdings": n},
                    "after": {"num_holdings": max(n + 5, 10)},
                    "improvement": "Μειώνει τον κίνδυνο από μεμονωμένο τίτλο.",
                },
                "confidence": "medium",
                "alternative": (
                    "Χρησιμοποιήστε ευρεία ETF (SPY, QQQ) για άμεση διαφοροποίηση "
                    "σε εκατοντάδες μετοχές με μία αγορά."
                ),
            })

        if "Sector" in title and severity != "ok":
            top_sector = detail.get("top_sector", "")
            top_pct = detail.get("top_sector_pct", 0.0)
            recs.append({
                "priority": _priority_for_severity(severity),
                "category": "Διαφοροποίηση (Diversification)",
                "title": f"Μείωση Συγκέντρωσης στον Κλάδο {top_sector}",
                "action": f"Μειώστε την έκθεση στον {top_sector} από {top_pct:.0f}% κάτω από 40%",
                "trades": [],
                "why": (
                    f"Ο κλάδος {top_sector} αντιπροσωπεύει {top_pct:.1f}% του χαρτοφυλακίου. "
                    "Αν ο κλάδος αυτός αντιμετωπίσει αντίξοες συνθήκες (ρύθμιση, επιτόκια, ανταγωνισμός), "
                    "η επίπτωση στο χαρτοφυλάκιο θα είναι ενισχυμένη."
                ),
                "impact": {
                    "before": {"top_sector_pct": round(top_pct, 1)},
                    "after": {"top_sector_pct": 40.0},
                    "improvement": f"Μειώνει σημαντικά τον κίνδυνο του κλάδου {top_sector}.",
                },
                "confidence": "medium",
                "alternative": "Στρέψτε κεφάλαια σε αμυντικούς κλάδους (Healthcare, Utilities).",
            })

        return recs

    def _rec_volatility(self, alert: dict, severity: str) -> list[dict]:
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
                    f"({target_vol:.1f}%). Το risk budget δεν αξιοποιείται πλήρως. "
                    "Περισσότερες μετοχές θα μπορούσαν να βελτιώσουν τις μακροπρόθεσμες αποδόσεις."
                ),
                "impact": {
                    "before": {"volatility_pct": current_vol},
                    "after": {"volatility_pct": target_vol},
                    "improvement": "Καλύτερη μακροπρόθεσμη αναμενόμενη απόδοση για την ίδια ανοχή κινδύνου.",
                },
                "confidence": "low",
                "alternative": "Διατηρήστε την τρέχουσα κατανομή αν προτιμάτε επιπλέον περιθώριο ασφαλείας.",
            }]

        # Over-volatility: suggest shifting to bonds
        total = self.portfolio.total_value
        # Approximate: each 10% shift to bonds from stocks reduces vol by ~3%
        pct_to_shift = min(deviation / 3.0 * 10.0, 30.0)
        shift_amount = pct_to_shift / 100.0 * total

        suggested_bond = self.BOND_ETFS[0]

        trades = [{
            "ticker": suggested_bond,
            "action": "BUY",
            "quantity": int(shift_amount / 82),  # approximate BND price
            "price": 82.0,
            "value": round(shift_amount, 2),
        }]

        return [{
            "priority": "critical" if severity == "alert" else "high",
            "category": "Μεταβλητότητα (Volatility)",
            "title": "Μείωση Μεταβλητότητας στον Στόχο",
            "action": f"Αγορά ${shift_amount:,.0f} ομολόγων ({suggested_bond}), "
                      f"χρηματοδοτούμενη από μείωση μετοχικών θέσεων",
            "trades": trades,
            "why": (
                f"Η μεταβλητότητα του χαρτοφυλακίου ({current_vol:.1f}%) υπερβαίνει τον στόχο "
                f"({target_vol:.1f}%) κατά {deviation:.1f} ποσοστιαίες μονάδες. "
                "Υψηλότερη μεταβλητότητα σημαίνει μεγαλύτερες διακυμάνσεις αξίας, "
                "αυξάνοντας τον κίνδυνο πανικόβλητων πωλήσεων στα χαμηλά της αγοράς."
            ),
            "impact": {
                "before": {"volatility_pct": current_vol},
                "after": {"volatility_pct": round(target_vol, 1)},
                "improvement": f"Η μεταβλητότητα μειώνεται από {current_vol:.1f}% σε ~{target_vol:.1f}%",
            },
            "confidence": "high",
            "alternative": (
                f"Στρέψτε σε εταιρικά ομόλογα για ελαφρώς υψηλότερη απόδοση "
                f"διατηρώντας τη μείωση της μεταβλητότητας."
            ),
        }]

    def _rec_drawdown(self, alert: dict, severity: str) -> list[dict]:
        detail = alert.get("detail", {})
        title = alert.get("title", "")

        if "Maximum Historical" in title or "Μέγιστη Ιστορική" in title:
            max_dd = detail.get("max_drawdown_pct", 0.0)
            total = self.portfolio.total_value
            defensive_amount = min(0.05 * total, 20000)

            return [{
                "priority": _priority_for_severity(severity),
                "category": "Drawdown",
                "title": "Προσθήκη Αμυντικού Αποθέματος έναντι Απωλειών",
                "action": f"Κατανείμετε ${defensive_amount:,.0f} σε αμυντικά assets (TLT ή GLD)",
                "trades": [
                    {
                        "ticker": "GLD",
                        "action": "BUY",
                        "quantity": int(defensive_amount / 2 / 185),
                        "price": 185.0,
                        "value": round(defensive_amount / 2, 2),
                    },
                    {
                        "ticker": "TLT",
                        "action": "BUY",
                        "quantity": int(defensive_amount / 2 / 95),
                        "price": 95.0,
                        "value": round(defensive_amount / 2, 2),
                    },
                ],
                "why": (
                    f"Η ιστορική μέγιστη απώλεια {max_dd:.1f}% υποδηλώνει σημαντικές "
                    "ζημιές σε συνθήκες αγοράς stres. "
                    "Ο χρυσός (GLD) και τα μακροπρόθεσμα ομόλογα ΗΠΑ (TLT) αποδίδουν ιστορικά "
                    "καλά σε πτώσεις μετοχών, λειτουργώντας ως αντισταθμιστικό μαξιλάρι."
                ),
                "impact": {
                    "before": {"max_drawdown_pct": max_dd},
                    "after": {"max_drawdown_pct_estimate": round(max_dd * 0.85, 1)},
                    "improvement": "Μειώνει την αναμενόμενη χειρότερη απώλεια κατά ~15%.",
                },
                "confidence": "medium",
                "alternative": "Χρησιμοποιήστε put options στο SPY ως εναλλακτική αντιστάθμιση (πιο σύνθετο).",
            }]

        return []

    def _rec_correlation(self, alert: dict, severity: str) -> list[dict]:
        detail = alert.get("detail", {})
        high_pairs = detail.get("high_corr_pairs", [])

        if not high_pairs:
            return []

        # Suggest removing the most duplicative position
        worst_pair = high_pairs[0]
        a, b = worst_pair["asset_a"], worst_pair["asset_b"]
        corr = worst_pair["correlation"]

        asset_a = next((x for x in self.portfolio.assets if x.ticker == a), None)
        asset_b = next((x for x in self.portfolio.assets if x.ticker == b), None)

        if asset_a is None or asset_b is None:
            return []

        # Recommend selling smaller of the two
        if asset_a.current_value >= asset_b.current_value:
            keep, reduce = asset_a, asset_b
        else:
            keep, reduce = asset_b, asset_a

        reduce_amount = reduce.current_value * 0.5  # sell half
        shares_to_sell = int(reduce_amount / reduce.current_price) if reduce.current_price > 0 else 0

        return [{
            "priority": _priority_for_severity(severity),
            "category": "Συσχέτιση (Correlation)",
            "title": f"Μείωση Επικαλυπτόμενης Έκθεσης ({a} / {b})",
            "action": f"Μειώστε τη θέση {reduce.ticker} κατά 50% (πώληση ${reduce_amount:,.0f})",
            "trades": [{
                "ticker": reduce.ticker,
                "action": "SELL",
                "quantity": shares_to_sell,
                "price": round(reduce.current_price, 2),
                "value": round(reduce_amount, 2),
            }],
            "why": (
                f"Τα {a} και {b} έχουν συσχέτιση {corr:.2f} — πολύ υψηλή. "
                "Η κατοχή και των δύο παρέχει ελάχιστη διαφοροποίηση: όταν το ένα πέφτει, "
                "πέφτει και το άλλο. Ουσιαστικά πληρώνετε διπλά για την ίδια έκθεση."
            ),
            "impact": {
                "before": {"pair_correlation": corr, "num_high_corr_pairs": len(high_pairs)},
                "after": {"description": f"Μειωμένη επικάλυψη μεταξύ {a} και {b}"},
                "improvement": "Απελευθερώνει κεφάλαια για πραγματικά μη συσχετισμένα assets.",
            },
            "confidence": "medium",
            "alternative": (
                f"Διατηρήστε και τα δύο αν έχετε διαφορετική εκτίμηση για {a} έναντι {b}, "
                "αλλά αντικαταστήστε το ένα με ETF από διαφορετικό κλάδο."
            ),
        }]

    def _rec_profile_drift(self, alert: dict, severity: str) -> list[dict]:
        detail = alert.get("detail", {})
        eq_drift = detail.get("equity_drift_pct", 0.0)
        current_eq = detail.get("current_equity_pct", 0.0)
        target_eq = detail.get("target_equity_pct", 0.0)
        current_fi = detail.get("current_fixed_income_pct", 0.0)
        target_fi = detail.get("target_fixed_income_pct", 0.0)

        total = self.portfolio.total_value
        shift_pct = abs(eq_drift)
        shift_amount = shift_pct / 100.0 * total

        rp_labels = {"conservative": "Συντηρητικό", "moderate": "Μέτριο", "aggressive": "Επιθετικό"}
        rp_label = rp_labels.get(detail.get("risk_profile", ""), detail.get("risk_profile", ""))

        if eq_drift > 0:
            action_str = (f"Πώληση ${shift_amount:,.0f} μετοχών, "
                          f"αγορά ${shift_amount:,.0f} ομολόγων (BND) για αναλογία "
                          f"{target_eq:.0f}/{target_fi:.0f} μετοχές/ομόλογα")
            why = (
                f"Το χαρτοφυλάκιο έχει {current_eq:.1f}% μετοχές αλλά ο στόχος είναι {target_eq:.1f}%. "
                "Αυτό είναι πιο επιθετικό από το δηλωμένο προφίλ κινδύνου. "
                "Η αναδιάρθρωση μειώνει τον κίνδυνο μεγάλης απώλειας σε πτώσεις αγοράς."
            )
        else:
            action_str = (f"Μεταφορά ${shift_amount:,.0f} από ομόλογα σε μετοχές (SPY) "
                          f"για αναλογία {target_eq:.0f}/{target_fi:.0f} μετοχές/ομόλογα")
            why = (
                f"Το χαρτοφυλάκιο έχει {current_eq:.1f}% μετοχές αλλά ο στόχος είναι {target_eq:.1f}%. "
                "Αυτό είναι πολύ συντηρητικό για το προφίλ σας, "
                "μειώνοντας πιθανώς τις μακροπρόθεσμες αποδόσεις περιττά."
            )

        return [{
            "priority": "critical" if severity == "alert" else "high",
            "category": "Απόκλιση Προφίλ (Profile Drift)",
            "title": "Αναδιάρθρωση για Εναρμόνιση με Προφίλ Κινδύνου",
            "action": action_str,
            "trades": [],
            "why": why,
            "impact": {
                "before": {
                    "equity_pct": round(current_eq, 1),
                    "fixed_income_pct": round(current_fi, 1),
                },
                "after": {
                    "equity_pct": round(target_eq, 1),
                    "fixed_income_pct": round(target_fi, 1),
                },
                "improvement": f"Εναρμόνιση χαρτοφυλακίου με προφίλ '{rp_label}'.",
            },
            "confidence": "high",
            "alternative": (
                "Αν οι συνθήκες της ζωής σας έχουν αλλάξει, "
                "εξετάστε ενημέρωση του προφίλ κινδύνου αντί για αναδιάρθρωση."
            ),
        }]

    # ------------------------------------------------------------------
    # Efficient-Frontier Optimisation Recommendation
    # ------------------------------------------------------------------

    def _optimizer_recommendation(self, rec_id: int) -> dict | None:
        """
        Use PyPortfolioOpt to find the max-Sharpe allocation and recommend trades.
        """
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt import risk_models, expected_returns
        except ImportError:
            logger.warning("PyPortfolioOpt not installed — skipping optimisation recommendation.")
            return None

        hist = self.portfolio.historical_prices
        tickers = [t for t in self.portfolio.tickers if t in hist.columns]

        if len(tickers) < 2 or hist.empty:
            return None

        prices = hist[tickers].dropna()
        if len(prices) < 60:
            return None

        try:
            mu = expected_returns.mean_historical_return(prices)
            cov = risk_models.sample_cov(prices)

            ef = EfficientFrontier(mu, cov, weight_bounds=(0, 0.40))
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            cleaned = ef.clean_weights()

            # Calculate Sharpe before and after
            ret, vol, sharpe_opt = ef.portfolio_performance(
                risk_free_rate=RISK_FREE_RATE, verbose=False
            )

            current_sharpe = self.portfolio.get_metrics().get("sharpe_ratio")

            # Build trade list
            total = self.portfolio.total_value
            current_weights = self.portfolio.get_weights_dict()
            changes = self.portfolio.get_allocation_changes_for_rebalance(cleaned)

            if not changes:
                return None

            trades = []
            for c in changes:
                ticker = c["ticker"]
                asset = next((a for a in self.portfolio.assets
                              if a.ticker == ticker), None)
                if asset is None:
                    continue
                shares = int(c["delta_value"] / asset.current_price) if asset.current_price > 0 else 0
                trades.append({
                    "ticker": ticker,
                    "action": c["action"],
                    "quantity": shares,
                    "price": round(asset.current_price, 2),
                    "value": c["delta_value"],
                    "current_pct": c["current_pct"],
                    "target_pct": c["target_pct"],
                })

            sharpe_before = round(current_sharpe, 3) if current_sharpe else None
            sharpe_after = round(sharpe_opt, 3)
            improvement = (
                round((sharpe_opt - current_sharpe) / abs(current_sharpe) * 100, 1)
                if current_sharpe else None
            )

            return {
                "priority": "medium",
                "category": "Optimisation",
                "title": "Efficient Frontier Portfolio Optimisation",
                "action": (
                    f"Rebalance {len(trades)} positions to maximise risk-adjusted return. "
                    f"Expected Sharpe ratio improvement: "
                    f"{sharpe_before} → {sharpe_after}"
                ),
                "trades": trades,
                "why": (
                    "Modern Portfolio Theory (MPT) identifies combinations of assets "
                    "that maximise return for a given level of risk. "
                    "The current portfolio sits below the efficient frontier — "
                    "meaning the same risk level could generate higher returns "
                    "with a different allocation mix."
                ),
                "impact": {
                    "before": {
                        "sharpe_ratio": sharpe_before,
                        "expected_return_pct": round(
                            self.portfolio.calculate_expected_annual_return() * 100, 2
                        ) if hasattr(self.portfolio, "calculate_expected_annual_return") else None,
                    },
                    "after": {
                        "sharpe_ratio": sharpe_after,
                        "expected_return_pct": round(ret * 100, 2),
                        "expected_volatility_pct": round(vol * 100, 2),
                    },
                    "improvement": (
                        f"Sharpe ratio improves by {improvement}% "
                        f"({sharpe_before} → {sharpe_after})"
                        if improvement else "Sharpe ratio optimised."
                    ),
                },
                "confidence": "medium",
                "alternative": (
                    "This optimisation is based on historical data. "
                    "Use 'minimum volatility' objective instead if capital preservation "
                    "is the primary goal."
                ),
                "optimal_weights": {
                    t: round(w * 100, 2) for t, w in cleaned.items() if w > 0.001
                },
            }

        except Exception as exc:
            logger.warning("Portfolio optimisation failed: %s", exc)
            return None

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
