"""
explainer.py

Converts raw numerical alerts, metrics, and recommendations into clear,
human-readable explanations.

This is the Explainable AI (XAI) layer of the system — the key differentiator
of the Digital Twin approach.
"""


def _fmt_pct(value, precision=1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}%"


def _fmt_money(value) -> str:
    if value is None:
        return "N/A"
    return f"${abs(value):,.0f}"


class Explainer:
    """
    Generates plain-English explanations for every piece of data in the system.
    """

    # ------------------------------------------------------------------
    # Metric Explanations
    # ------------------------------------------------------------------

    METRIC_EXPLANATIONS = {
        "volatility_annual_pct": {
            "name": "Annual Volatility",
            "what": (
                "Volatility measures how much the portfolio's value fluctuates. "
                "A 15% annual volatility means the portfolio is expected to move "
                "±15% from its average value in a typical year."
            ),
            "why_it_matters": (
                "Higher volatility means bigger swings — potentially larger gains "
                "but also larger losses. Matching volatility to your risk tolerance "
                "prevents panic-selling during downturns."
            ),
        },
        "sharpe_ratio": {
            "name": "Sharpe Ratio",
            "what": (
                "The Sharpe ratio measures return per unit of risk. "
                "Formula: (Portfolio Return − Risk-Free Rate) ÷ Volatility. "
                "A ratio > 1.0 is considered good; > 2.0 is excellent."
            ),
            "why_it_matters": (
                "A higher Sharpe ratio means you're being better compensated "
                "for the risk you take. Two portfolios with the same return but "
                "different risk levels will have different Sharpe ratios."
            ),
        },
        "beta": {
            "name": "Portfolio Beta",
            "what": (
                "Beta measures the portfolio's sensitivity to the overall market (S&P 500). "
                "Beta = 1.0 means the portfolio moves exactly with the market. "
                "Beta = 1.5 means it amplifies market moves by 50%."
            ),
            "why_it_matters": (
                "A high beta portfolio gains more in bull markets but loses more "
                "in bear markets. Conservative investors prefer beta < 1.0."
            ),
        },
        "max_drawdown_pct": {
            "name": "Maximum Drawdown",
            "what": (
                "The worst peak-to-trough loss experienced historically. "
                "A drawdown of -25% means the portfolio fell 25% from its peak "
                "before recovering."
            ),
            "why_it_matters": (
                "Large drawdowns test investor resolve. Many investors sell at the "
                "bottom after a big loss, locking in losses. Knowing the worst case "
                "helps plan for it emotionally and financially."
            ),
        },
        "diversification_ratio": {
            "name": "Diversification Ratio",
            "what": (
                "Ratio of weighted-average individual asset volatility to portfolio volatility. "
                "A ratio of 1.5 means diversification reduces portfolio risk by 33%."
            ),
            "why_it_matters": (
                "The higher the ratio, the more diversification is reducing your overall risk. "
                "A ratio near 1.0 means your assets are so correlated that diversification "
                "provides little benefit."
            ),
        },
        "var_95_monthly_pct": {
            "name": "Value-at-Risk (95%, Monthly)",
            "what": (
                "In 5% of months (roughly 1 month per year) the portfolio "
                "is expected to lose at least this percentage."
            ),
            "why_it_matters": (
                "VaR quantifies tail risk. It answers: how bad could a bad month get? "
                "Useful for liquidity planning — make sure you can afford the worst months."
            ),
        },
    }

    def explain_metric(self, metric_name: str, value, target=None) -> str:
        info = self.METRIC_EXPLANATIONS.get(metric_name, {})
        name = info.get("name", metric_name)
        what = info.get("what", "")
        why = info.get("why_it_matters", "")

        lines = [f"**{name}**: {value}"]
        if what:
            lines.append(f"\n*What it means:* {what}")
        if why:
            lines.append(f"\n*Why it matters:* {why}")
        if target is not None:
            diff = value - target if isinstance(value, (int, float)) else None
            if diff is not None:
                direction = "above" if diff > 0 else "below"
                lines.append(f"\n*Target:* {target} — currently {abs(diff):.2f} {direction} target.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Alert Explanations
    # ------------------------------------------------------------------

    def explain_alert(self, alert: dict) -> str:
        """
        Convert an alert dict from RiskMonitor into a clear plain-English explanation.
        """
        severity = alert.get("severity", "ok")
        category = alert.get("category", "")
        title = alert.get("title", "")
        description = alert.get("description", "")
        recommendation = alert.get("recommendation", "")

        severity_label = {
            "ok": "✅ OK",
            "caution": "⚠️ CAUTION",
            "alert": "🚨 ALERT",
        }.get(severity, severity.upper())

        lines = [
            f"{severity_label} — {category}: {title}",
            "",
            description,
        ]

        if recommendation and severity != "ok":
            lines += ["", f"**What to do:** {recommendation}"]

        return "\n".join(lines)

    def explain_all_alerts(self, analysis: dict) -> str:
        """Produce a full risk report from run_full_analysis() output."""
        summary = analysis.get("summary", {})
        lines = [
            "# Portfolio Risk Analysis",
            "",
            f"**Summary:** {summary.get('ok', 0)} OK | "
            f"{summary.get('caution', 0)} Caution | "
            f"{summary.get('alert', 0)} Alert",
            "",
            "---",
        ]

        for alert in analysis.get("alert", []):
            lines += ["", self.explain_alert(alert), ""]
        for alert in analysis.get("caution", []):
            lines += ["", self.explain_alert(alert), ""]
        for alert in analysis.get("ok", []):
            lines += ["", self.explain_alert(alert), ""]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Recommendation Explanations
    # ------------------------------------------------------------------

    def explain_recommendation(self, rec: dict) -> str:
        """
        Produce a detailed plain-English explanation for one recommendation.
        """
        priority = rec.get("priority", "medium").upper()
        title = rec.get("title", "")
        action = rec.get("action", "")
        why = rec.get("why", "")
        impact = rec.get("impact", {})
        confidence = rec.get("confidence", "medium").capitalize()
        alternative = rec.get("alternative", "")
        trades = rec.get("trades", [])

        # Impact section
        before = impact.get("before", {})
        after = impact.get("after", {})
        improvement = impact.get("improvement", "")

        impact_lines = []
        for key in before:
            b_val = before.get(key)
            a_val = after.get(key)
            if b_val is not None and a_val is not None:
                try:
                    change = a_val - b_val
                    direction = "↑" if change > 0 else "↓"
                    impact_lines.append(
                        f"  - {key.replace('_', ' ').title()}: "
                        f"{b_val} → {a_val} ({direction}{abs(change):.2f})"
                    )
                except TypeError:
                    pass

        if improvement:
            impact_lines.append(f"  - {improvement}")

        # Trades section
        trade_lines = []
        for t in trades:
            qty = t.get("quantity", 0)
            ticker = t.get("ticker", "")
            act = t.get("action", "")
            val = t.get("value", 0)
            price = t.get("price", 0)
            if qty and ticker:
                trade_lines.append(
                    f"  - {act} {qty} shares of {ticker} @ ${price:.2f} = ${val:,.0f}"
                )

        lines = [
            f"## [{priority}] {title}",
            "",
            f"**Action:** {action}",
            "",
            f"**Why:**",
            why,
            "",
        ]

        if trade_lines:
            lines += ["**Specific Trades:**"] + trade_lines + [""]

        if impact_lines:
            lines += ["**Expected Impact:**"] + impact_lines + [""]

        lines += [
            f"**Confidence:** {confidence}",
        ]

        if alternative:
            lines += ["", f"**Alternative:** {alternative}"]

        return "\n".join(lines)

    def explain_all_recommendations(self, recs: list[dict]) -> str:
        """Full formatted recommendation report."""
        if not recs:
            return "No recommendations — portfolio looks healthy!"

        lines = [
            "# Rebalancing Recommendations",
            f"*{len(recs)} recommendation(s) ranked by priority*",
            "",
        ]

        for rec in recs:
            lines += ["---", "", self.explain_recommendation(rec), ""]

        total_sell = sum(
            t.get("value", 0)
            for r in recs
            for t in r.get("trades", [])
            if t.get("action") == "SELL"
        )
        total_buy = sum(
            t.get("value", 0)
            for r in recs
            for t in r.get("trades", [])
            if t.get("action") == "BUY"
        )

        lines += [
            "---",
            "## Summary",
            f"- Total value to SELL: ${total_sell:,.0f}",
            f"- Total value to BUY:  ${total_buy:,.0f}",
            f"- Net cash required:   ${total_buy - total_sell:+,.0f}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scenario Explanations
    # ------------------------------------------------------------------

    def explain_scenario(self, comparison: dict) -> str:
        """
        Produce a plain-English analysis of a scenario comparison dict
        (output of ScenarioEngine.compare_portfolio_metrics).
        """
        name = comparison.get("scenario_name", "Scenario")
        desc = comparison.get("scenario_description", "")
        params = comparison.get("scenario_params", {})
        summary = comparison.get("summary", {})
        metrics = comparison.get("metrics", {})

        market_chg = params.get("market_change_pct", 0.0)
        rate_chg = params.get("rate_change_pct", 0.0)
        port_chg = summary.get("portfolio_change_pct", 0.0)
        resilience = summary.get("resilience_vs_market_pct", 0.0)

        curr_val = summary.get("current_total_value", 0)
        scen_val = summary.get("scenario_total_value", 0)

        resilience_comment = (
            "Το χαρτοφυλάκιο αποδίδει καλύτερα από την αγορά σε αυτό το σενάριο (καλή ανθεκτικότητα)."
            if resilience > 0 else
            "Το χαρτοφυλάκιο αποδίδει χειρότερα από την αγορά σε αυτό το σενάριο (χαμηλή ανθεκτικότητα)."
        )

        def _row(label: str, m: dict) -> str:
            curr = m.get("current")
            scen = m.get("scenario")
            chg = m.get("change")
            if curr is None or scen is None:
                return f"  - {label}: N/A"
            sign = "+" if chg and chg > 0 else ""
            return f"  - {label}: {curr} → {scen} ({sign}{chg})"

        lines = [
            f"## Σενάριο: {name}",
            f"*{desc}*",
            "",
            "**Παράμετροι Σεναρίου:**",
            f"  - Μεταβολή αγοράς: {market_chg:+.1f}%",
            f"  - Μεταβολή επιτοκίου: {rate_chg:+.2f}%",
            "",
            "**Επίπτωση στο Χαρτοφυλάκιο:**",
            f"  - Αξία χαρτοφυλακίου: ${curr_val:,.0f} → ${scen_val:,.0f} "
            f"({port_chg:+.1f}%)",
            "",
            "**Μεταβολές Βασικών Δεικτών:**",
            _row("Μεταβλητότητα — Volatility (%)", metrics.get("volatility_annual_pct", {})),
            _row("Δείκτης Sharpe", metrics.get("sharpe_ratio", {})),
            _row("Μέγιστη Απώλεια — Max Drawdown (%)", metrics.get("max_drawdown_pct", {})),
            _row("Beta", metrics.get("beta", {})),
            "",
            "**Αξιολόγηση Ανθεκτικότητας:**",
            f"  Μεταβολή χαρτοφυλακίου: {port_chg:+.1f}% έναντι αγοράς: {market_chg:+.1f}%",
            f"  Ανθεκτικότητα έναντι αγοράς: {resilience:+.1f}%",
            f"  {resilience_comment}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Glossary
    # ------------------------------------------------------------------

    GLOSSARY = {
        "Sharpe Ratio": "Return per unit of risk. Higher is better. Formula: (Return - RFR) / Volatility.",
        "Volatility": "Standard deviation of daily returns, annualised. Measures price fluctuation.",
        "Beta": "Portfolio sensitivity to the market. Beta=1 moves with market; Beta=2 moves twice as much.",
        "Max Drawdown": "Worst historical peak-to-trough loss. Indicates worst-case historical scenario.",
        "VaR": "Value-at-Risk: the loss level that is exceeded only X% of the time.",
        "Herfindahl Index": "Concentration measure. Σ(weight²). Range 0-1; >0.25 = highly concentrated.",
        "Efficient Frontier": "Set of portfolios that maximise return for each level of risk.",
        "Diversification Ratio": "Measures how much diversification reduces portfolio risk vs. individual assets.",
        "Correlation": "How two assets move together. Range -1 to +1. High correlation = less diversification benefit.",
        "Rebalancing": "Adjusting portfolio weights back to target allocation after market movements.",
        "Digital Twin": "A virtual replica of the portfolio that can be stress-tested without real trades.",
    }

    def explain_term(self, term: str) -> str:
        return self.GLOSSARY.get(term, f"No explanation available for '{term}'.")
