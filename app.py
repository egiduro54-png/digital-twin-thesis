"""
app.py

Σύστημα Συμβουλευτικής Επενδύσεων βασισμένο στο Digital Twin — Streamlit Dashboard
"""
from __future__ import annotations

import os
import io
import json
import logging
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Project modules
from src.utils import build_portfolio, format_currency, format_pct, format_ratio
from src.utils import severity_color, severity_emoji, priority_color, setup_logging
from src.risk_monitor import RiskMonitor
from src.scenario_engine import ScenarioEngine
from src.recommendations import RecommendationEngine
from src.explainer import Explainer
from src.validation import (
    ValidationExperiment,
    ValidationResults,
    PORTFOLIO_ARCHETYPES,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Σύστημα Συμβουλευτικής Επενδύσεων — Digital Twin",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging("WARNING")
logger = logging.getLogger(__name__)

EXPLAINER = Explainer()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PORTFOLIOS = {
    "Ισορροπημένο (Μέτριο)": str(DATA_DIR / "portfolio_moderate.csv"),
    "Δυναμικό (Επιθετικό)": str(DATA_DIR / "portfolio_aggressive.csv"),
    "Συντηρητικό (Εισόδημα)": str(DATA_DIR / "portfolio_conservative.csv"),
}

RISK_PROFILE_LABELS = {
    "liquidity_plus": "Liquidity Plus (Πολύ Χαμηλός)",
    "defensive":      "Defensive (Χαμηλός)",
    "flexible":       "Flexible (Μέτριος)",
    "growth":         "Growth (Υψηλός)",
    "dynamic":        "Dynamic (Πολύ Υψηλός)",
    # legacy
    "conservative": "Συντηρητικό",
    "moderate": "Ισορροπημένο",
    "aggressive": "Δυναμικό",
}

PRIORITY_LABELS = {
    "critical": "Κρίσιμο",
    "high": "Υψηλό",
    "medium": "Μέτριο",
    "low": "Χαμηλό",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "portfolio": None,
        "risk_analysis": None,
        "recommendations": None,
        "scenario_engine": None,
        "portfolio_name": "",
        "risk_profile": "moderate",
        "last_loaded": None,
        "validation_results": None,
        "user_archetypes": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_portfolio(csv_path: str, risk_profile: str, name: str):
    with st.spinner("Λήψη δεδομένων αγοράς από Yahoo Finance…"):
        try:
            portfolio = build_portfolio(
                csv_path=csv_path,
                risk_profile=risk_profile,
                portfolio_name=name,
                history_years=5,
            )
        except Exception as exc:
            st.error(f"Αποτυχία φόρτωσης χαρτοφυλακίου: {exc}")
            return

    try:
        monitor = RiskMonitor(portfolio)
        analysis = monitor.run_full_analysis()
    except Exception as exc:
        st.error(f"Αποτυχία ανάλυσης κινδύνου: {exc}")
        st.exception(exc)
        return

    try:
        engine = ScenarioEngine(portfolio)
    except Exception as exc:
        st.error(f"Αποτυχία μηχανισμού σεναρίων: {exc}")
        st.exception(exc)
        return

    try:
        rec_engine = RecommendationEngine(portfolio, monitor, use_optimizer=False)
        recs = rec_engine.generate_recommendations()
    except Exception as exc:
        st.warning(f"Οι προτάσεις δεν φορτώθηκαν (συνέχεια χωρίς αυτές): {exc}")
        recs = []

    st.session_state["portfolio"] = portfolio
    st.session_state["risk_analysis"] = analysis
    st.session_state["recommendations"] = recs
    st.session_state["scenario_engine"] = engine
    st.session_state["portfolio_name"] = name
    st.session_state["risk_profile"] = risk_profile
    st.session_state["last_loaded"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    st.success("Το χαρτοφυλάκιο φορτώθηκε επιτυχώς!")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("📊 Ρυθμίσεις Χαρτοφυλακίου")

    source = st.sidebar.radio(
        "Πηγή δεδομένων",
        ["Δείγμα χαρτοφυλακίου", "Ανέβασμα δικού μου CSV"],
        index=0,
    )

    risk_profile = st.sidebar.selectbox(
        "Προφίλ κινδύνου επενδυτή",
        ["liquidity_plus", "defensive", "flexible", "growth", "dynamic"],
        format_func=lambda x: RISK_PROFILE_LABELS[x],
        index=2,
    )

    if source == "Δείγμα χαρτοφυλακίου":
        # Auto-suggest sample based on risk profile
        _profile_to_sample = {
            "liquidity_plus": "Συντηρητικό (Εισόδημα)",
            "defensive":      "Συντηρητικό (Εισόδημα)",
            "flexible":       "Ισορροπημένο (Μέτριο)",
            "growth":         "Δυναμικό (Επιθετικό)",
            "dynamic":        "Δυναμικό (Επιθετικό)",
        }
        default_sample = _profile_to_sample.get(risk_profile, "Ισορροπημένο (Μέτριο)")
        default_idx = list(SAMPLE_PORTFOLIOS.keys()).index(default_sample)
        sample_name = st.sidebar.selectbox(
            "Επιλογή δείγματος",
            list(SAMPLE_PORTFOLIOS.keys()),
            index=default_idx,
        )
        csv_path = SAMPLE_PORTFOLIOS[sample_name]
        display_name = sample_name
    else:
        uploaded = st.sidebar.file_uploader(
            "Ανέβασμα CSV χαρτοφυλακίου",
            type=["csv"],
            help="Απαιτούμενες στήλες: ticker, quantity, entry_price",
        )
        if uploaded is None:
            st.sidebar.info("Ανεβάστε ένα CSV για να συνεχίσετε.")
            return
        tmp_path = str(DATA_DIR / "uploaded_portfolio.csv")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        csv_path = tmp_path
        display_name = uploaded.name.replace(".csv", "")

    if st.sidebar.button("Φόρτωση / Ανανέωση Χαρτοφυλακίου"):
        _load_portfolio(csv_path, risk_profile, display_name)

    if st.session_state.get("last_loaded"):
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Τελευταία ενημέρωση: {st.session_state['last_loaded']}")
        st.sidebar.caption("Πηγή δεδομένων: Yahoo Finance")

    with st.sidebar.expander("Μορφή αρχείου CSV"):
        st.markdown(
            "Το CSV πρέπει να έχει τις παρακάτω στήλες:\n"
            "```\nticker,quantity,entry_price\n"
            "AAPL,100,150.00\nMSFT,50,300.00\n```"
        )

    st.sidebar.markdown("---")
    st.sidebar.checkbox("📖 Τεκμηρίωση Συστήματος", key="show_docs")


# ---------------------------------------------------------------------------
# Tab 1: Επισκόπηση Χαρτοφυλακίου
# ---------------------------------------------------------------------------

def _risk_level(metrics: dict) -> tuple[str, str]:
    """Return (label, color) for a simple risk level badge."""
    vol = metrics.get("volatility_annual_pct") or 0
    beta = metrics.get("beta") or 1
    score = vol * 0.6 + beta * 5
    if score < 10:
        return "🟢 Χαμηλό", "#28a745"
    if score < 16:
        return "🟡 Μέτριο", "#ffc107"
    if score < 22:
        return "🟠 Υψηλό", "#fd7e14"
    return "🔴 Πολύ Υψηλό", "#dc3545"


def _portfolio_health_messages(portfolio) -> list[tuple[str, str]]:
    """
    Return list of (message, type) where type is 'success'|'warning'|'error'.
    These are the interpretive plain-language assessments.
    """
    msgs = []
    metrics = portfolio.get_metrics()
    composition = portfolio.get_composition()
    alignment = portfolio.get_risk_profile_alignment()

    vol = metrics.get("volatility_annual_pct") or 0
    tgt_vol = metrics.get("target_volatility_pct") or 12
    sharpe = metrics.get("sharpe_ratio") or 0
    hhi = composition.get("concentration", {}).get("herfindahl_index", 0)
    n = len(portfolio.assets)
    div_ratio = metrics.get("diversification_ratio") or 1
    max_drift = alignment.get("max_drift", 0)

    # Volatility
    if vol - tgt_vol > 4:
        msgs.append((f"Υψηλή μεταβλητότητα ({vol:.1f}% vs στόχο {tgt_vol:.1f}%) — το χαρτοφυλάκιο είναι πιο επικίνδυνο από το προφίλ σας.", "error"))
    elif vol - tgt_vol > 2:
        msgs.append((f"Μεταβλητότητα ({vol:.1f}%) ελαφρά πάνω από τον στόχο ({tgt_vol:.1f}%).", "warning"))
    else:
        msgs.append((f"Μεταβλητότητα ({vol:.1f}%) εντός στόχου ({tgt_vol:.1f}%). ✓", "success"))

    # Concentration
    if hhi > 0.25:
        msgs.append((f"Υψηλή συγκέντρωση (Herfindahl={hhi:.2f}) — ένας ή λίγοι τίτλοι κυριαρχούν στο χαρτοφυλάκιο.", "error"))
    elif hhi > 0.15:
        msgs.append((f"Μέτρια συγκέντρωση (Herfindahl={hhi:.2f}) — αξίζει επανεξέταση.", "warning"))

    # Diversification
    if n < 5:
        msgs.append((f"Χαμηλή διαφοροποίηση — μόνο {n} θέσεις. Συνιστάται τουλάχιστον 10.", "error"))
    elif n < 10:
        msgs.append((f"Περιορισμένη διαφοροποίηση ({n} θέσεις). Σκεφτείτε επέκταση.", "warning"))
    elif div_ratio > 1.3:
        msgs.append((f"Καλή διαφοροποίηση (Diversification Ratio={div_ratio:.2f}). ✓", "success"))

    # Sharpe
    if sharpe < 0:
        msgs.append(("Αρνητικός Sharpe Ratio — το χαρτοφυλάκιο αποδίδει χειρότερα από επένδυση χωρίς κίνδυνο.", "error"))
    elif sharpe < 0.5:
        msgs.append((f"Χαμηλός Sharpe Ratio ({sharpe:.2f}) — χαμηλή απόδοση για τον κίνδυνο που αναλαμβάνετε.", "warning"))
    elif sharpe > 1.0:
        msgs.append((f"Καλός Sharpe Ratio ({sharpe:.2f}) — ικανοποιητική αποζημίωση για τον κίνδυνο. ✓", "success"))

    # Profile drift
    if max_drift > 10:
        msgs.append((f"Σημαντική απόκλιση από το προφίλ κινδύνου ({max_drift:.1f}%) — απαιτείται rebalancing.", "error"))
    elif max_drift > 5:
        msgs.append((f"Μέτρια απόκλιση από το προφίλ κινδύνου ({max_drift:.1f}%).", "warning"))

    return msgs


def render_overview(portfolio):
    st.header("Επισκόπηση Χαρτοφυλακίου (Portfolio Overview)")

    metrics = portfolio.get_metrics()
    composition = portfolio.get_composition()

    # --- Risk level badge + interpretive health panel ---
    risk_label, risk_color = _risk_level(metrics)
    st.markdown(
        f"<div style='display:inline-block; background:{risk_color}22; "
        f"border:2px solid {risk_color}; border-radius:8px; padding:6px 18px; "
        f"font-size:1.1rem; font-weight:bold; color:{risk_color}; margin-bottom:12px;'>"
        f"Επίπεδο Κινδύνου: {risk_label}</div>",
        unsafe_allow_html=True,
    )

    health_msgs = _portfolio_health_messages(portfolio)
    for msg, mtype in health_msgs:
        if mtype == "success":
            st.success(msg)
        elif mtype == "warning":
            st.warning(msg)
        else:
            st.error(msg)

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Συνολική Αξία", format_currency(metrics["total_value"]))
    c2.metric(
        "Συνολική Απόδοση (Return)",
        format_pct(metrics["total_return_pct"], show_sign=True),
    )
    vol = metrics.get("volatility_annual_pct")
    tgt_vol = metrics.get("target_volatility_pct")
    c3.metric(
        "Μεταβλητότητα — Volatility (ετήσια)",
        format_pct(vol) if vol else "N/A",
        delta=format_pct((vol or 0) - (tgt_vol or 0), show_sign=True) + " έναντι στόχου"
        if vol and tgt_vol else None,
        delta_color="inverse",
    )
    c4.metric("Δείκτης Sharpe (Sharpe Ratio)", format_ratio(metrics.get("sharpe_ratio")))
    c5.metric("Beta", format_ratio(metrics.get("beta")))

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Θέσεις Χαρτοφυλακίου (Holdings)")
        rows = []
        for a in portfolio.assets:
            rows.append({
                "Ticker": a.ticker,
                "Όνομα": a.name[:25] if a.name else a.ticker,
                "Τεμάχια": int(a.quantity),
                "Τρέχουσα Τιμή": f"${a.current_price:,.2f}",
                "Αξία": format_currency(a.current_value),
                "Κατανομή %": format_pct(a.current_value / portfolio.total_value * 100),
                "Κέρδος/Ζημία (P&L)": format_pct(a.unrealized_pnl_pct, show_sign=True),
                "Κλάδος (Sector)": a.sector,
                "Κατηγορία": a.asset_class,
            })
        st.dataframe(pd.DataFrame(rows))

    with col_right:
        st.subheader("Σύνθεση (Composition)")
        _plot_composition(composition)

    st.markdown("---")
    st.subheader("Αναλυτικοί Δείκτες")

    # --- Period Returns ---
    period_rets = portfolio.calculate_period_returns()
    if period_rets:
        st.markdown("**Αποδόσεις ανά Περίοδο (Period Returns)**")
        pr1, pr2, pr3, pr4, pr5 = st.columns(5)
        def _pr(val):
            if val is None:
                return "N/A"
            return f"{val:+.2f}%"
        pr1.metric("MTD",  _pr(period_rets.get("mtd_pct")))
        pr2.metric("YTD",  _pr(period_rets.get("ytd_pct")))
        pr3.metric("1 Έτος",  _pr(period_rets.get("1y_pct")))
        pr4.metric("3 Έτη", _pr(period_rets.get("3y_pct")))
        pr5.metric("5 Έτη", _pr(period_rets.get("5y_pct")))
        st.markdown("---")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**Απόδοση (Performance)**")
        st.write(f"- Συνολική αξία: {format_currency(metrics['total_value'])}")
        st.write(f"- Κόστος κτήσης: {format_currency(metrics['total_cost'])}")
        st.write(f"- Αποτίμηση P&L: {format_pct(metrics['total_return_pct'], show_sign=True)}")
        st.write(f"- Αναμενόμενη ετήσια απόδοση: {format_pct(metrics.get('expected_annual_return_pct', 0), show_sign=True)}")
    with m2:
        st.markdown("**Risk-Adjusted Returns**")
        st.write(f"- Sharpe Ratio: {format_ratio(metrics.get('sharpe_ratio'))}")
        st.write(f"- Sortino Ratio: {format_ratio(metrics.get('sortino_ratio'))}")
        st.write(f"- Treynor Ratio: {format_ratio(metrics.get('treynor_ratio'))}")
        ir = metrics.get("information_ratio")
        st.write(f"- Information Ratio: {format_ratio(ir) if ir is not None else 'N/A'}")
    with m3:
        st.markdown("**Κίνδυνος (Risk)**")
        vol_3y = metrics.get("volatility_3y_pct")
        st.write(f"- Volatility 1Y: {format_pct(vol)} (στόχος: {format_pct(tgt_vol)})")
        st.write(f"- Volatility 3Y: {format_pct(vol_3y) if vol_3y else 'N/A'}")
        st.write(f"- Beta: {format_ratio(metrics.get('beta'))}")
        st.write(f"- Max Drawdown: {format_pct(metrics.get('max_drawdown_pct'), show_sign=True)}")
        st.write(f"- VaR 95% (μηνιαίο): {format_pct(metrics.get('var_95_monthly_pct'), show_sign=True)}")
        st.write(f"- Diversification Ratio: {format_ratio(metrics.get('diversification_ratio'))}")

    alignment = portfolio.get_risk_profile_alignment()
    st.subheader("Εναρμόνιση με Προφίλ Κινδύνου")
    a1, a2, a3 = st.columns(3)
    a1.metric("Μετοχές (Equity)", format_pct(alignment["current_equity_pct"]),
              delta=format_pct(alignment["equity_drift_pct"], show_sign=True) + " έναντι στόχου",
              delta_color="inverse")
    a2.metric("Ομόλογα (Fixed Income)", format_pct(alignment["current_fixed_income_pct"]),
              delta=format_pct(alignment["fixed_income_drift_pct"], show_sign=True) + " έναντι στόχου",
              delta_color="inverse")
    a3.metric("Στόχος Προφίλ", RISK_PROFILE_LABELS.get(alignment["risk_profile"], alignment["risk_profile"]))

    # --- Rebalancing button & before/after comparison ---
    st.markdown("---")
    st.subheader("Πρόταση Ανακατανομής (Rebalancing)")
    st.markdown(
        "Πατήστε το κουμπί για να δείτε τι αλλάζει αν το χαρτοφυλάκιο "
        "επαναφερθεί στην κατανομή-στόχο του προφίλ σας."
    )
    if st.button("Προτείνω Ανακατανομή Κεφαλαίων"):
        _render_rebalancing_comparison(portfolio, alignment, metrics)

    # --- Correlation Heatmap ---
    st.markdown("---")
    with st.expander("🔗 Πίνακας Συσχέτισης Τίτλων (Correlation Matrix)", expanded=False):
        _plot_correlation_heatmap(portfolio)


def _render_rebalancing_comparison(portfolio, alignment: dict, metrics: dict):
    total = portfolio.total_value
    rp = alignment["risk_profile"]

    from src.portfolio import Portfolio as _P
    targets = _P.TARGET_ALLOCATION.get(rp, {})

    current_eq = alignment["current_equity_pct"]
    current_fi = alignment["current_fixed_income_pct"]
    target_eq = targets.get("equity", 0) * 100
    target_fi = targets.get("fixed_income", 0) * 100
    target_alt = targets.get("alternative", 0) * 100

    eq_drift = current_eq - target_eq
    fi_drift = current_fi - target_fi

    if abs(eq_drift) < 2 and abs(fi_drift) < 2:
        st.success("Το χαρτοφυλάκιο είναι ήδη εντός 2% του στόχου — δεν χρειάζεται ανακατανομή.")
        return

    shift = abs(eq_drift) / 100 * total

    st.markdown("#### Σύγκριση Πριν / Μετά την Ανακατανομή")

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("**Τρέχουσα κατανομή (Πριν)**")
        st.write(f"- Μετοχές (Equity): **{current_eq:.1f}%**")
        st.write(f"- Ομόλογα (Fixed Income): **{current_fi:.1f}%**")
        vol_before = metrics.get("volatility_annual_pct") or 0
        sharpe_before = metrics.get("sharpe_ratio") or 0
        st.write(f"- Volatility: **{vol_before:.1f}%**")
        st.write(f"- Sharpe Ratio: **{sharpe_before:.2f}**")

    with col_after:
        st.markdown("**Κατανομή-στόχος (Μετά)**")
        st.write(f"- Μετοχές (Equity): **{target_eq:.1f}%**")
        st.write(f"- Ομόλογα (Fixed Income): **{target_fi:.1f}%**")
        # Rough estimates: each 10% shift to bonds reduces vol ~2.5% and adjusts sharpe
        vol_after = max(vol_before + (target_eq - current_eq) / 10 * 2.5, 1.0)
        sharpe_after = sharpe_before * (vol_before / vol_after) if vol_after > 0 else sharpe_before
        st.write(f"- Volatility (εκτίμηση): **{vol_after:.1f}%**")
        st.write(f"- Sharpe Ratio (εκτίμηση): **{sharpe_after:.2f}**")

    st.markdown("---")
    st.markdown("**Προτεινόμενες Κινήσεις:**")
    if eq_drift > 0:
        st.write(
            f"- Πώληση μετοχών αξίας **{format_currency(shift)}** "
            f"(μείωση Equity από {current_eq:.1f}% → {target_eq:.1f}%)"
        )
        st.write(
            f"- Αγορά ομολόγων αξίας **{format_currency(shift)}** "
            f"(αύξηση Fixed Income από {current_fi:.1f}% → {target_fi:.1f}%)"
        )
    else:
        st.write(
            f"- Αγορά μετοχών αξίας **{format_currency(shift)}** "
            f"(αύξηση Equity από {current_eq:.1f}% → {target_eq:.1f}%)"
        )
        st.write(
            f"- Πώληση ομολόγων αξίας **{format_currency(shift)}** "
            f"(μείωση Fixed Income από {current_fi:.1f}% → {target_fi:.1f}%)"
        )
    if target_alt > 0:
        st.info(
            f"Το προφίλ «{RISK_PROFILE_LABELS.get(rp, rp)}» προβλέπει επίσης "
            f"{target_alt:.0f}% σε εναλλακτικές κατηγορίες (π.χ. Real Estate, Commodities)."
        )
    st.caption(
        "Σημείωση: Οι εκτιμήσεις Volatility/Sharpe μετά το rebalancing είναι προσεγγιστικές. "
        "Για ακριβή ανάλυση δείτε τις αναλυτικές προτάσεις στην καρτέλα Προτάσεις Αναδιάρθρωσης."
    )


def _plot_correlation_heatmap(portfolio):
    """Render a colour-coded correlation matrix for portfolio assets."""
    corr = portfolio.calculate_correlation_matrix()
    if corr.empty or corr.shape[0] < 2:
        st.info("Χρειάζονται τουλάχιστον 2 τίτλοι με ιστορικά δεδομένα για τον πίνακα συσχέτισης.")
        return

    tickers = list(corr.columns)
    n = len(tickers)
    fig_size = max(5, n * 0.75)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    mat = corr.values
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(tickers, color="white", fontsize=9)

    # Annotate cells with correlation value
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            text_color = "black" if abs(val) < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    ax.set_title("Πίνακας Συσχέτισης (Pearson, ημερήσιες αποδόσεις)",
                 color="white", fontsize=11, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Interpretation guide
    st.caption(
        "🟢 Τιμές κοντά στο +1: ισχυρή θετική συσχέτιση (κινούνται μαζί) | "
        "🔴 Τιμές κοντά στο -1: αντίθετη κίνηση (καλή διαφοροποίηση) | "
        "⚪ Τιμές κοντά στο 0: ανεξάρτητη κίνηση"
    )

    # Flag highly correlated pairs
    high_corr = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(mat[i, j]) >= 0.8 and i != j:
                high_corr.append((tickers[i], tickers[j], mat[i, j]))

    if high_corr:
        st.warning(
            "**Υψηλή συσχέτιση** (≥ 0.80) — οι παρακάτω ζεύγοι κινούνται σχεδόν ταυτόχρονα, "
            "μειώνοντας το όφελος διαφοροποίησης:\n"
            + "\n".join(f"- {a} & {b}: {v:.2f}" for a, b, v in high_corr)
        )


def _plot_composition(composition: dict):
    by_class = composition.get("by_asset_class", {})
    if not by_class:
        st.info("Δεν υπάρχουν αρκετά δεδομένα για γράφημα σύνθεσης.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0e1117")

    def _pie(ax, data: dict, title: str):
        labels = [f"{k}\n{v:.1f}%" for k, v in data.items() if v > 0.5]
        values = [v for v in data.values() if v > 0.5]
        colors = plt.cm.Set3.colors[:len(values)]
        ax.pie(values, labels=labels, colors=colors, startangle=140,
               textprops={"color": "white", "fontsize": 8})
        ax.set_title(title, color="white", fontsize=10)
        ax.set_facecolor("#0e1117")

    _pie(axes[0], by_class, "Ανά Κατηγορία")

    by_sector = composition.get("by_sector", {})
    if by_sector:
        top = dict(sorted(by_sector.items(), key=lambda x: x[1], reverse=True)[:6])
        _pie(axes[1], top, "Ανά Κλάδο (top 6)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 2: Ανάλυση Σεναρίων
# ---------------------------------------------------------------------------

def render_scenarios(portfolio, engine: ScenarioEngine):
    st.header("Ανάλυση Σεναρίων — What-If Προσομοίωση")
    st.markdown(
        "Επιλέξτε ένα σενάριο για να δείτε πώς θα αντιδρούσε το χαρτοφυλάκιό σας "
        "υπό διαφορετικές συνθήκες αγοράς."
    )

    scenarios = engine.list_scenarios()
    categories = sorted(set(s["category"] for s in scenarios))
    selected_category = st.selectbox("Φίλτρο ανά κατηγορία", ["Όλα"] + categories)

    filtered = [s for s in scenarios
                if selected_category == "Όλα" or s["category"] == selected_category]

    scenario_names = {s["id"]: s["name"] for s in filtered}
    selected_id = st.selectbox(
        "Επιλογή σεναρίου",
        options=list(scenario_names.keys()),
        format_func=lambda x: scenario_names[x],
    )

    # Προσαρμοσμένο σενάριο
    st.markdown("**Ή φτιάξτε το δικό σας σενάριο:**")
    col_a, col_b, col_c = st.columns(3)
    custom_market = col_a.slider("Μεταβολή αγοράς (%)", -50, 30, 0, step=5) / 100
    custom_rate = col_b.slider("Μεταβολή επιτοκίου (%)", -2.0, 3.0, 0.0, step=0.25) / 100
    custom_vol = col_c.slider("Πολλαπλασιαστής μεταβλητότητας", 0.5, 5.0, 1.0, step=0.25)
    custom_name = st.text_input("Όνομα σεναρίου", "Προσαρμοσμένο Σενάριο")
    run_custom = st.button("Εκτέλεση Προσαρμοσμένου Σεναρίου")

    st.markdown("---")

    if run_custom:
        engine.create_custom_scenario(
            name=custom_name,
            market_change=custom_market,
            rate_change=custom_rate,
            volatility_multiplier=custom_vol,
        )
        custom_id = list(engine.scenarios.keys())[-1]
        comparison = engine.compare_portfolio_metrics(custom_id)
        _render_scenario_result(comparison, engine, custom_id)
    else:
        comparison = engine.compare_portfolio_metrics(selected_id)
        _render_scenario_result(comparison, engine, selected_id)

    st.markdown("---")
    st.subheader("Σύγκριση Πολλαπλών Σεναρίων")
    stress_ids = ["market_down_10", "market_down_20", "financial_crisis_2008",
                  "covid_crash_2020", "recession", "inflation_spike"]
    available_stress = [s for s in stress_ids if s in engine.scenarios]
    multi = engine.compare_multiple_scenarios(available_stress)
    _render_multi_scenario_table(multi)


def _render_scenario_result(comparison: dict, engine: ScenarioEngine, scenario_id: str):
    name = comparison["scenario_name"]
    desc = comparison["scenario_description"]
    params = comparison["scenario_params"]
    summary = comparison["summary"]
    metrics = comparison["metrics"]

    st.subheader(f"Αποτέλεσμα: {name}")
    st.caption(desc)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Μεταβολή αγοράς", format_pct(params["market_change_pct"], show_sign=True))
    pc2.metric("Μεταβολή επιτοκίου", format_pct(params["rate_change_pct"], show_sign=True))
    pc3.metric("Πολλαπλ. μεταβλητότητας", f"{params['volatility_multiplier']:.1f}x")
    if params.get("sector_overrides"):
        st.write("Επιπτώσεις ανά κλάδο: " + ", ".join(
            f"{sec}: {format_pct(chg * 100, show_sign=True)}"
            for sec, chg in params["sector_overrides"].items()
        ))

    port_chg = summary["portfolio_change_pct"]
    resilience = summary["resilience_vs_market_pct"]

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Τρέχουσα Αξία", format_currency(summary["current_total_value"]))
    rc2.metric(
        "Αξία στο Σενάριο",
        format_currency(summary["scenario_total_value"]),
        delta=format_pct(port_chg, show_sign=True),
        delta_color="normal" if port_chg >= 0 else "inverse",
    )
    resilience_label = "vs Market" if params["market_change_pct"] != 0 else ""
    rc3.metric(
        "Relative Performance vs Market",
        format_pct(resilience, show_sign=True) + f" {resilience_label}",
        help="Θετικό = outperformance έναντι αγοράς (portfolio return − market return). "
             "Αντικατοπτρίζει το όφελος διαφοροποίησης (alpha vs benchmark).",
    )

    st.subheader("Σύγκριση Δεικτών")
    table_rows = []
    metric_display = {
        "total_value": "Αξία Χαρτοφυλακίου ($)",
        "volatility_annual_pct": "Μεταβλητότητα — Volatility (%)",
        "sharpe_ratio": "Δείκτης Sharpe",
        "max_drawdown_pct": "Μέγιστη Απώλεια — Max Drawdown (%)",
        "beta": "Beta",
        "var_95_monthly_pct": "VaR 95% μηνιαίο (%)",
    }
    for key, label in metric_display.items():
        m = metrics.get(key, {})
        curr = m.get("current")
        scen = m.get("scenario")
        chg = m.get("change")
        chg_pct = m.get("change_pct")
        if curr is not None:
            table_rows.append({
                "Δείκτης": label,
                "Τρέχον": f"{curr:,.2f}",
                "Σενάριο": f"{scen:,.2f}" if scen is not None else "N/A",
                "Μεταβολή": f"{chg:+,.2f}" if chg is not None else "N/A",
                "Μεταβολή %": f"{chg_pct:+.1f}%" if chg_pct is not None else "N/A",
            })

    st.dataframe(pd.DataFrame(table_rows))

    st.subheader("Επίπτωση ανά Τίτλο (Asset)")
    impacts = engine.get_asset_impact(scenario_id)
    impact_df = pd.DataFrame(impacts)[
        ["ticker", "sector", "current_value", "scenario_value", "dollar_change", "pct_change"]
    ]
    impact_df.columns = [
        "Ticker", "Κλάδος (Sector)",
        "Τρέχουσα Αξία ($)", "Αξία Σεναρίου ($)",
        "Μεταβολή ($)", "Μεταβολή (%)"
    ]
    st.dataframe(impact_df)

    with st.expander("Επεξήγηση σεναρίου"):
        st.markdown(EXPLAINER.explain_scenario(comparison))


def _render_multi_scenario_table(multi: dict):
    rows = []
    for sid, data in multi.items():
        row = {
            "Σενάριο": data["name"],
            "Αξία ($)": format_currency(data["total_value"]),
            "Μεταβολή (%)": "—" if sid == "base" else format_pct(data["change_pct"], show_sign=True),
            "Volatility (%)": format_pct(data.get("volatility_pct")),
            "Sharpe": format_ratio(data.get("sharpe_ratio")),
            "Max Drawdown (%)": format_pct(data.get("max_drawdown_pct")),
        }
        rows.append(row)
    st.dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Tab 3: Παρακολούθηση Κινδύνου
# ---------------------------------------------------------------------------

def render_risk(analysis: dict):
    st.header("Παρακολούθηση Κινδύνου (Risk Monitoring)")

    summary = analysis["summary"]
    highest = analysis["highest_severity"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Σύνολο Ελέγχων", summary["total"])
    c2.metric("✅ Φυσιολογικό", summary["ok"])
    c3.metric("⚠️ Προσοχή", summary["caution"])
    c4.metric("🚨 Προειδοποίηση", summary["alert"])

    if highest == "alert":
        st.error("Το χαρτοφυλάκιο έχει κρίσιμες προειδοποιήσεις κινδύνου. Ελέγξτε τις προτάσεις άμεσα.")
    elif highest == "caution":
        st.warning("Υπάρχουν σημεία που χρήζουν προσοχής. Δείτε τις προτάσεις.")
    else:
        st.success("Το προφίλ κινδύνου του χαρτοφυλακίου είναι υγιές.")

    st.markdown("---")

    severity_labels = {
        "alert": "🚨 Προειδοποιήσεις (Alerts)",
        "caution": "⚠️ Σημεία Προσοχής (Cautions)",
        "ok": "✅ Φυσιολογικά",
    }
    for severity in ("alert", "caution", "ok"):
        bucket = analysis[severity]
        if not bucket:
            continue
        st.subheader(severity_labels[severity])
        for alert in bucket:
            _render_alert_card(alert)


def _render_alert_card(alert: dict):
    sev = alert["severity"]
    emoji = severity_emoji(sev)
    color = severity_color(sev)
    title = alert["title"]
    category = alert["category"]
    description = alert["description"]
    recommendation = alert.get("recommendation", "")
    detail = alert.get("detail", {})

    with st.container():
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding: 8px 16px; "
            f"margin-bottom: 12px; border-radius: 4px;'>"
            f"<strong>{emoji} {category}: {title}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.write(description)
        if recommendation and sev != "ok":
            st.info(f"**Σύσταση:** {recommendation}")

        if detail and sev != "ok":
            with st.expander("Αναλυτικά στοιχεία"):
                for k, v in detail.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        st.write(f"- **{k.replace('_', ' ').title()}**: {v}")
                    elif isinstance(v, dict):
                        st.write(f"- **{k.replace('_', ' ').title()}**: {v}")


# ---------------------------------------------------------------------------
# Tab 4: Προτάσεις Αναδιάρθρωσης
# ---------------------------------------------------------------------------

def render_recommendations(recs: list):
    st.header("Προτάσεις Αναδιάρθρωσης (Rebalancing Recommendations)")
    st.markdown(
        "Αυτόματα παραγόμενες, επεξηγήσιμες προτάσεις αναδιάρθρωσης χαρτοφυλακίου, "
        "ταξινομημένες κατά προτεραιότητα. Κάθε πρόταση αντιμετωπίζει συγκεκριμένο κίνδυνο."
    )

    if not recs:
        st.success("Δεν υπάρχουν προτάσεις — το χαρτοφυλάκιο είναι ισορροπημένο!")
        return

    by_priority = {}
    for r in recs:
        p = r.get("priority", "low")
        by_priority.setdefault(p, []).append(r)

    pc = st.columns(4)
    priority_labels_display = {
        "critical": "🔴 Κρίσιμο",
        "high": "🟠 Υψηλό",
        "medium": "🟡 Μέτριο",
        "low": "⚪ Χαμηλό",
    }
    for i, p in enumerate(["critical", "high", "medium", "low"]):
        count = len(by_priority.get(p, []))
        pc[i].metric(priority_labels_display[p], count)

    st.markdown("---")

    for rec in recs:
        _render_recommendation_card(rec)

    st.markdown("---")
    st.subheader("Σύνοψη Συναλλαγών (Trade Summary)")
    all_trades = [t for r in recs for t in r.get("trades", [])]
    if all_trades:
        sell_total = sum(t["value"] for t in all_trades if t.get("action") == "SELL")
        buy_total = sum(t["value"] for t in all_trades if t.get("action") == "BUY")
        net = buy_total - sell_total

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Σύνολο Πωλήσεων", format_currency(sell_total))
        tc2.metric("Σύνολο Αγορών", format_currency(buy_total))
        tc3.metric("Καθαρή Ανάγκη Κεφαλαίου", format_currency(net),
                   delta_color="inverse" if net > 0 else "normal")

        trades_df = pd.DataFrame(all_trades)
        if not trades_df.empty and "ticker" in trades_df.columns:
            display_cols = ["ticker", "action", "quantity", "price", "value"]
            display_cols = [c for c in display_cols if c in trades_df.columns]
            trades_df = trades_df[display_cols].rename(columns={
                "ticker": "Ticker",
                "action": "Ενέργεια",
                "quantity": "Τεμάχια",
                "price": "Τιμή ($)",
                "value": "Αξία ($)",
            })
            st.dataframe(trades_df)

    st.markdown("---")
    if st.button("Εξαγωγή Προτάσεων (κείμενο)"):
        text = EXPLAINER.explain_all_recommendations(recs)
        st.download_button(
            label="Λήψη recommendations.txt",
            data=text,
            file_name="recommendations.txt",
            mime="text/plain",
        )


def _render_recommendation_card(rec: dict):
    priority = rec.get("priority", "medium")
    color = priority_color(priority)
    title = rec.get("title", "")
    action = rec.get("action", "")
    why = rec.get("why", "")
    impact = rec.get("impact", {})
    confidence = rec.get("confidence", "medium")
    alternative = rec.get("alternative", "")
    trades = rec.get("trades", [])
    category = rec.get("category", "")
    rec_id = rec.get("id", "")

    confidence_labels = {"high": "Υψηλή", "medium": "Μέτρια", "low": "Χαμηλή"}
    conf_label = confidence_labels.get(confidence, confidence.capitalize())
    priority_label = PRIORITY_LABELS.get(priority, priority.capitalize())

    with st.expander(
        f"[{priority_label.upper()}] #{rec_id} — {title}  |  {category}  |  Εμπιστοσύνη: {conf_label}",
        expanded=(priority in ("critical", "high")),
    ):
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding: 8px 16px; "
            f"margin-bottom: 8px;'><strong>Ενέργεια:</strong> {action}</div>",
            unsafe_allow_html=True,
        )

        col_why, col_impact = st.columns(2)

        with col_why:
            st.markdown("**Γιατί αυτή η πρόταση:**")
            st.write(why)
            if alternative:
                st.markdown(f"**Εναλλακτικά:** {alternative}")

        with col_impact:
            st.markdown("**Αναμενόμενη επίπτωση:**")
            before = impact.get("before", {})
            after = impact.get("after", {})
            improvement = impact.get("improvement", "")

            for key in before:
                b = before.get(key)
                a = after.get(key)
                if b is not None and a is not None and isinstance(b, (int, float)):
                    try:
                        delta = a - b
                        st.write(f"- {key.replace('_', ' ').title()}: "
                                 f"{b:.2f} → {a:.2f} ({delta:+.2f})")
                    except Exception:
                        st.write(f"- {key}: {b} → {a}")

            if improvement:
                st.info(improvement)

        if trades:
            st.markdown("**Συγκεκριμένες συναλλαγές:**")
            for t in trades:
                qty = t.get("quantity", 0)
                action_label = "ΑΓΟΡΑ" if t.get("action") == "BUY" else "ΠΩΛΗΣΗ"
                if qty:
                    st.write(
                        f"- **{action_label}** {qty} τεμ. **{t.get('ticker')}** "
                        f"@ ${t.get('price', 0):.2f} = {format_currency(t.get('value', 0))}"
                    )


# ---------------------------------------------------------------------------
# Documentation Tab
# ---------------------------------------------------------------------------

def render_docs():
    st.header("📖 Τεκμηρίωση Συστήματος")
    st.markdown(
        "Πλήρης οδηγός χρήσης, αρχιτεκτονική κώδικα, στρατηγικές ανάλυσης "
        "και σενάρια χρήσης (use cases) του συστήματος."
    )

    # ------------------------------------------------------------------ stack
    with st.expander("🏗️ Τεχνολογική Στοίβα (Tech Stack)", expanded=False):
        st.markdown("""
**Frontend / UI**
- **Streamlit** — Python web framework για data apps. Κάθε αλληλεπίδραση του χρήστη (κλικ, slider) κάνει rerun ολόκληρου του script.

**Δεδομένα Αγοράς**
- **yfinance** — Ανακτά τιμές μετοχών από Yahoo Finance (τρέχουσες τιμές + ιστορικό 5 ετών).
- **pandas / numpy** — Επεξεργασία χρονοσειρών τιμών και υπολογισμός στατιστικών.

**Χρηματοοικονομική Ανάλυση**
- **PyPortfolioOpt** (`pyportfolioopt`) — Βελτιστοποίηση χαρτοφυλακίου με βάση τον Efficient Frontier (Modern Portfolio Theory).
- **scikit-learn** — Χρησιμοποιείται εσωτερικά για στατιστικές βοηθητικές λειτουργίες.

**Οπτικοποίηση**
- **matplotlib** — Γραφήματα (pie charts κατανομής, correlation heatmap).

**Αρχιτεκτονική Modules**
| Αρχείο | Ρόλος |
|---|---|
| `app.py` | Streamlit UI — όλο το dashboard |
| `src/data_loader.py` | Φόρτωση CSV, yfinance API calls |
| `src/portfolio.py` | Digital Twin — υπολογισμός όλων των μετρικών |
| `src/risk_monitor.py` | 6 κατηγορίες ελέγχου κινδύνου |
| `src/scenario_engine.py` | 14 προκαθορισμένα σενάρια + custom |
| `src/recommendations.py` | Παραγωγή προτάσεων αναδιάρθρωσης |
| `src/explainer.py` | XAI layer — plain-English/Greek επεξηγήσεις |
| `src/utils.py` | Βοηθητικές συναρτήσεις, formatters |
""")

    # ------------------------------------------------------------------ concept
    with st.expander("🧠 Έννοια του Digital Twin", expanded=False):
        st.markdown("""
**Τι είναι το Digital Twin στην επενδυτική συμβουλευτική;**

Το Digital Twin (Ψηφιακό Δίδυμο) είναι μια εικονική αναπαράσταση του πραγματικού χαρτοφυλακίου.
Αντί να εκτελείς πραγματικές συναλλαγές για να δεις τι θα γινόταν, το σύστημα:

1. **Δημιουργεί ένα ψηφιακό αντίγραφο** του χαρτοφυλακίου με τρέχουσες τιμές αγοράς
2. **Προσομοιώνει σενάρια** (π.χ. «τι γίνεται αν η αγορά πέσει 30%;») χωρίς πραγματικό κίνδυνο
3. **Παρακολουθεί κινδύνους** σε πραγματικό χρόνο και εντοπίζει αποκλίσεις
4. **Εξηγεί τις αποφάσεις** (Explainable AI) — κάθε πρόταση έχει αιτιολόγηση

**Ροή δεδομένων:**
```
CSV αρχείο (ticker, quantity, entry_price)
        ↓
yfinance → τρέχουσες τιμές + ιστορικό 5 ετών
        ↓
Portfolio object (Digital Twin) → μετρικές κινδύνου
        ↓
RiskMonitor → 6 ελέγχους  |  ScenarioEngine → προσομοιώσεις
        ↓
RecommendationEngine → κατατεταγμένες προτάσεις
        ↓
Explainer (XAI) → επεξηγήσεις σε φυσική γλώσσα
```
""")

    # ------------------------------------------------------------------ csv format
    with st.expander("📂 Μορφή Αρχείου CSV & Φόρτωση Χαρτοφυλακίου", expanded=False):
        st.markdown("""
**Απαιτούμενες στήλες:**

| Στήλη | Τύπος | Περιγραφή |
|---|---|---|
| `ticker` | text | Σύμβολο μετοχής (π.χ. AAPL, SPY, BTC-USD) |
| `quantity` | αριθμός | Αριθμός τεμαχίων που κατέχεις (>0) |
| `entry_price` | αριθμός | Τιμή κτήσης σε USD (>0) |

**Παράδειγμα:**
```csv
ticker,quantity,entry_price
AAPL,100,150.00
MSFT,50,300.00
SPY,200,380.00
BND,500,80.00
VXUS,300,58.00
GLD,40,185.00
```

**Τρόποι φόρτωσης (Sidebar):**
- **Δείγμα χαρτοφυλακίου** — 3 έτοιμα χαρτοφυλάκια: Συντηρητικό, Ισορροπημένο, Δυναμικό
- **Ανέβασμα CSV** — το δικό σου αρχείο

**Τα 3 δείγματα χαρτοφυλακίων:**

| Χαρτοφυλάκιο | Τίτλοι | Χαρακτήρας |
|---|---|---|
| Συντηρητικό | SPY, BND, TLT, VIG, JNJ, PG, KO, VYM, VXUS, GLD | Έμφαση σε ομόλογα, μερίσματα, χρυσό |
| Ισορροπημένο | AAPL, MSFT, GOOGL, JPM, JNJ, SPY, BND, VXUS, GLD, VNQ | Ισορροπία μετοχών/ομολόγων |
| Δυναμικό | AAPL, MSFT, GOOGL, TSLA, NVDA, META, AMZN, SPY, QQQ, BND | Έμφαση σε tech, υψηλή ανάπτυξη |

**Προφίλ κινδύνου επενδυτή:**
Επέλεξε ένα από τα 3 προφίλ — επηρεάζει τους στόχους μεταβλητότητας, τα όρια των
alerts, και τις προτάσεις rebalancing:

| Προφίλ | Στόχος Volatility | Μετοχές | Ομόλογα | Εναλλακτικά |
|---|---|---|---|---|
| Συντηρητικό | 7% | 40% | 50% | 10% |
| Ισορροπημένο | 12% | 60% | 30% | 10% |
| Δυναμικό | 18% | 80% | 10% | 10% |

Το προφίλ **δεν** αλλάζει τα πραγματικά δεδομένα — μόνο τους στόχους για σύγκριση.
""")

    # ------------------------------------------------------------------ tab 1
    with st.expander("📈 Tab 1 — Επισκόπηση Χαρτοφυλακίου", expanded=False):
        st.markdown("""
Η κεντρική καρτέλα του συστήματος. Εμφανίζεται αμέσως μόλις φορτωθεί χαρτοφυλάκιο
και δίνει μια **πλήρη εικόνα της κατάστασής του** — από απλά νούμερα έως ερμηνευτικές
αξιολογήσεις και πρόταση αναδιάρθρωσης.

---

**🔴🟡🟢 Επίπεδο Κινδύνου (Risk Level Badge)**

Στην κορυφή εμφανίζεται ένα χρωματιστό badge που δείχνει αυτόματα αν το χαρτοφυλάκιο
είναι **Χαμηλού / Μέτριου / Υψηλού / Πολύ Υψηλού κινδύνου**.

Υπολογίζεται συνδυάζοντας:
- Ετήσια Volatility (βαρύτητα 60%)
- Beta ως προς την αγορά (βαρύτητα 40%)

Έτσι ο επενδυτής καταλαβαίνει με μια ματιά πόσο επικίνδυνο είναι το χαρτοφυλάκιό του,
χωρίς να χρειαστεί να ερμηνεύσει ο ίδιος τους αριθμούς.

---

**💬 Ερμηνευτικά Μηνύματα Αξιολόγησης**

Κάτω από το badge εμφανίζονται αυτόματα χρωματιστά μηνύματα που εξηγούν **τι σημαίνουν
οι δείκτες** σε απλή γλώσσα:

| Χρώμα | Σημασία | Παράδειγμα |
|---|---|---|
| 🟢 Πράσινο | Εντός στόχου | «Μεταβλητότητα 11.2% εντός στόχου 12%. ✓» |
| 🟡 Κίτρινο | Αξίζει προσοχής | «Μέτρια συγκέντρωση (Herfindahl=0.19)» |
| 🔴 Κόκκινο | Απαιτεί άμεση δράση | «Υψηλή μεταβλητότητα 19% vs στόχο 12%» |

Οι έλεγχοι καλύπτουν: Volatility vs στόχο, Συγκέντρωση (Herfindahl), Διαφοροποίηση
(αριθμός θέσεων + Diversification Ratio), Sharpe Ratio, και Απόκλιση Προφίλ.

---

**📊 Βασικοί Δείκτες (KPIs) — 5 κουτιά στην κορυφή**

- **Συνολική Αξία** — τρέχουσα αγοραία αξία (quantity × current_price για κάθε τίτλο)
- **Συνολική Απόδοση (Return)** — % κέρδος/ζημία από την τιμή αγοράς
- **Μεταβλητότητα (Volatility)** — ετήσια, με ένδειξη πόσο απέχει από τον στόχο του προφίλ
- **Sharpe Ratio** — αποτελεσματικότητα: απόδοση σε σχέση με τον κίνδυνο
- **Beta** — πόσο το χαρτοφυλάκιο ακολουθεί ή ενισχύει τις κινήσεις του S&P 500

---

**📋 Πίνακας Θέσεων**

Κάθε γραμμή είναι ένας τίτλος (π.χ. AAPL, SPY) με:
- Τρέχουσα τιμή αγοράς
- Συνολική αξία θέσης
- % κατανομή στο χαρτοφυλάκιο
- Κέρδος/ζημία (P&L) από την τιμή αγοράς
- Κλάδος (Sector) και κατηγορία asset

---

**🥧 Γραφήματα Σύνθεσης**

Δύο pie charts που δείχνουν πώς κατανέμεται το χαρτοφυλάκιο:
- **Ανά κατηγορία asset** — Equity / Fixed Income / Real Estate / Commodity / Crypto
- **Ανά κλάδο (Sector)** — Technology / Healthcare / Finance κ.λπ. (top 6)

Βοηθούν να εντοπίσεις γρήγορα αν υπάρχει υπερβολική συγκέντρωση σε μία κατηγορία.

---

**📐 Αναλυτικοί Δείκτες**

Δύο στήλες με όλες τις μετρικές:

| Μετρική | Τι μετράει | Καλή τιμή |
|---|---|---|
| Volatility | Ετήσια τυπική απόκλιση log-αποδόσεων | Κοντά στον στόχο προφίλ |
| Sharpe Ratio | (Απόδοση − Risk-free rate) / Volatility | > 1.0 |
| Beta | Συσχέτιση κινήσεων με S&P 500 | 0.8–1.2 για μέτριο προφίλ |
| Max Drawdown | Χειρότερη ιστορική πτώση από κορυφή | > −20% |
| VaR 95% μηνιαίο | Ζημία που αναμένεται να ξεπεραστεί 1 φορά/χρόνο | > −10% |
| Diversification Ratio | Πόσο ο συνδυασμός μειώνει τον κίνδυνο | > 1.2 |
| Herfindahl Index | Συγκέντρωση (Σ βαρών²) | < 0.15 |

---

**🎯 Εναρμόνιση με Προφίλ Κινδύνου**

Τρία κουτιά που δείχνουν πόσο απέχει η τρέχουσα κατανομή από τον στόχο του επενδυτικού
προφίλ (Συντηρητικό / Ισορροπημένο / Δυναμικό):
- **Μετοχές (Equity %)** — τρέχον vs στόχος
- **Ομόλογα (Fixed Income %)** — τρέχον vs στόχος
- **Προφίλ** — ποιο προφίλ είναι επιλεγμένο

Οι αποκλίσεις εμφανίζονται ως delta (+ ή −) με κόκκινο χρώμα αν είναι αρνητικές.

---

**🔄 Κουμπί Πρότασης Ανακατανομής (Rebalancing)**

Στο κάτω μέρος της καρτέλας υπάρχει κουμπί **«Προτείνω Ανακατανομή Κεφαλαίων»**.

Όταν το πατάς, εμφανίζεται **σύγκριση πριν / μετά**:

| | Πριν | Μετά (εκτίμηση) |
|---|---|---|
| Equity % | π.χ. 80% | 60% (στόχος) |
| Fixed Income % | π.χ. 10% | 30% (στόχος) |
| Volatility | π.χ. 18.5% | ~14.2% |
| Sharpe Ratio | π.χ. 0.72 | ~0.91 |

Και τις συγκεκριμένες κινήσεις που χρειάζονται: «Πώληση μετοχών αξίας $X, Αγορά
ομολόγων αξίας $X».

Σημείωση: Οι τιμές μετά το rebalancing είναι **εκτιμήσεις** βασισμένες σε γραμμικά
μοντέλα. Για ακριβή ανάλυση trades ανά τίτλο, δες την καρτέλα Προτάσεις Αναδιάρθρωσης.
""")

    # ------------------------------------------------------------------ tab 2
    with st.expander("🔀 Tab 2 — Ανάλυση Σεναρίων (Scenario Analysis)", expanded=False):
        st.markdown("""
**Σκοπός:** Να δεις τι θα συμβεί στο χαρτοφυλάκιό σου σε υποθετικές συνθήκες αγοράς,
χωρίς να εκτελέσεις πραγματικές συναλλαγές.

---

**14 Προκαθορισμένα Σενάρια:**

| Κατηγορία | Σενάρια |
|---|---|
| Κρίσεις | Χρηματοπιστωτική Κρίση 2008 (−45%), COVID-19 Crash (−34%), Ύφεση 2022 |
| Bear Markets | Ήπια Διόρθωση (−10%), Μέτρια Πτώση (−20%), Σοβαρό Bear Market (−35%) |
| Bull Markets | Μέτρια Ανάκαμψη (+15%), Ισχυρό Bull Market (+30%) |
| Επιτόκια | Αύξηση επιτοκίων +1%, Αύξηση +2%, Μείωση −1% |
| Λοιπά | Spike Πληθωρισμού, Ύφεση + Αποπληθωρισμός, Flash Crash |

**Τι δείχνει κάθε σενάριο:**
- Νέα συνολική αξία χαρτοφυλακίου μετά το σενάριο
- % μεταβολή έναντι τρέχουσας αξίας
- Ανθεκτικότητα (Resilience) — πόσο καλύτερα/χειρότερα από την αγορά
- Μεταβολή μετρικών (Volatility, Sharpe, Max Drawdown, Beta)
- Αντίκτυπος ανά μεμονωμένο τίτλο

---

**Custom Σενάριο:**
Ορίζεις μόνος σου:
- **Μεταβολή αγοράς (%)** — slider από −60% έως +60%
- **Μεταβολή επιτοκίου (%)** — slider από −3% έως +3%

Κάθε τίτλος επηρεάζεται ανάλογα με το Beta του.

---

**Σύγκριση Σεναρίων:**
Εκτέλεσε πολλά σενάρια και συγκρίνετε side-by-side σε πίνακα.

**Χειρότερο Σενάριο:**
Το σύστημα εντοπίζει αυτόματα ποιο από τα 14 σενάρια έχει τη μεγαλύτερη ζημία.
""")

    # ------------------------------------------------------------------ tab 3
    with st.expander("⚠️ Tab 3 — Παρακολούθηση Κινδύνου (Risk Monitor)", expanded=False):
        st.markdown("""
**Σκοπός:** Αυτόματος εντοπισμός κινδύνων με βαθμολόγηση σοβαρότητας.

**Επίπεδα σοβαρότητας:**
- ✅ **OK** — εντός ασφαλών ορίων
- ⚠️ **CAUTION** — αξίζει προσοχής, δεν απαιτεί άμεση ενέργεια
- 🚨 **ALERT** — υπέρβαση κρίσιμου ορίου, απαιτεί δράση

---

**6 Κατηγορίες Ελέγχου:**

**1. Συγκέντρωση (Concentration)**
- Ελέγχει αν ένας τίτλος υπερβαίνει το 20-25% του χαρτοφυλακίου
- Ελέγχει αν οι top-5 τίτλοι υπερβαίνουν το 60-70%
- Herfindahl Index: μέτρο συγκέντρωσης (>0.25 = alert)

**2. Διαφοροποίηση (Diversification)**
- Αριθμός θέσεων: <10 = caution, <5 = alert
- Συγκέντρωση σε κλάδο (Sector): >50% = caution, >65% = alert
- Γεωγραφική διαφοροποίηση: <5% διεθνής έκθεση = caution

**3. Μεταβλητότητα (Volatility)**
- Συγκρίνει πραγματική volatility με τον στόχο του προφίλ σου
- Απόκλιση >2% = caution, >4% = alert
- Και υπό-χρησιμοποίηση risk budget (πολύ χαμηλή vol)

**4. Drawdown**
- Max Drawdown: <−15% = caution, <−25% = alert
- VaR 95% μηνιαίο: <−10% = caution, <−15% = alert

**5. Συσχέτιση (Correlation)**
- Μετράει πόσα ζεύγη τίτλων έχουν συσχέτιση ≥0.80
- >30% ζεύγη υψηλά συσχετισμένα = caution
- >50% = alert (διαφοροποίηση χωρίς ουσιαστικό όφελος)

**6. Απόκλιση Προφίλ (Profile Drift)**
- Συγκρίνει τρέχουσα κατανομή με τον στόχο του προφίλ κινδύνου
- Απόκλιση >5% = caution, >10% = alert
""")

    # ------------------------------------------------------------------ tab 4
    with st.expander("💡 Tab 4 — Προτάσεις Αναδιάρθρωσης (Recommendations)", expanded=False):
        st.markdown("""
Η καρτέλα αυτή μετατρέπει τους κινδύνους που εντοπίζει το Risk Monitor σε **συγκεκριμένες,
εξηγήσιμες οδηγίες δράσης**. Δεν λέει απλά «υπάρχει πρόβλημα» — λέει ακριβώς τι να κάνεις,
γιατί, πόσο θα κοστίσει, και τι θα αλλάξει μετά.

---

**Επίπεδα προτεραιότητας:**

| Επίπεδο | Χρώμα | Σημαίνει |
|---|---|---|
| CRITICAL | 🔴 Κόκκινο | Σοβαρός κίνδυνος — δράσε άμεσα |
| HIGH | 🟠 Πορτοκαλί | Σημαντικό ζήτημα — δράσε σύντομα |
| MEDIUM | 🟡 Κίτρινο | Καλό να αντιμετωπιστεί, όχι επείγον |
| LOW | ⚪ Γκρι | Μικρή βελτίωση, δεν επηρεάζει πολύ |

---

**Τι περιέχει κάθε κάρτα πρότασης:**

- **Τίτλος** — τι πρόβλημα αντιμετωπίζει (π.χ. «Μείωση Συγκέντρωσης SPY»)
- **Κατηγορία κινδύνου** — ποια από τις 6 κατηγορίες του Risk Monitor αφορά
- **Ενέργεια** — μία πρόταση σε μία γραμμή (π.χ. «Πώληση $12,000 από SPY»)
- **Συγκεκριμένες Συναλλαγές** — ακριβώς πόσα τεμάχια, σε ποια τιμή, συνολική αξία
- **Γιατί** — εξήγηση του κινδύνου που αντιμετωπίζει η πρόταση σε απλή γλώσσα
- **Αναμενόμενος Αντίκτυπος** — πίνακας «πριν → μετά» για τη σχετική μετρική
- **Εναλλακτική** — τι μπορείς να κάνεις αν διαφωνείς με την κύρια πρόταση
- **Βαθμός Εμπιστοσύνης** — high/medium/low ανάλογα με πόσο βέβαιη είναι η εκτίμηση

---

**Πώς παράγονται οι προτάσεις (λογική του συστήματος):**

1. Το Risk Monitor τρέχει και βρίσκει όλα τα alerts/cautions
2. Κάθε alert «δρομολογείται» στον αντίστοιχο generator (π.χ. alert συγκέντρωσης → `_rec_concentration()`)
3. Ο generator υπολογίζει τη συγκεκριμένη ενέργεια και τις συναλλαγές
4. Οι προτάσεις κατατάσσονται: CRITICAL πρώτα, μετά HIGH, MEDIUM, LOW
5. Μέσα σε κάθε επίπεδο, ταξινομούνται ανά κατηγορία κινδύνου

**Ένα παράδειγμα:**
Αν το SPY είναι 44% του χαρτοφυλακίου (πάνω από το όριο του 25%):
- Risk Monitor: 🚨 ALERT «Συγκέντρωση σε Μεμονωμένο Τίτλο»
- Recommendation: CRITICAL — «Πώληση 18 τεμαχίων SPY @ $520 = $9,360»
- Γιατί: «SPY αποτελεί 44% — άνω ασφαλούς ορίου 20%. Μείωση στο 20% μειώνει τον κίνδυνο απώλειας από ένα μόνο ETF»
- Impact: Concentration 44.0% → 20.0%

---

**Σύνοψη Συναλλαγών (Trade Summary)**

Στο κάτω μέρος της καρτέλας, πίνακας με όλες τις προτεινόμενες αγοραπωλησίες και
τα σύνολα: Συνολικές Πωλήσεις, Συνολικές Αγορές, Καθαρή Ανάγκη Κεφαλαίου.

---

**Εξαγωγή σε αρχείο:**
Κουμπί «Εξαγωγή Προτάσεων (κείμενο)» — κατεβάζει όλες τις προτάσεις ως `.txt` αρχείο
με πλήρεις επεξηγήσεις, κατάλληλο για αρχείο ή παρουσίαση.
""")

    # ------------------------------------------------------------------ xai
    with st.expander("🔍 Explainable AI (XAI) Layer", expanded=False):
        st.markdown("""
**Τι είναι το XAI (Explainable AI);**

Το πρόβλημα με τα περισσότερα συστήματα AI είναι ότι δίνουν αποτελέσματα χωρίς εξήγηση
(«black box»). Αυτό το σύστημα είναι σχεδιασμένο ως **white box** — κάθε αριθμός εξηγείται.

**Πώς υλοποιείται (src/explainer.py):**

| Μέθοδος | Τι κάνει |
|---|---|
| `explain_metric()` | Εξηγεί τι σημαίνει κάθε μετρική (Sharpe, Beta, κ.λπ.) |
| `explain_alert()` | Μετατρέπει ένα alert dict σε plain-text |
| `explain_all_alerts()` | Πλήρης αναφορά κινδύνου |
| `explain_recommendation()` | Λεπτομερής επεξήγηση μιας πρότασης |
| `explain_scenario()` | Ανάλυση αποτελέσματος σεναρίου |
| `explain_term()` | Γλωσσάρι χρηματοοικονομικών όρων |

**Γλωσσάρι:**
""")
        explainer = Explainer()
        for term, definition in explainer.GLOSSARY.items():
            st.markdown(f"- **{term}**: {definition}")

    # ------------------------------------------------------------------ scenarios strategy
    with st.expander("📐 Στρατηγική Σεναρίων — Πώς Λειτουργεί η Προσομοίωση", expanded=False):
        st.markdown("""
**Μοντέλο προσομοίωσης (src/scenario_engine.py):**

Κάθε σενάριο ορίζεται από δύο παραμέτρους:
- `market_change_pct` — μεταβολή της γενικής αγοράς (S&P 500)
- `rate_change_pct` — μεταβολή επιτοκίου (Fed Funds Rate)

**Πώς υπολογίζεται ο αντίκτυπος ανά τίτλο:**

```
Αντίκτυπος τίτλου = Beta × market_change_pct + rate_sensitivity × rate_change_pct
```

- **Μετοχές**: rate_sensitivity = −0.05 (ελαφρά αρνητική σχέση με επιτόκια)
- **Ομόλογα**: rate_sensitivity = −0.08 (ισχυρά αρνητική — duration effect)
- **Real Estate**: rate_sensitivity = −0.10
- **Commodities/Crypto**: rate_sensitivity ≈ 0 (ανεξάρτητα από επιτόκια)

**Metric scaling μετά το σενάριο:**
- Volatility κλιμακώνεται ανάλογα με το μέγεθος της αγοράς
- Sharpe Ratio υπολογίζεται εκ νέου με νέα τιμή
- Max Drawdown: worst-case σενάριο + υπάρχον drawdown
- Beta παραμένει σταθερό (δομικό χαρακτηριστικό)

**Ανθεκτικότητα (Resilience):**
```
Resilience = portfolio_change_pct − market_change_pct
```
Θετικό = το χαρτοφυλάκιο αντέχει καλύτερα από την αγορά (καλό σε bear markets).
Αρνητικό = υποαπόδοση έναντι αγοράς (ανησυχητικό σε πτώσεις).
""")

    # ------------------------------------------------------------------ tests
    with st.expander("🧪 Δοκιμές Κώδικα (Test Suite)", expanded=False):
        st.markdown("""
**Framework:** pytest

**Αρχεία δοκιμών (`tests/`):**

| Αρχείο | Τι δοκιμάζει |
|---|---|
| `test_data_loader.py` | Φόρτωση CSV, έλεγχος στηλών, τιμές <0 |
| `test_portfolio.py` | Υπολογισμός μετρικών, apply_scenario, weights |
| `test_risk_monitor.py` | Και τις 6 κατηγορίες ελέγχου, σωστά severity levels |
| `test_scenario_engine.py` | Προσομοίωση σεναρίων, custom scenarios, worst-case |
| `test_recommendations.py` | Παραγωγή προτάσεων, routing ανά κατηγορία |

**Εκτέλεση δοκιμών:**
```bash
cd d:/thesis
pytest tests/ -v
```

**Τι δοκιμάζεται:**
- Φόρτωση έγκυρου/άκυρου CSV → σωστά errors
- Υπολογισμός Sharpe, Volatility, Beta με γνωστά δεδομένα
- Alert severity: τιμή πάνω/κάτω από threshold → σωστό severity
- Σενάριο −30% με Beta=1.5 → αναμενόμενη απώλεια ~−45%
- Recommendations: alert συγκέντρωσης → παράγει SELL trade
""")

    # ------------------------------------------------------------------ use cases
    with st.expander("🎯 Σενάρια Χρήσης (Use Cases)", expanded=False):
        st.markdown("""
**Use Case 1: Επενδυτής με υπερσυγκεντρωμένο χαρτοφυλάκιο**

*Κατάσταση:* 60% του χαρτοφυλακίου σε μία μετοχή (π.χ. AAPL)

1. Φόρτωσε CSV → Tab 1 δείχνει Herfindahl Index > 0.25
2. Tab 3 → 🚨 ALERT «Συγκέντρωση σε Μεμονωμένο Τίτλο»
3. Tab 4 → CRITICAL πρόταση: «Πώληση X μετοχών AAPL, αγορά SPY/BND»
4. Tab 2 → Τρέξε σενάριο «Σοβαρό Bear Market −35%» → δες τι χάνεις vs διαφοροποιημένο χαρτοφυλάκιο

---

**Use Case 2: Συντηρητικός επενδυτής με υπερβολικά επιθετικό χαρτοφυλάκιο**

*Κατάσταση:* Επέλεξε «Συντηρητικό» προφίλ αλλά έχει 80% μετοχές

1. Tab 3 → ⚠️ CAUTION «Απόκλιση Προφίλ» — equity drift +40%
2. Tab 4 → HIGH πρόταση: «Μεταφορά $X από μετοχές σε BND»
3. Tab 1 → Volatility 18% vs στόχο 7% → ακόμα ένα alert
4. Tab 2 → Σενάριο «Αύξηση Επιτοκίων +2%» → δες αντίκτυπο στα ομόλογα

---

**Use Case 3: Stress test πριν από αβέβαιη περίοδο**

*Κατάσταση:* Θέλεις να γνωρίζεις το worst-case πριν ανακοινωθεί Fed απόφαση

1. Tab 2 → «Χειρότερο Σενάριο» → αυτόματος εντοπισμός
2. Custom σενάριο: αγορά −20%, επιτόκια +0.5%
3. Δες ανά τίτλο ποιος πλήττεται περισσότερο
4. Tab 4 → Αν υπάρχουν recommendations, εκτέλεσε τις πριν την απόφαση

---

**Use Case 4: Σύγκριση παλαιού vs νέου χαρτοφυλακίου**

1. Φόρτωσε το παλαιό CSV → κατέβασε export recommendations.txt
2. Εφάρμοσε τις προτάσεις στο CSV (νέες ποσότητες)
3. Φόρτωσε το νέο CSV → σύγκρινε Sharpe, Volatility, Max Drawdown
4. Tab 2 → Τρέξε τα ίδια σενάρια και στα δύο → ποιο αντέχει καλύτερα;
""")

    # ------------------------------------------------------------------ metrics deep dive
    with st.expander("📊 Αναλυτική Επεξήγηση Μετρικών", expanded=False):
        st.markdown("""
**Sharpe Ratio**
```
Sharpe = (Απόδοση χαρτοφυλακίου − Επιτόκιο αναφοράς) / Volatility
```
Επιτόκιο αναφοράς (Risk-Free Rate): 5.0% (10ετές ομόλογο ΗΠΑ).
- < 0: Το χαρτοφυλάκιο αποδίδει χειρότερα από το risk-free asset
- 0–1: Αποδεκτό
- 1–2: Καλό
- > 2: Εξαιρετικό

---

**Beta**
```
Beta = Cov(portfolio returns, market returns) / Var(market returns)
```
Αγορά αναφοράς: SPY (S&P 500). Ιστορικό: 5 χρόνια ημερήσιων αποδόσεων.
- Beta < 1: Λιγότερο ευμετάβλητο από την αγορά (π.χ. ομόλογα)
- Beta = 1: Κινείται παράλληλα με S&P 500
- Beta > 1: Ενισχύει τις κινήσεις της αγοράς (π.χ. tech μετοχές)

---

**Volatility (Μεταβλητότητα)**
```
Ετήσια Volatility = Std(ημερήσιες log-returns) × √252
```
252 = εργάσιμες μέρες/χρόνο. Log-returns χρησιμοποιούνται για καλύτερη στατιστική συμπεριφορά σε μακροπρόθεσμη ανάλυση.

---

**Maximum Drawdown**
```
Max Drawdown = min((τρέχουσα τιμή − μέγιστη προηγούμενη τιμή) / μέγιστη προηγούμενη τιμή)
```
Υπολογίζεται επί 5ετούς ιστορικού. Αρνητικός αριθμός πάντα.

---

**Value at Risk (VaR 95%, Μηνιαίο)**
```
VaR = percentile(μηνιαίες αποδόσεις, 5%)
```
Παραδοχή κανονικής κατανομής. Σημαίνει: σε 1 στους 20 μήνες η ζημία θα υπερβεί αυτό το ποσοστό.

---

**Diversification Ratio**
```
Diversification Ratio = Σ(βάρος_i × volatility_i) / volatility_χαρτοφυλακίου
```
Ratio > 1 σημαίνει ότι ο συνδυασμός assets μειώνει τον κίνδυνο. Ratio = 1.5 → 33% μείωση κινδύνου από διαφοροποίηση.

---

**Herfindahl-Hirschman Index (HHI)**
```
HHI = Σ(βάρος_i²)
```
- 0.0–0.15: Διαφοροποιημένο
- 0.15–0.25: Μέτρια συγκεντρωμένο
- > 0.25: Υψηλά συγκεντρωμένο
""")


# ---------------------------------------------------------------------------
# Tab 5: Πειραματική Αξιολόγηση (Experimental Validation)
# ---------------------------------------------------------------------------

def _csv_to_archetype(df: pd.DataFrame, name: str, risk_profile: str) -> dict:
    """Convert an uploaded portfolio CSV to a validation archetype dict."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce").fillna(0)
    df = df[(df["quantity"] > 0) & (df["entry_price"] > 0)]

    total_value = (df["quantity"] * df["entry_price"]).sum()
    if total_value == 0:
        raise ValueError("Μηδενική αξία χαρτοφυλακίου.")

    tickers = df["ticker"].tolist()
    weights = [(row["quantity"] * row["entry_price"]) / total_value
               for _, row in df.iterrows()]

    safe_id = "user_" + name.lower().replace(" ", "_")[:20]
    return {
        "id": safe_id,
        "label": f"★ {name}",
        "archetype": "Δικό μου",
        "tickers": tickers,
        "weights": weights,
        "risk_profile": risk_profile,
    }


def _val_metric_card(label: str, baseline_val, proposed_val,
                     higher_is_better: bool = True, fmt: str = ".3f"):
    """Render a side-by-side metric card: baseline vs proposed."""
    if baseline_val is None or proposed_val is None:
        return
    try:
        b = float(baseline_val)
        p = float(proposed_val)
    except (TypeError, ValueError):
        return

    delta = p - b
    if higher_is_better:
        color = "#28a745" if delta > 0 else ("#dc3545" if delta < 0 else "#6c757d")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
    else:
        color = "#28a745" if delta < 0 else ("#dc3545" if delta > 0 else "#6c757d")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")

    st.markdown(
        f"""
        <div style="background:#f8f9fa;border-radius:8px;padding:10px 14px;margin-bottom:8px">
          <div style="font-size:0.78rem;color:#6c757d;margin-bottom:4px">{label}</div>
          <div style="display:flex;gap:24px;align-items:center">
            <div>
              <span style="font-size:0.7rem;color:#888">Baseline</span><br>
              <strong style="font-size:1.1rem">{b:{fmt}}</strong>
            </div>
            <div>
              <span style="font-size:0.7rem;color:#888">Proposed</span><br>
              <strong style="font-size:1.1rem">{p:{fmt}}</strong>
            </div>
            <div style="color:{color};font-size:1.1rem">
              {arrow} <strong>{abs(delta):{fmt}}</strong>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_scatter_comparison(results: ValidationResults):
    """Scatter plot: predicted risk score vs actual drawdown, both systems."""
    import numpy as np

    valid_mask = [
        not (np.isnan(b) or np.isnan(p) or np.isnan(d))
        for b, p, d in zip(results.baseline_scores, results.proposed_scores,
                           results.actual_drawdowns)
    ]
    b_scores = [s for s, v in zip(results.baseline_scores, valid_mask) if v]
    p_scores = [s for s, v in zip(results.proposed_scores, valid_mask) if v]
    drawdowns = [d for d, v in zip(results.actual_drawdowns, valid_mask) if v]
    archetypes = [a for a, v in zip(results.portfolio_archetypes, valid_mask) if v]

    arch_colors = {
        "Συγκεντρωμένο":          "#dc3545",
        "Κλαδικό":                "#fd7e14",
        "Μέτρια Διαφοροποίηση":   "#ffc107",
        "Καλά Διαφοροποιημένο":   "#28a745",
        "Συντηρητικό":            "#17a2b8",
        "Δικό μου":               "#6f42c1",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, scores, title, method_color in [
        (ax1, b_scores, "Baseline (Στατικές Μετρικές)", "#6c757d"),
        (ax2, p_scores, "Proposed (Digital Twin)",       "#0d6efd"),
    ]:
        for arch, score, dd in zip(archetypes, scores, drawdowns):
            is_user = arch == "Δικό μου"
            color  = arch_colors.get(arch, "#888")
            marker = "*" if is_user else "o"
            size   = 200 if is_user else 60
            ax.scatter(score, dd, c=color, s=size, alpha=0.9,
                       marker=marker, edgecolors="white", linewidths=0.5,
                       zorder=5 if is_user else 3)

        # Trend line
        if len(scores) > 2:
            z = np.polyfit(scores, drawdowns, 1)
            xs = np.linspace(min(scores), max(scores), 100)
            ax.plot(xs, np.poly1d(z)(xs), "--", color=method_color, alpha=0.7, linewidth=1.5)

        # Spearman ρ annotation
        rho_key = "baseline_rho" if ax is ax1 else "proposed_rho"
        pval_key = "baseline_pval" if ax is ax1 else "proposed_pval"
        rho = results.metrics.get("spearman", {}).get(rho_key, float("nan"))
        pval = results.metrics.get("spearman", {}).get(pval_key, float("nan"))
        sig_star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        ax.text(
            0.05, 0.96,
            f"Spearman ρ = {rho:.3f}{sig_star}\np = {pval:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(-15, color="#dc3545", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Risk Score", fontsize=10)
        ax.set_ylabel("Πραγματικό Max Drawdown (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Legend
    legend_patches = []
    for arch, c in arch_colors.items():
        marker = "*" if arch == "Δικό μου" else "o"
        ms = 12 if arch == "Δικό μου" else 8
        legend_patches.append(
            plt.Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor=c, markersize=ms, label=arch)
        )
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def _plot_precision_recall_bar(results: ValidationResults):
    """Bar chart comparing precision, recall, F1 between baseline and proposed."""
    fd = results.metrics.get("fragile_detection", {})

    labels = ["Precision", "Recall", "F1-Score"]
    b_vals = [
        fd.get("baseline_precision") or 0,
        fd.get("baseline_recall")    or 0,
        fd.get("baseline_f1")        or 0,
    ]
    p_vals = [
        fd.get("proposed_precision") or 0,
        fd.get("proposed_recall")    or 0,
        fd.get("proposed_f1")        or 0,
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar(x - width / 2, b_vals, width, label="Baseline",
                    color="#6c757d", alpha=0.85, edgecolor="white")
    bars_p = ax.bar(x + width / 2, p_vals, width, label="Proposed (Digital Twin)",
                    color="#0d6efd", alpha=0.85, edgecolor="white")

    for bar in bars_b + bars_p:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        f"Εντοπισμός Ευάλωτων Χαρτοφυλακίων\n"
        f"(threshold: drawdown < {fd.get('threshold_pct', -15):.0f}%)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_risk_rank_comparison(results: ValidationResults):
    """Visualize how risk rankings compare between the two systems and ground truth."""
    import numpy as np

    valid_idx = [
        i for i in range(results.n_portfolios)
        if not (np.isnan(results.baseline_scores[i]) or
                np.isnan(results.proposed_scores[i]) or
                np.isnan(results.actual_drawdowns[i]))
    ]
    if len(valid_idx) < 4:
        return None

    b_scores  = [results.baseline_scores[i]  for i in valid_idx]
    p_scores  = [results.proposed_scores[i]  for i in valid_idx]
    drawdowns = [results.actual_drawdowns[i] for i in valid_idx]

    # Convert to ranks (1 = highest risk)
    def to_rank(vals, ascending=False):
        arr = np.array(vals)
        if ascending:
            return arr.argsort().argsort() + 1
        return (-arr).argsort().argsort() + 1

    b_rank  = to_rank(b_scores)
    p_rank  = to_rank(p_scores)
    dd_rank = to_rank(drawdowns)  # more negative = riskier

    labels  = [results.portfolio_labels[i] for i in valid_idx]
    n = len(valid_idx)

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)))
    y = np.arange(n)

    ax.scatter(b_rank,  y, marker="o", c="#6c757d", s=50, label="Baseline rank",  zorder=3)
    ax.scatter(p_rank,  y, marker="s", c="#0d6efd", s=50, label="Proposed rank",  zorder=3)
    ax.scatter(dd_rank, y, marker="D", c="#28a745", s=50, label="Actual DD rank", zorder=3)

    for i in range(n):
        ax.plot([b_rank[i], dd_rank[i]], [i, i], color="#6c757d", alpha=0.3, linewidth=1)
        ax.plot([p_rank[i], dd_rank[i]], [i, i], color="#0d6efd", alpha=0.3, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Risk Rank (1 = πιο επικίνδυνο)", fontsize=10)
    ax.set_title("Σύγκριση Κατάταξης Κινδύνου\n(Baseline vs Proposed vs Πραγματική Απόδοση)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def render_validation():
    """Tab 5: Πειραματική Αξιολόγηση — Σύγκριση Baseline vs Digital Twin."""
    st.subheader("🔬 Πειραματική Αξιολόγηση Συστήματος")
    st.markdown(
        """
        | Σύστημα | Περιγραφή |
        |---------|-----------|
        | **Baseline** | Στατικές μετρικές μόνο (volatility, Sharpe, drawdown, beta, HHI) |
        | **Proposed** | Digital Twin: RiskMonitor alerts + ScenarioEngine simulation |

        **Μεθοδολογία:** Walk-forward split → Analysis window (60%) | Evaluation window (40%)
        **Ground truth:** Πραγματικό max drawdown στο παράθυρο αξιολόγησης
        """
    )

    # ── Portfolio overview ─────────────────────────────────────────────────
    with st.expander("📋 30 Συνθετικά Χαρτοφυλάκια (Πλήρης Κατάλογος)", expanded=False):
        arch_df = pd.DataFrame([
            {
                "ID": a["id"],
                "Χαρτοφυλάκιο": a["label"],
                "Κατηγορία": a["archetype"],
                "Tickers": ", ".join(a["tickers"]),
                "Προφίλ": a["risk_profile"],
            }
            for a in PORTFOLIO_ARCHETYPES
        ])
        st.dataframe(arch_df, use_container_width=True, height=350)

    # ── Upload custom portfolios ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📂 Προσθήκη Δικών σου Χαρτοφυλακίων (Προαιρετικό)")
    st.caption(
        "Ανέβασε ένα ή περισσότερα CSV (ticker, quantity, entry_price) "
        "για να συμπεριληφθούν στο πείραμα ως επιπλέον σημεία (★)."
    )

    uploaded_files = st.file_uploader(
        "Ανέβασε CSV χαρτοφυλάκια",
        type=["csv"],
        accept_multiple_files=True,
        key="val_upload",
    )
    up_risk = st.selectbox(
        "Προφίλ κινδύνου για τα ανεβασμένα χαρτοφυλάκια",
        ["conservative", "moderate", "aggressive"],
        index=1,
        format_func=lambda x: {"conservative": "Συντηρητικό",
                                "moderate": "Ισορροπημένο",
                                "aggressive": "Δυναμικό"}[x],
        key="val_up_risk",
    )

    if uploaded_files:
        user_archetypes = []
        for f in uploaded_files:
            try:
                df_up = pd.read_csv(f)
                arch = _csv_to_archetype(
                    df_up,
                    name=f.name.replace(".csv", ""),
                    risk_profile=up_risk,
                )
                user_archetypes.append(arch)
                st.success(
                    f"✅ **{arch['label']}** — "
                    f"{len(arch['tickers'])} tickers: {', '.join(arch['tickers'])}"
                )
            except Exception as e:
                st.error(f"Σφάλμα στο {f.name}: {e}")
        st.session_state["user_archetypes"] = user_archetypes
    elif not st.session_state.get("user_archetypes"):
        st.session_state["user_archetypes"] = []

    # ── Run controls ───────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        run_btn = st.button(
            "🚀 Εκτέλεση Πειράματος",
            type="primary",
            help="Φορτώνει ιστορικά δεδομένα, βαθμολογεί 30 χαρτοφυλάκια και "
                 "υπολογίζει στατιστικές σύγκρισης. Διαρκεί ~3-5 λεπτά.",
        )
    with col_info:
        st.info(
            "Το πείραμα φορτώνει 5 έτη ιστορικών δεδομένων από Yahoo Finance "
            "για ~23 tickers και αξιολογεί 30 χαρτοφυλάκια. "
            "Τα αποτελέσματα αποθηκεύονται στη συνεδρία."
        )

    # ── Run or show cached results ─────────────────────────────────────────
    if run_btn or st.session_state.get("validation_results") is not None:
        if run_btn:
            # Clear previous results
            st.session_state["validation_results"] = None

            progress_bar = st.progress(0)
            status_text  = st.empty()
            total_steps  = len(PORTFOLIO_ARCHETYPES) + 6

            def _cb(step: int, total: int, msg: str):
                pct = int(step / total * 100)
                progress_bar.progress(pct)
                status_text.text(f"[{step}/{total}] {msg}")

            try:
                all_archetypes = (
                    PORTFOLIO_ARCHETYPES
                    + st.session_state.get("user_archetypes", [])
                )
                experiment = ValidationExperiment(
                    archetypes=all_archetypes,
                    history_years=5,
                    split_fraction=0.60,
                    progress_callback=_cb,
                )
                results = experiment.run()
                st.session_state["validation_results"] = results
                progress_bar.progress(100)
                status_text.text("✅ Ολοκληρώθηκε!")
                st.rerun()
            except Exception as exc:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Σφάλμα κατά την εκτέλεση: {exc}")
                st.exception(exc)
                return

        results: ValidationResults = st.session_state.get("validation_results")
        if results is None:
            return

        # ── Experiment metadata ────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Χαρτοφυλάκια", results.n_portfolios)
        m2.metric("Παράθυρο Ανάλυσης", f"{results.analysis_start} → {results.analysis_end}")
        m3.metric("Παράθυρο Αξιολόγησης", f"{results.eval_start} → {results.eval_end}")
        m4.metric("Έγκυρα αποτελέσματα", results.metrics.get("n_valid", "—"))

        # ── Key statistical metrics ────────────────────────────────────────
        st.markdown("### 📊 Κύριες Μετρικές Σύγκρισης")

        sp = results.metrics.get("spearman", {})
        fd = results.metrics.get("fragile_detection", {})
        mae = results.metrics.get("mae_proxy", {})
        kt  = results.metrics.get("kendall", {})

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Rank Correlation με Πραγματικό Drawdown")
            _val_metric_card(
                "Spearman ρ  (υψηλότερο = καλύτερο)",
                sp.get("baseline_rho"), sp.get("proposed_rho"),
                higher_is_better=True,
            )
            _val_metric_card(
                "Kendall τ  (υψηλότερο = καλύτερο)",
                kt.get("baseline_tau"), kt.get("proposed_tau"),
                higher_is_better=True,
            )

            p_b = sp.get("baseline_pval")
            p_p = sp.get("proposed_pval")
            sig_b = "p < 0.05 ✅" if (p_b and p_b < 0.05) else f"p = {p_b:.3f}" if p_b else "—"
            sig_p = "p < 0.05 ✅" if (p_p and p_p < 0.05) else f"p = {p_p:.3f}" if p_p else "—"
            st.caption(f"Baseline: {sig_b} | Proposed: {sig_p}")

            st.markdown("#### MAE Προβλεπόμενης vs Πραγματικής Απώλειας (%)")
            _val_metric_card(
                "MAE proxy (χαμηλότερο = καλύτερο)",
                mae.get("baseline_mae_pct"), mae.get("proposed_mae_pct"),
                higher_is_better=False, fmt=".2f",
            )

        with col_r:
            st.markdown(
                f"#### Εντοπισμός Ευάλωτων Χαρτοφυλακίων\n"
                f"*(drawdown < {fd.get('threshold_pct', -15):.0f}%)*"
            )
            _val_metric_card(
                "Precision (υψηλότερο = καλύτερο)",
                fd.get("baseline_precision"), fd.get("proposed_precision"),
                higher_is_better=True,
            )
            _val_metric_card(
                "Recall  (υψηλότερο = καλύτερο)",
                fd.get("baseline_recall"), fd.get("proposed_recall"),
                higher_is_better=True,
            )
            _val_metric_card(
                "F1-Score  (υψηλότερο = καλύτερο)",
                fd.get("baseline_f1"), fd.get("proposed_f1"),
                higher_is_better=True,
            )

            imp_f1  = fd.get("f1_improvement")
            imp_sp  = sp.get("improvement")
            imp_mae = mae.get("improvement_pct")

            def _fmt_imp(v, suffix="", decimals=3):
                if v is None:
                    return "—"
                sign = "+" if v > 0 else ""
                return f"{sign}{v:.{decimals}f}{suffix}"

            st.markdown(
                f"""
                <div style="background:#e8f5e9;border-radius:8px;padding:10px;margin-top:8px">
                  <strong>Βελτίωση Digital Twin vs Baseline</strong><br>
                  F1-Score: <strong>{_fmt_imp(imp_f1)}</strong> &nbsp;|&nbsp;
                  Spearman ρ: <strong>{_fmt_imp(imp_sp)}</strong> &nbsp;|&nbsp;
                  MAE: <strong>{_fmt_imp(-imp_mae if imp_mae is not None else None, '%', 2)}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Visualizations ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Οπτικοποίηση Αποτελεσμάτων")

        vtab1, vtab2, vtab3 = st.tabs([
            "🎯 Risk Score vs Actual Drawdown",
            "📊 Precision / Recall / F1",
            "🏆 Σύγκριση Κατάταξης",
        ])

        with vtab1:
            try:
                fig = _plot_scatter_comparison(results)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "Κάθε σημείο είναι ένα χαρτοφυλάκιο. Χρωματισμός κατά κατηγορία. "
                    "Διακεκομμένη κόκκινη γραμμή = threshold −15% (fragile). "
                    "Ρ > 0 σημαίνει θετική συσχέτιση μεταξύ προβλεπόμενου κινδύνου και πραγματικής απώλειας."
                )
            except Exception as e:
                st.warning(f"Σφάλμα γραφήματος: {e}")

        with vtab2:
            if fd.get("baseline_f1") is not None:
                try:
                    fig2 = _plot_precision_recall_bar(results)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                    st.caption(
                        f"Σύγκριση precision/recall/F1 για τον εντοπισμό χαρτοφυλακίων "
                        f"που υπέστησαν drawdown > {abs(fd.get('threshold_pct', 15)):.0f}% "
                        f"στην περίοδο αξιολόγησης. "
                        f"Ευάλωτα χαρτοφυλάκια: {fd.get('n_fragile', 0)}/{results.metrics.get('n_valid', 0)}."
                    )
                except Exception as e:
                    st.warning(f"Σφάλμα γραφήματος: {e}")
            else:
                st.info(
                    "Δεν υπάρχουν αρκετά 'fragile' χαρτοφυλάκια για αυτή την περίοδο αξιολόγησης. "
                    "Δοκιμάστε με διαφορετικό split ή περίοδο που περιλαμβάνει bear market."
                )

        with vtab3:
            try:
                fig3 = _plot_risk_rank_comparison(results)
                if fig3 is not None:
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)
                    st.caption(
                        "Κάθε χαρτοφυλάκιο παρουσιάζεται ως σειρά. "
                        "Κύκλος=Baseline rank, Τετράγωνο=Proposed rank, Διαμάντι=Πραγματικό DD rank. "
                        "Μικρότερη απόσταση μεταξύ τετραγώνου και διαμαντιού = ακριβέστερη πρόβλεψη."
                    )
            except Exception as e:
                st.warning(f"Σφάλμα γραφήματος: {e}")

        # ── Detailed table ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Αναλυτικός Πίνακας Αποτελεσμάτων")
        df = results.to_dataframe()

        # Color-code the actual drawdown column + highlight user rows
        def _color_dd(val):
            if pd.isna(val):
                return ""
            if val < -25:
                return "background-color: #f8d7da; color: #721c24"
            if val < -15:
                return "background-color: #fff3cd; color: #856404"
            if val < -5:
                return "background-color: #d1ecf1; color: #0c5460"
            return "background-color: #d4edda; color: #155724"

        def _highlight_user(row):
            if row.get("Κατηγορία") == "Δικό μου":
                return ["background-color: #f3e8ff; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style
              .map(_color_dd, subset=["Πραγματικό Drawdown (%)"])
              .apply(_highlight_user, axis=1),
            use_container_width=True,
            height=500,
        )

        # ── Export ─────────────────────────────────────────────────────────
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Λήψη αποτελεσμάτων CSV",
            data=csv_data,
            file_name="validation_results.csv",
            mime="text/csv",
        )

        # ── Statistical summary (raw metrics) ─────────────────────────────
        with st.expander("🔢 Πλήρη Στατιστικά (για παράρτημα διπλωματικής)", expanded=False):
            import json, math

            def _json_safe(obj):
                """Convert nan/inf to None for JSON serialization."""
                if isinstance(obj, dict):
                    return {k: _json_safe(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_json_safe(v) for v in obj]
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                return obj

            st.code(
                json.dumps(_json_safe(results.metrics), indent=2, ensure_ascii=False),
                language="json",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_state()
    try:
        render_sidebar()
    except Exception as exc:
        st.sidebar.error(f"Σφάλμα sidebar: {exc}")

    st.title("Σύστημα Συμβουλευτικής Επενδύσεων — Digital Twin")
    st.caption(
        "Προσομοίωση σεναρίων χαρτοφυλακίου, παρακολούθηση κινδύνου "
        "και επεξηγήσιμες προτάσεις αναδιάρθρωσης."
    )

    if st.session_state.get("show_docs"):
        render_docs()
        return

    portfolio = st.session_state.get("portfolio")

    # Tab 5 (Πειραματική Αξιολόγηση) is always available, independent of portfolio
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Επισκόπηση Χαρτοφυλακίου",
        "🔀 Ανάλυση Σεναρίων",
        "⚠️ Παρακολούθηση Κινδύνου",
        "💡 Προτάσεις Αναδιάρθρωσης",
        "🔬 Πειραματική Αξιολόγηση",
    ])

    with tab1:
        if portfolio is None:
            st.info(
                "Φορτώστε ένα χαρτοφυλάκιο από το πλαϊνό μενού για να ξεκινήσετε. "
                "Επιλέξτε δείγμα ή ανεβάστε το δικό σας CSV."
            )
            with st.expander("Μορφή αρχείου CSV"):
                st.code(
                    "ticker,quantity,entry_price\n"
                    "AAPL,100,150.00\n"
                    "MSFT,50,300.00\n"
                    "SPY,200,380.00\n"
                    "BND,500,80.00\n",
                    language="csv",
                )
            with st.expander("Γλωσσάρι Χρηματοοικονομικών Όρων"):
                explainer = Explainer()
                for term, definition in explainer.GLOSSARY.items():
                    st.write(f"**{term}**: {definition}")
        else:
            try:
                render_overview(portfolio)
            except Exception as e:
                st.error(f"Σφάλμα επισκόπησης: {e}")
                st.exception(e)

    with tab2:
        if portfolio is None:
            st.info("Φορτώστε χαρτοφυλάκιο για να δείτε την ανάλυση σεναρίων.")
        else:
            try:
                engine = st.session_state.get("scenario_engine")
                if engine:
                    render_scenarios(portfolio, engine)
            except Exception as e:
                st.error(f"Σφάλμα σεναρίου: {e}")
                st.exception(e)

    with tab3:
        if portfolio is None:
            st.info("Φορτώστε χαρτοφυλάκιο για να δείτε την παρακολούθηση κινδύνου.")
        else:
            try:
                analysis = st.session_state.get("risk_analysis")
                if analysis:
                    render_risk(analysis)
            except Exception as e:
                st.error(f"Σφάλμα παρακολούθησης κινδύνου: {e}")
                st.exception(e)

    with tab4:
        if portfolio is None:
            st.info("Φορτώστε χαρτοφυλάκιο για να δείτε τις προτάσεις.")
        else:
            try:
                recs = st.session_state.get("recommendations")
                if recs is not None:
                    render_recommendations(recs)
            except Exception as e:
                st.error(f"Σφάλμα προτάσεων: {e}")
                st.exception(e)

    with tab5:
        try:
            render_validation()
        except Exception as e:
            st.error(f"Σφάλμα αξιολόγησης: {e}")
            st.exception(e)

    st.markdown("---")
    st.caption(
        "📊 Σύστημα Συμβουλευτικής Επενδύσεων — Digital Twin | "
        "Δεδομένα αγοράς: Yahoo Finance (yfinance) | "
        "⚠️ Για εκπαιδευτικούς σκοπούς μόνο — δεν αποτελεί επενδυτική συμβουλή."
    )


if __name__ == "__main__":
    main()
