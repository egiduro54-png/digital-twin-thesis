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
import math
import textwrap

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must come first so errors can be shown on screen
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Twin Investment Advisory System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Heavy imports after set_page_config so any crash is visible on screen
_import_errors = []
try:
    import numpy as np
except Exception as _e:
    _import_errors.append(f"numpy: {_e}")
    np = None
try:
    import pandas as pd
except Exception as _e:
    _import_errors.append(f"pandas: {_e}")
    pd = None
if _import_errors:
    st.error("Crash during package import:")
    for err in _import_errors:
        st.code(err)
    st.stop()

# Project modules — imported after set_page_config so errors appear on screen
_import_errors = []
try:
    from src.utils import build_portfolio, format_currency, format_pct, format_ratio
    from src.utils import severity_color, severity_emoji, priority_color, setup_logging
except Exception as _e:
    _import_errors.append(f"src.utils: {_e}")
try:
    from src.risk_monitor import RiskMonitor
except Exception as _e:
    _import_errors.append(f"src.risk_monitor: {_e}")
try:
    from src.scenario_engine import ScenarioEngine
except Exception as _e:
    _import_errors.append(f"src.scenario_engine: {_e}")
try:
    from src.recommendations import RecommendationEngine
except Exception as _e:
    _import_errors.append(f"src.recommendations: {_e}")
try:
    from src.explainer import Explainer
except Exception as _e:
    _import_errors.append(f"src.explainer: {_e}")
try:
    from src.validation import ValidationExperiment, ValidationResults, PORTFOLIO_ARCHETYPES
except Exception as _e:
    _import_errors.append(f"src.validation: {_e}")
if _import_errors:
    st.error("Import errors detected:")
    for err in _import_errors:
        st.code(err)
    st.stop()

setup_logging("WARNING")
import logging
logger = logging.getLogger(__name__)

EXPLAINER = Explainer()


def _show_fig(fig, dpi: int = 120) -> None:
    """
    Render a matplotlib figure inline as a base64 PNG.

    st.pyplot serves images through Streamlit's in-memory media store; after a
    rerun or a Streamlit Cloud restart the browser may request an image that has
    already been evicted, leaving a broken-image icon. Embedding the PNG in the
    page avoids that second request entirely.
    """
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;height:auto">',
        unsafe_allow_html=True,
    )

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
        "optimization": None,
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


HISTORY_YEARS = 5  # covers the 2022 bear market for drawdown/recovery metrics


def _load_portfolio(csv_path: str, risk_profile: str, name: str):
    with st.spinner("Λήψη δεδομένων αγοράς από Yahoo Finance…"):
        try:
            portfolio = build_portfolio(
                csv_path=csv_path,
                risk_profile=risk_profile,
                portfolio_name=name,
                history_years=HISTORY_YEARS,
            )
        except Exception as exc:
            st.error(f"Αποτυχία φόρτωσης χαρτοφυλακίου: {exc}")
            return
    if _analyse_and_store(portfolio, risk_profile, name):
        st.success("Το χαρτοφυλάκιο φορτώθηκε επιτυχώς!")


def _analyse_and_store(portfolio, risk_profile: str, name: str) -> bool:
    """Run risk analysis, scenarios and recommendations for a loaded portfolio."""
    portfolio.risk_profile = risk_profile
    portfolio._metrics = None
    try:
        monitor = RiskMonitor(portfolio)
        analysis = monitor.run_full_analysis()
    except Exception as exc:
        st.error(f"Αποτυχία ανάλυσης κινδύνου: {exc}")
        st.exception(exc)
        return False

    try:
        engine = ScenarioEngine(portfolio)
    except Exception as exc:
        st.error(f"Αποτυχία μηχανισμού σεναρίων: {exc}")
        st.exception(exc)
        return False

    try:
        rec_engine = RecommendationEngine(portfolio, monitor)
        recs = rec_engine.generate_recommendations()
    except Exception as exc:
        st.warning(f"Οι προτάσεις δεν φορτώθηκαν (συνέχεια χωρίς αυτές): {exc}")
        recs = []
        rec_engine = None

    optimization = None
    if rec_engine is not None:
        try:
            optimization = rec_engine.optimization_recommendation()
        except Exception as exc:
            st.warning(f"Η βελτιστοποίηση Markowitz δεν ολοκληρώθηκε: {exc}")

    st.session_state["portfolio"] = portfolio
    st.session_state["risk_analysis"] = analysis
    st.session_state["recommendations"] = recs
    st.session_state["optimization"] = optimization
    st.session_state["plan_weights"] = dict(rec_engine.plan.w) if rec_engine is not None else None
    st.session_state["plan_trades"] = rec_engine.plan_trades() if rec_engine is not None else None
    st.session_state["scenario_engine"] = engine
    st.session_state["portfolio_name"] = name
    st.session_state["risk_profile"] = risk_profile
    st.session_state["last_loaded"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    return True


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

    # Keyed so the suitability questionnaire can set it
    st.session_state.setdefault("sidebar_profile", "flexible")
    risk_profile = st.sidebar.selectbox(
        "Προφίλ κινδύνου επενδυτή",
        ["liquidity_plus", "defensive", "flexible", "growth", "dynamic"],
        format_func=lambda x: RISK_PROFILE_LABELS[x],
        key="sidebar_profile",
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

    # Concentration: same verdict as the Risk Monitor (broad ETFs have their
    # own limit), so this tab and the risk tab never disagree
    analysis = st.session_state.get("risk_analysis") or RiskMonitor(portfolio).run_full_analysis()
    single = next((a for a in analysis.get("all", [])
                   if a.get("title") == "Συγκέντρωση σε Μεμονωμένο Τίτλο"), None)
    if single and single["severity"] != "ok":
        d = single["detail"]
        kind = "διαφοροποιημένο ETF" if d.get("is_fund") else "μεμονωμένος τίτλος"
        text = (f"συγκέντρωση: {d['largest_ticker']} ({kind}) αποτελεί "
                f"{d['largest_pct']:.1f}% του χαρτοφυλακίου (όριο {d['target_pct']:.0f}%).")
        if single["severity"] == "alert":
            msgs.append((f"Υψηλή {text}", "error"))
        else:
            msgs.append((f"Μέτρια {text}", "warning"))

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

    # --- Pointer to the coordinated rebalancing plan ---
    if max(abs(alignment.get("equity_drift_pct") or 0),
           abs(alignment.get("fixed_income_drift_pct") or 0)) > 5:
        st.info(
            "Το χαρτοφυλάκιο αποκλίνει από την κατανομή-στόχο του προφίλ. Το συντονισμένο σχέδιο "
            "ανακατανομής (ανά τίτλο, με τις επιπτώσεις του) βρίσκεται στην καρτέλα "
            "**Προτάσεις Αναδιάρθρωσης**."
        )

    # --- Correlation Heatmap ---
    st.markdown("---")
    with st.expander("🔗 Πίνακας Συσχέτισης Τίτλων (Correlation Matrix)", expanded=False):
        _plot_correlation_heatmap(portfolio)


def _plot_correlation_heatmap(portfolio):
    """Render a colour-coded correlation matrix for portfolio assets."""
    import matplotlib.pyplot as plt
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
    _show_fig(fig)
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
    import matplotlib.pyplot as plt
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
    _show_fig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 2: Ανάλυση Σεναρίων
# ---------------------------------------------------------------------------

def _generate_scenario_pdf(comparison: dict, impacts: list[dict]) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    params = comparison["scenario_params"]
    summary = comparison["summary"]
    metrics = comparison["metrics"]
    name = comparison["scenario_name"]
    desc = comparison["scenario_description"]
    ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    with PdfPages(buf) as pdf:
        # ── Page 1: Summary & Metrics ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.patch.set_facecolor("#f9f9f9")

        y = 0.96
        fig.text(0.5, y, "Scenario Analysis Report", ha="center", fontsize=18, fontweight="bold")
        y -= 0.05
        fig.text(0.5, y, name, ha="center", fontsize=13, color="#333333")
        y -= 0.04
        fig.text(0.5, y, desc, ha="center", fontsize=9, color="#666666")
        y -= 0.04
        fig.text(0.05, y, f"Export date: {ts}", fontsize=8, color="#999999")

        y -= 0.05
        fig.text(0.05, y, "Scenario Parameters", fontsize=11, fontweight="bold", color="#2c3e50")
        y -= 0.04
        param_lines = [
            f"Market Change:  {params['market_change_pct']:+.1f}%",
            f"Rate Change:    {params['rate_change_pct']:+.2f}%",
            f"Volatility Multiplier:  {params['volatility_multiplier']:.1f}x",
        ]
        if params.get("sector_overrides"):
            so = ", ".join(f"{k}: {v*100:+.0f}%" for k, v in params["sector_overrides"].items())
            param_lines.append(f"Sector Overrides:  {so}")
        for line in param_lines:
            fig.text(0.08, y, line, fontsize=9)
            y -= 0.035

        y -= 0.02
        fig.text(0.05, y, "Portfolio Summary", fontsize=11, fontweight="bold", color="#2c3e50")
        y -= 0.04
        port_chg = summary["portfolio_change_pct"]
        resilience = summary["resilience_vs_market_pct"]
        summary_lines = [
            (f"Current Value:    ${summary['current_total_value']:,.2f}", "black"),
            (f"Scenario Value:   ${summary['scenario_total_value']:,.2f}", "black"),
            (f"Portfolio Change: {port_chg:+.2f}%", "green" if port_chg >= 0 else "red"),
            (f"Relative vs Market: {resilience:+.2f}%", "green" if resilience >= 0 else "red"),
        ]
        for line, color in summary_lines:
            fig.text(0.08, y, line, fontsize=9, color=color)
            y -= 0.035

        # Metrics table
        y -= 0.02
        fig.text(0.05, y, "Metrics Comparison", fontsize=11, fontweight="bold", color="#2c3e50")
        y -= 0.02

        metric_labels = {
            "total_value": "Portfolio Value ($)",
            "volatility_annual_pct": "Annual Volatility (%)",
            "sharpe_ratio": "Sharpe Ratio",
            "max_drawdown_pct": "Max Drawdown (%)",
            "beta": "Beta",
            "var_95_monthly_pct": "VaR 95% Monthly (%)",
        }
        col_headers = ["Metric", "Current", "Scenario", "Change"]
        col_x = [0.05, 0.45, 0.62, 0.79]
        col_w = [0.4, 0.17, 0.17, 0.17]

        # Header row
        for hdr, x in zip(col_headers, col_x):
            fig.text(x, y, hdr, fontsize=9, fontweight="bold",
                     color="white",
                     bbox=dict(boxstyle="square,pad=0.3", facecolor="#2c3e50", edgecolor="none"))
        y -= 0.038

        row_bg = ["#ffffff", "#f2f2f2"]
        for ri, (key, label) in enumerate(metric_labels.items()):
            m = metrics.get(key, {})
            curr = m.get("current")
            scen = m.get("scenario")
            chg = m.get("change")
            if curr is None:
                continue
            bg = row_bg[ri % 2]
            vals = [
                label,
                f"{curr:,.2f}",
                f"{scen:,.2f}" if scen is not None else "N/A",
                f"{chg:+,.2f}" if chg is not None else "N/A",
            ]
            chg_color = "black"
            if chg is not None:
                # For drawdown and VaR, lower is better
                improve = chg > 0 if key not in ("volatility_annual_pct", "max_drawdown_pct", "var_95_monthly_pct") else chg < 0
                chg_color = "green" if improve else "red"

            for vi, (val, x) in enumerate(zip(vals, col_x)):
                color = chg_color if vi == 3 else "black"
                fig.text(x, y, val, fontsize=8, color=color,
                         bbox=dict(boxstyle="square,pad=0.25", facecolor=bg, edgecolor="none"))
            y -= 0.036

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Asset Impact ──────────────────────────────────────────
        if impacts:
            fig2, ax2 = plt.subplots(figsize=(11, 8.5))
            ax2.axis("off")
            fig2.patch.set_facecolor("#f9f9f9")
            fig2.text(0.5, 0.97, "Asset Impact", ha="center", fontsize=14, fontweight="bold")
            fig2.text(0.5, 0.94, name, ha="center", fontsize=10, color="#555555")

            headers = ["Ticker", "Sector", "Current Value ($)", "Scenario Value ($)", "Change ($)", "Change (%)"]
            rows = [
                [
                    d["ticker"],
                    d.get("sector", ""),
                    f"{d['current_value']:,.2f}",
                    f"{d['scenario_value']:,.2f}",
                    f"{d['dollar_change']:+,.2f}",
                    f"{d['pct_change']:+.2f}%",
                ]
                for d in impacts
            ]

            tbl = ax2.table(
                cellText=rows,
                colLabels=headers,
                loc="center",
                bbox=[0.0, 0.05, 1.0, 0.85],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.auto_set_column_width(list(range(len(headers))))
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f2f2f2")
                if row > 0 and col == 4:
                    try:
                        val = float(rows[row - 1][4].replace(",", "").replace("+", ""))
                        cell.set_text_props(color="green" if val >= 0 else "red")
                    except ValueError:
                        pass

            fig2.text(0.05, 0.02, f"Report: {name} | {ts}", fontsize=7, color="#aaaaaa")
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    buf.seek(0)
    return buf.read()


def _build_rebalanced_portfolio(portfolio, recs: list):
    """Build a copy of portfolio with the recommendations' trades applied."""
    import copy
    from src.portfolio import Asset, Portfolio as _Portfolio

    trades = _summary_trades(recs, portfolio)
    if not trades:
        return None

    bond_etfs = {"BND", "TLT", "IEF", "SHY", "AGG", "LQD", "HYG", "VCIT", "VCSH"}
    intl_etfs = {"VXUS", "EFA", "IEMG"}
    commodity_etfs = {"GLD", "SLV", "GSG"}

    assets_by_ticker = {a.ticker: copy.copy(a) for a in portfolio.assets}
    for trade in trades:
        ticker = trade["ticker"]
        qty = trade["quantity"]
        price = trade["price"]
        if trade["action"] == "SELL":
            if ticker in assets_by_ticker:
                assets_by_ticker[ticker].quantity = max(0.0, assets_by_ticker[ticker].quantity - qty)
        elif trade["action"] == "BUY":
            if ticker in assets_by_ticker:
                assets_by_ticker[ticker].quantity += qty
            else:
                if ticker in bond_etfs:
                    ac, sector = "Fixed Income", "Bond ETF"
                elif ticker in intl_etfs:
                    ac, sector = "International Equity", "International ETF"
                elif ticker in commodity_etfs:
                    ac, sector = "Commodity", "Commodities ETF"
                else:
                    ac, sector = "Equity", "Equity ETF"
                assets_by_ticker[ticker] = Asset(
                    ticker=ticker, quantity=float(qty),
                    entry_price=price, current_price=price,
                    name=ticker, sector=sector,
                    industry="Unknown", asset_class=ac, country="US",
                )

    new_assets = [a for a in assets_by_ticker.values() if a.quantity > 0]
    if not new_assets:
        return None
    return _Portfolio(
        assets=new_assets,
        historical_prices=portfolio.historical_prices,
        risk_profile=portfolio.risk_profile,
        name=f"{portfolio.name} (Αναδιαρθρωμένο)",
    )


def _render_rebalance_comparison(before_cmp: dict, after_cmp: dict):
    """Side-by-side current vs rebalanced portfolio under the same scenario."""
    b_sum = before_cmp.get("summary", {})
    a_sum = after_cmp.get("summary", {})
    b_met = before_cmp.get("metrics", {})
    a_met = after_cmp.get("metrics", {})

    st.subheader("Αναδιαρθρωμένο Χαρτοφυλάκιο: Σύγκριση στο ίδιο Σενάριο")
    st.caption(
        "Πώς θα αντεπεξερχόταν το αναδιαρθρωμένο χαρτοφυλάκιο (εφόσον εφαρμοστούν "
        "οι προτάσεις) υπό τις ίδιες συνθήκες σεναρίου."
    )

    c1, c2 = st.columns(2)
    b_val = b_sum.get("scenario_total_value")
    a_val = a_sum.get("scenario_total_value")
    b_chg = b_sum.get("portfolio_change_pct", 0)
    a_chg = a_sum.get("portfolio_change_pct", 0)
    c1.metric("Τρέχον — Αξία στο Σενάριο",
              format_currency(b_val) if b_val is not None else "N/A",
              f"{b_chg:+.1f}%")
    improvement = a_chg - b_chg
    c2.metric("Αναδιαρθρωμένο — Αξία στο Σενάριο",
              format_currency(a_val) if a_val is not None else "N/A",
              f"{a_chg:+.1f}% ({improvement:+.1f}% vs τρέχον)",
              delta_color="normal" if improvement >= 0 else "inverse")

    compare_defs = [
        ("volatility_annual_pct", "Μεταβλητότητα (%)", False),
        ("sharpe_ratio", "Sharpe Ratio", True),
        ("max_drawdown_pct", "Max Drawdown (%)", True),
        ("var_95_monthly_pct", "VaR 95% μηνιαίο (%)", True),
    ]
    rows = []
    for key, label, higher_better in compare_defs:
        b = b_met.get(key, {}).get("scenario")
        a = a_met.get(key, {}).get("scenario")
        if b is None or a is None:
            continue
        diff = a - b
        better = (diff > 0) == higher_better
        rows.append({
            "Δείκτης": label,
            "Τρέχον": f"{b:,.2f}",
            "Αναδιαρθρωμένο": f"{a:,.2f}",
            "Διαφορά": f"{diff:+.2f} {'✅' if better else '⚠️'}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows))


def _render_resilience_index(portfolio):
    """Composite Resilience Index with its six explained components."""
    from src.resilience import compute_resilience, resilience_band

    try:
        ri = compute_resilience(portfolio)
    except Exception as exc:
        st.warning(f"Ο Δείκτης Ανθεκτικότητας δεν υπολογίστηκε: {exc}")
        return

    st.subheader("Δείκτης Ανθεκτικότητας Χαρτοφυλακίου (Resilience Index)")
    st.caption(
        "Σύνθετος δείκτης 0–100 (υψηλότερο = πιο ανθεκτικό) από έξι συνιστώσες με "
        "σταθερά, προκαθορισμένα βάρη: αντοχή σε stress σενάρια, διαδοχικά shocks, "
        "κίνδυνος ουράς, ταχύτητα ανάκαμψης, διαφοροποίηση και συγκέντρωση."
    )
    raw = ri["raw"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Resilience Index", f"{ri['score']:.0f} / 100", resilience_band(ri["score"]),
              delta_color="off")
    c2.metric("Μέση απώλεια σε stress σενάρια", f"−{raw['mean_stress_loss_pct']:.1f}%")
    c3.metric("Διαδοχικά shocks (αγορά −20% → επιτόκια +2%)",
              f"−{raw['sequential_loss_pct']:.1f}%")

    raw_text = {
        "stress": f"μέση απώλεια {raw['mean_stress_loss_pct']:.1f}% σε 6 σενάρια",
        "sequential": f"απώλεια {raw['sequential_loss_pct']:.1f}%",
        "downside": (f"μέση ημερήσια απώλεια {raw['daily_cvar95_pct']:.2f}% στο χειρότερο 5% των ημερών"
                     if raw["daily_cvar95_pct"] is not None else "—"),
        "recovery": (
            f"{raw['recovery_days']:.0f} ημέρες ανάκαμψης"
            + ("" if raw["recovered"] else " (ανοιχτό drawdown, με ποινή)")
            + f", {raw['underwater_frac_pct']:.0f}% του χρόνου >5% κάτω από την κορυφή"
            if raw["underwater_frac_pct"] is not None else "—"
        ),
        "diversification": (f"Diversification Ratio {raw['diversification_ratio']:.2f}"
                            if raw["diversification_ratio"] is not None else "—"),
        "concentration": (f"{raw['effective_positions']:.1f} ισοδύναμες θέσεις"
                          if raw["effective_positions"] is not None else "—"),
    }
    rows = [{
        "Συνιστώσα": ri["labels"][k],
        "Βάρος": f"{ri['weights'][k]:.0%}",
        "Βαθμός (0–100)": round(v * 100) if np.isfinite(v) else None,
        "Συνεισφορά": round(100 * ri["weights"][k] * v, 1) if np.isfinite(v) else None,
        "Μέτρηση": raw_text[k],
    } for k, v in ri["components"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    weakest = min((k for k, v in ri["components"].items() if np.isfinite(v)),
                  key=lambda k: ri["components"][k], default=None)
    if weakest:
        st.info(f"Πιο αδύναμη διάσταση: **{ri['labels'][weakest]}** "
                f"({ri['components'][weakest] * 100:.0f}/100) — {raw_text[weakest]}.")
    st.markdown("---")


def render_scenarios(portfolio, engine: ScenarioEngine):
    st.header("Ανάλυση Σεναρίων — What-If Προσομοίωση")
    _render_resilience_index(portfolio)
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
        active_id = list(engine.scenarios.keys())[-1]
    else:
        active_id = selected_id

    comparison = engine.compare_portfolio_metrics(active_id)
    _render_scenario_result(comparison, engine, active_id)

    st.markdown("---")
    st.subheader("Σύγκριση Πολλαπλών Σεναρίων")
    stress_ids = ["market_down_10", "market_down_20", "financial_crisis_2008",
                  "covid_crash_2020", "recession", "inflation_spike"]
    available_stress = [s for s in stress_ids if s in engine.scenarios]
    multi = engine.compare_multiple_scenarios(available_stress)
    _render_multi_scenario_table(multi)

    st.markdown("---")
    st.subheader("Χειρότερο Σενάριο")
    st.caption("Εκτελεί και τα 14 προκαθορισμένα σενάρια και εντοπίζει αυτό με τη μεγαλύτερη απώλεια.")
    if st.button("Εντοπισμός χειρότερου σεναρίου"):
        predefined = [sid for sid, sc in engine.scenarios.items() if sc.get("category") != "Custom"]
        worst = engine.find_worst_scenario(predefined)
        if worst:
            w1, w2 = st.columns(2)
            w1.metric("Σενάριο", worst["scenario_name"])
            w2.metric("Απώλεια αξίας", format_currency(worst["portfolio_loss"]),
                      delta=f"{worst['portfolio_loss_pct']:.1f}%", delta_color="normal")
        else:
            st.info("Δεν ήταν δυνατός ο υπολογισμός των σεναρίων.")


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

def _render_learned_explanation(portfolio):
    """Per-portfolio explanation of the learned risk model (XAI)."""
    import json
    from src.learned_scorer import portfolio_features, explain_portfolio

    try:
        ex = json.loads((VALIDATION_DIR / "learned_explainer.json").read_text(encoding="utf-8"))
    except Exception:
        return  # no trained model shipped
    try:
        r = explain_portfolio(portfolio_features(portfolio), ex)
    except Exception as exc:
        st.caption(f"Η εξήγηση του μοντέλου δεν υπολογίστηκε: {exc}")
        return

    st.subheader("Γιατί αυτό το επίπεδο κινδύνου; (εξήγηση μοντέλου)")
    st.caption(
        f"Γραμμικό μοντέλο με μη αρνητικά βάρη, εκπαιδευμένο σε {ex['n_train']:,} χαρτοφυλάκια "
        "της αξιολόγησης να κατατάσσει τον κίνδυνο του επόμενου έτους από τις συνιστώσες του "
        "ψηφιακού διδύμου. Σύγκριση με τα 150 χαρτοφυλάκια του πιο πρόσφατου παραθύρου."
    )
    pct = r["percentile"]
    level = ("χαμηλότερο" if pct < 33 else "μέσο" if pct < 67 else "υψηλότερο")
    c1, c2 = st.columns([1, 2])
    c1.metric("Εκατοστημόριο κινδύνου", f"{pct:.0f}ο",
              help="Ποσοστό των χαρτοφυλακίων αναφοράς με χαμηλότερο εκτιμώμενο κίνδυνο.")
    c2.info(f"Το χαρτοφυλάκιο έχει **{level}** εκτιμώμενο κίνδυνο από τα χαρτοφυλάκια "
            f"αναφοράς: πιο επικίνδυνο από το {pct:.0f}% αυτών.")

    contrib = r["contributions"]
    contrib = contrib[contrib["contribution"].abs() > 1e-6]
    if contrib.empty:
        return
    st.bar_chart(contrib.set_index("label")["contribution"].sort_values(), horizontal=True)
    st.caption(
        "Συνεισφορά κάθε παράγοντα σε σχέση με ένα τυπικό χαρτοφυλάκιο: θετική = αυξάνει τον "
        "κίνδυνο, αρνητική = τον μειώνει. Επειδή το μοντέλο είναι γραμμικό, οι συνεισφορές "
        "αθροίζουν ακριβώς στο συνολικό σκορ (ισοδυναμούν με τιμές SHAP)."
    )
    top = contrib.iloc[0]
    direction = "αυξάνει" if top["contribution"] > 0 else "μειώνει"
    st.markdown(f"Ο παράγοντας με τη μεγαλύτερη επίδραση είναι **{top['label']}**, που "
                f"{direction} τον εκτιμώμενο κίνδυνο.")
    outside = contrib[contrib["outside_reference"]]
    if not outside.empty:
        st.caption("Εκτός του εύρους των χαρτοφυλακίων αναφοράς (η επίδραση έχει περιοριστεί "
                   "στις ±3 τυπικές αποκλίσεις): " + ", ".join(outside["label"]) + ".")
    st.markdown("---")


def render_risk(analysis: dict):
    st.header("Παρακολούθηση Κινδύνου (Risk Monitoring)")
    if st.session_state.get("portfolio") is not None:
        _render_learned_explanation(st.session_state["portfolio"])

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

def _summary_trades(recs: list, portfolio) -> list:
    """
    Trades for the summary: the net move from the current portfolio to the
    final coordinated plan. Summing each card's rounded trades drifts from it,
    so the per-card consolidation is only a fallback.
    """
    plan = st.session_state.get("plan_trades")
    if plan:
        return [dict(t) for t in plan]
    raw_trades = [t for r in recs for t in r.get("trades", [])]
    return _consolidate_trades(raw_trades, portfolio)


def _consolidate_trades(trades: list, portfolio) -> list:
    """
    Net trades per ticker (BUY minus SELL), cap SELLs to holdings,
    and scale BUYs so total BUY never exceeds total SELL proceeds.
    """
    holdings = {}
    if portfolio:
        holdings = {a.ticker: int(a.quantity) for a in portfolio.assets}

    # Net BUY against SELL per ticker: one step may sell what a later step buys
    net: dict[str, dict] = {}
    for t in trades:
        sign = 1 if t["action"] == "BUY" else -1
        row = net.setdefault(t["ticker"], {"quantity": 0, "price": t.get("price", 0.0)})
        row["quantity"] += sign * t.get("quantity", 0)

    sells, buys = [], []
    for ticker, data in net.items():
        action = "BUY" if data["quantity"] > 0 else "SELL"
        qty   = abs(data["quantity"])
        price = data["price"]
        if action == "SELL":
            if ticker in holdings:
                qty = min(qty, holdings[ticker])
            if qty > 0:
                sells.append({"ticker": ticker, "action": "SELL",
                               "quantity": qty, "price": price,
                               "value": round(qty * price, 2)})
        else:
            if qty > 0:
                buys.append({"ticker": ticker, "action": "BUY",
                              "quantity": qty, "price": price,
                              "value": round(qty * price, 2)})

    # Scale BUYs down if they exceed total sell proceeds
    total_sell = sum(t["value"] for t in sells)
    total_buy  = sum(t["value"] for t in buys)
    if total_buy > total_sell > 0:
        scale = total_sell / total_buy
        scaled = []
        for t in buys:
            new_qty = max(1, int(t["quantity"] * scale))
            scaled.append({"ticker": t["ticker"], "action": "BUY",
                            "quantity": new_qty, "price": t["price"],
                            "value": round(new_qty * t["price"], 2)})
        buys = scaled

    return sells + buys


def _generate_proposal_pdf(recs: list, all_trades: list, portfolio) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    port_name = portfolio.name if portfolio else "Portfolio"

    sell_total = sum(t["value"] for t in all_trades if t.get("action") == "SELL")
    buy_total  = sum(t["value"] for t in all_trades if t.get("action") == "BUY")
    net        = buy_total - sell_total

    with PdfPages(buf) as pdf:
        # ── Page 1: Cover ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.patch.set_facecolor("#1a252f")
        fig.text(0.5, 0.72, "Digital Twin Investment Advisory System",
                 ha="center", fontsize=20, fontweight="bold", color="white")
        fig.text(0.5, 0.62, "Portfolio Rebalancing Proposal",
                 ha="center", fontsize=15, color="#aaaaaa")
        fig.text(0.5, 0.52, port_name,
                 ha="center", fontsize=13, color="white")
        fig.text(0.5, 0.38, f"Date: {ts}",
                 ha="center", fontsize=10, color="#888888")
        fig.text(0.5, 0.28, f"Total Recommendations: {len(recs)}",
                 ha="center", fontsize=10, color="#aaaaaa")
        fig.text(0.5, 0.12,
                 "For informational purposes only. Not investment advice.",
                 ha="center", fontsize=8, color="#666666")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Trade Summary ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        fig.patch.set_facecolor("#f9f9f9")
        fig.text(0.5, 0.96, "Trade Summary", ha="center", fontsize=16, fontweight="bold")
        fig.text(0.05, 0.91, f"Portfolio: {port_name}   |   Date: {ts}", fontsize=9, color="#555")

        # Summary metrics
        fig.text(0.10, 0.85, f"Total Sells:  ${sell_total:,.2f}", fontsize=11, color="#c0392b")
        fig.text(0.40, 0.85, f"Total Buys:   ${buy_total:,.2f}", fontsize=11, color="#27ae60")
        net_color = "#27ae60" if net <= 0 else "#c0392b"
        fig.text(0.70, 0.85, f"Net Capital:  ${net:,.2f}", fontsize=11, color=net_color)

        # Trades table
        if all_trades:
            trade_rows = [
                [
                    t.get("ticker", ""),
                    t.get("action", ""),
                    str(int(t.get("quantity", 0))),
                    f"${t.get('price', 0):,.2f}",
                    f"${t.get('value', 0):,.2f}",
                ]
                for t in all_trades
            ]
            tbl = ax.table(
                cellText=trade_rows,
                colLabels=["Ticker", "Action", "Shares", "Price ($)", "Value ($)"],
                loc="center",
                bbox=[0.05, 0.20, 0.90, 0.60],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.auto_set_column_width(list(range(5)))
            for (row, col), cell in tbl.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f2f2f2")
                if row > 0 and col == 1:
                    action = trade_rows[row - 1][1]
                    cell.set_text_props(color="#c0392b" if action == "SELL" else "#27ae60",
                                       fontweight="bold")

        fig.text(0.05, 0.08,
                 "This proposal was generated automatically by the Digital Twin system.",
                 fontsize=7, color="#aaaaaa")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Pages 3+: Recommendations Detail ────────────────────────────
        priority_map = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        color_map = {"critical": "#dc3545", "high": "#fd7e14",
                     "medium": "#e6a817", "low": "#28a745"}
        LINE_H = 0.032   # height per text line
        MARGIN_BOTTOM = 0.06

        def _rec_height(rec):
            why_lines = len(textwrap.wrap(rec.get("why", ""), width=95))
            action_lines = len(textwrap.wrap(rec.get("action", ""), width=95))
            return LINE_H * (1 + action_lines + why_lines) + 0.025

        def _new_rec_page(page_num, total_pages):
            f, a = plt.subplots(figsize=(11, 8.5))
            a.axis("off")
            f.patch.set_facecolor("#f9f9f9")
            f.text(0.5, 0.96, "Recommendation Details", ha="center",
                   fontsize=16, fontweight="bold")
            f.text(0.5, 0.93, f"Page {page_num} of {total_pages}",
                   ha="center", fontsize=9, color="#888")
            return f, a

        # pre-calculate how many pages we need
        pages_needed = []
        current_page_recs = []
        y_used = 0.0
        available = 1.0 - 0.13 - MARGIN_BOTTOM  # top = 0.87 effective
        for rec in recs:
            h = _rec_height(rec)
            if y_used + h > available and current_page_recs:
                pages_needed.append(current_page_recs)
                current_page_recs = []
                y_used = 0.0
            current_page_recs.append(rec)
            y_used += h
        if current_page_recs:
            pages_needed.append(current_page_recs)
        if not pages_needed:
            pages_needed = [[]]

        total_pages = len(pages_needed)
        for page_num, chunk in enumerate(pages_needed, start=1):
            fig, ax = _new_rec_page(page_num, total_pages)
            y = 0.87
            for rec in chunk:
                priority = rec.get("priority", "medium")
                c = color_map.get(priority, "#333")
                fig.text(0.05, y,
                         f"[{priority_map.get(priority, '')}] {rec.get('title', '')}",
                         fontsize=10, fontweight="bold", color=c)
                y -= LINE_H

                for line in textwrap.wrap(f"Action: {rec.get('action', '')}", width=95):
                    fig.text(0.07, y, line, fontsize=9, color="#333")
                    y -= LINE_H

                for line in textwrap.wrap(f"Why: {rec.get('why', '')}", width=95):
                    fig.text(0.07, y, line, fontsize=8, color="#555")
                    y -= LINE_H

                y -= 0.025  # gap between recommendations

            fig.text(0.05, 0.03, f"Digital Twin Investment Advisory System | {ts}",
                     fontsize=7, color="#aaaaaa")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    return buf.read()


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
    portfolio = st.session_state.get("portfolio")
    all_trades = _summary_trades(recs, portfolio)
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

    _render_optimization_alternative(st.session_state.get("optimization"))

    st.markdown("---")
    if st.button("📄 Εξαγωγή PDF για Πελάτη"):
        with st.spinner("Δημιουργία PDF..."):
            try:
                pdf_bytes = _generate_proposal_pdf(recs, all_trades, portfolio)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="⬇️ Λήψη Trade Proposal PDF",
                    data=pdf_bytes,
                    file_name=f"trade_proposal_{ts}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Σφάλμα PDF: {e}")
                st.exception(e)


def _render_optimization_alternative(opt: dict | None):
    """Markowitz max-Sharpe allocation, shown as an alternative to the rule-based plan."""
    st.markdown("---")
    st.subheader("Εναλλακτική: Βελτιστοποίηση Markowitz (Efficient Frontier)")
    st.caption(
        "Πλήρης ανακατανομή που μεγιστοποιεί το Sharpe Ratio υπό τους περιορισμούς του "
        "προφίλ σας. Είναι εναλλακτική λύση: οι συναλλαγές της δεν προστίθενται στη "
        "Σύνοψη Συναλλαγών παραπάνω."
    )
    if not opt:
        st.info("Δεν προέκυψε εφικτή βελτιστοποιημένη κατανομή για αυτό το χαρτοφυλάκιο.")
        return
    _render_recommendation_card(opt)
    weights = opt.get("optimal_weights", {})
    if weights:
        df = pd.DataFrame(
            [{"Ticker": t, "Βάρος-στόχος (%)": w} for t, w in
             sorted(weights.items(), key=lambda kv: -kv[1])]
        )
        st.dataframe(df, hide_index=True)


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

def render_investor(portfolio):
    """Investor twin: suitability questionnaire and goal-based Monte Carlo."""
    from src.investor import QUESTIONS, score_questionnaire, simulate_goal, CMA_RETURNS
    from src.rebalance_plan import RebalancePlan

    st.header("Επενδυτής & Στόχοι")
    st.markdown(
        "Το ψηφιακό δίδυμο δεν αναπαριστά μόνο το χαρτοφυλάκιο αλλά και τον επενδυτή: "
        "το προφίλ κινδύνου του και τους στόχους του."
    )

    # ── Suitability questionnaire ─────────────────────────────────────────
    st.subheader("1. Ερωτηματολόγιο Καταλληλότητας")
    st.caption(
        "Επτά ερωτήσεις τύπου MiFID II. Οι ερωτήσεις ικανότητας ανάληψης ζημίας "
        "(ορίζοντας, εισόδημα, ρευστότητα) θέτουν ανώτατο όριο στο προφίλ, ανεξάρτητα "
        "από τη διάθεση για κίνδυνο."
    )
    with st.form("questionnaire"):
        answers = {}
        for q in QUESTIONS:
            label = q["text"] + ("  *(ικανότητα ζημίας)*" if q["capacity"] else "")
            choice = st.radio(label, q["options"], index=2, key=f"q_{q['id']}")
            answers[q["id"]] = q["options"].index(choice)
        submitted = st.form_submit_button("Υπολογισμός προφίλ")
    if submitted:
        st.session_state["questionnaire_result"] = score_questionnaire(answers)

    result = st.session_state.get("questionnaire_result")
    if result:
        prof = result["profile"]
        c1, c2 = st.columns(2)
        c1.metric("Βαθμολογία", f"{result['score']} / {result['max_score']}")
        c2.metric("Προτεινόμενο προφίλ", RISK_PROFILE_LABELS[prof])
        if result["capped_by"]:
            q = next(x for x in QUESTIONS if x["id"] == result["capped_by"])
            st.info(
                f"Η διάθεση για κίνδυνο αντιστοιχεί σε «{RISK_PROFILE_LABELS[result['uncapped_profile']]}», "
                f"αλλά η απάντηση στο «{q['text']}» περιορίζει το προφίλ σε «{RISK_PROFILE_LABELS[prof]}»."
            )
        current = st.session_state.get("risk_profile")
        if portfolio is not None and current and current != prof:
            st.warning(
                f"Το χαρτοφυλάκιο αναλύεται με προφίλ «{RISK_PROFILE_LABELS.get(current, current)}», "
                f"ενώ το ερωτηματολόγιο προτείνει «{RISK_PROFILE_LABELS[prof]}»."
            )
            if st.button("Εφαρμογή του προτεινόμενου προφίλ"):
                st.session_state["sidebar_profile"] = prof
                if _analyse_and_store(portfolio, prof, st.session_state.get("portfolio_name", "")):
                    st.rerun()
        elif portfolio is None:
            if st.button("Ορισμός ως προφίλ για την επόμενη φόρτωση"):
                st.session_state["sidebar_profile"] = prof
                st.rerun()

    st.markdown("---")

    # ── Goal-based Monte Carlo ────────────────────────────────────────────
    st.subheader("2. Στόχος και Πιθανότητα Επίτευξης (Monte Carlo)")
    if portfolio is None:
        st.info("Φορτώστε χαρτοφυλάκιο για να υπολογιστεί η πιθανότητα επίτευξης στόχου.")
        return

    total = portfolio.total_value
    g1, g2, g3 = st.columns(3)
    target = g1.number_input("Ποσό-στόχος ($)", min_value=1_000.0,
                             value=float(round(total * 2, -3)), step=10_000.0)
    years = int(g2.slider("Ορίζοντας (έτη)", 1, 40, 10))
    monthly = g3.number_input("Μηνιαία εισφορά ($)", min_value=0.0, value=500.0, step=100.0)

    plan = RebalancePlan(portfolio)
    groups = {t: plan.group(t) for t in portfolio.historical_prices.columns}
    allocations = {"Τρέχον χαρτοφυλάκιο": portfolio.get_weights_dict()}
    if st.session_state.get("plan_weights"):
        allocations["Μετά τις προτάσεις του DT"] = st.session_state["plan_weights"]

    goal = {"target": target, "years": years, "monthly": monthly}
    results = {}
    with st.spinner("Προσομοίωση 5.000 διαδρομών…"):
        for label, w in allocations.items():
            try:
                results[label] = simulate_goal(portfolio.historical_prices, w, groups,
                                               initial=total, monthly_contribution=monthly,
                                               years=years, target=target)
            except ValueError as exc:
                st.warning(f"{label}: {exc}")
    if results:
        _render_goal_results(results, CMA_RETURNS)

    st.markdown("---")
    _render_twin_timeline(portfolio)
    st.markdown("---")
    _render_twin_state(portfolio, goal)


def _render_goal_results(results: dict, CMA_RETURNS: dict):
    """Goal Monte Carlo table, fan chart and assumptions."""
    rows = [{
        "Κατανομή": label,
        "Πιθανότητα επίτευξης": f"{r['probability']:.0%}",
        "Διάμεση τελική αξία": format_currency(r["median"]),
        "Δυσμενές (5%)": format_currency(r["p5"]),
        "Ευνοϊκό (95%)": format_currency(r["p95"]),
        "Αναμ. ετήσια απόδοση": f"{r['expected_annual_return']:.1%}",
        "Ετήσια μεταβλητότητα": f"{r['annual_volatility']:.1%}",
    } for label, r in results.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    first = next(iter(results.values()))
    st.caption(f"Συνολικές εισφορές (αρχικό + μηνιαίες): {format_currency(first['invested'])}.")

    fan = pd.DataFrame({"Έτος": first["fan"]["year"]})
    for label, r in results.items():
        fan[f"{label} — διάμεσος"] = r["fan"]["p50"].values
        fan[f"{label} — δυσμενές 5%"] = r["fan"]["p5"].values
    st.line_chart(fan.set_index("Έτος"))

    st.caption(
        "Υποθέσεις: αναμενόμενες ετήσιες αποδόσεις μακροπρόθεσμες ανά κατηγορία "
        f"(μετοχές {CMA_RETURNS['equity']:.0%}, ομόλογα {CMA_RETURNS['fixed_income']:.0%}, "
        f"λοιπά {CMA_RETURNS['other']:.0%}), όχι ο μέσος του πρόσφατου ιστορικού· "
        "μεταβλητότητα και συσχετίσεις από τις ιστορικές μηνιαίες αποδόσεις· μηνιαία "
        "επανεξισορρόπηση· χωρίς κόστη και φόρους."
    )


def _render_twin_timeline(portfolio):
    """Month-by-month replay of the twin with the data available at each date."""
    from src.twin_timeline import timeline, first_alerts

    st.subheader("3. Χρονολόγιο του Διδύμου")
    st.caption(
        "Το δίδυμο «ξαναζεί» κάθε τέλος μήνα με μόνο όσα δεδομένα υπήρχαν τότε: πώς "
        "εξελίχθηκε ο κίνδυνος και πότε θα είχε σημάνει συναγερμός. Παραδοχή: οι τρέχουσες "
        "ποσότητες διακρατήθηκαν σε όλη την περίοδο (δεν υπάρχει ιστορικό συναλλαγών)."
    )
    key = (portfolio.name, portfolio.risk_profile, round(portfolio.total_value, 2))
    cached = st.session_state.get("twin_timeline")
    if cached is None or cached[0] != key:
        if not st.button("Υπολογισμός χρονολογίου"):
            return
        with st.spinner("Αναπαραγωγή του διδύμου μήνα προς μήνα…"):
            st.session_state["twin_timeline"] = (key, timeline(portfolio))
    tl = st.session_state["twin_timeline"][1]
    if tl.empty:
        st.info("Ανεπαρκές ιστορικό για χρονολόγιο (απαιτείται τουλάχιστον 1 έτος).")
        return

    chart = tl.set_index("date")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Resilience Index και drawdown**")
        st.line_chart(chart[["resilience", "drawdown_pct"]].rename(columns={
            "resilience": "Resilience Index", "drawdown_pct": "Drawdown από κορυφή (%)"}))
    with c2:
        st.markdown("**Ενεργά alerts και cautions**")
        st.bar_chart(chart[["n_alert", "n_caution"]].rename(columns={
            "n_alert": "Alerts", "n_caution": "Cautions"}))

    fa = first_alerts(tl)
    if not fa.empty:
        st.markdown("**Πρώτη εμφάνιση κάθε συναγερμού**")
        st.dataframe(fa.assign(
            date=fa["date"].dt.date,
            drawdown_pct=fa["drawdown_pct"].round(1),
            next_drawdown_pct=fa["next_drawdown_pct"].round(1),
        ).rename(columns={"date": "Ημερομηνία", "alert": "Συναγερμός",
                          "drawdown_pct": "Drawdown τότε (%)",
                          "next_drawdown_pct": "Drawdown που ακολούθησε, 6 μήνες (%)"}),
            hide_index=True)
        st.caption(
            "«Drawdown που ακολούθησε»: η χειρότερη πτώση στους επόμενους ~6 μήνες από την "
            "ημερομηνία του συναγερμού. Είναι πληροφορία εκ των υστέρων, μόνο για να κρίνετε "
            "αν ο συναγερμός άξιζε· το δίδυμο δεν τη χρησιμοποιεί."
        )
        if tl["next_drawdown_pct"].notna().any():
            with_alert = tl[tl["n_alert"] > 0]["next_drawdown_pct"].mean()
            without = tl[tl["n_alert"] == 0]["next_drawdown_pct"].mean()
            if np.isfinite(with_alert) and np.isfinite(without):
                st.caption(f"Μέσο drawdown 6 μηνών μετά από μήνες με ενεργό alert: {with_alert:.1f}% · "
                           f"χωρίς alert: {without:.1f}%.")
    worst = tl.loc[tl["drawdown_pct"].idxmin()]
    st.caption(
        f"Χειρότερο σημείο: {worst['date'].date()} με drawdown {worst['drawdown_pct']:.1f}% "
        f"και Resilience Index {worst['resilience']:.0f}."
    )


def _render_twin_state(portfolio, goal: dict):
    """Save the twin's state and compare with a previously saved one."""
    import json
    from src.twin_timeline import snapshot, compare_snapshots

    st.subheader("4. Αποθήκευση και σύγκριση με προηγούμενη κατάσταση")
    st.caption(
        "Αποθηκεύστε την κατάσταση του διδύμου (προφίλ, ερωτηματολόγιο, στόχος, θέσεις, "
        "βασικοί δείκτες). Την επόμενη φορά φορτώστε την για να δείτε τι άλλαξε από τότε."
    )
    current = snapshot(portfolio, st.session_state.get("questionnaire_result"), goal)
    st.download_button(
        "💾 Αποθήκευση κατάστασης (JSON)",
        data=json.dumps(current, ensure_ascii=False, indent=1),
        file_name=f"digital_twin_{datetime.date.today().isoformat()}.json",
        mime="application/json",
    )

    uploaded = st.file_uploader("Φόρτωση προηγούμενης κατάστασης", type=["json"],
                                key="twin_state_upload")
    if uploaded is None:
        return
    try:
        old = json.loads(uploaded.read().decode("utf-8"))
    except Exception as exc:
        st.error(f"Μη έγκυρο αρχείο κατάστασης: {exc}")
        return
    st.markdown(f"**Από την αποθήκευση της {old.get('saved_at', '—')} μέχρι σήμερα**")
    if old.get("risk_profile") != current["risk_profile"]:
        st.info(f"Το προφίλ άλλαξε: «{RISK_PROFILE_LABELS.get(old.get('risk_profile'), old.get('risk_profile'))}» "
                f"→ «{RISK_PROFILE_LABELS.get(current['risk_profile'], current['risk_profile'])}».")
    metrics, holdings = compare_snapshots(old, current)
    st.dataframe(metrics, hide_index=True)
    if not holdings.empty:
        st.markdown("**Μεταβολές βαρών (≥ 0.5 μ.)**")
        st.dataframe(holdings, hide_index=True)
    if old.get("goal") and old["goal"] != goal:
        g = old["goal"]
        st.caption(f"Προηγούμενος στόχος: {format_currency(g.get('target'))} σε {g.get('years')} έτη "
                   f"με {format_currency(g.get('monthly'))}/μήνα.")


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
- **SciPy** (`scipy.optimize`, SLSQP) — Βελτιστοποίηση Markowitz (max Sharpe) υπό περιορισμούς προφίλ κινδύνου και ορίου ανά θέση (`src/optimizer.py`).
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

**🟢 Συντηρητικό (Εισόδημα)** — Defensive / Liquidity Plus profile
| Ticker | Περιγραφή | Κατηγορία |
|---|---|---|
| SPY | S&P 500 ETF (iShares) | Equity ETF |
| BND | Vanguard Total Bond Market ETF | Fixed Income |
| TLT | iShares 20+ Year Treasury Bond ETF | Fixed Income |
| VIG | Vanguard Dividend Appreciation ETF | Equity ETF |
| JNJ | Johnson & Johnson | Healthcare |
| PG | Procter & Gamble | Consumer Staples |
| KO | Coca-Cola | Consumer Staples |
| VYM | Vanguard High Dividend Yield ETF | Equity ETF |
| VXUS | Vanguard Total International Stock ETF | International Equity |
| GLD | SPDR Gold Shares | Commodity |

**🟡 Ισορροπημένο (Μέτριο)** — Flexible profile
| Ticker | Περιγραφή | Κατηγορία |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| GOOGL | Alphabet (Google) | Technology |
| JPM | JPMorgan Chase | Financials |
| JNJ | Johnson & Johnson | Healthcare |
| SPY | S&P 500 ETF | Equity ETF |
| BND | Vanguard Total Bond Market ETF | Fixed Income |
| VXUS | Vanguard Total International Stock ETF | International Equity |
| GLD | SPDR Gold Shares | Commodity |
| VNQ | Vanguard Real Estate ETF | Real Estate |

**🔴 Δυναμικό (Επιθετικό)** — Growth / Dynamic profile
| Ticker | Περιγραφή | Κατηγορία |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| GOOGL | Alphabet (Google) | Technology |
| TSLA | Tesla Inc. | Consumer Discretionary |
| NVDA | NVIDIA Corp. | Technology |
| META | Meta Platforms | Technology |
| AMZN | Amazon.com | Consumer Discretionary |
| SPY | S&P 500 ETF | Equity ETF |
| QQQ | Invesco NASDAQ-100 ETF | Equity ETF |
| BND | Vanguard Total Bond Market ETF | Fixed Income |

**Προφίλ κινδύνου επενδυτή:**
Επέλεξε ένα από τα 5 προφίλ — επηρεάζουν τους στόχους μεταβλητότητας, τα όρια alerts, τις προτάσεις rebalancing και τον υπολογισμό του Information Ratio:

| Προφίλ | Στόχος Volatility | Μετοχές | Ομόλογα | Cash |
|---|---|---|---|---|
| Liquidity Plus | 2% | 5% | 60% | 35% |
| Defensive | 4% | 25% | 55% | 20% |
| Flexible | 7% | 45% | 45% | 10% |
| Growth | 10% | 65% | 30% | 5% |
| Dynamic | 14% | 85% | 10% | 5% |

Το επιλεγμένο δείγμα χαρτοφυλακίου **αλλάζει αυτόματα** ανάλογα με το προφίλ.
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

**📅 Αποδόσεις ανά Περίοδο**

| Περίοδος | Περιγραφή |
|---|---|
| MTD | Month-to-Date — από αρχή τρέχοντος μήνα |
| YTD | Year-to-Date — από αρχή τρέχοντος έτους |
| 1 Έτος | Απόδοση τελευταίων 12 μηνών |
| 3 Έτη | Απόδοση τελευταίων 3 ετών |
| 5 Έτη | Απόδοση τελευταίων 5 ετών |

---

**📐 Αναλυτικοί Δείκτες**

Τρεις στήλες με όλες τις μετρικές:

**Απόδοση:**

| Μετρική | Τι μετράει |
|---|---|
| Volatility 1Y / 3Y | Ετήσια τυπική απόκλιση αποδόσεων (1 & 3 ετών) |
| Beta | Συσχέτιση κινήσεων με S&P 500 |
| Max Drawdown | Χειρότερη ιστορική πτώση από κορυφή |
| VaR 95% μηνιαίο | Ζημία που αναμένεται να ξεπεραστεί 1 φορά/χρόνο |
| Diversification Ratio | Πόσο ο συνδυασμός μειώνει τον κίνδυνο |

**Risk-Adjusted Returns:**

| Μετρική | Τύπος | Καλή τιμή |
|---|---|---|
| Sharpe Ratio | (Απόδοση − Rf) / Volatility | > 1.0 |
| Sortino Ratio | (Απόδοση − Rf) / Downside Deviation | > 1.0 |
| Treynor Ratio | (Απόδοση − Rf) / Beta | > 0.10 |
| Information Ratio | Active Return / Tracking Error vs benchmark | > 0.5 |

Benchmark για Information Ratio: MSCI ACWI + Bloomberg Aggregate, blended ανά profile weights.

---

**🔗 Correlation Heatmap**

Collapsible expander στο κάτω μέρος της καρτέλας. Δείχνει τον πίνακα συσχέτισης Pearson
μεταξύ όλων των τίτλων του χαρτοφυλακίου:
- 🟢 Τιμές κοντά στο +1: κινούνται μαζί (μειώνει τη διαφοροποίηση)
- 🔴 Τιμές κοντά στο −1: αντίθετη κίνηση (ενισχύει τη διαφοροποίηση)
- Αυτόματη προειδοποίηση για ζεύγη με συσχέτιση ≥ 0.80

---

**🎯 Εναρμόνιση με Προφίλ Κινδύνου**

Τρία κουτιά που δείχνουν πόσο απέχει η τρέχουσα κατανομή από τον στόχο του επενδυτικού
προφίλ (Liquidity Plus / Defensive / Flexible / Growth / Dynamic):
- **Μετοχές (Equity %)** — τρέχον vs στόχος
- **Ομόλογα (Fixed Income %)** — τρέχον vs στόχος
- **Προφίλ** — ποιο προφίλ είναι επιλεγμένο

Οι αποκλίσεις εμφανίζονται ως delta (+ ή −) με κόκκινο χρώμα αν είναι αρνητικές.

---

**🔄 Ανακατανομή**

Αν η κατανομή αποκλίνει πάνω από 5 μονάδες από τον στόχο του προφίλ, εμφανίζεται
παραπομπή στην καρτέλα **Προτάσεις Αναδιάρθρωσης**, όπου βρίσκεται το ενιαίο,
συντονισμένο σχέδιο ανακατανομής ανά τίτλο.
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
- Relative Performance vs Market — πόσο καλύτερα/χειρότερα από την αγορά (= alpha)
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
# Tab 6: What-If Ανάλυση Βαρών
# ---------------------------------------------------------------------------

def _compute_metrics_for_weights(
    historical_prices: "pd.DataFrame",
    weights: dict[str, float],
    risk_free_rate: float = 0.04,
) -> dict:
    """Compute portfolio metrics for an arbitrary weight dict using historical prices."""
    available = [t for t in weights if t in historical_prices.columns]
    if not available:
        return {}

    prices = historical_prices[available].dropna()
    if len(prices) < 20:
        return {}

    daily_ret = np.log(prices / prices.shift(1)).dropna()
    w = np.array([weights[t] for t in available], dtype=float)
    total_w = w.sum()
    if total_w == 0:
        return {}
    w = w / total_w

    port_ret = (daily_ret * w).sum(axis=1)

    ann_ret = float(port_ret.mean() * 252)
    vol = float(port_ret.std() * np.sqrt(252))
    sharpe = (ann_ret - risk_free_rate) / vol if vol > 0 else float("nan")

    cum = (1 + port_ret).cumprod()
    rolling_max = cum.cummax()
    max_dd = float(((cum - rolling_max) / rolling_max).min())

    var_95 = float(np.percentile(port_ret, 5) * np.sqrt(21))

    downside = port_ret[port_ret < 0]
    if len(downside) >= 5:
        dd_dev = float(downside.std() * np.sqrt(252))
        sortino = (ann_ret - risk_free_rate) / dd_dev if dd_dev > 0 else float("nan")
    else:
        sortino = float("nan")

    return {
        "ann_return_pct": ann_ret * 100,
        "volatility_pct": vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "var_95_monthly_pct": var_95 * 100,
    }


def render_whatif(portfolio):
    st.header("What-If Ανάλυση Βαρών")
    st.markdown(
        "Αλλάξτε τα βάρη των assets με τα sliders και δείτε αμέσως πώς επηρεάζονται "
        "τα metrics του χαρτοφυλακίου σε σύγκριση με το τρέχον."
    )

    assets = portfolio.assets
    current_weights = portfolio.get_weights_dict()
    tickers = list(current_weights.keys())
    n = len(tickers)

    if n == 0:
        st.warning("Δεν βρέθηκαν assets.")
        return

    st.markdown("### Τρέχοντα Βάρη → Προτεινόμενα Βάρη")
    st.caption(
        "Ορίστε το νέο βάρος (%) για κάθε asset. "
        "Το άθροισμα θα κανονικοποιηθεί αυτόματα στο 100% κατά τον υπολογισμό."
    )

    # Render sliders — one per row of 2 columns
    new_weights_raw = {}
    col_pairs = [st.columns(2) for _ in range((n + 1) // 2)]
    for i, ticker in enumerate(tickers):
        col = col_pairs[i // 2][i % 2]
        asset = next((a for a in assets if a.ticker == ticker), None)
        label = asset.name if (asset and asset.name and asset.name != ticker) else ticker
        current_pct = round(current_weights[ticker] * 100, 1)
        new_pct = col.slider(
            f"{label} ({ticker})",
            min_value=0.0,
            max_value=100.0,
            value=current_pct,
            step=0.5,
            key=f"whatif_slider_{ticker}",
        )
        new_weights_raw[ticker] = new_pct

    # Show sum & normalised weights
    total_raw = sum(new_weights_raw.values())
    if total_raw == 0:
        st.error("Το άθροισμα των βαρών είναι 0. Ορίστε τουλάχιστον ένα θετικό βάρος.")
        return

    new_weights_norm = {t: v / total_raw for t, v in new_weights_raw.items()}

    # Summary bar
    remaining_col, _ = st.columns([1, 2])
    delta_color = "normal" if abs(total_raw - 100) < 0.5 else "off"
    remaining_col.metric(
        "Άθροισμα Βαρών",
        f"{total_raw:.1f}%",
        delta=f"{total_raw - 100:+.1f}% από 100%",
        delta_color=delta_color,
        help="Κανονικοποιείται αυτόματα στο 100% κατά τον υπολογισμό.",
    )

    st.markdown("---")

    # Compute metrics
    hist = portfolio.historical_prices
    current_m = _compute_metrics_for_weights(hist, current_weights)
    proposed_m = _compute_metrics_for_weights(hist, new_weights_norm)

    if not current_m or not proposed_m:
        st.warning("Ανεπαρκή ιστορικά δεδομένα για υπολογισμό metrics.")
        return

    # Side-by-side comparison
    st.subheader("Σύγκριση Τρέχοντος vs Προτεινόμενου Χαρτοφυλακίου")

    metric_defs = [
        ("Αναμ. Ετήσια Απόδοση (%)", "ann_return_pct", True),
        ("Ετήσια Μεταβλητότητα (%)", "volatility_pct", False),
        ("Sharpe Ratio", "sharpe", True),
        ("Sortino Ratio", "sortino", True),
        ("Max Drawdown (%)", "max_drawdown_pct", True),
        ("VaR 95% Μηνιαίο (%)", "var_95_monthly_pct", True),
    ]

    col_labels = st.columns([2, 1, 1, 1])
    col_labels[0].markdown("**Δείκτης**")
    col_labels[1].markdown("**Τρέχον**")
    col_labels[2].markdown("**Προτεινόμενο**")
    col_labels[3].markdown("**Μεταβολή**")

    for label, key, higher_is_better in metric_defs:
        curr_val = current_m.get(key)
        prop_val = proposed_m.get(key)
        cols = st.columns([2, 1, 1, 1])
        cols[0].write(label)
        cols[1].write(f"{curr_val:.2f}" if curr_val is not None else "N/A")
        if prop_val is not None and curr_val is not None:
            diff = prop_val - curr_val
            improved = (diff > 0) == higher_is_better
            sign = "+" if diff >= 0 else ""
            diff_str = f"{sign}{diff:.2f}"
            cols[2].write(f"{prop_val:.2f}")
            cols[3].markdown(
                f"<span style='color:{'green' if improved else 'red'}'>{diff_str}</span>",
                unsafe_allow_html=True,
            )
        else:
            cols[2].write("N/A")
            cols[3].write("—")

    st.markdown("---")

    # Weight comparison table
    st.subheader("Κατανομή Βαρών")
    weight_rows = []
    for ticker in tickers:
        asset = next((a for a in assets if a.ticker == ticker), None)
        name = asset.name if (asset and asset.name and asset.name != ticker) else ticker
        weight_rows.append({
            "Ticker": ticker,
            "Όνομα": name,
            "Τρέχον Βάρος (%)": f"{current_weights[ticker] * 100:.1f}%",
            "Προτεινόμενο Βάρος (%)": f"{new_weights_norm[ticker] * 100:.1f}%",
            "Διαφορά (%)": f"{(new_weights_norm[ticker] - current_weights[ticker]) * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(weight_rows))

    # Cumulative return chart
    st.subheader("Ιστορική Εξέλιξη (Cumulative Return)")
    available_curr = [t for t in tickers if t in hist.columns]
    if available_curr:
        prices = hist[available_curr].dropna()
        daily_ret = np.log(prices / prices.shift(1)).dropna()

        # Current weights cumulative return
        w_curr = np.array([current_weights.get(t, 0) for t in available_curr], dtype=float)
        w_curr = w_curr / w_curr.sum() if w_curr.sum() > 0 else w_curr
        port_curr = (daily_ret * w_curr).sum(axis=1)
        cum_curr = (1 + port_curr).cumprod() - 1

        # Proposed weights cumulative return
        w_prop = np.array([new_weights_norm.get(t, 0) for t in available_curr], dtype=float)
        w_prop = w_prop / w_prop.sum() if w_prop.sum() > 0 else w_prop
        port_prop = (daily_ret * w_prop).sum(axis=1)
        cum_prop = (1 + port_prop).cumprod() - 1

        chart_df = pd.DataFrame({
            "Τρέχον Χαρτοφυλάκιο": cum_curr * 100,
            "Προτεινόμενο Χαρτοφυλάκιο": cum_prop * 100,
        })
        st.line_chart(chart_df)
        st.caption("Σωρευτική απόδοση (%) βασισμένη σε ιστορικά δεδομένα με τα νέα βάρη.")


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
    import matplotlib.pyplot as plt
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


def _plot_auroc_bar(results: ValidationResults):
    import matplotlib.pyplot as plt
    fd = results.metrics.get("fragile_detection", {})

    labels = ["Baseline", "Digital Twin"]
    vals = [
        fd.get("baseline_auroc") or 0,
        fd.get("proposed_auroc") or 0,
    ]
    colors = ["#6c757d", "#0d6efd"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", width=0.4)

    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.5)")
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(
        f"AUROC — Εντοπισμός Ευάλωτων Χαρτοφυλακίων\n"
        f"(threshold: drawdown < {fd.get('threshold_pct', -15):.0f}%)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_risk_rank_comparison(results: ValidationResults):
    import matplotlib.pyplot as plt
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
    dd_rank = to_rank(drawdowns, ascending=True)  # more negative = riskier → rank 1

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


VALIDATION_DIR = Path(__file__).resolve().parent / "results" / "validation"
HOLDOUT_DIR = Path(__file__).resolve().parent / "results" / "validation_holdout"

_SCORER_LABELS = {
    "baseline": "Baseline (στατικές μετρικές, σταθερά βάρη)",
    "proposed": "Digital Twin scorer (alerts + σενάρια + drift, σταθερά βάρη)",
    "alerts_scenarios": "DT χωρίς drift (alerts + σενάρια)",
    "resilience": "Resilience Index",
    "hist_vol": "Ιστορική μεταβλητότητα (μόνη)",
    "hist_cvar95": "Ιστορικό CVaR 95% (μόνο)",
    "alerts_only": "Μόνο alerts",
    "scenarios_only": "Μόνο σενάρια",
    "drift_only": "Μόνο απόκλιση προφίλ",
    "equity_share": "Ποσοστό μετοχών",
    "hhi": "HHI",
    "hist_maxdd": "Ιστορικό max drawdown",
    "ri_stress": "RI: stress σενάρια",
    "ri_sequential": "RI: διαδοχικά shocks",
    "ri_downside": "RI: CVaR",
    "ri_recovery": "RI: ανάκαμψη",
    "ri_diversification": "RI: διαφοροποίηση",
    "ri_concentration": "RI: συγκέντρωση",
    "learned_ridge_dt": "Μαθημένο Ridge — συνιστώσες DT",
    "learned_gbm_dt": "Μαθημένο GBM — συνιστώσες DT",
    "learned_ridge_dt+static": "Μαθημένο Ridge — DT + στατικές",
    "learned_gbm_dt+static": "Μαθημένο GBM — DT + στατικές",
    "learned_ridge_pos_dt": "Μαθημένο Ridge (μη αρνητικά βάρη) — συνιστώσες DT",
    "learned_ridge_pos_dt+static": "Μαθημένο Ridge (μη αρνητικά βάρη) — DT + στατικές",
}
_RQ3_LABELS = {
    "hold": "Καμία ενέργεια",
    "profile_only": "Μόνο εναρμόνιση προφίλ",
    "dt": "Όλες οι προτάσεις DT",
    "markowitz": "Βελτιστοποίηση Markowitz",
    "hold_bond_mix": "Τρέχον + ομόλογα (ίδιος κίνδυνος με DT)",
}


def _read_validation_csv(name: str, base: Path | None = None) -> pd.DataFrame | None:
    path = (base or VALIDATION_DIR) / name
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


def _significance(ci: str) -> str:
    """'[lo, hi]' → whether the 95% CI of a difference excludes zero."""
    try:
        lo, hi = (float(x) for x in str(ci).strip("[]").split(","))
    except ValueError:
        return ""
    if lo > 0:
        return "σημαντικά καλύτερο"
    if hi < 0:
        return "σημαντικά χειρότερο"
    return "μη σημαντική διαφορά"


def _read_original_bootstrap() -> dict | None:
    import json
    try:
        return json.loads((VALIDATION_DIR / "original_bootstrap.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def _render_ranking_table(summary: pd.DataFrame):
    """Scorers vs the static baseline and vs historical volatility alone."""
    main = ["baseline", "resilience", "proposed", "alerts_scenarios",
            "hist_vol", "hist_cvar95"]
    has_vol = "spearman_ci_vs_vol" in summary.columns
    has_uw = "spearman_underwater" in summary.columns
    rows = []
    for s in main + [s for s in summary["scorer"] if s not in main]:
        r = summary[summary["scorer"] == s]
        if r.empty:
            continue
        r = r.iloc[0]
        rec = {
            "Σύστημα": _SCORER_LABELS.get(s, s),
            "Spearman ρ": round(r["spearman"], 3),
            "Δ vs Baseline (95% CI)": "—" if s == "baseline"
                else f"{r['spearman_diff']:+.3f} {r['spearman_ci']}",
            "Αποτέλεσμα vs Baseline": "" if s == "baseline" else _significance(r["spearman_ci"]),
        }
        if has_vol:
            vol_ci = r.get("spearman_ci_vs_vol")
            rec["Δ vs μεταβλητότητα (95% CI)"] = ("—" if s == "hist_vol" or pd.isna(vol_ci)
                                                  else f"{r['spearman_diff_vs_vol']:+.3f} {vol_ci}")
            rec["Αποτέλεσμα vs μεταβλητότητα"] = ("" if s == "hist_vol" or pd.isna(vol_ci)
                                                  else _significance(vol_ci))
        rec["AUROC (DD<−15%)"] = round(r["auroc_fixed"], 3)
        rec["Δ AUROC vs Baseline (95% CI)"] = ("" if s == "baseline"
                                               else f"{r['auroc_fixed_diff']:+.3f} {r['auroc_fixed_ci']}")
        if has_uw:
            rec["ρ με χρόνο κάτω από κορυφή"] = round(r["spearman_underwater"], 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    st.dataframe(df.head(len(main)), hide_index=True)
    with st.expander("Ablation: κάθε συνιστώσα ξεχωριστά"):
        st.dataframe(df.iloc[len(main):], hide_index=True)
    st.caption(
        "Spearman ρ: συμφωνία της κατάταξης κινδύνου με την πραγματική απώλεια του επόμενου "
        "έτους (μέσος όρος παραθύρων). AUROC: διάκριση των χαρτοφυλακίων με drawdown < −15%. "
        "«Χρόνος κάτω από κορυφή»: δεύτερο ground truth για την ανάκαμψη, το ποσοστό των "
        "ημερών του επόμενου έτους με απώλεια > 5% από την κορυφή."
    )


def render_robust_validation():
    """Pre-computed rolling-window evaluation (scripts/run_robust_validation.py)."""
    summary = _read_validation_csv("summary.csv")
    if summary is None:
        st.info("Δεν βρέθηκαν αποτελέσματα εκτεταμένης αξιολόγησης "
                "(εκτελέστε `python -m scripts.run_robust_validation`).")
        return
    import json
    try:
        windows = json.loads((VALIDATION_DIR / "windows.json").read_text(encoding="utf-8"))
    except Exception:
        windows = []

    st.markdown("### Εκτεταμένη αξιολόγηση: κυλιόμενα παράθυρα")
    n_win = len(windows) or "12"
    span = (f"{windows[0]['analysis_start']} → {windows[-1]['eval_end']}" if windows else "")
    st.markdown(
        f"**{n_win} κυλιόμενα παράθυρα** ({span}): 3 έτη ανάλυσης → 1 έτος αξιολόγησης, "
        "με **150 τυχαία χαρτοφυλάκια** σε κάθε παράθυρο. Κάθε σύστημα βαθμολογεί μόνο με "
        "δεδομένα του παραθύρου ανάλυσης· το ground truth είναι το πραγματικό max drawdown "
        "του επόμενου έτους. Οι διαφορές από το Baseline έχουν 95% διάστημα εμπιστοσύνης "
        "από paired cluster bootstrap (1.000 επαναδειγματοληψίες)."
    )

    rt1, rt2, rt3, rt4 = st.tabs(["Κατάταξη κινδύνου", "ΕΕ3: Βελτιώνουν οι προτάσεις;",
                                  "Scorer που μαθαίνει (AI/XAI)",
                                  "Έλεγχος σε νέους τίτλους (holdout)"])

    with rt1:
        _render_ranking_table(summary)

    with rt2:
        rq3 = _read_validation_csv("rq3_rebalancing.csv")
        if rq3 is None:
            st.info("Δεν βρέθηκαν αποτελέσματα ΕΕ3.")
        else:
            st.markdown(
                "Στο τέλος κάθε παραθύρου ανάλυσης εφαρμόζεται μία από πέντε ενέργειες και "
                "το χαρτοφυλάκιο διακρατείται για το επόμενο έτος. Η ενέργεια «τρέχον + ομόλογα» "
                "παίρνει **τον ίδιο εκτιμώμενο κίνδυνο** με το σχέδιο του DT απλώς αναμειγνύοντας "
                "ομόλογα: αν το DT δεν τη νικά, η αξία του είναι μόνο «λιγότερες μετοχές» και όχι "
                "οι συγκεκριμένες επιλογές του."
            )
            levels = rq3[rq3["comparison"].isin(_RQ3_LABELS)]
            st.dataframe(pd.DataFrame([{
                "Ενέργεια": _RQ3_LABELS[r["comparison"]],
                "Max drawdown (%)": round(r["maxdd"], 1),
                "Μεταβλητότητα (%)": round(r["vol"], 1),
                "CVaR 95% ημερ. (%)": round(r["cvar"], 2),
                "Ετήσια απόδοση (%)": round(r["ret"], 1),
                "Sharpe": round(r["sharpe"], 2),
            } for _, r in levels.iterrows()]), hide_index=True)

            diffs = rq3[~rq3["comparison"].isin(_RQ3_LABELS)]
            st.markdown("**Διαφορές (95% CI, παράθυρα όπου η διαφορά είναι θετική)**")
            st.dataframe(pd.DataFrame([{
                "Σύγκριση": " − ".join(_RQ3_LABELS.get(x.strip(), x.strip())
                                       for x in r["comparison"].split("−")),
                "Max drawdown (μ.)": f"{r['maxdd']:+.2f} {r['maxdd_ci']} ({r['maxdd_windows_up']})",
                "Απόδοση (μ.)": f"{r['ret']:+.2f} {r['ret_ci']} ({r['ret_windows_up']})",
                "Sharpe": f"{r['sharpe']:+.3f} {r['sharpe_ci']} ({r['sharpe_windows_up']})",
            } for _, r in diffs.iterrows()]), hide_index=True)
            st.caption(
                "Θετική διαφορά στο max drawdown σημαίνει μικρότερη απώλεια (καλύτερο). "
                "Αγορές και πωλήσεις χωρίς κόστη συναλλαγών."
            )

    with rt3:
        learned = _read_validation_csv("learned_summary.csv")
        imp = _read_validation_csv("learned_importance.csv")
        if learned is None:
            st.info("Δεν βρέθηκαν αποτελέσματα (εκτελέστε `python -m scripts.run_learned_scorer`).")
        else:
            n_test = int(learned["windows"].iloc[0])
            st.markdown(
                "Τα βάρη μαθαίνονται από τα δεδομένα αντί να ορίζονται a priori. Για κάθε "
                "παράθυρο k, το μοντέλο εκπαιδεύεται **μόνο στα παράθυρα πριν από το k** "
                f"(walk-forward) και αξιολογείται στο k ({n_test} παράθυρα εκτός δείγματος)."
            )
            st.dataframe(pd.DataFrame([{
                "Σύστημα": _SCORER_LABELS.get(r["scorer"], r["scorer"]),
                "Spearman ρ": round(r["spearman"], 3),
                "Δ vs Baseline (95% CI)": "—" if r["scorer"] == "baseline"
                    else f"{r['spearman_diff']:+.3f} {r['spearman_ci']}",
                "Αποτέλεσμα": "" if r["scorer"] == "baseline" else _significance(r["spearman_ci"]),
                "AUROC": round(r["auroc_fixed"], 3),
            } for _, r in learned.iterrows()]), hide_index=True)
            if imp is not None:
                st.markdown("**Επεξηγησιμότητα: τι μετράει στην πρόβλεψη** "
                            "(permutation importance, Ridge, DT + στατικές)")
                sub = imp[imp["feature_set"] == "dt+static"].sort_values("importance")
                st.bar_chart(sub.set_index("label")["importance"], horizontal=True)
                st.caption("Πτώση του μέσου Spearman ρ όταν το χαρακτηριστικό ανακατεύεται "
                           "μέσα σε κάθε παράθυρο: όσο μεγαλύτερη, τόσο πιο χρήσιμο.")

    with rt4:
        _render_holdout_tab()


def _render_holdout_verdict(holdout: pd.DataFrame, main: pd.DataFrame | None):
    """State, from the numbers, whether the Resilience Index generalises."""
    def row(df, scorer):
        if df is None:
            return None
        r = df[df["scorer"] == scorer]
        return None if r.empty else r.iloc[0]

    ri_h, ri_m = row(holdout, "resilience"), row(main, "resilience")
    if ri_h is None or "spearman_ci_vs_vol" not in ri_h:
        return
    vs_base = _significance(ri_h["spearman_ci"])
    vs_vol = _significance(ri_h["spearman_ci_vs_vol"])
    generalises = vs_base == "σημαντικά καλύτερο" and vs_vol != "σημαντικά χειρότερο"

    main_txt = (f" (στο κύριο σύνολο: {ri_m['spearman']:.3f}, {ri_m['spearman_diff']:+.3f} έναντι "
                f"Baseline)" if ri_m is not None else "")
    lines = [
        f"Resilience Index: Spearman **{ri_h['spearman']:.3f}**{main_txt}.",
        f"Έναντι Baseline: {ri_h['spearman_diff']:+.3f} {ri_h['spearman_ci']}, **{vs_base}**.",
        f"Έναντι ιστορικής μεταβλητότητας: {ri_h['spearman_diff_vs_vol']:+.3f} "
        f"{ri_h['spearman_ci_vs_vol']}, **{vs_vol}**.",
    ]
    if generalises:
        st.success("**Συμπέρασμα: ο Resilience Index γενικεύεται σε νέους τίτλους.**\n\n"
                   + "\n\n".join(lines))
    else:
        dn = row(holdout, "ri_downside")
        cvar = ""
        if dn is not None and _significance(dn["spearman_ci"]) == "σημαντικά καλύτερο":
            vs = _significance(dn.get("spearman_ci_vs_vol", ""))
            rel = {"σημαντικά καλύτερο": "καλύτερα από τη μεταβλητότητα",
                   "σημαντικά χειρότερο": "αλλά χειρότερα από τη μεταβλητότητα"}.get(
                vs, "όσο και η ιστορική μεταβλητότητα, όχι καλύτερα")
            cvar = (f" Η συνιστώσα CVaR μόνη της γενικεύει έναντι του Baseline "
                    f"({dn['spearman_diff']:+.3f} {dn['spearman_ci']}), {rel} "
                    f"({dn['spearman_diff_vs_vol']:+.3f} {dn['spearman_ci_vs_vol']}). Αυτό που "
                    "δεν γενικεύει είναι ο συνδυασμός των υπόλοιπων συνιστωσών.")
        st.warning(
            "**Συμπέρασμα: ο Resilience Index δεν γενικεύεται ως προβλέπτης κινδύνου.** Το "
            "πλεονέκτημά του στο κύριο σύνολο δεν επαναλαμβάνεται σε τίτλους που δεν "
            "χρησιμοποιήθηκαν στον σχεδιασμό, πιθανότατα επειδή τα βάρη του ορίστηκαν με "
            "γνώση των αποτελεσμάτων του κύριου συνόλου." + cvar + " Παραμένει χρήσιμος ως "
            "επεξηγηματικός δείκτης (δείχνει πού είναι αδύναμο ένα χαρτοφυλάκιο), όχι ως "
            "καλύτερος προβλέπτης.\n\n" + "\n\n".join(lines)
        )


def _render_holdout_tab():
    summary = _read_validation_csv("summary.csv", HOLDOUT_DIR)
    if summary is None:
        st.info("Δεν βρέθηκαν αποτελέσματα holdout "
                "(`python -m scripts.run_robust_validation --universe holdout`).")
        return
    from src.robust_validation import HOLDOUT_META

    examples = [t for t in ("CSCO", "MRK", "CAT", "NEE", "IEF", "LQD", "IYR", "SLV")
                if t in HOLDOUT_META]
    st.markdown(
        f"Το ίδιο πείραμα σε **{len(HOLDOUT_META)} τίτλους που δεν χρησιμοποιήθηκαν ποτέ** κατά "
        f"τον σχεδιασμό του συστήματος (π.χ. {', '.join(examples)}). Τα βάρη του Resilience "
        "Index ορίστηκαν αφού είχε φανεί ότι το CVaR τα πάει καλά στο κύριο σύνολο τίτλων· εδώ "
        "ελέγχεται αν το αποτέλεσμα γενικεύεται."
    )
    _render_holdout_verdict(summary, _read_validation_csv("summary.csv"))
    _render_ranking_table(summary)
    learned = _read_validation_csv("learned_summary.csv", HOLDOUT_DIR)
    if learned is not None:
        st.markdown("**Scorers που μαθαίνουν (walk-forward) στους νέους τίτλους**")
        st.dataframe(pd.DataFrame([{
            "Σύστημα": _SCORER_LABELS.get(r["scorer"], r["scorer"]),
            "Spearman ρ": round(r["spearman"], 3),
            "Δ vs Baseline (95% CI)": "—" if r["scorer"] == "baseline"
                else f"{r['spearman_diff']:+.3f} {r['spearman_ci']}",
            "Αποτέλεσμα": "" if r["scorer"] == "baseline" else _significance(r["spearman_ci"]),
        } for _, r in learned.iterrows()]), hide_index=True)
    rq3 = _read_validation_csv("rq3_rebalancing.csv", HOLDOUT_DIR)
    if rq3 is not None:
        d = rq3[rq3["comparison"] == "dt − hold_bond_mix"]
        if not d.empty:
            r = d.iloc[0]
            st.caption(
                f"ΕΕ3 στους νέους τίτλους, DT − (τρέχον + ομόλογα, ίδιος κίνδυνος): "
                f"max drawdown {r['maxdd']:+.2f} μ. {r['maxdd_ci']}, "
                f"Sharpe {r['sharpe']:+.3f} {r['sharpe_ci']}."
            )


def render_validation():
    """Tab 5: Πειραματική Αξιολόγηση — Σύγκριση Baseline vs Digital Twin."""
    st.subheader("🔬 Πειραματική Αξιολόγηση Συστήματος")
    render_robust_validation()
    st.markdown("---")
    st.markdown("### Αρχικό πείραμα: 30 χαρτοφυλάκια, ένα παράθυρο (διαδραστικό)")
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
        st.dataframe(arch_df, height=350)

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
            def _fmt_pval(p):
                if p is None:
                    return "—"
                return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            st.caption(
                f"Έλεγχος ρ ≠ 0 — Baseline: {_fmt_pval(p_b)} | Proposed: {_fmt_pval(p_p)}. "
                "Δείχνει μόνο ότι κάθε σύστημα σχετίζεται με τον κίνδυνο, όχι ότι το ένα "
                "είναι καλύτερο από το άλλο· αυτό το δείχνει το διάστημα εμπιστοσύνης της διαφοράς."
            )

            st.markdown("#### MAE Κατάταξης Κινδύνου (κανονικοποιημένο, 0–1)")
            _val_metric_card(
                "MAE κανονικοποιημένο (χαμηλότερο = καλύτερο)",
                mae.get("baseline_mae_pct"), mae.get("proposed_mae_pct"),
                higher_is_better=False, fmt=".3f",
            )

        with col_r:
            st.markdown(
                f"#### Εντοπισμός Ευάλωτων Χαρτοφυλακίων\n"
                f"*(drawdown < {fd.get('threshold_pct', -15):.0f}%)*"
            )
            _val_metric_card(
                "AUROC (υψηλότερο = καλύτερο, 0.5 = τυχαίο)",
                fd.get("baseline_auroc"), fd.get("proposed_auroc"),
                higher_is_better=True, fmt=".3f",
            )
            n_frag = fd.get("n_fragile", results.metrics.get("n_fragile", "—"))
            n_tot  = results.metrics.get("n_valid", 30)
            st.caption(
                f"Ευάλωτα χαρτοφυλάκια: {n_frag}/{n_tot} "
                f"(drawdown < {fd.get('threshold_pct', -15):.0f}%). "
                f"AUROC > 0.5 σημαίνει καλύτερο από τυχαίο· 1.0 = τέλειος διαχωρισμός."
            )

            imp_auroc = fd.get("auroc_improvement")
            imp_sp    = sp.get("improvement")
            imp_mae   = mae.get("improvement_pct")

            def _fmt_imp(v, suffix="", decimals=3):
                if v is None:
                    return "—"
                sign = "+" if v > 0 else ""
                return f"{sign}{v:.{decimals}f}{suffix}"

            st.markdown(
                f"**Διαφορά Digital Twin − Baseline:** AUROC {_fmt_imp(imp_auroc)} | "
                f"Spearman ρ {_fmt_imp(imp_sp)} | MAE (norm) {_fmt_imp(imp_mae, '', 3)}"
            )
            boot = _read_original_bootstrap()
            if boot:
                st.caption(
                    "Paired bootstrap 95% CI της διαφοράς (2.000 επαναδειγματοληψίες): "
                    f"Spearman {boot['spearman']['diff']:+.3f} {boot['spearman']['ci95']}, "
                    f"AUROC {boot['auroc_fixed']['diff']:+.3f} {boot['auroc_fixed']['ci95']}. "
                    "Με 30 χαρτοφυλάκια σε ένα παράθυρο το διάστημα περιλαμβάνει το μηδέν: "
                    "η διαφορά δεν είναι στατιστικά σημαντική. Δείτε την εκτεταμένη αξιολόγηση παραπάνω."
                )

        # ── Visualizations ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Οπτικοποίηση Αποτελεσμάτων")

        import matplotlib.pyplot as plt
        vtab1, vtab2, vtab3 = st.tabs([
            "🎯 Risk Score vs Actual Drawdown",
            "📊 AUROC — Εντοπισμός Ευάλωτων",
            "🏆 Σύγκριση Κατάταξης",
        ])

        with vtab1:
            try:
                fig = _plot_scatter_comparison(results)
                _show_fig(fig)
                plt.close(fig)
                st.caption(
                    "Κάθε σημείο είναι ένα χαρτοφυλάκιο. Χρωματισμός κατά κατηγορία. "
                    "Διακεκομμένη κόκκινη γραμμή = threshold −15% (fragile). "
                    "Ρ > 0 σημαίνει θετική συσχέτιση μεταξύ προβλεπόμενου κινδύνου και πραγματικής απώλειας."
                )
            except Exception as e:
                st.warning(f"Σφάλμα γραφήματος: {e}")

        with vtab2:
            if fd.get("baseline_auroc") is not None:
                try:
                    fig2 = _plot_auroc_bar(results)
                    _show_fig(fig2)
                    plt.close(fig2)
                    st.caption(
                        f"AUROC (Area Under ROC Curve): μετράει πόσο καλά κάθε σύστημα "
                        f"διαχωρίζει τα ευάλωτα από τα μη ευάλωτα χαρτοφυλάκια "
                        f"(threshold: drawdown < {abs(fd.get('threshold_pct', 15)):.0f}%) "
                        f"ανεξάρτητα από κατώφλι αποφάσεων."
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
                    _show_fig(fig3)
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

    st.title("Digital Twin Investment Advisory System")
    st.caption(
        "Προσομοίωση σεναρίων χαρτοφυλακίου, παρακολούθηση κινδύνου "
        "και επεξηγήσιμες προτάσεις αναδιάρθρωσης."
    )

    if st.session_state.get("show_docs"):
        render_docs()
        return

    portfolio = st.session_state.get("portfolio")

    # Tab 5 (Πειραματική Αξιολόγηση) is always available, independent of portfolio
    tab1, tab2, tab3, tab4, tab_inv, tab5 = st.tabs([
        "📈 Επισκόπηση Χαρτοφυλακίου",
        "🔀 Ανάλυση Σεναρίων",
        "⚠️ Παρακολούθηση Κινδύνου",
        "💡 Προτάσεις Αναδιάρθρωσης",
        "👤 Επενδυτής & Στόχοι",
        "🔬 Πειραματική Αξιολόγηση",
    ])

    with tab_inv:
        try:
            render_investor(portfolio)
        except Exception as e:
            st.error(f"Σφάλμα καρτέλας επενδυτή: {e}")
            st.exception(e)

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
        "📊 Digital Twin Investment Advisory System | "
        "Δεδομένα αγοράς: Yahoo Finance (yfinance) | "
        "⚠️ Για εκπαιδευτικούς σκοπούς μόνο — δεν αποτελεί επενδυτική συμβουλή."
    )


if __name__ == "__main__":
    main()
