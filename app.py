"""
app.py
------
Streamlit web interface for the Insider Trading Signal Detector.

Run with:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from src.data_loader import fetch_market_data, fetch_fundamentals, fetch_index_data
from src.features    import build_features
from src.model       import load_model, predict_fair_value
from src.detector    import compute_inflation, score_insider_probability, generate_report


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Insider Trading Detector",
    page_icon  = "📈",
    layout     = "wide",
)


# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .score-critical { color: #ef5350; font-size: 2.5rem; font-weight: 800; }
    .score-high     { color: #ff7043; font-size: 2.5rem; font-weight: 800; }
    .score-medium   { color: #ffd600; font-size: 2.5rem; font-weight: 800; }
    .score-low      { color: #66bb6a; font-size: 2.5rem; font-weight: 800; }
    .verdict-box {
        background: #1e2130;
        border-left: 4px solid #ef5350;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        font-size: 1.05rem;
    }
    .stProgress > div > div { background: linear-gradient(90deg, #66bb6a, #ffd600, #ef5350); }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
st.title("📈 Insider Trading Signal Detector")
st.caption("Detects potential price inflation using fair-value estimation and anomaly scoring.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ticker_input = st.text_input(
        "NSE Ticker Symbol",
        placeholder="e.g. NUCLEUS.NS, ADANIENT.NS",
        help="Enter any NSE ticker with .NS suffix"
    ).strip().upper()

    if ticker_input and not ticker_input.endswith(".NS"):
        ticker_input += ".NS"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    with col2:
        end_date   = st.date_input("End Date",   value=pd.to_datetime("2024-12-31"))

    st.divider()
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")

    st.divider()
    st.caption("⚠️ For research purposes only. Does not constitute legal advice.")

# Hardcoded optimal thresholds
RESID_THRESH = 2.0
VOL_THRESH   = 1.5


# ── Model check ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None

bundle = get_model()

if bundle is None:
    st.error("""
    ⚠️ **No trained model found.**

    Please train the model first by running in your terminal:
    ```
    python main.py --train
    ```
    Then refresh this page.
    """)
    st.stop()

st.success(f"✅ Model loaded — Training R² = **{bundle['metrics']['r2']:.4f}**")


# ── Analysis ──────────────────────────────────────────────────
if analyze_btn and ticker_input:

    with st.spinner(f"Downloading data for **{ticker_input}** from yfinance..."):
        index_df = fetch_index_data(str(start_date), str(end_date))
        mkt      = fetch_market_data(ticker_input, str(start_date), str(end_date))
        fund     = fetch_fundamentals(ticker_input)

    if mkt.empty or len(mkt) < 30:
        st.error(f"❌ Could not fetch data for **{ticker_input}**. Check the ticker symbol and try again.")
        st.stop()

    with st.spinner("Building features and predicting fair value..."):
        feat_df   = build_features(mkt, index_df, fund)
        predicted = predict_fair_value(feat_df, bundle)
        result_df = compute_inflation(feat_df["close"], predicted)
        result_df = score_insider_probability(result_df, feat_df["volume_z"])
        report    = generate_report(ticker_input, result_df, fund)

    # ── Top metrics row ───────────────────────────────────────
    st.markdown(f"## 🏢 {ticker_input} — Analysis Results")
    st.markdown(f"*{len(result_df)} trading days analysed from {str(start_date)} to {str(end_date)}*")

    max_score = report["max_insider_score"]
    if max_score >= 75:
        score_class = "score-critical"
        verdict_color = "#ef5350"
    elif max_score >= 50:
        score_class = "score-high"
        verdict_color = "#ff7043"
    elif max_score >= 25:
        score_class = "score-medium"
        verdict_color = "#ffd600"
    else:
        score_class = "score-low"
        verdict_color = "#66bb6a"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🎯 Max Insider Score",
                  f"{report['max_insider_score']}%",
                  help="Highest single-day insider probability detected")

    with c2:
        st.metric("📅 Peak Suspicious Date",
                  report["peak_date"])

    with c3:
        st.metric("💹 Peak Price Inflation",
                  f"{report['peak_inflation_pct']}%",
                  delta=f"₹{report['peak_inflation_rs']} above fair value",
                  delta_color="inverse")

    with c4:
        st.metric("📊 Avg Score (1yr)",
                  f"{report['avg_score_last_1yr']}%")

    # ── Verdict banner ────────────────────────────────────────
    st.markdown(f"""
    <div class="verdict-box" style="border-color: {verdict_color}">
        <b>VERDICT:</b> {report['verdict']}
    </div>
    """, unsafe_allow_html=True)

    # Score progress bar
    st.markdown(f"**Overall Risk Level: {max_score}%**")
    st.progress(int(max_score) / 100)

    st.divider()

    # ── Alert breakdown ───────────────────────────────────────
    st.subheader("🚨 Alert Level Breakdown")
    alert_cols = st.columns(4)
    for i, (level, color) in enumerate([
        ("CRITICAL", "#ef5350"),
        ("HIGH",     "#ff7043"),
        ("MEDIUM",   "#ffd600"),
        ("LOW",      "#66bb6a"),
    ]):
        count = report["alert_counts"].get(level, 0)
        pct   = round(count / len(result_df) * 100, 1)
        with alert_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:{color}; font-size:1.8rem; font-weight:800">{count}</div>
                <div style="color:{color}; font-weight:600">{level}</div>
                <div style="color:#888; font-size:0.85rem">{pct}% of days</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Fundamentals ──────────────────────────────────────────
    st.subheader("📋 Fundamentals")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.metric("PE Ratio",        report["pe_ratio"]        or "N/A")
    with f2:
        st.metric("PEG Ratio",       report["peg_ratio"]       or "N/A")
    with f3:
        st.metric("EPS",             report["eps"]             or "N/A")
    with f4:
        st.metric("Earnings Growth", f"{round(report['earnings_growth']*100, 1)}%" if report["earnings_growth"] else "N/A")

    st.divider()

    # ── Charts ────────────────────────────────────────────────
    st.subheader("📈 Analysis Charts")

    df = result_df.copy()
    df.index = pd.to_datetime(df.index)

    # Panel 1: Actual vs Predicted
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Actual vs Fair Value",
        "💰 Price Inflation",
        "🎯 Insider Score",
        "📊 Volume Anomaly"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e2130")
        ax.plot(df.index, df["predicted_price"], color="#42a5f5",
                linewidth=1.5, linestyle="--", label="Predicted Fair Value")
        ax.plot(df.index, df["actual_price"], color="#ffffff",
                linewidth=1.2, alpha=0.9, label="Actual Price")
        ax.fill_between(df.index, df["actual_price"], df["predicted_price"],
                        where=(df["actual_price"] >= df["predicted_price"]),
                        alpha=0.4, color="#ef5350", label="Above fair value (suspicious)")
        ax.fill_between(df.index, df["actual_price"], df["predicted_price"],
                        where=(df["actual_price"] < df["predicted_price"]),
                        alpha=0.25, color="#66bb6a", label="Below fair value")
        critical = df[df["alert_level"].isin(["CRITICAL", "HIGH"])]
        if not critical.empty:
            ax.scatter(critical.index, critical["actual_price"],
                       color="#ef5350", zorder=5, s=40,
                       label=f"HIGH/CRITICAL alerts ({len(critical)})")
        ax.set_ylabel("Price (₹)", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30, ha="right", color="white")
        ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
        ax.spines[:].set_color("#2d3250")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e2130")
        inf = df["inflation_pct"]
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.fill_between(df.index, inf, 0, where=(inf >= 0),
                        color="#ef5350", alpha=0.7, label="Price inflated above fair value")
        ax.fill_between(df.index, inf, 0, where=(inf < 0),
                        color="#66bb6a", alpha=0.5, label="Price below fair value")
        ax.set_ylabel("Inflation %", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30, ha="right", color="white")
        ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
        ax.spines[:].set_color("#2d3250")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e2130")
        score = df["insider_score"]
        cmap  = cm.get_cmap("RdYlGn_r")
        norm  = Normalize(vmin=0, vmax=100)
        points = np.array([mdates.date2num(df.index), score.values]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        lc     = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.5)
        lc.set_array(score.values)
        ax.add_collection(lc)
        ax.set_xlim(df.index.min(), df.index.max())
        ax.set_ylim(0, 105)
        ax.axhline(75, color="#ef5350", linewidth=0.8, linestyle=":", label="CRITICAL (75%)")
        ax.axhline(50, color="#ff7043", linewidth=0.8, linestyle=":", label="HIGH (50%)")
        ax.axhline(25, color="#ffd600", linewidth=0.8, linestyle=":", label="MEDIUM (25%)")
        ax.set_ylabel("Insider Score %", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30, ha="right", color="white")
        ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
        ax.spines[:].set_color("#2d3250")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
        cbar.ax.tick_params(colors="white")
        cbar.set_label("Score %", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e2130")
        vol_z  = df["volume_z"]
        colors = ["#ef5350" if v > 1.5 else "#42a5f5" for v in vol_z]
        ax.bar(df.index, vol_z, color=colors, width=1.5, alpha=0.8)
        ax.axhline(1.5, color="#ef5350", linewidth=1, linestyle="--",
                   label="Spike threshold (1.5σ)")
        ax.set_ylabel("Volume Z-Score", color="white")
        ax.set_xlabel("Date", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30, ha="right", color="white")
        ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
        ax.spines[:].set_color("#2d3250")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Raw data table ────────────────────────────────────────
    st.subheader("📄 Detailed Signal Data")

    # Show top suspicious days
    top_days = result_df.sort_values("insider_score", ascending=False).head(20)
    top_days = top_days[[
        "actual_price", "predicted_price",
        "inflation_rs", "inflation_pct",
        "insider_score", "alert_level", "volume_z"
    ]].round(2)
    top_days.index = pd.to_datetime(top_days.index).strftime("%Y-%m-%d")
    top_days.columns = [
        "Actual Price (₹)", "Fair Value (₹)",
        "Inflation (₹)", "Inflation %",
        "Insider Score %", "Alert Level", "Volume Z"
    ]

    def color_alert(val):
        colors = {
            "CRITICAL": "background-color: #b71c1c; color: white",
            "HIGH":     "background-color: #e64a19; color: white",
            "MEDIUM":   "background-color: #f57f17; color: black",
            "LOW":      "background-color: #1b5e20; color: white",
        }
        return colors.get(str(val), "")

    st.dataframe(
        top_days.style.applymap(color_alert, subset=["Alert Level"]),
        use_container_width=True,
        height=400,
    )

    # Download button
    csv = result_df.to_csv()
    st.download_button(
        label     = "⬇️ Download Full Data as CSV",
        data      = csv,
        file_name = f"{ticker_input}_signals.csv",
        mime      = "text/csv",
    )

elif analyze_btn and not ticker_input:
    st.warning("⚠️ Please enter a ticker symbol in the sidebar.")

else:
    # Landing state
    st.markdown("""
    ### How to use
    1. Enter an NSE ticker in the sidebar (e.g. `ADANIENT.NS`, `NUCLEUS.NS`, `ZOMATO.NS`)
    2. Set your date range
    3. Click **Analyze**

    The app will:
    - Download live data from yfinance
    - Predict the **fair value** of the stock using the trained ML model
    - Compare actual price vs fair value
    - Score each day with an **insider trading probability (0–100%)**
    - Show charts of price inflation, anomaly scores, and volume spikes

    ---
    > ⚠️ Make sure you have run `python main.py --train` before using this app.
    """)

    st.info("👈 Enter a ticker in the sidebar and click **Analyze** to begin.")
