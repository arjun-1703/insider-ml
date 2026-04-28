"""
plotter.py
----------
Generates the analysis chart:
  - Actual price vs predicted fair-value price
  - Inflation % over time
  - Insider score heatmap on price chart
  - Volume spikes highlighted
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_analysis(ticker: str, result_df: pd.DataFrame, report: dict):
    """
    Master plot: 4 panels showing:
      1. Actual vs Predicted price with insider-score color overlay
      2. Price inflation % over time
      3. Insider trading score over time (0-100%)
      4. Volume z-score with anomaly markers
    """
    df = result_df.copy()
    df.index = pd.to_datetime(df.index)

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
    fig.suptitle(
        f"{ticker}  —  Insider Trading Analysis\n"
        f"Max Score: {report['max_insider_score']}%  |  "
        f"Verdict: {report['verdict']}",
        fontsize=13, fontweight="bold", y=0.98
    )

    # ── Panel 1: Actual vs Predicted price ────────────────────
    ax1 = axes[0]
    ax1.plot(df.index, df["predicted_price"], color="#1565C0",
             linewidth=1.5, linestyle="--", label="Predicted Fair Value", zorder=2)
    ax1.plot(df.index, df["actual_price"], color="#212121",
             linewidth=1.2, alpha=0.7, label="Actual Price", zorder=1)

    # Colour-fill the gap between actual and predicted by score
    ax1.fill_between(
        df.index,
        df["actual_price"],
        df["predicted_price"],
        where=(df["actual_price"] >= df["predicted_price"]),
        alpha=0.35, color="#E53935", label="Price above fair value (suspicious)"
    )
    ax1.fill_between(
        df.index,
        df["actual_price"],
        df["predicted_price"],
        where=(df["actual_price"] < df["predicted_price"]),
        alpha=0.20, color="#43A047", label="Price below fair value (undervalued)"
    )

    # Mark CRITICAL / HIGH alert days
    critical = df[df["alert_level"].isin(["CRITICAL", "HIGH"])]
    if not critical.empty:
        ax1.scatter(critical.index, critical["actual_price"],
                    color="red", zorder=5, s=50,
                    label=f"HIGH/CRITICAL alerts ({len(critical)} days)")

    ax1.set_ylabel("Price (₹)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Actual Price vs Predicted Fair Value", fontsize=10, loc="left")

    # ── Panel 2: Inflation % ───────────────────────────────────
    ax2 = axes[1]
    inf_pct = df["inflation_pct"]
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(df.index, inf_pct, 0,
                     where=(inf_pct >= 0), color="#E53935", alpha=0.6, label="Inflated above fair")
    ax2.fill_between(df.index, inf_pct, 0,
                     where=(inf_pct < 0),  color="#43A047", alpha=0.4, label="Below fair value")
    ax2.set_ylabel("Inflation %", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_title("Price Inflation vs Fair Value (%)", fontsize=10, loc="left")

    # ── Panel 3: Insider score 0-100% ─────────────────────────
    ax3 = axes[2]
    score = df["insider_score"]

    # Colour the score line by risk level
    cmap   = cm.get_cmap("RdYlGn_r")
    norm   = Normalize(vmin=0, vmax=100)
    points = np.array([mdates.date2num(df.index), score.values]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc     = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(score.values)
    ax3.add_collection(lc)
    ax3.set_xlim(df.index.min(), df.index.max())
    ax3.set_ylim(0, 105)

    ax3.axhline(75, color="#E53935", linewidth=0.8, linestyle=":", alpha=0.8, label="CRITICAL (75%)")
    ax3.axhline(50, color="#FB8C00", linewidth=0.8, linestyle=":", alpha=0.8, label="HIGH (50%)")
    ax3.axhline(25, color="#FDD835", linewidth=0.8, linestyle=":", alpha=0.7, label="MEDIUM (25%)")

    ax3.set_ylabel("Insider Score %", fontsize=10)
    ax3.legend(loc="upper left", fontsize=7)
    ax3.set_title("Insider Trading Probability Score (0–100%)", fontsize=10, loc="left")

    # Colorbar for score
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, orientation="vertical", pad=0.01, fraction=0.02)
    cbar.set_label("Score %", fontsize=8)

    # ── Panel 4: Volume z-score ────────────────────────────────
    ax4 = axes[3]
    vol_z = df["volume_z"]
    colors = ["#E53935" if v > 1.5 else "#90CAF9" for v in vol_z]
    ax4.bar(df.index, vol_z, color=colors, width=1.5, alpha=0.8)
    ax4.axhline(1.5, color="#E53935", linewidth=0.8, linestyle="--", label="Spike threshold (1.5σ)")
    ax4.set_ylabel("Volume Z-Score", fontsize=10)
    ax4.set_xlabel("Date", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.set_title("Volume Anomaly (red bars = unusual volume)", fontsize=10, loc="left")

    # Format x-axis dates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    path = os.path.join(PLOTS_DIR, f"{ticker}_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")
    return path
