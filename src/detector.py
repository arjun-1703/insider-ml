"""
detector.py
-----------
Core detection logic:
  1. Predicts fair value for a given stock
  2. Computes price inflation = actual - predicted
  3. Scores insider trading probability (0-100%)
  4. Generates detailed report
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore, percentileofscore


def compute_inflation(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """
    Compute how much the actual price deviates from the fair-value prediction.

    Returns DataFrame with:
      - actual_price
      - predicted_price
      - inflation_rs    : actual - predicted in rupees
      - inflation_pct   : inflation as % of predicted price
      - inflation_z     : z-score of inflation over the full period
    """
    df = pd.DataFrame({
        "actual_price":    actual,
        "predicted_price": predicted,
    }).dropna()

    df["inflation_rs"]  = df["actual_price"] - df["predicted_price"]
    df["inflation_pct"] = (df["inflation_rs"] / df["predicted_price"]) * 100
    df["inflation_z"]   = zscore(df["inflation_rs"])

    return df


def score_insider_probability(
    inflation_df: pd.DataFrame,
    volume_z: pd.Series,
) -> pd.DataFrame:
    """
    Compute a 0–100% insider trading probability score for each day.

    Score is driven by:
      - How extreme the price inflation is (z-score)
      - Whether volume spiked at the same time
      - Whether inflation is positive (price pumped up, not down)
      - Whether the spike is sudden (not a gradual drift)

    Returns DataFrame with 'insider_score' and 'alert_level' columns added.
    """
    df = inflation_df.copy()
    df["volume_z"] = volume_z.reindex(df.index).fillna(0)

    # --- Component 1: Price inflation magnitude (0-40 points) ---
    # How many sigmas above normal is the price?
    inf_z_abs = df["inflation_z"].abs()
    price_score = np.clip(inf_z_abs / 4.0, 0, 1) * 40   # max 40 pts at 4σ+

    # --- Component 2: Volume spike (0-30 points) ---
    vol_score = np.clip(df["volume_z"].abs() / 4.0, 0, 1) * 30  # max 30 pts at 4σ+

    # --- Component 3: Positive inflation direction (0-20 points) ---
    # Insider trading typically inflates price UP before an announcement
    direction_score = np.where(df["inflation_pct"] > 0, 20, 5)

    # --- Component 4: Suddenness — short-term spike vs rolling avg (0-10 points) ---
    rolling_inf = df["inflation_rs"].rolling(10).mean().fillna(0)
    suddenness  = (df["inflation_rs"] - rolling_inf).abs()
    sudden_z    = pd.Series(zscore(suddenness.fillna(0)), index=df.index)
    sudden_score = np.clip(sudden_z / 3.0, 0, 1) * 10

    # --- Total score ---
    df["insider_score"] = (
        price_score + vol_score + direction_score + sudden_score
    ).clip(0, 100).round(1)

    # Alert levels
    df["alert_level"] = pd.cut(
        df["insider_score"],
        bins     = [0, 25, 50, 75, 100],
        labels   = ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True,
    )

    return df


def generate_report(ticker: str, result_df: pd.DataFrame, fundamentals: dict) -> dict:
    """
    Summarise the analysis into a human-readable report dict.
    """
    # Focus on the most recent 252 trading days (1 year) for summary
    recent = result_df.tail(252)

    # Peak suspicious day
    peak_idx   = result_df["insider_score"].idxmax()
    peak_row   = result_df.loc[peak_idx]

    # Days flagged at each alert level
    alert_counts = result_df["alert_level"].value_counts().to_dict()

    # Average recent score
    avg_score  = recent["insider_score"].mean()
    max_score  = result_df["insider_score"].max()

    # Overall verdict
    if max_score >= 75:
        verdict = "SUSPICIOUS — significant price inflation detected with volume anomalies"
    elif max_score >= 50:
        verdict = "MODERATE — some unusual price-volume patterns detected"
    elif max_score >= 25:
        verdict = "LOW CONCERN — minor deviations from fair value"
    else:
        verdict = "NORMAL — price tracks fair value closely"

    report = {
        "ticker":           ticker,
        "verdict":          verdict,
        "max_insider_score":       round(float(max_score), 1),
        "avg_score_last_1yr":      round(float(avg_score), 1),
        "peak_date":               str(peak_idx.date()),
        "peak_actual_price":       round(float(peak_row["actual_price"]), 2),
        "peak_predicted_price":    round(float(peak_row["predicted_price"]), 2),
        "peak_inflation_rs":       round(float(peak_row["inflation_rs"]), 2),
        "peak_inflation_pct":      round(float(peak_row["inflation_pct"]), 2),
        "peak_volume_z":           round(float(peak_row["volume_z"]), 2),
        "alert_counts":            alert_counts,
        "pe_ratio":                fundamentals.get("pe_ratio"),
        "peg_ratio":               fundamentals.get("peg_ratio"),
        "eps":                     fundamentals.get("eps"),
        "earnings_growth":         fundamentals.get("earnings_growth"),
    }
    return report


def print_report(report: dict):
    """Pretty-print the analysis report to console."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  INSIDER TRADING ANALYSIS: {report['ticker']}")
    print(sep)
    print(f"  VERDICT      : {report['verdict']}")
    print(f"  MAX SCORE    : {report['max_insider_score']}% insider probability")
    print(f"  AVG SCORE    : {report['avg_score_last_1yr']}% (last 1 year)")
    print(f"\n  PEAK SUSPICIOUS DATE : {report['peak_date']}")
    print(f"  Actual Price         : ₹{report['peak_actual_price']}")
    print(f"  Predicted Fair Price : ₹{report['peak_predicted_price']}")
    print(f"  Price Inflation      : ₹{report['peak_inflation_rs']} ({report['peak_inflation_pct']}%)")
    print(f"  Volume Z-Score       : {report['peak_volume_z']}σ")
    print(f"\n  ALERT LEVEL BREAKDOWN:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = report["alert_counts"].get(level, 0)
        bar   = "█" * min(count // 5, 30)
        print(f"    {level:<10} {count:>5} days  {bar}")
    print(f"\n  FUNDAMENTALS:")
    print(f"    PE Ratio       : {report['pe_ratio']}")
    print(f"    PEG Ratio      : {report['peg_ratio']}")
    print(f"    EPS            : {report['eps']}")
    print(f"    Earnings Growth: {report['earnings_growth']}")
    print(sep)
