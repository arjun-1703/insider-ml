"""
features.py
-----------
Builds all features for the fair-value model.
Key fundamentals: PE, PEG, EPS, growth rates, price-to-book, ROE.
Key market: returns, moving averages, volatility, beta vs index.
"""

import numpy as np
import pandas as pd
from scipy import stats


def build_features(market_df: pd.DataFrame,
                   index_df: pd.DataFrame,
                   fundamentals: dict) -> pd.DataFrame:
    """
    Master feature builder.
    Returns DataFrame with all features + target (next-day close).
    """
    df = market_df.copy()

    # ── 1. Price features ─────────────────────────────────────
    df["return"]       = df["close"].pct_change()
    df["log_return"]   = np.log(df["close"] / df["close"].shift(1))
    df["ma_5"]         = df["close"].rolling(5).mean()
    df["ma_10"]        = df["close"].rolling(10).mean()
    df["ma_20"]        = df["close"].rolling(20).mean()
    df["ma_50"]        = df["close"].rolling(50).mean()
    df["volatility_10"]= df["return"].rolling(10).std()
    df["volatility_20"]= df["return"].rolling(20).std()
    df["price_ma20_ratio"] = df["close"] / (df["ma_20"] + 1e-9)  # how far from 20-day avg

    # ── 2. Volume features ────────────────────────────────────
    df["vol_ma_10"]  = df["volume"].rolling(10).mean()
    roll_vol         = df["volume"].rolling(20)
    df["volume_z"]   = (df["volume"] - roll_vol.mean()) / (roll_vol.std() + 1e-9)

    # ── 3. Index / risk features ──────────────────────────────
    if not index_df.empty:
        df = df.join(index_df[["index_return"]], how="left")
        df["index_return"] = df["index_return"].ffill().fillna(0)
    else:
        df["index_return"] = 0.0

    # Rolling 30-day beta vs index
    df["beta_30"] = _rolling_beta(df["return"], df["index_return"], window=30)

    # ── 4. Fundamental features (broadcast as constants) ──────
    fund_fields = [
        "pe_ratio", "forward_pe", "peg_ratio",
        "eps", "eps_forward",
        "earnings_growth", "revenue_growth",
        "price_to_book", "roe", "debt_to_equity",
    ]
    for f in fund_fields:
        df[f] = fundamentals.get(f)   # None → NaN

    # ── 5. Derived features ───────────────────────────────────
    # Volatility compression: short vol / long vol
    df["vol_compression"] = df["volatility_10"] / (df["volatility_20"] + 1e-9)
    # Price momentum: 5-day return
    df["momentum_5"]  = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)

    # ── 6. Target: next-day close price ──────────────────────
    df["target"] = df["close"].shift(-1)
    df = df.iloc[:-1]          # drop last row (no target)
    df = df.dropna(subset=["target", "return", "ma_20"])

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns that are actually present and have data."""
    candidates = [
        "return", "log_return",
        "ma_5", "ma_10", "ma_20", "ma_50",
        "volatility_10", "volatility_20",
        "price_ma20_ratio",
        "vol_ma_10", "volume_z",
        "index_return", "beta_30",
        "pe_ratio", "forward_pe", "peg_ratio",
        "eps", "eps_forward",
        "earnings_growth", "revenue_growth",
        "price_to_book", "roe", "debt_to_equity",
        "vol_compression", "momentum_5", "momentum_10",
    ]
    # Only keep cols that exist and have at least some non-NaN values
    return [c for c in candidates if c in df.columns and df[c].notna().sum() > 10]


def _rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling OLS beta of stock vs market."""
    betas = [np.nan] * len(stock_ret)
    for i in range(window, len(stock_ret)):
        y = stock_ret.iloc[i - window: i].values
        x = mkt_ret.iloc[i - window: i].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 10:
            continue
        slope, *_ = stats.linregress(x[mask], y[mask])
        betas[i] = slope
    return pd.Series(betas, index=stock_ret.index)
