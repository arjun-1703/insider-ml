"""
data_loader.py
--------------
Downloads OHLCV market data and fundamental data from yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data. Returns cleaned DataFrame."""
    print(f"    Downloading market data: {ticker} ...")
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"    WARNING: Could not download {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def fetch_fundamentals(ticker: str) -> dict:
    """
    Pull fundamental metrics from yfinance.
    Returns a dict of scalar values — PE, PEG, EPS, growth rates etc.
    """
    print(f"    Downloading fundamentals: {ticker} ...")
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"    WARNING: Could not fetch fundamentals for {ticker}: {e}")
        return {}

    pe  = info.get("trailingPE")
    eg  = info.get("earningsGrowth")   # decimal e.g. 0.15 = 15%

    # Compute PEG = PE / (EPS growth %)
    peg = None
    if pe and eg and eg > 0:
        peg = pe / (eg * 100)

    return {
        "pe_ratio":        pe,
        "forward_pe":      info.get("forwardPE"),
        "peg_ratio":       peg if peg else info.get("pegRatio"),
        "eps":             info.get("trailingEps"),
        "eps_forward":     info.get("forwardEps"),
        "earnings_growth": eg,
        "revenue_growth":  info.get("revenueGrowth"),
        "price_to_book":   info.get("priceToBook"),
        "roe":             info.get("returnOnEquity"),
        "debt_to_equity":  info.get("debtToEquity"),
        "beta":            info.get("beta"),
    }


def fetch_index_data(start: str, end: str) -> pd.DataFrame:
    """Download NIFTY 50 or SENSEX as market index."""
    print("    Downloading NIFTY 50 index ...")
    tickers_to_try = ["^NSEI", "^BSESN", "NSEI.NS"]
    for t in tickers_to_try:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df.columns = [c.lower() for c in df.columns]
                df.index.name = "date"
                df = df[["close"]].rename(columns={"close": "index_close"}).dropna()
                df["index_return"] = df["index_close"].pct_change()
                print(f"    Index fetched using: {t}")
                return df
        except Exception:
            continue
    print("    WARNING: Could not fetch index data.")
    return pd.DataFrame()


def load_tickers(path: str) -> list:
    """Read tickers from file, skipping comments and blank lines."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]
