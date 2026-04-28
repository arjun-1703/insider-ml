"""
main.py
-------
Insider Trading Detector — Two modes:

  MODE 1: Train the fair-value model on multiple companies
    python main.py --train

  MODE 2: Analyse a specific company for insider trading signals
    python main.py --analyze TICKER

Examples:
    python main.py --train
    python main.py --analyze NUCLEUS.NS
    python main.py --analyze RELIANCE.NS
    python main.py --analyze ZOMATO.NS --start 2022-01-01
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import fetch_market_data, fetch_fundamentals, fetch_index_data, load_tickers
from src.features    import build_features, get_feature_columns
from src.model       import train_model, load_model, predict_fair_value
from src.detector    import compute_inflation, score_insider_probability, generate_report, print_report
from src.plotter     import plot_analysis


# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Insider Trading Signal Detector")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",   action="store_true",
                       help="Train the fair-value model on training_tickers.txt")
    group.add_argument("--analyze", metavar="TICKER",
                       help="Analyse a specific NSE ticker, e.g. NUCLEUS.NS")
    p.add_argument("--start",   default="2020-01-01", help="Start date for data (default: 2020-01-01)")
    p.add_argument("--end",     default="2024-12-31", help="End date for data   (default: 2024-12-31)")
    p.add_argument("--tickers", default="training_tickers.txt", help="Training tickers file")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
def run_training(args):
    """
    Step 1: Download data for all training companies and train the model.
    """
    print("=" * 60)
    print("  TRAINING FAIR-VALUE MODEL")
    print("=" * 60)
    print(f"  Period  : {args.start} → {args.end}")

    tickers  = load_tickers(args.tickers)
    print(f"  Tickers : {tickers}\n")

    # Download index data once
    index_df = fetch_index_data(args.start, args.end)

    # Download and feature-engineer each training company
    all_featured = []
    for ticker in tickers:
        print(f"\n  [{ticker}]")
        try:
            mkt  = fetch_market_data(ticker, args.start, args.end)
            fund = fetch_fundamentals(ticker)

            if mkt.empty or len(mkt) < 100:
                print(f"    SKIP: not enough data")
                continue

            feat_df = build_features(mkt, index_df, fund)
            if len(feat_df) > 50:
                all_featured.append(feat_df)
                print(f"    OK: {len(feat_df)} rows, {len(get_feature_columns(feat_df))} features")
            else:
                print(f"    SKIP: too few clean rows ({len(feat_df)})")

        except Exception as e:
            print(f"    ERROR: {e} — skipping")
            continue

    if not all_featured:
        print("\n  ERROR: No training data collected. Check internet connection.")
        return

    # Train model on combined data
    bundle = train_model(all_featured)
    print("\n  ✅  Training complete!")
    print("  Now run:  python main.py --analyze <TICKER>")


# ──────────────────────────────────────────────────────────────
def run_analysis(args):
    """
    Step 2: Analyse a new company using the trained model.
    """
    ticker = args.analyze.upper()
    if not ticker.endswith(".NS"):
        ticker = ticker + ".NS"

    print("=" * 60)
    print(f"  ANALYSING: {ticker}")
    print("=" * 60)

    # Load trained model
    try:
        bundle = load_model()
        print(f"  Model loaded. Features: {len(bundle['feature_cols'])}")
        print(f"  Training R² = {bundle['metrics']['r2']:.4f}")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return

    # Download data for target company
    print(f"\n  Downloading data for {ticker} ...")
    index_df = fetch_index_data(args.start, args.end)
    mkt      = fetch_market_data(ticker, args.start, args.end)
    fund     = fetch_fundamentals(ticker)

    if mkt.empty or len(mkt) < 30:
        print(f"  ERROR: Could not download sufficient data for {ticker}")
        print("  Check that the ticker is correct (e.g. ZOMATO.NS, NUCLEUS.NS)")
        return

    print(f"  Downloaded {len(mkt)} trading days of data")

    # Build features
    feat_df = build_features(mkt, index_df, fund)
    if len(feat_df) < 20:
        print(f"  ERROR: Not enough clean data after feature engineering ({len(feat_df)} rows)")
        return

    # Predict fair value
    print(f"\n  Predicting fair value ...")
    predicted = predict_fair_value(feat_df, bundle)

    # Compute inflation and score
    result_df = compute_inflation(feat_df["close"], predicted)
    result_df = score_insider_probability(result_df, feat_df["volume_z"])

    # Generate and print report
    report = generate_report(ticker, result_df, fund)
    print_report(report)

    # Save results CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", f"{ticker}_signals.csv")
    result_df.to_csv(csv_path)
    print(f"\n  Results saved → {csv_path}")

    # Plot
    print(f"  Generating analysis chart ...")
    plot_path = plot_analysis(ticker, result_df, report)

    print(f"\n  ✅  Analysis complete!")
    print(f"  Chart  → {plot_path}")
    print(f"  Data   → {csv_path}\n")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    if args.train:
        run_training(args)
    else:
        run_analysis(args)
