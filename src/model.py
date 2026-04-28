"""
model.py
--------
Trains a fair-value price model on data from MULTIPLE companies combined.
The model learns what a "normal" price looks like given fundamentals + market data.
When you later run a new company through it, deviations = potential anomalies.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import get_feature_columns

MODEL_PATH  = os.path.join("outputs", "models", "fair_value_model.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    """Extract X, y from a featured DataFrame. Impute NaNs safely."""
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Fill NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


def train_model(all_data: list) -> dict:
    """
    Train fair-value model on a combined dataset from multiple companies.

    Parameters
    ----------
    all_data : list of DataFrames, one per training company

    Returns
    -------
    dict with model, scaler, feature_cols, metrics
    """
    print("\n  Building combined training dataset ...")

    # Collect feature columns common across all companies
    common_features = None
    clean_dfs = []

    for df in all_data:
        if df is None or len(df) < 50:
            continue
        cols = get_feature_columns(df)
        if common_features is None:
            common_features = set(cols)
        else:
            common_features = common_features.intersection(set(cols))
        clean_dfs.append(df)

    if not clean_dfs or not common_features:
        raise ValueError("Not enough training data collected.")

    feature_cols = sorted(list(common_features))
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Feature list : {feature_cols}")

    # Combine all company data
    combined_X = []
    combined_y = []

    for df in clean_dfs:
        X, y = prepare_xy(df, feature_cols)
        if len(X) > 20:
            combined_X.append(X)
            combined_y.append(y)

    X_all = pd.concat(combined_X, ignore_index=True)
    y_all = pd.concat(combined_y, ignore_index=True)

    print(f"  Total training samples: {len(X_all)} rows from {len(combined_X)} companies")

    # Chronological split within combined data (last 20% = test)
    n    = len(X_all)
    cut  = int(n * 0.80)
    X_tr, X_te = X_all.iloc[:cut], X_all.iloc[cut:]
    y_tr, y_te = y_all.iloc[:cut], y_all.iloc[cut:]

    # Scale
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # Model: Gradient Boosting (captures non-linear PE/PEG relationships well)
    print("  Training Gradient Boosting model ...")
    model = GradientBoostingRegressor(
        n_estimators  = 400,
        learning_rate = 0.05,
        max_depth     = 4,
        subsample     = 0.8,
        random_state  = 42,
    )
    model.fit(X_tr_s, y_tr)

    # Evaluate
    pred_te = model.predict(X_te_s)
    mae  = mean_absolute_error(y_te, pred_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred_te))
    r2   = r2_score(y_te, pred_te)

    print(f"\n  ── Model Training Results ──")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top 5 features:")
    for feat, imp in fi.head(5).items():
        print(f"    {feat:<25} {imp:.4f}")

    # Save model bundle
    bundle = {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "metrics":      {"mae": mae, "rmse": rmse, "r2": r2},
        "feature_importance": fi,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Model saved → {MODEL_PATH}")

    return bundle


def load_model() -> dict:
    """Load previously trained model bundle."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            "Run: python main.py --train   first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_fair_value(df: pd.DataFrame, bundle: dict) -> pd.Series:
    """
    Run fair-value predictions on a new company's featured DataFrame.
    Returns a Series of predicted prices aligned to df's index.
    """
    feature_cols = bundle["feature_cols"]
    scaler       = bundle["model"].__class__  # just for reference
    model        = bundle["model"]
    sc           = bundle["scaler"]

    # Keep only the features the model was trained on
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  NOTE: {len(missing)} features missing for this ticker, filling with 0: {missing}")

    X = df.reindex(columns=feature_cols, fill_value=0).copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    X_s   = sc.transform(X)
    preds = model.predict(X_s)
    return pd.Series(preds, index=df.index, name="predicted_price")
