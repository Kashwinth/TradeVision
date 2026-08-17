"""
XGBoost training — CSE next-day return regressor (cross-sectional).

Train on every CSV in ml/data/ and save the artifact the API loads. All features
come from services/features.py — the SAME module the serving code uses — so a
model built here is guaranteed to receive identical inputs in production.

Cross-sectional: there is no Asset_ID in the feature set, so the model learns
"this pattern of returns/RSI/relative volume implies X return about tomorrow" without
caring which company produced it. That is what lets it predict ANY CSE ticker,
including ones never seen in training. The number of files in ml/data/ therefore
has no effect on which symbols the API can predict — it only changes the volume
of training data.

Run from the backend root (PYTHONPATH=/app in Docker):
    python -m ml.train_model

Expects CSVs in ml/data/ (see ml/fetch_data.py) with at least Date, Open, High,
Low, Close, Volume. Writes models/srilanka_stock_classifier.json.
"""

import glob
import os

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb
# pyrefly: ignore [missing-import]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from services.features import (
    FEATURE_COLUMNS,
    add_indicators,
    add_model_features,
    drop_non_trading_rows,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_HERE, "data"))
MODEL_OUT = os.getenv(
    "MODEL_OUT", os.path.join(_HERE, "..", "models", "srilanka_stock_classifier.json")
)
PLOT_OUT = os.getenv("PLOT_OUT", os.path.join(_HERE, "roc_curve.png"))

TRAIN_SPLIT = 0.8

# A single-day move larger than this is not a market move. CSE applies price
# bands, and prices here are fetched with auto_adjust=False, so a 5x or 10x jump
# is an unadjusted split or scrip issue showing up as a fake return. Sri Lankan
# banks issue scrip dividends most years, so this is expected, not exotic.
#
# The contamination spreads: a bad return at row i also poisons Return_Lag1..3 at
# rows i+1..i+3 and the label at row i-1, so the whole neighbourhood is dropped,
# not just the offending row.
MAX_ABS_DAILY_RETURN = 0.40


def _drop_split_artifacts(df: "pd.DataFrame", name: str) -> "pd.DataFrame":
    """Remove rows whose features or label were built from a split-sized jump."""
    contaminated = df["Current_Return"].abs() > MAX_ABS_DAILY_RETURN

    # Same test applied to each lag: a lag column IS an earlier Current_Return,
    # so this covers rows i+1..i+2 without index arithmetic.
    for lag in ("Return_Lag1", "Return_Lag2"):
        contaminated |= df[lag].abs() > MAX_ABS_DAILY_RETURN

    # Row i-1's label is the return at row i, so a bad return poisons it too.
    contaminated |= contaminated.shift(-1, fill_value=False)

    n_bad = int(contaminated.sum())
    if n_bad:
        print(f"    (dropped {n_bad} rows around {name}'s split/scrip jumps)")
    return df[~contaminated]


def load_and_prepare(path: str) -> "pd.DataFrame":
    """One CSV -> a frame carrying the model features and the direction target."""
    name = os.path.basename(path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Identical rule to services/price_data.py, from the same shared function:
    # a bar counts only if it traded. Yahoo forward-fills the CSE feed, and
    # those rows manufacture zero-return observations AND zero-return labels.
    before = len(df)
    df = drop_non_trading_rows(df).reset_index(drop=True)
    if len(df) < before:
        print(f"    (dropped {before - len(df)} non-trading rows)")

    # Recompute indicators with the shared code — the same path the API uses.
    df = add_indicators(df)
    df = add_model_features(df)

    # Target: the exact percentage return for the next day.
    df["Target_Return"] = df["Current_Return"].shift(-1)

    # The newest row has no next day; the shift above scored it NaN,
    # which is a fabricated label rather than a measured one.
    df = df.iloc[:-1]

    df = _drop_split_artifacts(df, name)

    return df.dropna(subset=FEATURE_COLUMNS + ["Target_Return"])


def main() -> None:
    print("Step 1: Building features with services/features.py (shared with the API)...")

    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_paths:
        raise SystemExit(
            f"No training CSVs found in {DATA_DIR}. Run `python -m ml.fetch_data` "
            f"first, or point DATA_DIR at a folder containing them."
        )
    print(f"Found {len(csv_paths)} CSVs\n")

    train_dfs, test_dfs, skipped = [], [], []

    for path in csv_paths:
        name = os.path.basename(path)
        try:
            df = load_and_prepare(path)
        except Exception as e:
            print(f"  {name:<28} FAILED  {e}")
            skipped.append(name)
            continue

        if len(df) < 50:
            print(f"  {name:<28} SKIPPED (only {len(df)} usable rows)")
            skipped.append(name)
            continue

        # Split each asset on its own timeline before concatenating, so no
        # stock's future leaks into another's training window.
        split_row = int(len(df) * TRAIN_SPLIT)
        train_dfs.append(df.iloc[:split_row])
        test_dfs.append(df.iloc[split_row:])
        print(f"  {name:<28} OK      {len(df):>6} rows")

    if not train_dfs:
        raise SystemExit(f"No usable training data in {DATA_DIR}.")

    master_train = pd.concat(train_dfs, ignore_index=True).sort_values("Date")
    master_test = pd.concat(test_dfs, ignore_index=True).sort_values("Date")

    X_train, y_train = master_train[FEATURE_COLUMNS], master_train["Target_Return"]
    X_test, y_test = master_test[FEATURE_COLUMNS], master_test["Target_Return"]

    print(
        f"\nStep 2: Training XGBoost Regressor on {len(X_train):,} rows from "
        f"{len(csv_paths) - len(skipped)} assets, {len(FEATURE_COLUMNS)} features..."
    )
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n=================== PIPELINE RESULTS ===================")
    print(f"Mean Absolute Error (MAE): {mae:.5f}")
    print(f"Mean Squared Error (MSE):  {mse:.5f}")
    print(f"R-Squared (R2):            {r2:.5f}")
    print("=========================================================")

    os.makedirs(os.path.dirname(os.path.abspath(MODEL_OUT)), exist_ok=True)
    model.save_model(MODEL_OUT)
    print(f"\n-> Model saved to {os.path.abspath(MODEL_OUT)}")

    saved_features = model.get_booster().feature_names
    print(f"-> Artifact feature names: {saved_features}")
    if saved_features != FEATURE_COLUMNS:
        print("!! MISMATCH: artifact features differ from services/features.py")


if __name__ == "__main__":
    main()
