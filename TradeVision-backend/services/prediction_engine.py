"""
Stock prediction engine — XGBoost inference + sentiment blend + price sizing.

The trained artifact is an XGBRegressor: it returns the expected percentage return
for the next day. The price figures in the API response are derived directly
from this expected return:

    expected_return = model.predict(feature_row)[0]
    expected_return += SENTIMENT_INFLUENCE * sentiment_score
    predicted   = latest_close * (1 + expected_return)

The model has no sentiment input (its feature set is fixed), so the
blend happens AFTER inference.
"""

import os

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import xgboost as xgb

from services import price_data
from services.features import (
    add_indicators,
    build_feature_row,
    latest_technicals,
    realized_volatility,
)
from services.ticker_registry import Ticker

DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "srilanka_stock_classifier.json")
)

# How strongly FinBERT sentiment ([-1, 1]) nudges the model's predicted return.
# 0.005 * score: fully bullish news moves expected return by +0.5%.
SENTIMENT_INFLUENCE = float(os.getenv("SENTIMENT_INFLUENCE", "0.005"))

TREND_UP_THRESHOLD = float(os.getenv("TREND_UP_THRESHOLD", "0.002"))
TREND_DOWN_THRESHOLD = float(os.getenv("TREND_DOWN_THRESHOLD", "-0.002"))


class ModelNotLoadedError(RuntimeError):
    """The model artifact is missing or could not be loaded."""


def _trend_label(p_adjusted: float) -> str:
    if p_adjusted > TREND_UP_THRESHOLD:
        return "Upward"
    if p_adjusted < TREND_DOWN_THRESHOLD:
        return "Downward"
    return "Neutral"


class StockPredictionEngine:
    def __init__(self, model_path: str | None = None):
        # Resolve relative to the repo root (default path is ../models/...)
        self.model_path = os.path.abspath(model_path or DEFAULT_MODEL_PATH)
        self._model = None
        self._load_error: str | None = None
        self._load_model()

    # ──────────────────────────────────────────────
    #  Loading
    # ──────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Best-effort load. A missing or corrupt artifact must not take the whole
        API down: the engine records the problem and every later call returns a
        "model not loaded" result instead of raising.
        """
        if not os.path.exists(self.model_path):
            self._load_error = f"Model file not found: {self.model_path}"
            return

        try:
            # XGBRegressor is the sklearn wrapper the artifact was saved from.
            model = xgb.XGBRegressor()
            model.load_model(self.model_path)
            self._model = model
        except Exception as e:
            self._load_error = f"Failed to load model from {self.model_path}: {e}"
            self._model = None
            return

        self._load_error = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def reload(self, model_path: str | None = None) -> None:
        """Re-point and reload, so a freshly-added artifact needs no container restart."""
        if model_path is not None:
            self.model_path = os.path.abspath(model_path)
        self._model = None
        self._load_error = None
        self._load_model()

    # ──────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────

    def predict_price(self, ticker: Ticker, sentiment_score: float = 0.0) -> dict:
        """
        Full prediction for one ticker. Returns the prediction block plus the
        technical summary the route needs, in a single call.
        """
        if not self.is_loaded:
            return {
                "price_prediction": None,
                "model_status": "model_not_loaded",
                "model_error": self.load_error,
                "technical_summary": None,
            }

        # 1. Prices (full history for converging indicators) -> features for the
        #    LAST row. No company identity is passed: the model is cross-sectional.
        df = price_data.get_prediction_history(ticker)
        feature_row = build_feature_row(df)
        indicators = add_indicators(df)

        # 2. Model: Predicted percentage return for tomorrow.
        pred_return = float(self._model.predict(feature_row)[0])

        # 3. Sentiment blend (post-hoc — the model has no sentiment feature).
        expected_return = pred_return + SENTIMENT_INFLUENCE * sentiment_score

        # 4. Volatility context (optional, mainly for deriving a mock 'confidence' score)
        sigma = realized_volatility(df)
        if sigma > 0:
            # Map return to a 0-100 bullishness score: 
            # expected_return of +sigma -> ~100 confidence
            mock_bullishness = min(max((expected_return / sigma * 50.0) + 50.0, 0.0), 100.0)
        else:
            mock_bullishness = 50.0

        latest = price_data.latest_close(df)
        predicted_close = float(latest * (1.0 + expected_return))
        change_percent = expected_return * 100.0
        trend = _trend_label(expected_return)

        return {
            "price_prediction": {
                "predicted_close": round(predicted_close, 4),
                "change_percent": round(change_percent, 4),
                "trend": trend,
                "probability_up": None,
                "probability_up_adjusted": None,
                "confidence": round(mock_bullishness, 2),
                "model_status": "loaded",
            },
            "model_status": "loaded",
            "model_error": None,
            "technical_summary": latest_technicals(indicators),
        }

    def feature_row(self, ticker: Ticker) -> dict:
        """The exact feature vector the model will see, for debugging/parity checks."""
        df = price_data.get_prediction_history(ticker)
        row = build_feature_row(df)
        return row.to_dict(orient="records")[0]
