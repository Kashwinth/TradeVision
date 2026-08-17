"""
Pydantic response models for the analysis endpoint.

These exist for the generated OpenAPI docs as much as for validation: with them,
http://localhost:8000/docs renders the exact payload shape, which makes the
endpoint testable in the browser as well as in Postman.
"""

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


class ArticleBlock(BaseModel):
    headline: str
    article_url: str
    source: str
    age_days: int | None = None
    sentiment_score: float | None = None
    scraped_at: str


class SentimentBlock(BaseModel):
    score: float = Field(..., description="FinBERT sentiment, -1.0 (bearish) to 1.0 (bullish).")
    label: str = Field(..., description="Bullish, Bearish or Neutral.")
    headline_count: int = Field(..., description="Articles that were successfully scored.")
    status: str = Field(
        ...,
        description=(
            "'ok', or why the score fell back to neutral "
            "('no_articles_found', 'skipped', 'error: ...')."
        ),
    )
    articles: list[ArticleBlock] = Field(
        default_factory=list, description="Raw articles scraped for this ticker."
    )


class PricePredictionBlock(BaseModel):
    predicted_close: float
    change_percent: float
    trend: str = Field(..., description="Upward, Downward or Neutral.")

    # The underlying model is now a Regressor, so direction probabilities are null.
    # predicted_close is derived directly from the model's expected return.
    probability_up: float | None = Field(None, description="Raw P(up) from XGBoost, before sentiment.")
    probability_up_adjusted: float | None = Field(None, description="P(up) after the sentiment blend.")
    confidence: float = Field(..., description="Bullishness score derived from expected return vs volatility.")
    model_status: str


class TechnicalSummary(BaseModel):
    rsi_3: float | None = None
    sma_3: float | None = None
    ema_3: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None


class StockAnalysisResponse(BaseModel):
    symbol: str
    company_name: str
    as_of: str | None = Field(None, description="Date of the latest price bar (YYYY-MM-DD).")
    latest_price: float | None = None
    sentiment_analysis: SentimentBlock
    # Null when the model artifact is absent — the rest of the payload still
    # populates so the endpoint stays testable before the model is dropped in.
    price_prediction: PricePredictionBlock | None = None
    technical_summary: TechnicalSummary | None = None
    model_status: str = Field(..., description="'loaded' or 'model_not_loaded'.")
    warnings: list[str] = Field(default_factory=list)


class SymbolInfo(BaseModel):
    symbol: str
    name: str
    # None for symbols outside the curated list. Historical metadata only — the
    # cross-sectional model does not consume it.
    asset_id: int | None = None
    yahoo_symbol: str


class SymbolListResponse(BaseModel):
    count: int
    symbols: list[SymbolInfo]
