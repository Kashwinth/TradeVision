"""
Deep AI analysis powered by DeepSeek.

This module takes the ALREADY-COMPUTED TradeVision data (XGBoost prediction,
FinBERT sentiment, live CSE quote, technicals) and feeds it into the DeepSeek
model with a carefully engineered prompt. The model is instructed to:

  1. Synthesise all provided data into a structured, multi-layered analysis.
  2. Return a detailed JSON response.

WHY THIS IS A SEPARATE MODULE
The Gemini chat (services/gemini_chat.py) powers the conversational AI Chat
page. This module powers a one-shot, deep-dive analysis triggered by the
"AI Prediction" button on the Stock Analyzer page.

TIMEOUT
The HTTP client timeout is set to 60 seconds.
"""

import json
import os
import threading

# pyrefly: ignore [missing-import]
import requests

from services.cache import TTLCache

# ---------- Configuration (from .env) ----------

_API_URL = os.getenv("AI_ANALYSIS_API_URL", "https://api.deepseek.com/v1/chat/completions")
_API_KEY = os.getenv("AI_ANALYSIS_API_KEY", "")
_MODEL   = os.getenv("AI_ANALYSIS_MODEL", "deepseek-chat")

# DeepSeek usually responds within 30-60 seconds.
_TIMEOUT = int(os.getenv("AI_ANALYSIS_TIMEOUT", "60"))

# Cache the analysis for 30 minutes — the same stock re-clicked should not
# re-run a long model call.
_CACHE_TTL = float(os.getenv("AI_ANALYSIS_CACHE_TTL", "1800"))
_cache = TTLCache(_CACHE_TTL)

_session_lock = threading.Lock()
_session: "requests.Session | None" = None


class AiAnalysisError(RuntimeError):
    """The AI analysis could not be completed."""


class AiAnalysisUnavailable(RuntimeError):
    """No API key or URL configured."""


def is_available() -> bool:
    return bool(_API_URL.strip()) and bool(_API_KEY.strip())


def _get_session() -> "requests.Session":
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_API_KEY}",
            })
    return _session


# ---------- Prompt Engineering ----------

_SYSTEM_PROMPT = """\
You are TradeVision's Deep Analysis Engine — a senior financial analyst AI \
specialising in the Colombo Stock Exchange (CSE) in Sri Lanka.

You will receive structured data about a stock including:
- Live CSE price and market data
- An XGBoost machine learning model's next-day direction prediction
- FinBERT news sentiment analysis score
- Technical indicators (RSI, MACD, etc.)

YOUR TASK — perform a multi-layered analysis:

1. **SYNTHESISE** all the provided data into a structured analysis.
2. Provide your expert interpretation on what these metrics mean collectively.
3. **RETURN VALID JSON** matching this exact schema:
```json
{
  "overall_verdict": "Bullish" | "Bearish" | "Neutral",
  "verdict_confidence": "High" | "Medium" | "Low",
  "summary": "1-2 sentence executive summary",
  "key_insights": [
    "Insight 1 (e.g. Price action & XGBoost)",
    "Insight 2 (e.g. Sentiment & news)",
    "Insight 3 (e.g. Technicals)"
  ],
  "recent_news": [
    {
      "headline": "Article title",
      "source": "Publication name",
      "url": "https://...",
      "date": "YYYY-MM-DD or approximate",
      "relevance": "1 short sentence on why this matters"
    }
  ],
  "risk_factors": ["risk 1", "risk 2"],
  "recommendation_note": "Not financial advice. This is model output for informational purposes only."
}
```

RULES:
- Every number you cite MUST come from the data provided. Do NOT hallucinate prices.
- You will be provided with a list of recently scraped news articles. Use these to populate the `recent_news` array. Do NOT invent or hallucinate URLs or articles that were not provided to you. If no articles are provided, leave the array empty `[]`.
- Prices are in Sri Lankan Rupees (LKR). Write as "Rs X.XX".
- Keep `key_insights` to exactly 3-5 short, punchy bullet points.
- Keep the `summary` strictly to 1-2 sentences.
- The overall_verdict must be supported by the evidence you present.
- RETURN ONLY THE JSON OBJECT. No markdown fences, no preamble, no commentary.

"""


def _build_user_prompt(
    symbol: str,
    company_name: str,
    prediction: dict,
    sentiment: dict,
    live_quote: dict,
    technicals: dict | None,
) -> str:
    """Inject all TradeVision data into the user message."""

    price_pred = prediction.get("price_prediction") or {}
    
    # Format the scraped articles for the prompt
    articles = sentiment.get("articles", [])
    news_lines = []
    if articles:
        for i, a in enumerate(articles, 1):
            age = f"{a.get('age_days')} days ago" if a.get('age_days') is not None else "Unknown date"
            news_lines.append(f"{i}. Headline: {a.get('headline')}\n   URL: {a.get('article_url')}\n   Source: {a.get('source')} ({age})\n")
        news_section = "\n".join(news_lines)
    else:
        news_section = "No recent news articles were found."

    return f"""\
Analyse this CSE stock for me:

## Company
- Symbol: {symbol}
- Name: {company_name}

## Live CSE Market Data
- Current Price: Rs {live_quote.get('price', 'N/A')}
- Change: {live_quote.get('change', 'N/A')} ({live_quote.get('change_percent', 'N/A')}%)
- Volume: {live_quote.get('volume', 'N/A')}
- Day High: {live_quote.get('day_high', 'N/A')}
- Day Low: {live_quote.get('day_low', 'N/A')}
- Previous Close: {live_quote.get('previous_close', 'N/A')}
- Market Cap: {live_quote.get('market_cap', 'N/A')}

## XGBoost Price Prediction (next trading day)
- Predicted Close: Rs {price_pred.get('predicted_close', 'N/A')}
- Predicted Change: {price_pred.get('change_percent', 'N/A')}%
- Trend: {price_pred.get('trend', 'N/A')}
- Model Confidence: {price_pred.get('confidence', 'N/A')}%
- Raw P(up): {price_pred.get('probability_up', 'N/A')}
- Adjusted P(up) after sentiment: {price_pred.get('probability_up_adjusted', 'N/A')}

## FinBERT News Sentiment
- Score: {sentiment.get('score', 0)} (range: -1 bearish to +1 bullish)
- Label: {sentiment.get('label', 'Neutral')}
- Headlines Analysed: {sentiment.get('headline_count', 0)}

## Technical Indicators
{json.dumps(technicals, indent=2) if technicals else 'Not available'}

## Recent News Articles (Scraped from the Web)
{news_section}

## As-of Date
- Prediction computed from data as of: {prediction.get('as_of', 'N/A')}

Based on this data, provide your full structured JSON analysis.
"""


# ---------- API Call ----------

def analyze(
    symbol: str,
    company_name: str,
    prediction: dict,
    sentiment: dict,
    live_quote: dict,
    technicals: dict | None,
) -> dict:
    """
    Run the full DeepSeek deep analysis. Blocking — takes 30-60 seconds.
    Returns the parsed JSON analysis dict.
    """
    if not is_available():
        raise AiAnalysisUnavailable(
            "AI deep analysis is not configured: set AI_ANALYSIS_API_URL and "
            "AI_ANALYSIS_API_KEY in the backend .env file."
        )

    # Check cache first
    cache_key = f"ai_analysis:{symbol}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    user_prompt = _build_user_prompt(
        symbol, company_name, prediction, sentiment, live_quote, technicals
    )

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    try:
        resp = _get_session().post(
            _API_URL,
            json=payload,
            timeout=_TIMEOUT,
        )
    except requests.Timeout:
        raise AiAnalysisError(
            "The AI analysis request timed out. The agent may be performing "
            "a deep web search — please try again."
        )
    except requests.ConnectionError as e:
        raise AiAnalysisError(f"Cannot reach the AI analysis API: {e}")
    except requests.RequestException as e:
        raise AiAnalysisError(f"AI analysis request failed: {e}")

    if resp.status_code != 200:
        detail = resp.text[:300] if resp.text else f"HTTP {resp.status_code}"
        raise AiAnalysisError(f"AI analysis API returned an error: {detail}")

    try:
        body = resp.json()
    except ValueError:
        raise AiAnalysisError("AI analysis API returned non-JSON response.")

    # Extract the assistant's message content
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AiAnalysisError(
            "Unexpected response format from the AI analysis API."
        )

    # Parse the JSON from the model's response
    # The model might wrap it in markdown fences, so strip those
    text = content.strip()
    if text.startswith("```"):
        # Remove ```json ... ``` wrapping
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text in a wrapper
        result = {
            "overall_verdict": "Neutral",
            "verdict_confidence": "Low",
            "summary": text[:500],
            "key_insights": ["Analysis could not be fully parsed."],
            "recent_news": [],
            "risk_factors": [],
            "recommendation_note": "Not financial advice.",
            "_raw": text,
        }

    # Cache successful results
    _cache.set(cache_key, result)
    return result
