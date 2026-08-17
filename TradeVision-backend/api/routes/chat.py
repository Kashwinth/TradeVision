"""
AI chat endpoint.

    POST /api/v1/chat
    { "messages": [{"role": "user", "content": "..."}], "context": {"symbol": "JKH.N0000"} }

Gemini answers, but it is not allowed to answer from memory: every figure it
states comes from a tool call executed here during the request. This module owns
those tool implementations; services/gemini_chat.py owns the model transport and
the schemas the model sees.

WHY THE HANDLERS LIVE IN THE ROUTE LAYER
They reach api/dependencies.py for the shared XGBoost engine and FinBERT
analyzer, and api/routes/prediction.py for the analysis orchestration. Putting
them in services/ would make services/ import api/ — backwards. So the service
declares the tools and the route supplies the functions.

Everything here is blocking (requests, yfinance, FinBERT, the Gemini SDK), so the
entire chat turn goes through run_in_threadpool for the reasons documented at the
top of routes/prediction.py.
"""

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
# pyrefly: ignore [missing-import]
from starlette.concurrency import run_in_threadpool

from api.routes.prediction import analyze_sync
from api.schemas.chat import ChatRequest, ChatResponse
from services import cse_api, gemini_chat, deepseek_chat
from services.gemini_chat import GeminiChatError, GeminiUnavailableError
from services.deepseek_chat import DeepseekChatError, DeepseekUnavailableError
from services.ticker_registry import UnknownTickerError, resolve_open

router = APIRouter(prefix="/api/v1", tags=["chat"])

# Long conversations cost tokens on every turn and add nothing after a while. The
# client resends the whole history, so this is also the guard against a client
# posting an unbounded payload.
MAX_HISTORY = 20

# Movers lists are 10 rows each; trimming the per-row payload keeps a
# three-list answer from crowding out the model's own reasoning space.
_QUOTE_KEYS = ("symbol", "name", "price", "change", "change_percent", "volume", "turnover")


def _slim(quote: dict) -> dict:
    """A movers row with the fields a chat answer actually reads."""
    return {k: quote[k] for k in _QUOTE_KEYS if k in quote}


def _tool_get_stock_analysis(symbol: str, include_news: bool = True) -> dict:
    """
    The `/analyze` pipeline, unchanged.

    include_news defaults to True so that every chat analysis automatically
    fetches FinBERT sentiment, as requested by the user.
    """
    try:
        ticker = resolve_open(symbol)
    except UnknownTickerError as e:
        return {"error": str(e)}

    try:
        result = analyze_sync(ticker, include_news)
    except HTTPException as e:
        # Surface the message, not the status code: the model has to explain this
        # to a person, and "no price history for X" is the useful half.
        return {"error": str(e.detail)}

    payload = result.model_dump()

    # A live CSE price alongside a possibly-months-old prediction is the single
    # most useful thing this tool can add, and it is exactly what the warnings
    # array is trying to tell the user about.
    try:
        payload["live_cse_quote"] = _slim(cse_api.quote(ticker.symbol))
    except cse_api.CseApiError as e:
        payload["live_cse_quote"] = {"error": str(e)}

    return payload


def _tool_get_quote(symbol: str) -> dict:
    try:
        return cse_api.quote(symbol)
    except cse_api.CseApiError as e:
        return {"error": str(e)}


def _movers_tool(fetch, label: str):
    """topGainers/topLooses/mostActiveTrades share one shape and one failure mode."""
    def run() -> dict:
        try:
            return {label: [_slim(q) for q in fetch()]}
        except cse_api.CseApiError as e:
            return {"error": str(e)}
    return run


def _tool_get_market_overview() -> dict:
    """Open/closed plus ASPI and session totals. Each half degrades separately."""
    payload: dict = {}
    try:
        payload["status"] = cse_api.market_status()
    except cse_api.CseApiError as e:
        payload["status_error"] = str(e)
    try:
        payload["indices"] = cse_api.market_indices()
    except cse_api.CseApiError as e:
        payload["indices_error"] = str(e)
    return payload


def _tool_list_symbols() -> dict:
    """
    Symbol + name for every listed company, so the model can turn "John Keells"
    into JKH.N0000.

    ~285 rows, which is why only two fields are returned and the result is passed
    through compact(): the full quote payload for the whole exchange would eat the
    context window for a lookup that needs a name and a ticker.
    """
    try:
        rows = [{"symbol": q["symbol"], "name": q["name"]} for q in cse_api.all_companies()]
    except cse_api.CseApiError as e:
        return {"error": str(e)}
    return gemini_chat.compact(rows)


def _tool_search_symbols(query: str) -> dict:
    """
    Search for a CSE ticker symbol by company name or symbol.
    """
    if not query or not query.strip():
        return {"error": "Query cannot be empty."}
        
    try:
        q = query.lower().strip()
        rows = [
            {"symbol": comp["symbol"], "name": comp["name"]} 
            for comp in cse_api.all_companies()
            if q in comp["name"].lower() or q in comp["symbol"].lower()
        ]
    except cse_api.CseApiError as e:
        return {"error": str(e)}
    
    return {"results": rows[:10]}


TOOL_HANDLERS = {
    "get_stock_analysis": _tool_get_stock_analysis,
    "get_quote": _tool_get_quote,
    "get_top_gainers": _movers_tool(cse_api.top_gainers, "gainers"),
    "get_top_losers": _movers_tool(cse_api.top_losers, "losers"),
    "get_most_active": _movers_tool(cse_api.most_active, "most_active"),
    "get_market_overview": _tool_get_market_overview,
    "list_symbols": _tool_list_symbols,
    "search_symbols": _tool_search_symbols,
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the TradeVision assistant about the CSE",
)
async def chat(request: ChatRequest):
    """
    Answer the last message, grounded in live CSE data and the prediction model.

    Can take a while: the model may chain several tool calls, and a question about
    news triggers a scrape. Slow is the correct trade here — a fast wrong price is
    worse than a slow right one.
    """
    messages = [m.model_dump() for m in request.messages][-MAX_HISTORY:]

    if not any(m["role"] == "user" and m["content"].strip() for m in messages):
        raise HTTPException(status_code=422, detail="No user message to respond to.")

    symbol = request.context.symbol if request.context else None

    # Fallback to DeepSeek if Gemini fails or is unavailable
    try:
        result = await run_in_threadpool(
            gemini_chat.chat, messages, TOOL_HANDLERS, symbol
        )
        return ChatResponse(**result)
    except (GeminiUnavailableError, GeminiChatError) as gemini_err:
        try:
            result = await run_in_threadpool(
                deepseek_chat.chat, messages, TOOL_HANDLERS, symbol
            )
            return ChatResponse(**result)
        except DeepseekUnavailableError as ds_err:
            # If both are unavailable, throw 503
            if isinstance(gemini_err, GeminiUnavailableError):
                raise HTTPException(
                    status_code=503, 
                    detail="AI chat is not configured. Neither GEMINI_API_KEY nor AI_ANALYSIS_API_KEY is set."
                ) from ds_err
            # If Gemini failed but DeepSeek is unavailable, raise the Gemini error
            raise HTTPException(status_code=502, detail=f"Gemini API failed and DeepSeek is not configured. (Gemini error: {gemini_err})") from ds_err
        except DeepseekChatError as ds_err:
            raise HTTPException(status_code=502, detail=f"Both AI APIs failed. DeepSeek error: {ds_err}") from ds_err


@router.get("/chat/status", summary="Whether AI chat is configured on this server")
async def chat_status():
    """
    Lets the UI hide or disable the chat entry points instead of offering a button
    that always 503s. Deliberately reports nothing about the key beyond its
    presence.
    """
    gemini_available = gemini_chat.is_available()
    deepseek_available = deepseek_chat.is_available()
    available = gemini_available or deepseek_available
    
    if gemini_available:
        model = gemini_chat.MODEL
    elif deepseek_available:
        model = deepseek_chat._MODEL
    else:
        model = None
        
    return {
        "available": available,
        "model": model,
        "detail": None if available else "Neither GEMINI_API_KEY nor AI_ANALYSIS_API_KEY is set on the server.",
    }
