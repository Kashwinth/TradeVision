"""
Live CSE market data endpoints.

    GET /api/v1/market/summary                     gainers, losers, most active, indices
    GET /api/v1/market/quote?symbol=JKH.N0000      one live quote
    GET /api/v1/market/symbols                     every listed company (~285)
    GET /api/v1/market/history?symbol=…&days=90    daily OHLC for the chart
    GET /api/v1/market/intraday?symbol=…           today's tick series

WHY EVERY CSE CALL GOES THROUGH run_in_threadpool
Same rule the prediction route documents: `requests` is synchronous, so calling
it directly from an async handler stalls the event loop and freezes every other
request. services/cse_api.py is deliberately sync — it is the threadpool's job to
keep it off the loop.

WHY /history IS SERVED FROM YAHOO AND NOT THE CSE
The CSE's chart endpoint returns the same 5 daily bars whether you request 7 days
or 365 (verified against period=7/30/90/180/365), so it cannot draw a 1M or 3M
chart. price_data.get_price_history() is used instead, which has the side benefit
that the chart and the model see identical bars — the honest thing to show next to
a prediction. Its stale_days is passed through so the UI can label the gap.
"""

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, Query
# pyrefly: ignore [missing-import]
from starlette.concurrency import run_in_threadpool

from api.schemas.market import (
    MarketIndices,
    MarketQuote,
    MarketSummaryResponse,
    PriceHistoryResponse,
    PricePoint,
    SymbolListResponse,
)
from services import cse_api, price_data
from services.cse_api import CseApiError
from services.features import InsufficientHistoryError
from services.price_data import PriceDataError
from services.ticker_registry import UnknownTickerError, resolve_open

router = APIRouter(prefix="/api/v1/market", tags=["market"])


def _summary_sync() -> dict:
    """
    Blocking: up to five CSE calls, most served from one cached tradeSummary.

    Each list degrades independently. A failed gainers fetch should not blank the
    whole page when losers and indices came back fine, so failures become
    warnings and an empty list rather than a 502.
    """
    warnings: list[str] = []

    def safe(label: str, fn, default):
        try:
            return fn()
        except CseApiError as e:
            warnings.append(f"{label} unavailable: {e}")
            return default

    return {
        "status": safe("Market status", cse_api.market_status, "Unknown"),
        "gainers": safe("Top gainers", cse_api.top_gainers, []),
        "losers": safe("Top losers", cse_api.top_losers, []),
        "most_active": safe("Most active", cse_api.most_active, []),
        "indices": safe("Market indices", cse_api.market_indices, {}),
        "warnings": warnings,
    }


@router.get(
    "/summary",
    response_model=MarketSummaryResponse,
    summary="Live CSE market summary: gainers, losers, most active, ASPI",
)
async def market_summary():
    """
    The CSE caps each mover list at 10 rows. Rows are enriched from the trade
    summary, because the movers endpoints omit company name and volume — and
    mostActiveTrades omits price entirely.
    """
    data = await run_in_threadpool(_summary_sync)

    return MarketSummaryResponse(
        status=data["status"],
        gainers=[MarketQuote(**q) for q in data["gainers"]],
        losers=[MarketQuote(**q) for q in data["losers"]],
        most_active=[MarketQuote(**q) for q in data["most_active"]],
        indices=MarketIndices(**data["indices"]),
        warnings=data["warnings"],
    )


@router.get(
    "/quote",
    response_model=MarketQuote,
    summary="Live quote for one CSE symbol",
)
async def market_quote(
    symbol: str = Query(
        ...,
        description="CSE ticker, e.g. JKH.N0000 (short form 'JKH' also accepted).",
        examples=["JKH.N0000"],
    ),
):
    try:
        row = await run_in_threadpool(cse_api.quote, symbol)
    except CseApiError as e:
        # Ambiguous upstream: either the symbol is not listed or the CSE is down.
        # 404 is the useful answer for the client either way, and the detail text
        # distinguishes them.
        raise HTTPException(status_code=404, detail=str(e)) from e
    return MarketQuote(**row)


@router.get(
    "/symbols",
    response_model=SymbolListResponse,
    summary="Every company listed on the CSE, with live quotes",
)
async def market_symbols():
    """
    Backs the frontend's search box. Returns full quotes rather than bare symbols
    so search results can show price and change without a second round trip.
    """
    try:
        rows = await run_in_threadpool(cse_api.all_companies)
    except CseApiError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return SymbolListResponse(count=len(rows), symbols=[MarketQuote(**r) for r in rows])


def _history_sync(symbol: str, days: int) -> dict:
    ticker = resolve_open(symbol)
    
    # User requested to only show 7 days (5 trading bars) and strictly use the 
    # official CSE API to guarantee live, accurate data without Yahoo staleness.
    points = []
    warnings: list[str] = []
    as_of = None
    stale_days = 0

    try:
        live_data = cse_api.historical_5day(symbol)
        if live_data:
            for p in live_data:
                if p.get("time"):
                    dt = pd.to_datetime(p["time"], unit="ms").strftime("%Y-%m-%d")
                    close = float(p["price"])
                    high = float(p["high"]) if p.get("high") is not None else close
                    low = float(p["low"]) if p.get("low") is not None else close
                    vol = float(p["quantity"]) if p.get("quantity") is not None else 0.0
                    
                    points.append(PricePoint(
                        timestamp=dt,
                        open=close, 
                        high=high,
                        low=low,
                        close=close,
                        volume=vol
                    ))
            
            points.sort(key=lambda x: x.timestamp)
            
            if points:
                as_of = points[-1].timestamp
                last_dt = pd.to_datetime(as_of).date()
                import datetime
                stale_days = (datetime.date.today() - last_dt).days
        else:
            warnings.append(f"No recent CSE historical data for {symbol}.")
    except Exception as e:
        warnings.append(f"Failed to fetch CSE historical data: {e}")

    return {
        "symbol": ticker.symbol,
        "source": "cse_official",
        "points": points,
        "as_of": as_of,
        "stale_days": stale_days,
        "warnings": warnings,
    }


@router.get(
    "/history",
    response_model=PriceHistoryResponse,
    summary="Daily OHLC history for the price chart",
)
async def market_history(
    symbol: str = Query(..., description="CSE ticker.", examples=["JKH.N0000"]),
    days: int = Query(90, ge=2, le=730, description="Trading days to return."),
):
    try:
        data = await run_in_threadpool(_history_sync, symbol, days)
    except UnknownTickerError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PriceDataError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except InsufficientHistoryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PriceHistoryResponse(**data)


@router.get(
    "/intraday",
    summary="Today's per-trade tick series for one symbol",
)
async def market_intraday(
    symbol: str = Query(..., description="CSE ticker.", examples=["JKH.N0000"]),
):
    """
    The one CSE chart mode that returns real data. Separate from /history because
    it is a tick series, not daily OHLC, and returns nothing when the market has
    not traded today.
    """
    try:
        points = await run_in_threadpool(cse_api.intraday, symbol)
    except CseApiError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"symbol": symbol, "count": len(points), "points": points}
