import type {
  StockPrice,
  StockQuote,
  MarketSummary,
  MarketIndices,
  WatchlistItem,
  UserPortfolio,
  PredictionResult,
  ChatMessage,
} from '../types/stock';
import { supabase } from '../lib/supabase';

/**
 * Backend client.
 *
 * Everything market-related is live: quotes, gainers/losers and the search list
 * come from the CSE via the backend, charts come from the backend's Yahoo
 * history, and predictions come from the real XGBoost + FinBERT pipeline.
 *
 * The backend speaks snake_case (matching Python convention); the components
 * speak camelCase. This file is the only place that translation happens.
 *
 * Portfolio and watchlist are still mock — they need per-user persistence, which
 * arrives with auth and the database.
 */

const API_BASE = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const { data: { session } } = await supabase.auth.getSession();
  const headers = new Headers(init?.headers);
  if (!headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  if (session?.access_token) {
    headers.set('Authorization', `Bearer ${session.access_token}`);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
    });
  } catch {
    // fetch only rejects on network-level failure, which in practice means the
    // backend is not running. Say that instead of "Failed to fetch".
    throw new Error(
      `Cannot reach the TradeVision API at ${API_BASE}. Is the backend running?`
    );
  }

  if (!response.ok) {
    // FastAPI puts the useful message in `detail`.
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      /* non-JSON error body; the status line is all we have */
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

// --- wire types (snake_case, as the backend sends them) ----------------------

interface ApiQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  turnover?: number;
  previous_close?: number | null;
  day_high?: number | null;
  day_low?: number | null;
  day_open?: number | null;
  market_cap?: string | null;
}

interface ApiIndices {
  aspi: number | null;
  aspi_change: number | null;
  aspi_change_percent: number | null;
  turnover: number | null;
  share_volume: number | null;
  trades: number | null;
  listed_companies: number | null;
}

interface ApiMarketSummary {
  status: string;
  gainers: ApiQuote[];
  losers: ApiQuote[];
  most_active: ApiQuote[];
  indices: ApiIndices;
  warnings: string[];
}

interface ApiPricePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ApiAnalysis {
  symbol: string;
  company_name: string;
  as_of: string | null;
  latest_price: number | null;
  sentiment_analysis: {
    score: number;
    label: string;
    headline_count: number;
    status: string;
  };
  price_prediction: {
    predicted_close: number;
    change_percent: number;
    trend: string;
    probability_up: number | null;
    probability_up_adjusted: number | null;
    confidence: number;
    model_status: string;
  } | null;
  model_status: string;
  warnings: string[];
}

// --- mappers ------------------------------------------------------------------

const mapQuote = (q: ApiQuote): StockQuote => ({
  ticker: q.symbol,
  name: q.name,
  price: q.price,
  change: q.change,
  changePercent: q.change_percent,
  volume: q.volume,
  marketCap: q.market_cap ?? undefined,
  turnover: q.turnover,
  previousClose: q.previous_close ?? undefined,
  dayHigh: q.day_high ?? undefined,
  dayLow: q.day_low ?? undefined,
  dayOpen: q.day_open ?? undefined,
});

const mapIndices = (i: ApiIndices): MarketIndices => ({
  aspi: i.aspi,
  aspiChange: i.aspi_change,
  aspiChangePercent: i.aspi_change_percent,
  turnover: i.turnover,
  shareVolume: i.share_volume,
  trades: i.trades,
  listedCompanies: i.listed_companies,
});

/** The backend's 'Upward' | 'Downward' | 'Neutral' -> the UI's lowercase form. */
const mapTrend = (trend: string | undefined): 'up' | 'down' | 'neutral' => {
  const key = (trend ?? '').toLowerCase();
  if (key.startsWith('up')) return 'up';
  if (key.startsWith('down')) return 'down';
  return 'neutral';
};

// --- market data --------------------------------------------------------------

export const fetchMarketSummary = async (): Promise<MarketSummary> => {
  const data = await request<ApiMarketSummary>('/api/v1/market/summary');
  return {
    gainers: data.gainers.map(mapQuote),
    losers: data.losers.map(mapQuote),
    mostActive: data.most_active.map(mapQuote),
    status: data.status,
    indices: mapIndices(data.indices),
    warnings: data.warnings ?? [],
  };
};

export const fetchStockQuote = async (ticker: string): Promise<StockQuote> => {
  const data = await request<ApiQuote>(
    `/api/v1/market/quote?symbol=${encodeURIComponent(ticker)}`
  );
  return mapQuote(data);
};

export const fetchStockHistory = async (
  ticker: string,
  days: number
): Promise<StockPrice[]> => {
  const data = await request<{ points: ApiPricePoint[] }>(
    `/api/v1/market/history?symbol=${encodeURIComponent(ticker)}&days=${days}`
  );
  return data.points;
};

/**
 * The full company list, cached in-module.
 *
 * /market/symbols returns all ~285 listed companies. The analyzer's search box
 * calls this on every debounced keystroke, so refetching each time would be
 * wasteful; filtering happens client-side against one cached list. The TTL is
 * short enough that prices in the results stay roughly current.
 */
const SYMBOL_CACHE_MS = 60_000;
let symbolCache: { at: number; quotes: StockQuote[] } | null = null;
let symbolInFlight: Promise<StockQuote[]> | null = null;

const loadSymbols = async (): Promise<StockQuote[]> => {
  if (symbolCache && Date.now() - symbolCache.at < SYMBOL_CACHE_MS) {
    return symbolCache.quotes;
  }
  // Share one request between concurrent callers rather than firing several on
  // the first render.
  if (!symbolInFlight) {
    symbolInFlight = request<{ symbols: ApiQuote[] }>('/api/v1/market/symbols')
      .then((data) => {
        const quotes = data.symbols.map(mapQuote);
        symbolCache = { at: Date.now(), quotes };
        return quotes;
      })
      .finally(() => {
        symbolInFlight = null;
      });
  }
  return symbolInFlight;
};

export const searchStocks = async (query: string): Promise<StockQuote[]> => {
  const all = await loadSymbols();
  const q = query.trim().toLowerCase();

  if (!q) {
    // Empty query backs the "popular stocks" list and the home marquee, so show
    // the most heavily traded names rather than an alphabetical slice.
    return [...all].sort((a, b) => (b.turnover ?? 0) - (a.turnover ?? 0)).slice(0, 25);
  }

  return all
    .filter(
      (s) =>
        s.ticker.toLowerCase().includes(q) || s.name.toLowerCase().includes(q)
    )
    .slice(0, 50);
};

// --- prediction ---------------------------------------------------------------

/**
 * Real model output from GET /api/v1/stocks/analyze.
 *
 * `includeNews` defaults to false deliberately. With news on, the backend
 * scrapes headlines and runs FinBERT over them, which costs 30-90 seconds on a
 * cold cache — far too long for a page load. The analyzer requests sentiment
 * separately, on demand.
 */
export const fetchPrediction = async (
  ticker: string,
  includeNews = false
): Promise<PredictionResult> => {
  const data = await request<ApiAnalysis>(
    `/api/v1/stocks/analyze?symbol=${encodeURIComponent(ticker)}` +
      `&include_news=${includeNews}`
  );

  const p = data.price_prediction;
  return {
    ticker: data.symbol,
    companyName: data.company_name,
    nextDayPrice: p?.predicted_close ?? null,
    confidence: p?.confidence ?? 0,
    trend: mapTrend(p?.trend),
    changePercent: p?.change_percent ?? null,
    probabilityUp: p?.probability_up ?? null,
    probabilityUpAdjusted: p?.probability_up_adjusted ?? null,
    asOf: data.as_of,
    latestPrice: data.latest_price,
    sentimentScore: data.sentiment_analysis.score,
    sentimentLabel: data.sentiment_analysis.label,
    headlineCount: data.sentiment_analysis.headline_count,
    sentimentStatus: data.sentiment_analysis.status,
    modelStatus: data.model_status,
    warnings: data.warnings ?? [],
    generatedAt: new Date().toISOString(),
  };
};

// --- AI deep analysis (GPT-5.6-Sol) ------------------------------------------

import type { AiAnalysisResult } from '../types/stock';

/**
 * Deep AI analysis powered by GPT-5.6-Sol with internet news search.
 *
 * This can take 1-3 minutes because the agent router searches the web for
 * recent news. The timeout is set to 5 minutes to be safe.
 */
export const fetchAiAnalysis = async (ticker: string): Promise<AiAnalysisResult> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); // 5 min

  try {
    let response: Response;
    try {
      response = await fetch(
        `${API_BASE}/api/v1/stocks/ai-analysis?symbol=${encodeURIComponent(ticker)}`,
        {
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
        }
      );
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new Error('AI analysis timed out after 5 minutes. Please try again.');
      }
      throw new Error(
        `Cannot reach the TradeVision API at ${API_BASE}. Is the backend running?`
      );
    }

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const body = await response.json();
        if (typeof body?.detail === 'string') detail = body.detail;
      } catch {
        /* non-JSON error body */
      }
      throw new Error(detail);
    }

    return (await response.json()) as AiAnalysisResult;
  } finally {
    clearTimeout(timeoutId);
  }
};

// --- AI chat ------------------------------------------------------------------

export interface ChatReply {
  reply: string;
  /** Already display-formatted by the backend, e.g. `get_quote(symbol=JKH.N0000)`. */
  toolsUsed: string[];
  warnings: string[];
}

export const sendChatMessage = async (
  messages: Pick<ChatMessage, 'role' | 'content'>[],
  symbol?: string
): Promise<ChatReply> => {
  const data = await request<{
    reply: string;
    tools_used: string[];
    warnings: string[];
  }>('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      context: symbol ? { symbol } : undefined,
    }),
  });

  return {
    reply: data.reply,
    toolsUsed: data.tools_used ?? [],
    warnings: data.warnings ?? [],
  };
};

/**
 * Whether the server has a Gemini key.
 *
 * Lets the UI disable the composer up front instead of letting the user type a
 * question and then hit a 503. Never throws — an unreachable backend is reported
 * as simply unavailable, which is true.
 */
export const fetchChatStatus = async (): Promise<{ available: boolean; detail: string | null }> => {
  try {
    const data = await request<{ available: boolean; detail: string | null }>(
      '/api/v1/chat/status'
    );
    return { available: data.available, detail: data.detail };
  } catch (e) {
    return { available: false, detail: e instanceof Error ? e.message : 'Chat is unavailable.' };
  }
};

// --- Portfolio & Watchlist (Backend integration) --------------------------------

interface ApiWatchlistItem {
  id: string;
  symbol: string;
}

interface ApiPortfolioItem {
  id: string;
  symbol: string;
  quantity: number;
  price: number;
  date_acquired: string;
}

export const fetchWatchlist = async (): Promise<WatchlistItem[]> => {
  const [data, allSymbols] = await Promise.all([
    request<ApiWatchlistItem[]>('/api/v1/watchlist/'),
    loadSymbols().catch(() => []) // fail gracefully if market data is down
  ]);

  return data.map(item => {
    const quote = allSymbols.find(s => s.ticker === item.symbol);
    return {
      ticker: item.symbol,
      name: quote?.name || item.symbol, // UI needs a name, use market name if found
      targetPrice: null,
      alertEnabled: false,
      id: item.id,
      currentPrice: quote?.price,
      dayChange: quote?.change,
      dayChangePct: quote?.changePercent,
    };
  });
};

export const addToWatchlist = async (symbol: string): Promise<void> => {
  await request('/api/v1/watchlist/', {
    method: 'POST',
    body: JSON.stringify({ symbol })
  });
};

export const removeFromWatchlist = async (id: string): Promise<void> => {
  await request(`/api/v1/watchlist/${id}`, { method: 'DELETE' });
};

export const fetchPortfolio = async (): Promise<UserPortfolio> => {
  const [data, allSymbols] = await Promise.all([
    request<ApiPortfolioItem[]>('/api/v1/portfolio/'),
    loadSymbols().catch(() => [])
  ]);
  
  let totalValue = 0;
  let totalCost = 0;
  let dayChangeAbs = 0;
  let prevTotalValue = 0;
  
  const grouped = new Map<string, {
    ticker: string;
    name: string;
    transactions: { id: string; shares: number; price: number; dateAcquired: string }[];
    currentPrice: number;
    quoteChange: number;
  }>();

  for (const item of data) {
    if (!grouped.has(item.symbol)) {
      const quote = allSymbols.find(s => s.ticker === item.symbol);
      grouped.set(item.symbol, {
        ticker: item.symbol,
        name: quote?.name || item.symbol,
        transactions: [],
        currentPrice: quote?.price || item.price,
        quoteChange: quote?.change || 0
      });
    }
    grouped.get(item.symbol)!.transactions.push({
      id: item.id,
      shares: item.quantity,
      price: item.price,
      dateAcquired: item.date_acquired || 'N/A'
    });
  }

  const holdings = Array.from(grouped.values()).map(group => {
    let groupShares = 0;
    let groupCost = 0;
    
    for (const t of group.transactions) {
      groupShares += t.shares;
      groupCost += t.shares * t.price;
    }
    
    const avgCost = groupShares > 0 ? groupCost / groupShares : 0;
    const itemTotalValue = group.currentPrice * groupShares;
    const itemTotalCost = groupCost;
    const itemDayChange = group.quoteChange * groupShares;
    
    totalValue += itemTotalValue;
    totalCost += itemTotalCost;
    dayChangeAbs += itemDayChange;
    prevTotalValue += (group.currentPrice - group.quoteChange) * groupShares;
    
    return {
      ticker: group.ticker,
      name: group.name,
      totalShares: groupShares,
      avgCost,
      currentPrice: group.currentPrice,
      transactions: group.transactions.sort((a, b) => new Date(b.dateAcquired).getTime() - new Date(a.dateAcquired).getTime())
    };
  });
  
  const dayChangePercent = prevTotalValue > 0 ? (dayChangeAbs / prevTotalValue) * 100 : 0;
  
  return {
    holdings,
    totalValue,
    dayChange: dayChangeAbs,
    dayChangePercent: Number(dayChangePercent.toFixed(2))
  };
};

export const addToPortfolio = async (symbol: string, quantity: number, price: number, date_acquired: string): Promise<void> => {
  await request('/api/v1/portfolio/', {
    method: 'POST',
    body: JSON.stringify({ symbol, quantity, price, date_acquired })
  });
};

export const removeFromPortfolio = async (id: string): Promise<void> => {
  await request(`/api/v1/portfolio/${id}`, { method: 'DELETE' });
};
