export interface StockPrice {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockQuote {
  ticker: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  // Optional because the CSE's mover endpoints omit these; they are filled by
  // the backend's trade-summary join when a match exists.
  marketCap?: string;
  turnover?: number;
  previousClose?: number;
  dayHigh?: number;
  dayLow?: number;
  dayOpen?: number;
}

export interface MarketIndices {
  aspi: number | null;
  aspiChange: number | null;
  aspiChangePercent: number | null;
  turnover: number | null;
  shareVolume: number | null;
  trades: number | null;
  listedCompanies: number | null;
}

export interface MarketSummary {
  gainers: StockQuote[];
  losers: StockQuote[];
  mostActive: StockQuote[];
  /** 'Market Open' / 'Market Closed' / etc., straight from the CSE. */
  status: string;
  indices: MarketIndices;
  /** Per-list upstream failures, so a partial page can explain itself. */
  warnings: string[];
}

export interface WatchlistItem {
  id?: string;
  ticker: string;
  name: string;
  targetPrice: number | null;
  alertEnabled: boolean;
  currentPrice?: number;
  dayChange?: number;
  dayChangePct?: number;
}

export interface PortfolioTransaction {
  id: string;
  shares: number;
  price: number;
  dateAcquired: string;
}

export interface PortfolioHolding {
  ticker: string;
  name: string;
  totalShares: number;
  avgCost: number;
  currentPrice: number;
  transactions: PortfolioTransaction[];
}

export interface UserPortfolio {
  holdings: PortfolioHolding[];
  totalValue: number;
  dayChange: number;
  dayChangePercent: number;
}

export interface PredictionResult {
  ticker: string;
  companyName: string;

  /** Null when the model artifact is not loaded on the backend. */
  nextDayPrice: number | null;
  /** probability_up_adjusted as a percentage. */
  confidence: number;
  trend: 'up' | 'down' | 'neutral';
  changePercent: number | null;
  probabilityUp: number | null;
  probabilityUpAdjusted: number | null;

  /**
   * Date of the price bar the prediction was computed from. This can be months
   * behind today: Yahoo has stopped updating CSE symbols, and the backend
   * discards the forward-filled placeholder rows rather than modelling them. The
   * prediction is for the day after `asOf`, not necessarily tomorrow.
   */
  asOf: string | null;
  /** Latest close from the model's feed — NOT the live CSE price. */
  latestPrice: number | null;

  sentimentScore: number;
  sentimentLabel: string;
  headlineCount: number;
  sentimentStatus: string;

  modelStatus: string;
  /** Backend-generated caveats, staleness chief among them. Render these. */
  warnings: string[];
  generatedAt: string;
}

export interface ChatMessage {
  /** Stable key for the list — message content is not unique enough. */
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /**
   * Data lookups the backend ran for this answer, already formatted for display
   * (e.g. `get_quote(symbol=JKH.N0000)`). Rendered as chips so the grounding is
   * visible: the assistant is not allowed to state a figure it did not fetch.
   */
  toolsUsed?: string[];
  /** Tool failures or truncation notices attached to this answer. */
  warnings?: string[];
  isError?: boolean;
}

export interface AiNewsItem {
  headline: string;
  source: string;
  url: string;
  date: string;
  relevance: string;
}

export interface AiAnalysisResult {
  overall_verdict: 'Bullish' | 'Bearish' | 'Neutral';
  verdict_confidence: 'High' | 'Medium' | 'Low';
  summary: string;
  key_insights: string[];
  recent_news: AiNewsItem[];
  risk_factors: string[];
  recommendation_note: string;
}
