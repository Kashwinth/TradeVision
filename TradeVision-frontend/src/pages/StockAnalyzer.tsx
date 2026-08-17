import React, { useCallback, useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Search, Brain, TrendingUp, TrendingDown, Loader2, AlertCircle, Newspaper, RefreshCw,
  Sparkles, ExternalLink, ShieldAlert, Eye, ChevronRight,
} from 'lucide-react';
import { StockCard } from '../components/StockCard';
import { LiveChart } from '../components/LiveChart';
import type { StockQuote, StockPrice, PredictionResult, AiAnalysisResult } from '../types/stock';
import { searchStocks, fetchStockQuote, fetchStockHistory, fetchPrediction, fetchAiAnalysis } from '../services/api';
import { useAuth } from '../context/AuthContext';

export const StockAnalyzer: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<StockQuote[]>([]);
  const [selectedStock, setSelectedStock] = useState<StockQuote | null>(null);
  const [chartData, setChartData] = useState<StockPrice[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [isLoadingNews, setIsLoadingNews] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  
  // Hardcoded to 7 days to strictly use the official CSE 5-day live data
  const timeRange = 7;

  // AI Deep Analysis state
  const [aiAnalysis, setAiAnalysis] = useState<AiAnalysisResult | null>(null);
  const [isLoadingAiAnalysis, setIsLoadingAiAnalysis] = useState(false);
  const [aiAnalysisError, setAiAnalysisError] = useState<string | null>(null);

  // Handle Search
  useEffect(() => {
    let stale = false;
    const timer = setTimeout(() => {
      setIsSearching(true);
      searchStocks(searchQuery)
        .then((results) => {
          if (!stale) setSearchResults(results);
        })
        .catch((e) => {
          if (!stale) setError(e instanceof Error ? e.message : 'Search failed');
        })
        .finally(() => {
          if (!stale) setIsSearching(false);
        });
    }, 300);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [searchQuery]);

  const ticker = searchParams.get('ticker') || 'JKH.N0000';

  // Quote + chart
  useEffect(() => {
    let stale = false;
    const loadStockDetails = async () => {
      setIsLoadingDetails(true);
      setError(null);
      try {
        const [quote, history] = await Promise.all([
          fetchStockQuote(ticker),
          fetchStockHistory(ticker, timeRange),
        ]);
        if (stale) return;
        setSelectedStock(quote);
        setChartData(history);
      } catch (e) {
        if (!stale) setError(e instanceof Error ? e.message : 'Failed to load stock details');
      } finally {
        if (!stale) setIsLoadingDetails(false);
      }
    };
    void loadStockDetails();
    return () => { stale = true; };
  }, [ticker, timeRange]);

  // Prediction
  useEffect(() => {
    let stale = false;
    const loadPrediction = async () => {
      setIsLoadingPrediction(true);
      setPredictionError(null);
      setPrediction(null);
      setAiAnalysis(null);
      setAiAnalysisError(null);
      try {
        const result = await fetchPrediction(ticker);
        if (!stale) setPrediction(result);
      } catch (e) {
        if (!stale) setPredictionError(e instanceof Error ? e.message : 'Prediction failed');
      } finally {
        if (!stale) setIsLoadingPrediction(false);
      }
    };
    void loadPrediction();
    return () => { stale = true; };
  }, [ticker]);

  const loadNewsSentiment = useCallback(async () => {
    setIsLoadingNews(true);
    setPredictionError(null);
    try {
      const result = await fetchPrediction(ticker, true);
      setPrediction(result);
    } catch (e) {
      setPredictionError(e instanceof Error ? e.message : 'News sentiment failed');
    } finally {
      setIsLoadingNews(false);
    }
  }, [ticker]);

  const loadAiAnalysis = useCallback(async () => {
    if (!user) {
      navigate('/login');
      return;
    }

    setIsLoadingAiAnalysis(true);
    setAiAnalysisError(null);
    setAiAnalysis(null);
    try {
      const result = await fetchAiAnalysis(ticker);
      setAiAnalysis(result);
    } catch (e) {
      setAiAnalysisError(e instanceof Error ? e.message : 'AI analysis failed');
    } finally {
      setIsLoadingAiAnalysis(false);
    }
  }, [ticker, user, navigate]);

  const handleSelectStock = (next: string) => {
    setSearchParams({ ticker: next });
  };

  const hasSentiment = prediction != null && prediction.sentimentStatus !== 'skipped';

  const verdictColor = (verdict: string) => {
    if (verdict === 'Bullish') return 'text-accent-green';
    if (verdict === 'Bearish') return 'text-accent-red';
    return 'text-yellow-500';
  };

  const verdictBg = (verdict: string) => {
    if (verdict === 'Bullish') return 'bg-accent-green/10 border-accent-green/30';
    if (verdict === 'Bearish') return 'bg-accent-red/10 border-accent-red/30';
    return 'bg-yellow-500/10 border-yellow-500/30';
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 h-[calc(100vh-64px)] flex flex-col md:flex-row gap-6">

      {/* Left Sidebar - Search & List */}
      <div className="w-full md:w-1/3 lg:w-1/4 flex flex-col h-full bg-primary border border-border rounded-2xl overflow-hidden shadow-sm">
        <div className="p-4 border-b border-border bg-secondary/50">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
            <input
              type="text"
              placeholder="Search CSE stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-primary border border-border rounded-xl text-text-primary focus:outline-none focus:ring-2 focus:ring-accent-green focus:border-transparent transition-all"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {isSearching && searchResults.length === 0 ? (
            <div className="flex justify-center p-8">
              <Loader2 className="w-6 h-6 animate-spin text-text-secondary" />
            </div>
          ) : searchResults.length > 0 ? (
            searchResults.map((quote) => (
              <StockCard
                key={quote.ticker}
                quote={quote}
                selected={selectedStock?.ticker === quote.ticker}
                onClick={() => handleSelectStock(quote.ticker)}
              />
            ))
          ) : (
            <p className="text-center text-text-secondary mt-8">No stocks found.</p>
          )}
        </div>
      </div>

      {/* Right Main Content - Chart & Prediction */}
      <div className="w-full md:w-2/3 lg:w-3/4 flex flex-col h-full space-y-6 overflow-y-auto pb-8">

        {/* Main Chart Card */}
        <div className="shrink-0 bg-primary border border-border rounded-2xl p-6 shadow-sm">
          {isLoadingDetails ? (
            <div className="h-[500px] flex items-center justify-center">
              <Loader2 className="w-8 h-8 animate-spin text-accent-green" />
            </div>
          ) : error || !selectedStock ? (
            <div className="h-[500px] flex flex-col items-center justify-center text-center px-6">
              <AlertCircle className="w-10 h-10 text-accent-red mb-4" />
              <p className="font-bold text-text-primary mb-2">Could not load {ticker}</p>
              <p className="text-text-secondary max-w-md">{error ?? 'No data available.'}</p>
            </div>
          ) : (
            <>
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h1 className="text-3xl font-bold text-text-primary">{selectedStock.name}</h1>
                  <p className="text-text-secondary">{selectedStock.ticker} • CSE</p>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-bold text-text-primary">Rs {selectedStock.price.toFixed(2)}</p>
                  <p className={`font-medium text-lg flex items-center justify-end ${selectedStock.change >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {selectedStock.change >= 0 ? <TrendingUp className="w-5 h-5 mr-1" /> : <TrendingDown className="w-5 h-5 mr-1" />}
                    {Math.abs(selectedStock.change).toFixed(2)} ({Math.abs(selectedStock.changePercent).toFixed(2)}%)
                  </p>
                  <p className="text-xs text-text-secondary mt-1">Live CSE price</p>
                </div>
              </div>

              <div className="flex space-x-2 mb-4">
                <div className="px-4 py-1.5 rounded-lg text-sm font-medium bg-text-primary text-primary">
                  1W (Live)
                </div>
              </div>

              <LiveChart data={chartData} ticker={selectedStock.ticker} />
            </>
          )}
        </div>

        {/* AI Prediction Panel */}
        <div className="shrink-0 bg-gradient-to-br from-primary to-secondary border border-border rounded-2xl p-6 shadow-sm relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-accent-green/5 rounded-full -translate-y-1/2 translate-x-1/2 pointer-events-none blur-2xl" />

          <div className="flex items-center flex-wrap gap-3 mb-6">
            <Brain className="w-6 h-6 text-accent-green" />
            <h2 className="text-xl font-bold text-text-primary">AI Prediction Analysis</h2>
            {prediction?.asOf && (
              <span className="px-2 py-1 text-xs font-medium bg-secondary text-text-secondary rounded-md">
                Based on close of {prediction.asOf}
              </span>
            )}
            {prediction && prediction.modelStatus !== 'loaded' && (
              <span className="px-2 py-1 text-xs font-medium bg-accent-red/10 text-accent-red rounded-md">
                Model not loaded
              </span>
            )}
          </div>

          {isLoadingPrediction ? (
            <div className="h-32 flex items-center justify-center text-text-secondary">
              <Loader2 className="w-6 h-6 animate-spin mr-3 text-accent-green" />
              Running the model…
            </div>
          ) : predictionError ? (
            <div className="flex items-start px-4 py-3 rounded-xl bg-accent-red/10 border border-accent-red/20 text-sm text-accent-red">
              <AlertCircle className="w-4 h-4 mr-2 mt-0.5 shrink-0" />
              <span>{predictionError}</span>
            </div>
          ) : prediction ? (
            <>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-primary/50 p-4 rounded-xl border border-border">
                  <p className="text-sm text-text-secondary mb-1">Next Day Predicted Close</p>
                  <p className="text-2xl font-bold text-text-primary">
                    {prediction.nextDayPrice !== null ? `Rs ${prediction.nextDayPrice.toFixed(2)}` : '—'}
                  </p>
                  <p className={`text-sm mt-1 ${
                    prediction.trend === 'up' ? 'text-accent-green'
                      : prediction.trend === 'down' ? 'text-accent-red' : 'text-text-secondary'
                  }`}>
                    {prediction.trend === 'up' ? 'Expected Uptrend'
                      : prediction.trend === 'down' ? 'Expected Downtrend' : 'No clear direction'}
                    {prediction.changePercent !== null && ` (${prediction.changePercent >= 0 ? '+' : ''}${prediction.changePercent.toFixed(2)}%)`}
                  </p>
                </div>

                <div className="bg-primary/50 p-4 rounded-xl border border-border">
                  <div className="flex justify-between items-end mb-2">
                    <p className="text-sm text-text-secondary">Model Confidence</p>
                    <p className="text-xl font-bold text-text-primary">{prediction.confidence.toFixed(1)}%</p>
                  </div>
                  <div className="w-full h-3 bg-border rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ${
                        prediction.confidence > 75 ? 'bg-accent-green'
                          : prediction.confidence > 50 ? 'bg-yellow-500' : 'bg-accent-red'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, prediction.confidence))}%` }}
                    />
                  </div>
                  {prediction.probabilityUp !== null && (
                    <p className="text-xs text-text-secondary mt-2">
                      Raw P(up) {(prediction.probabilityUp * 100).toFixed(1)}% before the sentiment blend.
                    </p>
                  )}
                </div>

                <div className="bg-primary/50 p-4 rounded-xl border border-border">
                  <p className="text-sm text-text-secondary mb-1">News Sentiment</p>
                  {hasSentiment ? (
                    <>
                      <p className={`text-2xl font-bold ${
                        prediction.sentimentScore > 0.05 ? 'text-accent-green'
                          : prediction.sentimentScore < -0.05 ? 'text-accent-red' : 'text-text-primary'
                      }`}>
                        {prediction.sentimentLabel}
                      </p>
                      <p className="text-xs text-text-secondary mt-1">
                        FinBERT {prediction.sentimentScore.toFixed(3)} over {prediction.headlineCount} headline(s)
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-sm text-text-secondary mb-3">
                        Skipped — scraping and scoring headlines takes up to a minute.
                      </p>
                      <button
                        onClick={() => void loadNewsSentiment()}
                        disabled={isLoadingNews}
                        className="inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-medium bg-accent-green/10 text-accent-green hover:bg-accent-green/20 disabled:opacity-60 transition-colors"
                      >
                        {isLoadingNews
                          ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing…</>
                          : <><Newspaper className="w-4 h-4 mr-2" /> Analyze news</>}
                      </button>
                    </>
                  )}
                </div>
              </div>

              {prediction.warnings.length > 0 && (
                <div className="mt-6 space-y-2">
                  {prediction.warnings.map((w) => (
                    <div
                      key={w}
                      className="flex items-start px-4 py-3 rounded-xl bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-600 dark:text-yellow-500"
                    >
                      <AlertCircle className="w-4 h-4 mr-2 mt-0.5 shrink-0" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}

              <p className="text-xs text-text-secondary mt-4">
                Model output from an XGBoost direction classifier blended with FinBERT news
                sentiment. Not financial advice.
              </p>

              {/* ── AI Deep Analysis Button ── */}
              {!aiAnalysis && !isLoadingAiAnalysis && (
                <div className="mt-6 pt-6 border-t border-border">
                  <button
                    id="ai-prediction-button"
                    onClick={() => void loadAiAnalysis()}
                    disabled={isLoadingAiAnalysis}
                    className="w-full group relative inline-flex items-center justify-center px-6 py-4 rounded-xl text-base font-semibold
                      bg-emerald-600 text-white
                      hover:bg-emerald-700
                      shadow-lg shadow-accent-green/25 hover:shadow-accent-green/40
                      transition-all duration-300 hover:scale-[1.02]
                      disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100"
                  >
                    {user ? 'AI Deep Analysis Generate Insights' : 'Login for AI Deep Analysis'}
                  </button>
                  <p className="text-xs text-text-secondary text-center mt-2">
                    Uses DeepSeek AI to perform multi-layered analysis. Usually takes 30-60 seconds.
                  </p>
                </div>
              )}
            </>
          ) : (
            <button
              onClick={() => setSearchParams({ ticker })}
              className="inline-flex items-center text-sm font-medium text-text-secondary hover:text-accent-green transition-colors"
            >
              <RefreshCw className="w-4 h-4 mr-2" /> Retry prediction
            </button>
          )}
        </div>

        {/* ── AI Deep Analysis Loading State ── */}
        {isLoadingAiAnalysis && (
          <div className="shrink-0 bg-gradient-to-br from-accent-green/10 to-emerald-900/10 border border-accent-green/30 rounded-2xl p-8 shadow-sm">
            <div className="flex flex-col items-center justify-center text-center space-y-4">
              <div className="relative">
                <Sparkles className="w-10 h-10 text-accent-green animate-pulse" />
                <div className="absolute inset-0 w-10 h-10 bg-accent-green/20 rounded-full animate-ping" />
              </div>
              <h3 className="text-lg font-semibold text-text-primary">AI Analysis in Progress…</h3>
              <p className="text-sm text-text-secondary max-w-md">
                Analysing market data, evaluating technicals, and generating insights.
                This typically takes 30-60 seconds.
              </p>
              <div className="flex items-center gap-3 text-xs text-text-secondary">
                <Loader2 className="w-4 h-4 animate-spin text-accent-green" />
                <span>Powered by DeepSeek</span>
              </div>
            </div>
          </div>
        )}

        {/* ── AI Deep Analysis Error ── */}
        {aiAnalysisError && !isLoadingAiAnalysis && (
          <div className="shrink-0 bg-accent-red/5 border border-accent-red/20 rounded-2xl p-6">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-accent-red mr-3 mt-0.5 shrink-0" />
              <div>
                <p className="font-medium text-accent-red mb-1">AI Analysis Failed</p>
                <p className="text-sm text-text-secondary">{aiAnalysisError}</p>
                <button
                  onClick={() => void loadAiAnalysis()}
                  className="mt-3 inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium bg-accent-green/10 text-accent-green hover:bg-accent-green/20 transition-colors"
                >
                  <RefreshCw className="w-4 h-4 mr-2" /> Retry Analysis
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── AI Deep Analysis Results ── */}
        {aiAnalysis && !isLoadingAiAnalysis && (
          <div className="shrink-0 bg-gradient-to-br from-primary to-secondary border border-accent-green/20 rounded-2xl shadow-sm overflow-hidden">

            {/* Header with Verdict */}
            <div className="p-6 bg-gradient-to-r from-accent-green/10 to-emerald-500/10 border-b border-accent-green/10">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-3">
                  <Sparkles className="w-6 h-6 text-accent-green" />
                  <h2 className="text-xl font-bold text-text-primary">AI Deep Analysis</h2>
                </div>
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border font-bold text-lg ${verdictBg(aiAnalysis.overall_verdict)} ${verdictColor(aiAnalysis.overall_verdict)}`}>
                  {aiAnalysis.overall_verdict === 'Bullish' ? <TrendingUp className="w-5 h-5" /> : aiAnalysis.overall_verdict === 'Bearish' ? <TrendingDown className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  {aiAnalysis.overall_verdict}
                  <span className="text-sm font-normal opacity-75">({aiAnalysis.verdict_confidence} confidence)</span>
                </div>
              </div>
              {aiAnalysis.summary && (
                <p className="mt-4 text-text-secondary leading-relaxed">{aiAnalysis.summary}</p>
              )}
            </div>

            <div className="p-6 space-y-6">
              {aiAnalysis.key_insights && aiAnalysis.key_insights.length > 0 && (
                <div>
                  <h3 className="flex items-center gap-2 text-sm font-semibold text-accent-green uppercase tracking-wider mb-3">
                    <Sparkles className="w-4 h-4" /> Key Insights
                  </h3>
                  <ul className="space-y-2">
                    {aiAnalysis.key_insights.map((insight, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-accent-green shrink-0" />
                        <span className="text-sm text-text-secondary leading-relaxed">{insight}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Recent News with Links */}
              {aiAnalysis.recent_news && aiAnalysis.recent_news.length > 0 && (
                <div>
                  <h3 className="flex items-center gap-2 text-sm font-semibold text-accent-green uppercase tracking-wider mb-3">
                    <Newspaper className="w-4 h-4" /> Recent News
                  </h3>
                  <div className="space-y-3">
                    {aiAnalysis.recent_news.map((news, idx) => (
                      <a
                        key={idx}
                        href={news.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-start gap-3 p-3 rounded-xl bg-primary/50 border border-border hover:border-accent-green/30 hover:bg-accent-green/5 transition-all group"
                      >
                        <ExternalLink className="w-4 h-4 text-accent-green mt-1 shrink-0 group-hover:scale-110 transition-transform" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-text-primary group-hover:text-accent-green transition-colors line-clamp-2">
                            {news.headline}
                          </p>
                          <p className="text-xs text-text-secondary mt-1">
                            {news.source} {news.date && `• ${news.date}`}
                          </p>
                          {news.relevance && (
                            <p className="text-xs text-accent-green/70 mt-1">{news.relevance}</p>
                          )}
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Risk Factors */}
              {aiAnalysis.risk_factors && aiAnalysis.risk_factors.length > 0 && (
                <div>
                  <h3 className="flex items-center gap-2 text-sm font-semibold text-yellow-500 uppercase tracking-wider mb-2">
                    <ShieldAlert className="w-4 h-4" /> Risk Factors
                  </h3>
                  <ul className="space-y-1.5">
                    {aiAnalysis.risk_factors.map((risk, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm text-text-secondary">
                        <span className="text-yellow-500 mt-0.5">•</span>
                        {risk}
                      </li>
                    ))}
                  </ul>
                </div>
              )}



              {/* Disclaimer */}
              <div className="pt-4 border-t border-border">
                <p className="text-xs text-text-secondary italic">
                  {aiAnalysis.recommendation_note || 'Not financial advice. This is model output for informational purposes only.'}
                </p>
                <p className="text-xs text-text-secondary mt-1">
                  Powered by DeepSeek • Analysis based on supplied market data
                </p>
              </div>
            </div>
          </div>
        )}

      </div>

    </div>
  );
};
