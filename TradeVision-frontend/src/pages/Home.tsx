import React from 'react';
import { Link } from 'react-router-dom';
import { BrainCircuit, BellRing, PieChart, ArrowRight, Zap, TrendingUp, TrendingDown } from 'lucide-react';
import { MarqueeTicker } from '../components/MarqueeTicker';
import { TradingViewWidget } from '../components/TradingViewWidget';
import { useStockContext } from '../context/StockContext';
import { StockCard } from '../components/StockCard';

/** 1_522_182_240 -> "Rs 1.52B". */
const formatRupees = (value: number): string => {
  for (const [divisor, suffix] of [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']] as const) {
    if (value >= divisor) return `Rs ${(value / divisor).toFixed(2)}${suffix}`;
  }
  return `Rs ${value.toFixed(0)}`;
};

export const Home: React.FC = () => {
  const { marketSummary } = useStockContext();

  const tickerItems = marketSummary
    ? [...marketSummary.gainers, ...marketSummary.losers]
    : [];

  const indices = marketSummary?.indices;
  
  // Get top 3 movers for the new section
  const topGainers = marketSummary?.gainers.slice(0, 3) || [];
  const topLosers = marketSummary?.losers.slice(0, 3) || [];

  return (
    <div className="flex flex-col min-h-[calc(100vh-64px)]">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-primary py-20 lg:py-32 flex-grow flex flex-col justify-center animate-fade-in-up">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-accent-green/10 via-primary to-primary pointer-events-none" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary text-sm font-medium text-text-primary mb-8 border border-border">
            <Zap className="w-4 h-4 text-accent-green" />
            <span className="text-text-secondary">Now live with real-time AI predictions</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-extrabold text-text-primary tracking-tight mb-6">
            Predict the Future of <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-green to-emerald-500">
              Sri Lankan Stocks
            </span>
          </h1>
          <p className="mt-4 max-w-2xl mx-auto text-xl text-text-secondary mb-10">
            Advanced AI-powered analysis for the Colombo Stock Exchange. 
            Make smarter decisions with real-time predictions and portfolio tracking.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/register"
              className="inline-flex justify-center items-center px-8 py-4 border border-transparent text-lg font-medium rounded-xl text-white bg-accent-green hover:bg-accent-green/90 shadow-lg shadow-accent-green/30 transition-all duration-300 hover:scale-105"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
            <Link
              to="/top-stocks"
              className="inline-flex justify-center items-center px-8 py-4 border-2 border-border text-lg font-medium rounded-xl text-text-primary hover:border-text-secondary transition-all duration-300 bg-secondary/50 backdrop-blur hover:scale-105"
            >
              Explore Stocks
            </Link>
          </div>
        </div>
      </section>

      {/* Live Ticker */}
      <MarqueeTicker items={tickerItems} />

      {/* Stats Bar */}
      <section className="py-12 border-b border-border bg-primary animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-8 text-center divide-x divide-border">
            <div>
              <p className="text-4xl font-extrabold text-text-primary mb-2">
                {indices?.listedCompanies ?? '—'}
              </p>
              <p className="text-text-secondary font-medium">Listed Companies</p>
            </div>
            <div>
              <p className="text-4xl font-extrabold text-text-primary mb-2">
                {indices?.aspi
                  ? indices.aspi.toLocaleString(undefined, { maximumFractionDigits: 0 })
                  : '—'}
                {indices?.aspiChangePercent != null && (
                  <span
                    className={`ml-2 text-base font-bold ${
                      indices.aspiChangePercent >= 0 ? 'text-accent-green' : 'text-accent-red'
                    }`}
                  >
                    {indices.aspiChangePercent >= 0 ? '+' : ''}
                    {indices.aspiChangePercent.toFixed(2)}%
                  </span>
                )}
              </p>
              <p className="text-text-secondary font-medium">ASPI Index</p>
            </div>
            <div className="col-span-2 md:col-span-1 border-t md:border-t-0 md:border-l border-border pt-8 md:pt-0">
              <p className="text-4xl font-extrabold text-text-primary mb-2">
                {indices?.turnover ? formatRupees(indices.turnover) : '—'}
              </p>
              <p className="text-text-secondary font-medium">Market Turnover</p>
            </div>
          </div>
        </div>
      </section>
      
      {/* Live Market Overview (New Section) */}
      {(topGainers.length > 0 || topLosers.length > 0) && (
        <section className="py-16 bg-secondary border-b border-border animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-end mb-8">
              <div>
                <h2 className="text-2xl font-bold text-text-primary">Today's Top Movers</h2>
                <p className="text-text-secondary mt-1">Live from the Colombo Stock Exchange</p>
              </div>
              <Link to="/top-stocks" className="hidden sm:inline-flex items-center text-sm font-semibold text-accent-green hover:text-accent-green/80 transition-colors">
                View All <ArrowRight className="ml-1 w-4 h-4" />
              </Link>
            </div>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-accent-green" />
                  <h3 className="font-semibold text-text-primary">Top Gainers</h3>
                </div>
                <div className="space-y-3">
                  {topGainers.map(stock => (
                    <StockCard key={stock.ticker} quote={stock} />
                  ))}
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <TrendingDown className="w-5 h-5 text-accent-red" />
                  <h3 className="font-semibold text-text-primary">Top Losers</h3>
                </div>
                <div className="space-y-3">
                  {topLosers.map(stock => (
                    <StockCard key={stock.ticker} quote={stock} />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Features Section */}
      <section className="py-24 bg-primary animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-text-primary">Why Choose TradeVision?</h2>
            <p className="mt-4 text-text-secondary">Powerful tools designed for the modern investor.</p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-secondary p-8 rounded-2xl border border-border hover:border-accent-green/50 hover:shadow-2xl hover:-translate-y-2 transition-all duration-300 group">
              <div className="w-14 h-14 bg-accent-green/10 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <BrainCircuit className="w-8 h-8 text-accent-green" />
              </div>
              <h3 className="text-xl font-bold text-text-primary mb-3">AI Predictions</h3>
              <p className="text-text-secondary leading-relaxed">
                Our models analyze historical data and market sentiment to forecast short-term price movements with high confidence.
              </p>
            </div>

            <div className="bg-secondary p-8 rounded-2xl border border-border hover:border-accent-green/50 hover:shadow-2xl hover:-translate-y-2 transition-all duration-300 group">
              <div className="w-14 h-14 bg-accent-green/10 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <BellRing className="w-8 h-8 text-accent-green" />
              </div>
              <h3 className="text-xl font-bold text-text-primary mb-3">Smart Alerts</h3>
              <p className="text-text-secondary leading-relaxed">
                Never miss an opportunity. Set custom price targets and get instant notifications when your conditions are met.
              </p>
            </div>

            <div className="bg-secondary p-8 rounded-2xl border border-border hover:border-accent-green/50 hover:shadow-2xl hover:-translate-y-2 transition-all duration-300 group">
              <div className="w-14 h-14 bg-accent-green/10 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <PieChart className="w-8 h-8 text-accent-green" />
              </div>
              <h3 className="text-xl font-bold text-text-primary mb-3">Portfolio Tracking</h3>
              <p className="text-text-secondary leading-relaxed">
                Monitor your investments in real-time. View detailed performance metrics and balance your portfolio effectively.
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* CTA Section (New Section) */}
      <section className="py-24 bg-gradient-to-b from-primary to-secondary border-t border-border animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-extrabold text-text-primary mb-6">
            Ready to transform your trading?
          </h2>
          <p className="text-lg text-text-secondary mb-10 max-w-2xl mx-auto">
            Join TradeVision today and get access to cutting-edge AI insights for the Colombo Stock Exchange.
          </p>
          <Link
            to="/register"
            className="inline-flex justify-center items-center px-10 py-5 border border-transparent text-lg font-bold rounded-xl text-white bg-accent-green hover:bg-emerald-600 shadow-xl shadow-accent-green/20 transition-all duration-300 hover:-translate-y-1"
          >
            Create Free Account
            <ArrowRight className="ml-2 w-6 h-6" />
          </Link>
        </div>
      </section>
    </div>
  );
};
