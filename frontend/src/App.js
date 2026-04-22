import React, { useState } from 'react';
import { 
  Loader2, 
  Sparkles, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  ArrowRight
} from 'lucide-react';

const App = () => {
  const [text, setText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Use relative URL for same-origin requests
  const API_URL = '';

  const checkFacts = async () => {
    if (!text.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to connect to fact-checking service. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (type) => {
    const examples = {
      mixed: `The speed of light is 300000 kilometers per second.
Humans only use 10 percent of their brain.
The Amazon rainforest is in Africa.`,
      basics: `The Eiffel Tower is located in Paris.
The Earth revolves around the Sun.
Water boils at 100 degrees Celsius.`,
      myths: `Drinking coffee stunts your growth.
Bats are blind.
The Great Wall of China is visible from space.`
    };
    setText(examples[type] || '');
    setResults(null);
  };

  const getVerdictIcon = (verdict) => {
    switch(verdict?.toUpperCase()) {
      case 'SUPPORTED': 
        return <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />;
      case 'REFUTED': 
        return <XCircle className="w-5 h-5 text-rose-400 flex-shrink-0" />;
      default: 
        return <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0" />;
    }
  };

  const getVerdictColor = (verdict) => {
    switch(verdict?.toUpperCase()) {
      case 'SUPPORTED': 
        return 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300';
      case 'REFUTED': 
        return 'bg-rose-500/10 border-rose-500/20 text-rose-300';
      default: 
        return 'bg-amber-500/10 border-amber-500/20 text-amber-300';
    }
  };

  const getVerdictLabel = (verdict) => {
    switch(verdict?.toUpperCase()) {
      case 'SUPPORTED': return 'Verified';
      case 'REFUTED': return 'False';
      case 'NOT ENOUGH INFO': return 'Uncertain';
      default: return verdict;
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white overflow-x-hidden">
      {/* Background Effects */}
      <div className="fixed inset-0 bg-gradient-to-br from-violet-950/30 via-[#0a0a0f] to-cyan-950/20 pointer-events-none" />
      <div className="fixed top-0 left-1/4 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed bottom-0 right-1/4 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />
      
      {/* Subtle Grid Pattern */}
      <div 
        className="fixed inset-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }}
      />
      
      <div className="relative z-10 max-w-4xl mx-auto px-6 py-16 md:py-20">
        {/* Live Badge */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm text-zinc-400 backdrop-blur-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" />
            Live · Wikipedia + NLI
          </div>
        </div>

        {/* Header */}
        <div className="text-center mb-12 md:mb-16">
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white via-violet-200 to-violet-400 bg-clip-text text-transparent leading-tight">
            Detect hallucinations<br />before they spread.
          </h1>
          <p className="text-base md:text-lg text-zinc-400 max-w-2xl mx-auto leading-relaxed px-4">
            Paste any text. We'll extract factual claims, retrieve evidence from 
            Wikipedia, and verify each one with a natural-language inference model.
          </p>
        </div>

        {/* Input Card */}
        <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.06] rounded-2xl p-6 mb-8 shadow-2xl shadow-black/50">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to fact-check..."
            className="w-full h-40 bg-transparent text-zinc-200 placeholder-zinc-600 resize-none focus:outline-none text-base leading-relaxed"
          />
          
          {/* Buttons Row */}
          <div className="flex flex-wrap items-center justify-between gap-4 mt-4 pt-4 border-t border-white/[0.06]">
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => loadExample('mixed')}
                className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-zinc-300 text-sm transition-all duration-200 border border-white/[0.06]"
              >
                Try mixed claims
              </button>
              <button
                onClick={() => loadExample('basics')}
                className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-zinc-300 text-sm transition-all duration-200 border border-white/[0.06]"
              >
                Try basics
              </button>
              <button
                onClick={() => loadExample('myths')}
                className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-zinc-300 text-sm transition-all duration-200 border border-white/[0.06]"
              >
                Try myths
              </button>
              <button
                onClick={() => { setText(''); setResults(null); }}
                className="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-500 hover:text-zinc-400 text-sm transition-all duration-200 border border-white/[0.06]"
              >
                Clear
              </button>
            </div>
            
            <button
              onClick={checkFacts}
              disabled={loading || !text.trim()}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-violet-500 hover:bg-violet-400 disabled:bg-violet-500/50 disabled:cursor-not-allowed text-white font-medium transition-all duration-200 shadow-lg shadow-violet-500/25 hover:shadow-violet-500/40"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Fact-check
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm animate-fade-in">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="animate-fade-in">
            <div className="flex items-center gap-3 mb-6">
              <Sparkles className="w-5 h-5 text-violet-400" />
              <h2 className="text-xl font-semibold text-zinc-200">
                Results 
                <span className="text-zinc-500 font-normal ml-2">
                  ({results.length} {results.length === 1 ? 'claim' : 'claims'})
                </span>
              </h2>
            </div>
            
            {results.length === 0 ? (
              <div className="p-8 text-center text-zinc-500 bg-white/[0.02] rounded-xl border border-white/[0.06] border-dashed">
                <AlertCircle className="w-8 h-8 mx-auto mb-3 text-zinc-600" />
                No factual claims detected in the input.
                <p className="text-sm text-zinc-600 mt-2">
                  Try entering statements with facts, dates, or locations.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {results.map((result, idx) => (
                  <div
                    key={idx}
                    className={`p-5 rounded-xl border backdrop-blur-sm transition-all duration-300 hover:border-opacity-40 ${getVerdictColor(result.final_verdict)}`}
                  >
                    {/* Claim Header */}
                    <div className="flex items-start gap-3 mb-4">
                      {getVerdictIcon(result.final_verdict)}
                      <div className="flex-1 min-w-0">
                        <p className="text-zinc-100 font-medium mb-2 leading-relaxed">
                          {result.claim}
                        </p>
                        <div className="flex flex-wrap items-center gap-3 text-sm">
                          <span className={`font-semibold px-2 py-0.5 rounded bg-white/10`}>
                            {getVerdictLabel(result.final_verdict)}
                          </span>
                          <span className="text-zinc-500">·</span>
                          <span className="text-zinc-400">
                            {Math.round((result.confidence || 0) * 100)}% confidence
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Note */}
                    {result.note && (
                      <p className="text-sm text-zinc-400 italic mb-3 pl-8">
                        {result.note}
                      </p>
                    )}
                    
                    {/* Evidence */}
                    {result.evidence && result.evidence.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <p className="text-xs text-zinc-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                          <ArrowRight className="w-3 h-3" />
                          Evidence from Wikipedia
                        </p>
                        <ul className="space-y-2">
                          {result.evidence.slice(0, 3).map((ev, i) => (
                            <li 
                              key={i} 
                              className="text-sm text-zinc-400 pl-3 border-l-2 border-violet-500/30 leading-relaxed"
                            >
                              {ev.length > 180 ? ev.slice(0, 180) + '...' : ev}
                            </li>
                          ))}
                        </ul>
                        {result.evidence.length > 3 && (
                          <p className="text-xs text-zinc-600 mt-2 pl-3">
                            +{result.evidence.length - 3} more evidence sentences
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="mt-20 pt-8 border-t border-white/[0.06] text-center">
          <p className="text-sm text-zinc-600 flex items-center justify-center gap-2 flex-wrap">
            <span>Powered by</span>
            <span className="px-2 py-1 rounded bg-white/5 text-zinc-400 text-xs border border-white/5">spaCy</span>
            <span className="text-zinc-700">·</span>
            <span className="px-2 py-1 rounded bg-white/5 text-zinc-400 text-xs border border-white/5">BAAI/bge-base-en-v1.5</span>
            <span className="text-zinc-700">·</span>
            <span className="px-2 py-1 rounded bg-white/5 text-zinc-400 text-xs border border-white/5">nli-roberta-base</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default App;
