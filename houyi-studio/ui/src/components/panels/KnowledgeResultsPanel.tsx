/**
 * Knowledge Results Panel - Displays search results from RAG with highlighting
 *
 * Features:
 * - Quality Summary collapsible section
 * - Score distribution visualization
 * - Relevance/Coverage/Confidence assessment
 * - Improvement suggestions
 */
import React, { useMemo, useState } from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import {
  FileText,
  Search,
  Database,
  Copy,
  Check,
  Download,
  Zap,
  Cpu,
  ChevronDown,
  ChevronRight,
  BarChart3,
  Lightbulb,
} from 'lucide-react';

interface KnowledgeResultsPanelProps {
  className?: string;
}

/**
 * Highlights query terms in text by wrapping them in <mark> tags
 */
const highlightText = (text: string, query: string): React.ReactNode => {
  if (!query.trim()) return text;

  // Split query into terms (simple tokenization)
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 2); // Only highlight terms > 2 chars

  if (terms.length === 0) return text;

  // Create regex pattern for all terms
  const pattern = new RegExp(`(${terms.map(escapeRegExp).join('|')})`, 'gi');
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    const isMatch = terms.some((term) => part.toLowerCase() === term);
    if (isMatch) {
      return (
        <mark key={i} className="bg-yellow-500/30 text-yellow-200 rounded px-0.5">
          {part}
        </mark>
      );
    }
    return part;
  });
};

// Escape special regex characters
const escapeRegExp = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');


const QualityIndicator: React.FC<{ level: string; label: string }> = ({ level, label }) => {
  const colors = {
    high: 'text-green-400',
    medium: 'text-yellow-400',
    low: 'text-red-400',
    unknown: 'text-gray-500',
  };
  const icons = {
    high: '🟢',
    medium: '🟡',
    low: '🔴',
    unknown: '⚪',
  };
  return (
    <span className={`flex items-center gap-1 text-[10px] ${colors[level as keyof typeof colors] || colors.unknown}`}>
      <span>{icons[level as keyof typeof icons] || icons.unknown}</span>
      <span>{label}</span>
      <span className="capitalize">{level}</span>
    </span>
  );
};


const ScoreDistributionBar: React.FC<{ distribution: Record<string, number>; total: number }> = ({
  distribution,
  total,
}) => {
  if (total === 0) return null;

  const buckets = [
    { key: '80-100', label: '80-100%', color: 'bg-green-500' },
    { key: '60-80', label: '60-80%', color: 'bg-green-400' },
    { key: '40-60', label: '40-60%', color: 'bg-yellow-500' },
    { key: '20-40', label: '20-40%', color: 'bg-orange-500' },
    { key: '0-20', label: '0-20%', color: 'bg-red-500' },
  ];

  return (
    <div className="space-y-1">
      <div className="text-[10px] text-gray-400 mb-1">Score Distribution</div>
      <div className="flex h-3 rounded overflow-hidden bg-gray-700">
        {buckets.map(({ key, color }) => {
          const count = distribution[key] || 0;
          const width = (count / total) * 100;
          if (width === 0) return null;
          return (
            <div
              key={key}
              className={`${color} transition-all`}
              style={{ width: `${width}%` }}
              title={`${key}: ${count} results`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[9px] text-gray-500">
        {buckets.map(({ key, label }) => {
          const count = distribution[key] || 0;
          if (count === 0) return null;
          return (
            <span key={key}>
              {label}: {count}
            </span>
          );
        })}
      </div>
    </div>
  );
};

export const KnowledgeResultsPanel: React.FC<KnowledgeResultsPanelProps> = ({ className }) => {
  const {
    knowledgeSearchResults,
    knowledgeSearchQuery,
    knowledgeSearchModeUsed,
    knowledgeSearchStrategiesUsed,
    knowledgeSearchQuality,
    isSearchingKnowledge,
    selectedLibraryId,
    knowledgeLibraries,
  } = useConsoleStore();

  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [exportedFormat, setExportedFormat] = useState<string | null>(null);
  const [showQuality, setShowQuality] = useState(true);

  const selectedLibrary = knowledgeLibraries.find((lib) => lib.library_id === selectedLibraryId);

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1) + '%';
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.5) return 'text-yellow-400';
    return 'text-gray-400';
  };

  // Extract filename from full path
  const getFileName = (filePath: string) => {
    const parts = filePath.split('/');
    return parts[parts.length - 1] || filePath;
  };

  const handleCopyContent = async (content: string, id: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // Ignore clipboard errors
    }
  };

  const handleExportJSON = async () => {
    const exportData = {
      query: knowledgeSearchQuery,
      library: selectedLibrary?.name || 'unknown',
      exported_at: new Date().toISOString(),
      quality: knowledgeSearchQuality,
      results: knowledgeSearchResults.map((r) => ({
        content: r.content,
        score: r.score,
        source: r.source,
        metadata: r.metadata,
      })),
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(exportData, null, 2));
      setExportedFormat('json');
      setTimeout(() => setExportedFormat(null), 2000);
    } catch {
      // Ignore clipboard errors
    }
  };

  const handleExportMarkdown = async () => {
    const lines = [
      `# Search Results: "${knowledgeSearchQuery}"`,
      `Library: ${selectedLibrary?.name || 'unknown'}`,
      `Results: ${knowledgeSearchResults.length}`,
      '',
    ];
    knowledgeSearchResults.forEach((r, i) => {
      lines.push(`## Result ${i + 1} (${formatScore(r.score)})`);
      if (r.source?.file_path) {
        lines.push(`Source: ${r.source.file_path}`);
      }
      lines.push('');
      lines.push(r.content);
      lines.push('');
      lines.push('---');
      lines.push('');
    });
    try {
      await navigator.clipboard.writeText(lines.join('\n'));
      setExportedFormat('md');
      setTimeout(() => setExportedFormat(null), 2000);
    } catch {
      // Ignore clipboard errors
    }
  };

  // Memoize highlighted results for performance
  const highlightedResults = useMemo(() => {
    return knowledgeSearchResults.map((result) => ({
      ...result,
      highlightedContent: highlightText(
        result.content.length > 500 ? result.content.slice(0, 500) + '...' : result.content,
        knowledgeSearchQuery
      ),
    }));
  }, [knowledgeSearchResults, knowledgeSearchQuery]);

  // Calculate score statistics (fallback if no quality from backend)
  const scoreStats = useMemo(() => {
    if (knowledgeSearchResults.length === 0) return null;

    // Use backend quality if available
    if (knowledgeSearchQuality) {
      return {
        min: knowledgeSearchQuality.min_score,
        max: knowledgeSearchQuality.max_score,
        avg: knowledgeSearchQuality.avg_score,
        distribution: knowledgeSearchQuality.score_distribution || {},
        total: knowledgeSearchQuality.total_count,
        relevance: knowledgeSearchQuality.relevance,
        coverage: knowledgeSearchQuality.coverage,
        confidence: knowledgeSearchQuality.confidence_level,
        suggestion: knowledgeSearchQuality.suggestion,
        aboveThreshold: knowledgeSearchQuality.above_threshold_count,
      };
    }

    // Fallback: calculate locally
    const scores = knowledgeSearchResults.map((r) => r.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;

    // Score distribution buckets
    const distribution: Record<string, number> = {
      '80-100': 0,
      '60-80': 0,
      '40-60': 0,
      '20-40': 0,
      '0-20': 0,
    };
    let aboveThreshold = 0;
    scores.forEach((s) => {
      if (s >= 0.8) distribution['80-100']++;
      else if (s >= 0.6) distribution['60-80']++;
      else if (s >= 0.4) distribution['40-60']++;
      else if (s >= 0.2) distribution['20-40']++;
      else distribution['0-20']++;
      if (s >= 0.6) aboveThreshold++;
    });

    // Assess relevance
    const relevance = avg >= 0.7 ? 'high' : avg >= 0.5 ? 'medium' : 'low';
    const aboveRatio = aboveThreshold / scores.length;
    const coverage = aboveRatio >= 0.6 ? 'high' : aboveRatio >= 0.3 ? 'medium' : 'low';
    const scoreRange = max - min;
    const confidence =
      scoreRange < 0.3 && avg >= 0.6 ? 'high' : scoreRange < 0.5 ? 'medium' : 'low';

    return {
      min,
      max,
      avg,
      distribution,
      total: scores.length,
      relevance,
      coverage,
      confidence,
      suggestion: avg < 0.5 ? 'Consider refining your query or expanding the knowledge base' : null,
      aboveThreshold,
    };
  }, [knowledgeSearchResults, knowledgeSearchQuality]);

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'agentic':
        return <Zap size={10} className="text-blue-400" />;
      case 'indexed':
        return <Cpu size={10} className="text-green-400" />;
      default:
        return <Database size={10} className="text-purple-400" />;
    }
  };

  const getModeLabel = (mode: string) => {
    switch (mode) {
      case 'agentic':
        return 'Agentic';
      case 'indexed':
        return 'Indexed';
      case 'auto':
        return 'Auto';
      default:
        return mode || 'Unknown';
    }
  };

  if (isSearchingKnowledge) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center">
          <div className="animate-pulse flex items-center justify-center gap-2 text-blue-400">
            <Search size={16} className="animate-bounce" />
            <span className="text-sm">Searching knowledge base...</span>
          </div>
          {knowledgeSearchQuery && (
            <div className="text-xs text-gray-500 mt-2">Query: "{knowledgeSearchQuery}"</div>
          )}
        </div>
      </div>
    );
  }

  if (!knowledgeSearchQuery) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center text-gray-500">
          <Database size={32} className="mx-auto mb-2 opacity-50" />
          <div className="text-sm">No search query</div>
          <div className="text-xs mt-1">Use the Knowledge panel to search your libraries</div>
        </div>
      </div>
    );
  }

  if (knowledgeSearchResults.length === 0) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <div className="text-center text-gray-500">
          <Search size={32} className="mx-auto mb-2 opacity-50" />
          <div className="text-sm">No results found</div>
          <div className="text-xs mt-1">Try different keywords or check your knowledge library</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700 bg-gray-800/50">
        <div className="flex items-center gap-2">
          <Search size={14} className="text-gray-400" />
          <span className="text-xs text-gray-300">
            {knowledgeSearchResults.length} results for "{knowledgeSearchQuery}"
          </span>
        </div>
        <div className="flex items-center gap-3">
          {/* Mode indicator */}
          {knowledgeSearchModeUsed && (
            <div className="flex items-center gap-1 text-[10px] text-gray-400 bg-gray-700/50 px-2 py-0.5 rounded">
              {getModeIcon(knowledgeSearchModeUsed)}
              <span>{getModeLabel(knowledgeSearchModeUsed)}</span>
            </div>
          )}
          {/* Strategies indicator */}
          {knowledgeSearchStrategiesUsed && knowledgeSearchStrategiesUsed.length > 0 && (
            <div className="flex items-center gap-1 text-[10px] text-gray-500 bg-gray-700/30 px-2 py-0.5 rounded">
              <span>{knowledgeSearchStrategiesUsed.map(s => s.toUpperCase()).join('+')}</span>
            </div>
          )}
          {selectedLibrary && (
            <div className="flex items-center gap-1 text-[10px] text-gray-500">
              <Database size={10} />
              <span>{selectedLibrary.name}</span>
            </div>
          )}
        </div>
      </div>

      {/* Quality Summary (collapsible) */}
      {scoreStats && (
        <div className="border-b border-gray-700">
          <button
            onClick={() => setShowQuality(!showQuality)}
            className="w-full flex items-center justify-between px-4 py-2 text-xs text-gray-400 hover:bg-gray-800/50"
          >
            <span className="flex items-center gap-2">
              <BarChart3 size={12} className="text-purple-400" />
              Quality Summary
            </span>
            <div className="flex items-center gap-3">
              {/* Quick stats always visible */}
              <span className="text-[10px] text-gray-500">
                {formatScore(scoreStats.min)} - {formatScore(scoreStats.max)} (avg{' '}
                {formatScore(scoreStats.avg)})
              </span>
              <span className={`text-[10px] ${getScoreColor(scoreStats.avg)}`}>
                {scoreStats.aboveThreshold}/{scoreStats.total} &gt; 60%
              </span>
              {showQuality ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </div>
          </button>

          {showQuality && (
            <div className="px-4 py-3 bg-gray-900/30 space-y-3">
              {/* Score Distribution */}
              <ScoreDistributionBar
                distribution={scoreStats.distribution}
                total={scoreStats.total}
              />

              {/* Assessment indicators */}
              <div className="flex items-center gap-4 pt-2 border-t border-gray-700/50">
                <QualityIndicator level={scoreStats.relevance} label="Relevance" />
                <QualityIndicator level={scoreStats.coverage} label="Coverage" />
                <QualityIndicator level={scoreStats.confidence} label="Confidence" />
              </div>

              {/* Suggestion */}
              {scoreStats.suggestion && (
                <div className="flex items-start gap-2 p-2 bg-blue-500/10 rounded border border-blue-500/20 text-[10px] text-blue-300">
                  <Lightbulb size={12} className="flex-shrink-0 mt-0.5" />
                  <span>{scoreStats.suggestion}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Results List */}
      <div className="flex-1 overflow-y-auto">
        {highlightedResults.map((result, index: number) => (
          <div
            key={result.chunk_id || index}
            className="border-b border-gray-700/50 hover:bg-gray-800/30 transition-colors group"
          >
            <div className="p-3">
              {/* Source Info */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-xs">
                  <FileText size={12} className="text-gray-500 flex-shrink-0" />
                  {result.source ? (
                    <span
                      className="text-gray-400 truncate max-w-[300px] cursor-default"
                      title={result.source.file_path}
                    >
                      {getFileName(result.source.file_path)}
                      {result.source.location && (
                        <span className="text-gray-600"> : {result.source.location}</span>
                      )}
                    </span>
                  ) : (
                    <span className="text-gray-500">Unknown source</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() =>
                      handleCopyContent(result.content, result.chunk_id || String(index))
                    }
                    className="p-1 hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Copy content"
                  >
                    {copiedId === (result.chunk_id || String(index)) ? (
                      <Check size={12} className="text-green-400" />
                    ) : (
                      <Copy size={12} />
                    )}
                  </button>
                  <div className={`text-xs font-mono ${getScoreColor(result.score)}`}>
                    {formatScore(result.score)}
                  </div>
                </div>
              </div>

              {/* Content with highlighting */}
              <div className="text-xs text-gray-300 leading-relaxed">
                {result.highlightedContent}
              </div>

              {/* Snippet highlight if available */}
              {result.source?.snippet && result.source.snippet !== result.content && (
                <div className="mt-2 p-2 bg-gray-900/50 rounded border-l-2 border-blue-500">
                  <div className="text-[10px] text-gray-500 mb-1">Matched snippet:</div>
                  <div className="text-xs text-gray-400 italic">
                    "{highlightText(result.source.snippet, knowledgeSearchQuery)}"
                  </div>
                </div>
              )}

              {/* Metadata tags */}
              {result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {Object.entries(result.metadata)
                    .filter(([key]) => !['chunk_id', 'doc_id'].includes(key))
                    .slice(0, 3)
                    .map(([key, value]) => (
                      <span
                        key={key}
                        className="px-1.5 py-0.5 bg-gray-700 rounded text-[10px] text-gray-400"
                      >
                        {key}: {String(value).slice(0, 20)}
                      </span>
                    ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Footer with quick stats and export */}
      <div className="px-4 py-2 border-t border-gray-700 bg-gray-800/50 text-[10px] text-gray-500 flex items-center justify-between">
        <span>
          Showing {knowledgeSearchResults.length} results
          {knowledgeSearchResults.length > 0 && (
            <>
              {' '}
              &middot; Best: {formatScore(Math.max(...knowledgeSearchResults.map((r) => r.score)))}
            </>
          )}
        </span>
        <div className="flex items-center gap-1">
          {exportedFormat && (
            <span className="text-green-400 mr-1">
              <Check size={10} className="inline" /> Copied {exportedFormat.toUpperCase()}
            </span>
          )}
          <button
            onClick={handleExportMarkdown}
            className="px-2 py-0.5 hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300 transition-colors"
            title="Copy as Markdown"
          >
            MD
          </button>
          <button
            onClick={handleExportJSON}
            className="px-2 py-0.5 hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300 transition-colors"
            title="Copy as JSON"
          >
            <Download size={10} className="inline mr-0.5" />
            JSON
          </button>
        </div>
      </div>
    </div>
  );
};
