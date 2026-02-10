/**
 * Knowledge Search - Search input for knowledge base with history and advanced options
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { Search, X, Loader2, Clock, Settings2 } from 'lucide-react';
import type { RAGMode } from '@/types/ir';

const SEARCH_HISTORY_KEY = 'houyi_knowledge_search_history';
const MAX_HISTORY_ITEMS = 10;

interface KnowledgeSearchProps {
  onSearch?: (query: string) => void;
}

export const KnowledgeSearch: React.FC<KnowledgeSearchProps> = ({ onSearch }) => {
  const {
    selectedLibraryId,
    knowledgeSearchQuery,
    isSearchingKnowledge,
    searchKnowledge,
    clearKnowledgeSearch,
    knowledgeLibraries,
  } = useConsoleStore();

  const [query, setQuery] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [modeOverride, setModeOverride] = useState<RAGMode | null>(null);  // null = use library config
  const [topK, setTopK] = useState(10);
  const inputRef = useRef<HTMLInputElement>(null);
  const historyRef = useRef<HTMLDivElement>(null);

  // Get selected library for mode hint
  const selectedLibrary = knowledgeLibraries.find(
    (lib) => lib.library_id === selectedLibraryId
  );

  // Effective mode: user override or library config
  const effectiveMode = modeOverride || selectedLibrary?.mode || 'auto';

  // Load search history from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(SEARCH_HISTORY_KEY);
      if (stored) {
        setSearchHistory(JSON.parse(stored));
      }
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Sync local state with store
  useEffect(() => {
    setQuery(knowledgeSearchQuery);
  }, [knowledgeSearchQuery]);

  // Close history dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const addToHistory = useCallback((searchQuery: string) => {
    setSearchHistory((prev) => {
      const filtered = prev.filter((q) => q !== searchQuery);
      const updated = [searchQuery, ...filtered].slice(0, MAX_HISTORY_ITEMS);
      try {
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(updated));
      } catch {
        // Ignore localStorage errors
      }
      return updated;
    });
  }, []);

  const removeFromHistory = useCallback((searchQuery: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSearchHistory((prev) => {
      const updated = prev.filter((q) => q !== searchQuery);
      try {
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(updated));
      } catch {
        // Ignore localStorage errors
      }
      return updated;
    });
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      addToHistory(query.trim());
      // Only pass mode if user explicitly overrode it
      searchKnowledge(query.trim(), selectedLibraryId || undefined, modeOverride || undefined, topK);
      onSearch?.(query.trim());
      setShowHistory(false);
    }
  };

  const handleHistorySelect = (historyQuery: string) => {
    setQuery(historyQuery);
    addToHistory(historyQuery);
    searchKnowledge(historyQuery, selectedLibraryId || undefined, modeOverride || undefined, topK);
    onSearch?.(historyQuery);
    setShowHistory(false);
  };

  const handleClear = () => {
    setQuery('');
    clearKnowledgeSearch();
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      if (showHistory) {
        setShowHistory(false);
      } else {
        handleClear();
      }
    } else if (e.key === 'ArrowDown' && searchHistory.length > 0) {
      setShowHistory(true);
    }
  };

  const filteredHistory = searchHistory.filter((h) =>
    h.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="p-3 border-t border-gray-700">
      <form onSubmit={handleSubmit}>
        <div className="relative" ref={historyRef}>
          {isSearchingKnowledge ? (
            <Loader2
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-blue-400 animate-spin"
            />
          ) : (
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          )}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => searchHistory.length > 0 && setShowHistory(true)}
            className="w-full bg-gray-900 border border-gray-700 rounded-lg pl-9 pr-16 py-2 text-xs text-gray-200 placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
            placeholder={selectedLibraryId ? 'Search knowledge...' : 'Select a library first'}
            disabled={!selectedLibraryId || isSearchingKnowledge}
          />
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {query && (
              <button
                type="button"
                onClick={handleClear}
                className="p-1 hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300"
              >
                <X size={12} />
              </button>
            )}
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={`p-1 hover:bg-gray-700 rounded transition-colors ${
                showAdvanced ? 'text-blue-400' : 'text-gray-500 hover:text-gray-300'
              }`}
              title="Advanced options"
            >
              <Settings2 size={12} />
            </button>
          </div>

          {/* Search History Dropdown */}
          {showHistory && filteredHistory.length > 0 && (
            <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg overflow-hidden">
              <div className="px-2 py-1 text-[10px] text-gray-500 border-b border-gray-700 flex items-center gap-1">
                <Clock size={10} />
                <span>Recent searches</span>
              </div>
              {filteredHistory.map((historyQuery, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleHistorySelect(historyQuery)}
                  className="w-full px-3 py-2 text-xs text-left text-gray-300 hover:bg-gray-700 flex items-center justify-between group"
                >
                  <span className="truncate">{historyQuery}</span>
                  <button
                    type="button"
                    onClick={(e) => removeFromHistory(historyQuery, e)}
                    className="p-0.5 hover:bg-gray-600 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X size={10} className="text-gray-500" />
                  </button>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="mt-2 p-2 bg-gray-800/50 rounded-lg border border-gray-700 space-y-2">
            {/* Mode Selection */}
            <div>
              <label className="block text-[10px] text-gray-500 mb-1">
                Search Mode {selectedLibrary && `(library: ${selectedLibrary.mode})`}
              </label>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => setModeOverride(null)}
                  className={`flex-1 px-2 py-1 rounded text-[10px] transition-colors ${
                    modeOverride === null
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                >
                  Default
                </button>
                {(['agentic', 'indexed'] as RAGMode[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setModeOverride(m)}
                    className={`flex-1 px-2 py-1 rounded text-[10px] transition-colors ${
                      modeOverride === m
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                    }`}
                  >
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
              <div className="text-[10px] text-gray-500 mt-1">
                Will use: {effectiveMode}
              </div>
            </div>

            {/* Top K Slider */}
            <div>
              <label className="block text-[10px] text-gray-500 mb-1">
                Results: {topK}
              </label>
              <input
                type="range"
                min="1"
                max="50"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
              />
              <div className="flex justify-between text-[9px] text-gray-600">
                <span>1</span>
                <span>50</span>
              </div>
            </div>
          </div>
        )}

        {!selectedLibraryId && (
          <p className="text-[10px] text-gray-500 mt-1">
            Select a knowledge library above to search
          </p>
        )}
      </form>
    </div>
  );
};
