/**
 * SearchModal: global search modal for conversation history.
 *
 * Opens via Header search icon or Cmd+K / Ctrl+K shortcut.
 * Searches conversation titles and message content via backend API.
 * Clicking a result navigates to the conversation (and message).
 *
 * Features:
 * - Date-grouped results (MM/DD headers)
 * - Keyword highlighting in snippets
 * - Sort by created time or last updated
 * - Shows conversation title + matched snippet with context
 */
import React from 'react';
import { Search, X, MessageSquare, FileText, Bookmark } from 'lucide-react';
import { useChatStore } from '@/stores/useChatStore';
import { CenterStage } from '@/components/CenterStage';

const API_BASE = '/api/chat';

interface SearchResult {
  conversation_id: string;
  title: string;
  match_type: 'title' | 'message';
  message_id: string | null;
  role: string | null;
  snippet: string;
  bookmarked?: boolean;
  created_at: number;
  updated_at: number;
}

type SortMode = 'created' | 'updated';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
}

/** Highlight query terms in text by wrapping matches in <mark> */
function highlightText(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text;
  const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'));
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase()
      ? <mark key={i} className="bg-yellow-500/30 text-yellow-200 rounded-sm px-0.5">{part}</mark>
      : part
  );
}

/** Group results by date (MM/DD) using the specified timestamp key */
function groupByDate(results: SearchResult[], tsKey: 'created_at' | 'updated_at' = 'created_at'): Map<string, SearchResult[]> {
  const groups = new Map<string, SearchResult[]>();
  for (const r of results) {
    const ts = r[tsKey] || r.created_at;
    const d = new Date(ts * 1000);
    const key = `${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(r);
  }
  return groups;
}

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = React.useState('');
  const [results, setResults] = React.useState<SearchResult[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [sortMode, setSortMode] = React.useState<SortMode>('created');
  const [resultCount, setResultCount] = React.useState(0);
  const [searchTime, setSearchTime] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const debounceRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const loadConversation = useChatStore((s) => s.loadConversation);

  React.useEffect(() => {
    if (isOpen) {
      setQuery('');
      setResults([]);
      setResultCount(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const doSearch = React.useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      setResultCount(0);
      return;
    }
    setLoading(true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}&limit=50`);
      if (res.ok) {
        const data = await res.json();
        const items = data.results || [];
        setResults(items);
        setResultCount(items.length);
        setSearchTime(Math.round(performance.now() - t0));
      }
    } catch {
      setResults([]);
      setResultCount(0);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(val), 250);
  };

  const handleSelectResult = (result: SearchResult) => {
    loadConversation(result.conversation_id, result.message_id ?? undefined);
    onClose();
  };

  // Sort and group results based on sortMode
  const sortKey = sortMode === 'updated' ? 'updated_at' : 'created_at';
  const sorted = [...results].sort((a, b) => (b[sortKey] || b.created_at) - (a[sortKey] || a.created_at));
  const grouped = groupByDate(sorted, sortKey);

  return (
    <CenterStage isOpen={isOpen} onClose={onClose} size="S" title="Search Conversations">
      {/* Search input */}
      <div className="flex items-center gap-2 pb-3 border-b border-gray-700 -mt-1">
        <Search size={16} className="text-gray-500 shrink-0" />
        <input
          ref={inputRef}
          value={query}
          onChange={handleInputChange}
          placeholder="Search conversations..."
          className="flex-1 bg-transparent text-[13px] text-gray-200 placeholder:text-gray-500 focus:outline-none"
        />
        {query && (
          <button
            onClick={() => { setQuery(''); setResults([]); setResultCount(0); }}
            className="p-0.5 hover:bg-gray-700 rounded text-gray-500 hover:text-gray-300"
            type="button"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {/* Sort toggle + result count */}
      {query && results.length > 0 && (
        <div className="flex items-center justify-between py-1.5 border-b border-gray-700/50">
          <span className="text-[10px] text-gray-500">
            Found {resultCount} results in {(searchTime / 1000).toFixed(3)} seconds
          </span>
          <div className="flex items-center gap-1 bg-gray-800 rounded-md p-0.5">
            <button
              onClick={() => setSortMode('created')}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                sortMode === 'created' ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
              }`}
              type="button"
            >
              Created
            </button>
            <button
              onClick={() => setSortMode('updated')}
              className={`px-2 py-0.5 rounded text-[10px] transition-colors ${
                sortMode === 'updated' ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
              }`}
              type="button"
            >
              Last Updated
            </button>
          </div>
        </div>
      )}

      {/* Results */}
      <div className="mt-2">
        {loading && (
          <div className="py-3 text-[12px] text-gray-500 animate-pulse">Searching...</div>
        )}

        {!loading && query && results.length === 0 && (
          <div className="py-6 text-center text-[12px] text-gray-500">
            No results for &ldquo;{query}&rdquo;
          </div>
        )}

        {!loading && results.length > 0 && (
          <div className="py-1">
            {Array.from(grouped.entries()).map(([dateKey, items]) => (
              <div key={dateKey}>
                {/* Date group header */}
                <div className="pt-3 pb-1">
                  <span className="text-[13px] font-semibold text-gray-400">{dateKey}</span>
                </div>
                {items.map((result, i) => (
                  <button
                    key={`${result.conversation_id}-${result.message_id || 'title'}-${i}`}
                    onClick={() => handleSelectResult(result)}
                    className="w-full text-left px-2 py-2 -mx-2 rounded-md hover:bg-gray-700/50 transition-colors flex items-start gap-3"
                    type="button"
                  >
                    <div className="shrink-0 mt-0.5">
                      {result.bookmarked ? (
                        <Bookmark size={14} className="text-yellow-500 fill-yellow-500" />
                      ) : result.match_type === 'title' ? (
                        <FileText size={14} className="text-gray-500" />
                      ) : (
                        <MessageSquare size={14} className="text-gray-500" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between">
                        <div className="text-[12px] text-gray-200 font-medium truncate">
                          {highlightText(result.title, query)}
                        </div>
                        <span className="text-[10px] text-gray-600 shrink-0 ml-2">
                          {new Date(result.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                      {result.snippet && (
                        <div className="text-[11px] text-gray-500 mt-0.5 line-clamp-2">
                          {result.role && (
                            <span className="text-gray-600 mr-1">[{result.role}]</span>
                          )}
                          {highlightText(result.snippet, query)}
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}

        {!loading && !query && (
          <div className="py-6 text-center text-[12px] text-gray-600">
            Type to search conversation history
          </div>
        )}
      </div>
    </CenterStage>
  );
};
