/**
 * Chunk View Dialog - Shows chunks for a document with content preview
 */
import React, { useEffect, useState } from 'react';
import { X, RefreshCw, Loader2, FileText, Hash, ChevronDown, ChevronRight } from 'lucide-react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import type { KnowledgeDocument } from '@/types/ir';

interface ChunkViewDialogProps {
  isOpen: boolean;
  libraryId: string;
  document: KnowledgeDocument | null;
  onClose: () => void;
}

export const ChunkViewDialog: React.FC<ChunkViewDialogProps> = ({
  isOpen,
  libraryId,
  document,
  onClose,
}) => {
  const {
    chunks,
    isLoadingChunks,
    requestChunks,
    clearChunks,
  } = useConsoleStore();

  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');

  // Load chunks when dialog opens
  useEffect(() => {
    if (isOpen && document && libraryId) {
      requestChunks(libraryId, document.doc_id);
    }
    return () => {
      clearChunks();
    };
  }, [isOpen, document, libraryId, requestChunks, clearChunks]);

  if (!isOpen || !document) return null;

  // Filter chunks by search query
  const filteredChunks = chunks.filter(chunk =>
    chunk.content.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleChunk = (chunkId: string) => {
    const newExpanded = new Set(expandedChunks);
    if (newExpanded.has(chunkId)) {
      newExpanded.delete(chunkId);
    } else {
      newExpanded.add(chunkId);
    }
    setExpandedChunks(newExpanded);
  };

  const expandAll = () => {
    setExpandedChunks(new Set(filteredChunks.map(c => c.chunk_id)));
  };

  const collapseAll = () => {
    setExpandedChunks(new Set());
  };

  const truncateContent = (content: string, maxLength: number = 150) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  const highlightSearch = (text: string) => {
    if (!searchQuery) return text;
    const parts = text.split(new RegExp(`(${searchQuery})`, 'gi'));
    return parts.map((part, i) =>
      part.toLowerCase() === searchQuery.toLowerCase() ? (
        <mark key={i} className="bg-yellow-500/30 text-yellow-200">{part}</mark>
      ) : (
        part
      )
    );
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-50"
        onClick={onClose}
      />

      {/* Dialog */}
      <div className="fixed inset-0 flex items-center justify-center z-50 pointer-events-none">
        <div
          className="bg-gray-800 border border-gray-600 rounded-lg shadow-xl w-[800px] max-h-[85vh] flex flex-col pointer-events-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <div>
              <h3 className="text-sm font-semibold text-gray-200">Document Chunks</h3>
              <div className="flex items-center gap-2 mt-1">
                <FileText size={12} className="text-gray-400" />
                <span className="text-xs text-gray-400">{document.file_name}</span>
                <span className="text-xs text-gray-500">·</span>
                <span className="text-xs text-gray-500">{chunks.length} chunks</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => requestChunks(libraryId, document.doc_id)}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
                title="Refresh"
              >
                <RefreshCw size={14} className={isLoadingChunks ? 'animate-spin' : ''} />
              </button>
              <button
                onClick={onClose}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Toolbar */}
          <div className="p-3 border-b border-gray-700 flex items-center gap-3">
            <div className="relative flex-1">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search in chunks..."
                className="w-full pl-3 pr-3 py-2 bg-gray-900 border border-gray-700 rounded text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="flex gap-1">
              <button
                onClick={expandAll}
                className="px-2 py-1.5 text-[10px] bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
              >
                Expand All
              </button>
              <button
                onClick={collapseAll}
                className="px-2 py-1.5 text-[10px] bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
              >
                Collapse All
              </button>
            </div>
          </div>

          {/* Chunk List */}
          <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
            {isLoadingChunks ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 size={24} className="text-gray-400 animate-spin" />
              </div>
            ) : filteredChunks.length === 0 ? (
              <div className="text-center py-8">
                <Hash size={32} className="text-gray-600 mx-auto mb-2" />
                <p className="text-xs text-gray-500">
                  {searchQuery ? 'No matching chunks' : 'No chunks found'}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {filteredChunks.map((chunk) => {
                  const isExpanded = expandedChunks.has(chunk.chunk_id);
                  return (
                    <div
                      key={chunk.chunk_id}
                      className="bg-gray-900 border border-gray-700 rounded overflow-hidden"
                    >
                      {/* Chunk Header */}
                      <div
                        className="flex items-center gap-2 p-2 cursor-pointer hover:bg-gray-800"
                        onClick={() => toggleChunk(chunk.chunk_id)}
                      >
                        {isExpanded ? (
                          <ChevronDown size={14} className="text-gray-400" />
                        ) : (
                          <ChevronRight size={14} className="text-gray-400" />
                        )}
                        <span className="text-[10px] font-mono text-gray-500 w-12">
                          #{chunk.chunk_index}
                        </span>
                        <span className="text-xs text-gray-300 flex-1 truncate">
                          {isExpanded ? '' : highlightSearch(truncateContent(chunk.content))}
                        </span>
                        <span className="text-[10px] text-gray-500">
                          {chunk.content.length} chars
                        </span>
                      </div>

                      {/* Chunk Content (expanded) */}
                      {isExpanded && (
                        <div className="border-t border-gray-700 p-3">
                          <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
                            {highlightSearch(chunk.content)}
                          </pre>
                          <div className="flex items-center gap-4 mt-3 pt-2 border-t border-gray-700 text-[10px] text-gray-500">
                            <span>Chars: {chunk.start_char} - {chunk.end_char}</span>
                            <span>Length: {chunk.content.length}</span>
                            {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                              <span>Metadata: {JSON.stringify(chunk.metadata)}</span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-3 border-t border-gray-700 flex justify-between items-center">
            <div className="text-[10px] text-gray-500">
              {filteredChunks.length} of {chunks.length} chunks
              {searchQuery && ` matching "${searchQuery}"`}
            </div>
            <button
              onClick={onClose}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-200"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </>
  );
};
