/**
 * Knowledge Config Dialog - Create/edit knowledge library
 *
 * Features:
 * - Indexing Options (BM25/Vector/Graph)
 * - Embedding Model selection
 * - Contextual Retrieval toggle
 */
import React from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import {
  X,
  Database,
  FolderOpen,
  ChevronDown,
  ChevronRight,
  Zap,
  Search,
  GitBranch,
  Sparkles,
} from 'lucide-react';
import type { RAGMode } from '@/types/ir';

interface KnowledgeConfigDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

type RetrievalStrategy = 'bm25' | 'vector' | 'graph';
type EmbeddingProvider = 'local' | 'openai' | 'gemini';

export const KnowledgeConfigDialog: React.FC<KnowledgeConfigDialogProps> = ({
  isOpen,
  onClose,
}) => {
  const { createKnowledgeLibrary } = useConsoleStore();

  // Basic fields
  const [name, setName] = React.useState('');
  const [description, setDescription] = React.useState('');
  const [mode, setMode] = React.useState<RAGMode>('auto');
  const [knowledgeDir, setKnowledgeDir] = React.useState('');
  const [isSubmitting, setIsSubmitting] = React.useState(false);

 
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [strategies, setStrategies] = React.useState<RetrievalStrategy[]>(['bm25', 'vector']);
  const [embeddingProvider, setEmbeddingProvider] = React.useState<EmbeddingProvider>('local');
  const [useContextual, setUseContextual] = React.useState(false);

  const toggleStrategy = (strategy: RetrievalStrategy) => {
    setStrategies((prev) => {
      if (prev.includes(strategy)) {
        // Don't allow empty strategies
        if (prev.length === 1) return prev;
        return prev.filter((s) => s !== strategy);
      }
      return [...prev, strategy];
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!name.trim()) {
      return;
    }

    setIsSubmitting(true);

    try {
      createKnowledgeLibrary({
        name: name.trim(),
        description: description.trim(),
        mode,
        knowledge_dir: knowledgeDir.trim() || './knowledge',
       
        strategies: mode === 'indexed' ? strategies : undefined,
        embedding_provider: mode === 'indexed' ? embeddingProvider : undefined,
        contextual_retrieval: mode === 'indexed' ? useContextual : undefined,
      });
      onClose();
      // Reset form
      setName('');
      setDescription('');
      setMode('auto');
      setKnowledgeDir('');
      setStrategies(['bm25', 'vector']);
      setEmbeddingProvider('local');
      setUseContextual(false);
      setShowAdvanced(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  const isIndexedMode = mode === 'indexed';

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onKeyDown={handleKeyDown}
    >
      <div className="bg-gray-800 rounded-lg w-[420px] shadow-xl border border-gray-700 max-h-[90vh] overflow-y-auto custom-scrollbar">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 sticky top-0 bg-gray-800">
          <div className="flex items-center gap-2">
            <Database size={16} className="text-blue-400" />
            <span className="text-sm font-medium text-gray-200">Create Knowledge Library</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
          >
            <X size={16} />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
              placeholder="My Knowledge Library"
              autoFocus
              required
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 focus:border-blue-500 focus:outline-none resize-none"
              placeholder="Brief description of this knowledge library..."
              rows={2}
            />
          </div>

          {/* Mode */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Mode</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: 'auto', label: 'Auto', desc: 'Smart selection' },
                { value: 'agentic', label: 'Agentic', desc: 'LLM-driven' },
                { value: 'indexed', label: 'Indexed', desc: 'Vector search' },
              ].map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setMode(option.value as RAGMode)}
                  className={`
                    p-2 rounded border text-center transition-colors
                    ${
                      mode === option.value
                        ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                        : 'border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-600'
                    }
                  `}
                >
                  <div className="text-xs font-medium">{option.label}</div>
                  <div className="text-[10px] text-gray-500 mt-0.5">{option.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Knowledge Directory */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Knowledge Directory</label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <FolderOpen
                  size={14}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
                />
                <input
                  type="text"
                  value={knowledgeDir}
                  onChange={(e) => setKnowledgeDir(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded pl-9 pr-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 focus:border-blue-500 focus:outline-none"
                  placeholder="./knowledge"
                />
              </div>
            </div>
            <p className="text-[10px] text-gray-500 mt-1">
              Path to the directory containing your documents
            </p>
          </div>

          {/* Advanced Options (Indexed mode only) */}
          {isIndexedMode && (
            <div className="border border-gray-700 rounded-lg overflow-hidden">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between px-3 py-2 bg-gray-900 text-xs text-gray-400 hover:bg-gray-800"
              >
                <span className="flex items-center gap-2">
                  <Sparkles size={12} className="text-purple-400" />
                  Advanced Indexing Options
                </span>
                {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </button>

              {showAdvanced && (
                <div className="p-3 space-y-3 bg-gray-900/50">
                  {/* Retrieval Strategies - compact inline */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-1.5">Strategies</label>
                    <div className="flex gap-1.5">
                      {[
                        { value: 'bm25' as RetrievalStrategy, label: 'BM25', icon: Search },
                        { value: 'vector' as RetrievalStrategy, label: 'Vector', icon: Zap },
                        { value: 'graph' as RetrievalStrategy, label: 'Graph', icon: GitBranch },
                      ].map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => toggleStrategy(option.value)}
                          className={`
                            flex items-center gap-1 px-2 py-1 rounded border text-[11px] transition-colors
                            ${
                              strategies.includes(option.value)
                                ? 'border-green-500 bg-green-500/10 text-green-400'
                                : 'border-gray-700 bg-gray-900 text-gray-500 hover:border-gray-600'
                            }
                          `}
                        >
                          <option.icon size={10} />
                          <span>{option.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Embedding + Contextual in one row */}
                  <div className="flex items-center gap-3">
                    {/* Embedding Provider - compact dropdown style */}
                    <div className="flex-1">
                      <label className="block text-xs text-gray-400 mb-1.5">Embedding</label>
                      <div className="flex gap-1">
                        {[
                          { value: 'local' as EmbeddingProvider, label: 'Local' },
                          { value: 'openai' as EmbeddingProvider, label: 'OpenAI' },
                          { value: 'gemini' as EmbeddingProvider, label: 'Gemini' },
                        ].map((option) => (
                          <button
                            key={option.value}
                            type="button"
                            onClick={() => setEmbeddingProvider(option.value)}
                            className={`
                              px-2 py-1 rounded border text-[11px] transition-colors
                              ${
                                embeddingProvider === option.value
                                  ? 'border-purple-500 bg-purple-500/10 text-purple-400'
                                  : 'border-gray-700 bg-gray-900 text-gray-500 hover:border-gray-600'
                              }
                            `}
                          >
                            {option.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Contextual Retrieval - compact toggle */}
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-gray-400">Contextual</span>
                      <button
                        type="button"
                        onClick={() => setUseContextual(!useContextual)}
                        className={`
                          w-8 h-4 rounded-full transition-colors relative
                          ${useContextual ? 'bg-blue-500' : 'bg-gray-600'}
                        `}
                      >
                        <div
                          className={`
                            absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform
                            ${useContextual ? 'translate-x-4' : 'translate-x-0.5'}
                          `}
                        />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-1.5 rounded text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-700"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim() || isSubmitting}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 rounded text-xs text-white"
            >
              {isSubmitting ? 'Creating...' : 'Create Library'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
