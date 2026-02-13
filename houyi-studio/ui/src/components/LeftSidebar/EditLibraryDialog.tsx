/**
 * Edit Knowledge Library Dialog - Enhanced with chunking and search settings
 */
import React, { useState, useEffect } from 'react';
import { X, ChevronDown, ChevronRight, Settings2, Layers, Search } from 'lucide-react';
import type { KnowledgeLibrary, RAGMode } from '@/types/ir';

interface LibrarySettings {
  chunk_size: number;
  chunk_overlap: number;
  chunking_strategy: 'recursive' | 'sentence' | 'paragraph';
  top_k: number;
  score_threshold: number;
  enable_rerank: boolean;
}

interface EditLibraryDialogProps {
  isOpen: boolean;
  library: KnowledgeLibrary | null;
  onSave: (libraryId: string, updates: Record<string, any>) => void;
  onCancel: () => void;
}

const DEFAULT_SETTINGS: LibrarySettings = {
  chunk_size: 512,
  chunk_overlap: 50,
  chunking_strategy: 'recursive',
  top_k: 10,
  score_threshold: 0.5,
  enable_rerank: true,
};

export const EditLibraryDialog: React.FC<EditLibraryDialogProps> = ({
  isOpen,
  library,
  onSave,
  onCancel,
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [mode, setMode] = useState<RAGMode>('auto');

  // Advanced settings
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [settings, setSettings] = useState<LibrarySettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    if (library) {
      setName(library.name);
      setDescription(library.description || '');
      setMode(library.mode);

      // Load settings from metadata if available
      const meta = library.metadata || {};
      setSettings({
        chunk_size: meta.chunk_size ?? DEFAULT_SETTINGS.chunk_size,
        chunk_overlap: meta.chunk_overlap ?? DEFAULT_SETTINGS.chunk_overlap,
        chunking_strategy: meta.chunking_strategy ?? DEFAULT_SETTINGS.chunking_strategy,
        top_k: meta.top_k ?? DEFAULT_SETTINGS.top_k,
        score_threshold: meta.score_threshold ?? DEFAULT_SETTINGS.score_threshold,
        enable_rerank: meta.enable_rerank ?? DEFAULT_SETTINGS.enable_rerank,
      });
    }
  }, [library]);

  if (!isOpen || !library) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    const updates: Record<string, any> = {};
    if (name !== library.name) updates.name = name;
    if (description !== (library.description || '')) updates.description = description;
    if (mode !== library.mode) updates.mode = mode;

    // Include settings in metadata
    const oldMeta = library.metadata || {};
    const newMeta = {
      ...oldMeta,
      chunk_size: settings.chunk_size,
      chunk_overlap: settings.chunk_overlap,
      chunking_strategy: settings.chunking_strategy,
      top_k: settings.top_k,
      score_threshold: settings.score_threshold,
      enable_rerank: settings.enable_rerank,
    };

    // Check if metadata changed
    if (JSON.stringify(newMeta) !== JSON.stringify(oldMeta)) {
      updates.metadata = newMeta;
    }

    if (Object.keys(updates).length > 0) {
      onSave(library.library_id, updates);
    } else {
      onCancel();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onCancel();
    }
  };

  const updateSettings = (key: keyof LibrarySettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onKeyDown={handleKeyDown}
    >
      <div className="bg-gray-800 rounded-lg shadow-xl w-full max-w-lg mx-4 max-h-[85vh] overflow-y-auto custom-scrollbar">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 sticky top-0 bg-gray-800">
          <h3 className="text-sm font-semibold text-gray-50">Edit Knowledge Library</h3>
          <button
            onClick={onCancel}
            className="text-gray-400 hover:text-gray-50 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-xs text-gray-50 placeholder-gray-500 focus:outline-none focus:border-blue-500"
              autoFocus
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description"
              className="w-full h-20 px-3 py-2 bg-gray-900 border border-gray-700 rounded text-xs text-gray-50 placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
            />
          </div>

          {/* Mode */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Mode</label>
            <div className="flex gap-2">
              {(['auto', 'agentic', 'indexed'] as RAGMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`flex-1 px-3 py-2 rounded text-xs transition-colors ${
                    mode === m
                      ? 'bg-blue-600 text-gray-50'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-xs text-gray-400 hover:text-gray-50 transition-colors w-full py-2"
          >
            {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <Settings2 size={14} />
            <span>Advanced Settings</span>
          </button>

          {/* Advanced Settings Panel */}
          {showAdvanced && (
            <div className="space-y-3 p-3 bg-gray-900/50 rounded-lg border border-gray-700">
              {/* Chunking Settings - more compact */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Layers size={12} className="text-gray-500" />
                  <span className="text-[11px] font-medium text-gray-300">Chunking</span>
                </div>

                {/* Chunk Size + Overlap in row */}
                <div className="grid grid-cols-2 gap-3 mb-2">
                  <div>
                    <label className="block text-[10px] text-gray-500 mb-1">
                      Size: {settings.chunk_size}
                    </label>
                    <input
                      type="number"
                      min="128"
                      max="2048"
                      step="64"
                      value={settings.chunk_size}
                      onChange={(e) => updateSettings('chunk_size', Number(e.target.value))}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-[10px] text-gray-50"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-gray-500 mb-1">
                      Overlap: {settings.chunk_overlap}
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="200"
                      step="10"
                      value={settings.chunk_overlap}
                      onChange={(e) => updateSettings('chunk_overlap', Number(e.target.value))}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-[10px] text-gray-50"
                    />
                  </div>
                </div>

                {/* Chunking Strategy */}
                <div>
                  <label className="block text-[10px] text-gray-500 mb-1">Strategy</label>
                  <div className="flex gap-1">
                    {(['recursive', 'sentence', 'paragraph'] as const).map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => updateSettings('chunking_strategy', s)}
                        className={`flex-1 px-2 py-1 rounded text-[10px] transition-colors ${
                          settings.chunking_strategy === s
                            ? 'bg-blue-600 text-gray-50'
                            : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                        }`}
                      >
                        {s.charAt(0).toUpperCase() + s.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Search Settings - more compact */}
              <div className="pt-2 border-t border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Search size={12} className="text-gray-500" />
                  <span className="text-[11px] font-medium text-gray-300">Search</span>
                </div>

                {/* Top K + Score Threshold in row */}
                <div className="grid grid-cols-2 gap-3 mb-2">
                  <div>
                    <label className="block text-[10px] text-gray-500 mb-1">
                      Results: {settings.top_k}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="50"
                      value={settings.top_k}
                      onChange={(e) => updateSettings('top_k', Number(e.target.value))}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-[10px] text-gray-50"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-gray-500 mb-1">
                      Min Score: {(settings.score_threshold * 100).toFixed(0)}%
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={Math.round(settings.score_threshold * 100)}
                      onChange={(e) => updateSettings('score_threshold', Number(e.target.value) / 100)}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-[10px] text-gray-50"
                    />
                  </div>
                </div>

                {/* Enable Rerank */}
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-gray-400">Enable Rerank</span>
                  <button
                    type="button"
                    onClick={() => updateSettings('enable_rerank', !settings.enable_rerank)}
                    className={`w-8 h-4 rounded-full transition-colors relative ${
                      settings.enable_rerank ? 'bg-blue-500' : 'bg-gray-600'
                    }`}
                  >
                    <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                      settings.enable_rerank ? 'translate-x-4' : 'translate-x-0.5'
                    }`} />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Info */}
          <div className="text-[10px] text-gray-500">
            Library ID: {library.library_id}
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2 sticky bottom-0 bg-gray-800 py-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-xs text-gray-300 hover:text-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-gray-50 text-xs rounded transition-colors"
            >
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
