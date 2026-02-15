/**
 * Knowledge detail panel shown in the right sidebar.
 *
 * Displays selected library metadata:
 *   - name, description, mode
 *   - document and chunk counts
 *   - index status
 *   - created and updated timestamps
 *
 * Provides actions to configure the library and rebuild index.
 */
import React from 'react';
import { useConsoleStore } from '../../stores/useConsoleStore';
import { Database, FileText, Layers, Settings, RefreshCw } from 'lucide-react';

const MODE_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  agentic: { bg: 'bg-purple-500/20', text: 'text-purple-300', label: 'Agentic' },
  indexed: { bg: 'bg-blue-500/20', text: 'text-blue-300', label: 'Indexed' },
  auto: { bg: 'bg-green-500/20', text: 'text-green-300', label: 'Auto' },
};

function getIndexStatus(docCount: number, chunkCount: number): { label: string; color: string } {
  if (docCount === 0) return { label: 'Empty', color: 'text-gray-400' };
  if (chunkCount === 0) return { label: 'Not Indexed', color: 'text-yellow-400' };
  return { label: 'Indexed', color: 'text-green-400' };
}

function formatDate(iso: string): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export interface KnowledgeDetailPanelProps {
  /** Called when the user clicks [Configure...]. */
  onConfigure?: (libraryId: string) => void;
  /** Called when the user clicks [Rebuild Index]. */
  onRebuildIndex?: (libraryId: string) => void;
}

export const KnowledgeDetailPanel: React.FC<KnowledgeDetailPanelProps> = ({
  onConfigure,
  onRebuildIndex,
}) => {
  const selectedLibraryId = useConsoleStore((s) => s.selectedLibraryId);
  const knowledgeLibraries = useConsoleStore((s) => s.knowledgeLibraries);
  const isIngesting = useConsoleStore((s) => s.isIngesting);
  const ingestLibraryId = useConsoleStore((s) => s.ingestLibraryId);
  const ingestProgress = useConsoleStore((s) => s.ingestProgress);

  const library = React.useMemo(
    () => knowledgeLibraries.find((lib) => lib.library_id === selectedLibraryId) ?? null,
    [knowledgeLibraries, selectedLibraryId],
  );

  if (!library) {
    return (
      <div className="p-4 text-center text-gray-400 text-sm mt-4">
        <Database size={28} className="mx-auto mb-2 opacity-50" />
        <p>Select a knowledge library to view details</p>
      </div>
    );
  }

  const modeStyle = MODE_STYLES[library.mode] ?? MODE_STYLES.auto;
  const indexStatus = getIndexStatus(library.doc_count, library.chunk_count);
  const isCurrentIngesting = isIngesting && ingestLibraryId === library.library_id;

  return (
    <div className="flex flex-col gap-3 p-3 text-xs">
      {/* Header */}
      <div className="flex items-start gap-2">
        <Database size={16} className="text-blue-400 mt-0.5 shrink-0" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-gray-100 truncate" title={library.name}>
            {library.name}
          </h3>
          <span className={`inline-block mt-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${modeStyle.bg} ${modeStyle.text}`}>
            {modeStyle.label}
          </span>
        </div>
      </div>

      {/* Description */}
      {library.description && (
        <p className="text-gray-400 leading-relaxed">{library.description}</p>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-gray-700/50 rounded p-2 flex items-center gap-2">
          <FileText size={14} className="text-gray-400 shrink-0" />
          <div>
            <div className="text-gray-200 font-medium">{library.doc_count}</div>
            <div className="text-[10px] text-gray-500">Documents</div>
          </div>
        </div>
        <div className="bg-gray-700/50 rounded p-2 flex items-center gap-2">
          <Layers size={14} className="text-gray-400 shrink-0" />
          <div>
            <div className="text-gray-200 font-medium">{library.chunk_count}</div>
            <div className="text-[10px] text-gray-500">Chunks</div>
          </div>
        </div>
      </div>

      {/* Index Status */}
      <div className="bg-gray-700/50 rounded p-2">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Index Status</span>
          <span className={`font-medium ${indexStatus.color}`}>{indexStatus.label}</span>
        </div>
        {isCurrentIngesting && (
          <div className="mt-2">
            <div className="flex items-center justify-between text-[10px] text-gray-400 mb-1">
              <span>Ingesting...</span>
              <span>{Math.round(ingestProgress)}%</span>
            </div>
            <div className="w-full bg-gray-600 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full transition-all"
                style={{ width: `${ingestProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="space-y-1 text-gray-400">
        <div className="flex justify-between">
          <span>Directory</span>
          <span className="text-gray-300 font-mono text-[10px] truncate max-w-[60%]" title={library.knowledge_dir}>
            {library.knowledge_dir}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Created</span>
          <span className="text-gray-300">{formatDate(library.created_at)}</span>
        </div>
        <div className="flex justify-between">
          <span>Updated</span>
          <span className="text-gray-300">{formatDate(library.updated_at)}</span>
        </div>
      </div>

      {/* Action bar */}
      <div className="flex gap-2 pt-1 border-t border-gray-700">
        <button
          onClick={() => onConfigure?.(library.library_id)}
          className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-200 transition-colors"
          title="Open full configuration in Center Stage"
        >
          <Settings size={12} />
          Configure...
        </button>
        <button
          onClick={() => onRebuildIndex?.(library.library_id)}
          disabled={isCurrentIngesting}
          className="flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title="Rebuild knowledge index"
        >
          <RefreshCw size={12} className={isCurrentIngesting ? 'animate-spin' : ''} />
          Rebuild
        </button>
      </div>
    </div>
  );
};
