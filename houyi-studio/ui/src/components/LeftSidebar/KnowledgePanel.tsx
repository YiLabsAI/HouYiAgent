/**
 * Knowledge Panel - Displays knowledge libraries and provides search
 */
import React, { useState } from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { Plus, Database, Trash2, RefreshCw, Upload, CheckCircle, AlertCircle, Loader2, Circle, Settings2, RotateCw, FileText, MoreVertical } from 'lucide-react';
import type { KnowledgeLibrary } from '@/types/ir';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { ImportFilesDialog, IngestProgress } from './ImportFilesDialog';
import { EditLibraryDialog } from './EditLibraryDialog';
import { DocumentListDialog } from './DocumentListDialog';

interface KnowledgePanelProps {
  onOpenCreateDialog: () => void;
}

export const KnowledgePanel: React.FC<KnowledgePanelProps> = ({ onOpenCreateDialog }) => {
  const {
    knowledgeLibraries,
    selectedLibraryId,
    isLoadingLibraries,
    selectKnowledgeLibrary,
    deleteKnowledgeLibrary,
    requestKnowledgeLibraries,
    ingestKnowledgeFiles,
    isIngesting,
    ingestProgress,
    ingestCurrentFile,
    ingestFilesProcessed,
    ingestTotalFiles,
    editKnowledgeLibrary,
    rebuildKnowledgeIndex,
    cancelIngest,
  } = useConsoleStore();

  // State for delete confirmation dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [libraryToDelete, setLibraryToDelete] = useState<KnowledgeLibrary | null>(null);

  // State for import dialog
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [libraryToImport, setLibraryToImport] = useState<KnowledgeLibrary | null>(null);

  // State for edit dialog
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [libraryToEdit, setLibraryToEdit] = useState<KnowledgeLibrary | null>(null);

  // State for document list dialog
  const [documentListOpen, setDocumentListOpen] = useState(false);
  const [libraryForDocs, setLibraryForDocs] = useState<KnowledgeLibrary | null>(null);

  // State for action menu
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
      }
    };
    if (menuOpenId) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [menuOpenId]);

  React.useEffect(() => {
    // Request libraries on mount
    requestKnowledgeLibraries();
  }, [requestKnowledgeLibraries]);

  const handleDeleteClick = (e: React.MouseEvent, library: KnowledgeLibrary) => {
    e.stopPropagation();
    setLibraryToDelete(library);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = () => {
    if (libraryToDelete) {
      deleteKnowledgeLibrary(libraryToDelete.library_id);
    }
    setDeleteDialogOpen(false);
    setLibraryToDelete(null);
  };

  const handleCancelDelete = () => {
    setDeleteDialogOpen(false);
    setLibraryToDelete(null);
  };

  const handleImportClick = (e: React.MouseEvent, library: KnowledgeLibrary) => {
    e.stopPropagation();
    setLibraryToImport(library);
    setImportDialogOpen(true);
  };

  const handleImport = (paths: string[]) => {
    if (libraryToImport) {
      ingestKnowledgeFiles(libraryToImport.library_id, paths);
    }
    setImportDialogOpen(false);
    setLibraryToImport(null);
  };

  const handleCancelImport = () => {
    setImportDialogOpen(false);
    setLibraryToImport(null);
  };

  const handleEditClick = (e: React.MouseEvent, library: KnowledgeLibrary) => {
    e.stopPropagation();
    setLibraryToEdit(library);
    setEditDialogOpen(true);
  };

  const handleSaveEdit = (libraryId: string, updates: Record<string, any>) => {
    editKnowledgeLibrary(libraryId, updates);
    setEditDialogOpen(false);
    setLibraryToEdit(null);
  };

  const handleCancelEdit = () => {
    setEditDialogOpen(false);
    setLibraryToEdit(null);
  };

  const handleRebuildClick = (e: React.MouseEvent, library: KnowledgeLibrary) => {
    e.stopPropagation();
    const incremental = !e.shiftKey;
    rebuildKnowledgeIndex(library.library_id, incremental);
  };

  const handleViewDocsClick = (e: React.MouseEvent, library: KnowledgeLibrary) => {
    e.stopPropagation();
    setLibraryForDocs(library);
    setDocumentListOpen(true);
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
        return mode;
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case 'agentic':
        return 'bg-blue-600';
      case 'indexed':
        return 'bg-green-600';
      case 'auto':
        return 'bg-purple-600';
      default:
        return 'bg-gray-600';
    }
  };

  const formatRelativeTime = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  // Get library status based on state
  const getLibraryStatus = (library: KnowledgeLibrary) => {
    // Check if this library is currently being ingested
    if (isIngesting && ingestLibraryId === library.library_id) {
      return 'indexing';
    }
    // Check explicit status field
    const status = (library as any).status;
    if (status) {
      return status;
    }
    // Infer from doc_count
    if ((library.doc_count ?? 0) === 0 && (library.chunk_count ?? 0) === 0) {
      return 'empty';
    }
    return 'ready';
  };

  const getStatusIndicator = (library: KnowledgeLibrary) => {
    const status = getLibraryStatus(library);
    switch (status) {
      case 'empty':
        return { icon: Circle, color: 'text-gray-500', label: 'Empty' };
      case 'indexing':
        return { icon: Loader2, color: 'text-yellow-400 animate-spin', label: 'Indexing' };
      case 'ready':
        return { icon: CheckCircle, color: 'text-green-400', label: 'Ready' };
      case 'error':
        return { icon: AlertCircle, color: 'text-red-400', label: 'Error' };
      default:
        return { icon: Circle, color: 'text-gray-500', label: status };
    }
  };

  const ingestLibraryId = useConsoleStore((state) => state.ingestLibraryId);

  return (
    <div className="p-3 text-xs text-gray-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs font-semibold text-gray-200">Knowledge Libraries</div>
        <div className="flex gap-1">
          <button
            onClick={() => requestKnowledgeLibraries()}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
            title="Refresh"
          >
            <RefreshCw size={14} className={isLoadingLibraries ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={onOpenCreateDialog}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
            title="Create Library"
          >
            <Plus size={14} />
          </button>
        </div>
      </div>

      {/* Library List */}
      <div className="space-y-2">
        {isLoadingLibraries && knowledgeLibraries.length === 0 ? (
          <div className="text-[11px] text-gray-500 text-center py-4">
            Loading libraries...
          </div>
        ) : knowledgeLibraries.length === 0 ? (
          <div className="bg-gray-900 border border-gray-700 rounded p-3 text-center">
            <Database className="w-8 h-8 text-gray-600 mx-auto mb-2" />
            <div className="text-[11px] text-gray-400">No knowledge libraries</div>
            <button
              onClick={onOpenCreateDialog}
              className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-[11px] text-white"
            >
              Create Library
            </button>
          </div>
        ) : (
          knowledgeLibraries.map((library: KnowledgeLibrary) => {
            const statusInfo = getStatusIndicator(library);
            const StatusIcon = statusInfo.icon;
            return (
            <div
              key={library.library_id}
              onClick={() => selectKnowledgeLibrary(
                selectedLibraryId === library.library_id ? null : library.library_id
              )}
              className={`
                bg-gray-900 border rounded p-2 cursor-pointer transition-colors
                ${selectedLibraryId === library.library_id
                  ? 'border-blue-500 bg-gray-800'
                  : 'border-gray-700 hover:border-gray-600'
                }
              `}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <Database size={12} className="text-gray-400 flex-shrink-0" />
                    <span className="text-[11px] font-medium text-gray-200 truncate">
                      {library.name}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[9px] text-white ${getModeColor(library.mode)}`}>
                      {getModeLabel(library.mode)}
                    </span>
                  </div>
                  {library.description && (
                    <div className="text-[10px] text-gray-500 mt-1 truncate">
                      {library.description}
                    </div>
                  )}
                  <div className="flex items-center gap-3 mt-1 text-[10px] text-gray-500">
                    <div className="flex items-center gap-1" title={statusInfo.label}>
                      <StatusIcon size={10} className={statusInfo.color} />
                      <span className={statusInfo.color}>{statusInfo.label}</span>
                    </div>
                    <span>{library.doc_count ?? 0} docs</span>
                    <span>{library.chunk_count ?? 0} chunks</span>
                    {library.updated_at && (
                      <span title={new Date(library.updated_at).toLocaleString()}>
                        {formatRelativeTime(library.updated_at)}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex gap-0.5 flex-shrink-0">
                  <button
                    onClick={(e) => handleImportClick(e, library)}
                    className="p-1 hover:bg-gray-700 rounded text-gray-500 hover:text-blue-400"
                    title="Import files"
                  >
                    <Upload size={12} />
                  </button>
                  <div className="relative" ref={menuOpenId === library.library_id ? menuRef : null}>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(menuOpenId === library.library_id ? null : library.library_id);
                      }}
                      className={`p-1 hover:bg-gray-700 rounded transition-colors ${
                        menuOpenId === library.library_id ? 'text-blue-400 bg-gray-700' : 'text-gray-500 hover:text-gray-300'
                      }`}
                      title="More actions"
                    >
                      <MoreVertical size={12} />
                    </button>
                    {menuOpenId === library.library_id && (
                      <div className="absolute right-0 top-full mt-1 z-50 bg-gray-800 border border-gray-700 rounded shadow-lg py-1 min-w-[140px]">
                        <button
                          onClick={(e) => { handleViewDocsClick(e, library); setMenuOpenId(null); }}
                          className="w-full px-3 py-1.5 text-left text-[11px] text-gray-300 hover:bg-gray-700 flex items-center gap-2"
                        >
                          <FileText size={12} /> View Documents
                        </button>
                        <button
                          onClick={(e) => { handleEditClick(e, library); setMenuOpenId(null); }}
                          className="w-full px-3 py-1.5 text-left text-[11px] text-gray-300 hover:bg-gray-700 flex items-center gap-2"
                        >
                          <Settings2 size={12} /> Edit Settings
                        </button>
                        <button
                          onClick={(e) => { handleRebuildClick(e, library); setMenuOpenId(null); }}
                          className="w-full px-3 py-1.5 text-left text-[11px] text-gray-300 hover:bg-gray-700 flex items-center gap-2"
                          disabled={isIngesting}
                        >
                          <RotateCw size={12} /> Rebuild Index
                        </button>
                        <div className="border-t border-gray-700 my-1" />
                        <button
                          onClick={(e) => { handleDeleteClick(e, library); setMenuOpenId(null); }}
                          className="w-full px-3 py-1.5 text-left text-[11px] text-red-400 hover:bg-gray-700 flex items-center gap-2"
                        >
                          <Trash2 size={12} /> Delete
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          );})
        )}

        {/* Ingest Progress */}
        <IngestProgress
          isIngesting={isIngesting}
          progress={ingestProgress}
          currentFile={ingestCurrentFile}
          filesProcessed={ingestFilesProcessed}
          totalFiles={ingestTotalFiles}
          onCancel={cancelIngest}
        />
      </div>

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={deleteDialogOpen}
        title="Delete Knowledge Library"
        message="Are you sure you want to delete this knowledge library? This action cannot be undone."
        itemName={libraryToDelete?.name}
        confirmText="Delete"
        variant="danger"
        onConfirm={handleConfirmDelete}
        onCancel={handleCancelDelete}
      />

      {/* Import Files Dialog */}
      <ImportFilesDialog
        isOpen={importDialogOpen}
        libraryName={libraryToImport?.name || ''}
        libraryId={libraryToImport?.library_id || ''}
        onImport={handleImport}
        onCancel={handleCancelImport}
      />

      {/* Edit Library Dialog */}
      <EditLibraryDialog
        isOpen={editDialogOpen}
        library={libraryToEdit}
        onSave={handleSaveEdit}
        onCancel={handleCancelEdit}
      />

      {/* Document List Dialog */}
      <DocumentListDialog
        isOpen={documentListOpen}
        library={libraryForDocs}
        onClose={() => {
          setDocumentListOpen(false);
          setLibraryForDocs(null);
        }}
      />
    </div>
  );
};
