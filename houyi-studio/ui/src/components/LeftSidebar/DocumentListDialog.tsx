/**
 * Document List Dialog - Shows documents in a knowledge library
 */
import React, { useState, useEffect } from 'react';
import { X, FileText, Trash2, Eye, EyeOff, RefreshCw, AlertCircle, CheckCircle, Loader2, Clock, Search, Hash } from 'lucide-react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import type { KnowledgeDocument, KnowledgeLibrary } from '@/types/ir';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { ChunkViewDialog } from './ChunkViewDialog';

interface DocumentListDialogProps {
  isOpen: boolean;
  library: KnowledgeLibrary | null;
  onClose: () => void;
}

export const DocumentListDialog: React.FC<DocumentListDialogProps> = ({
  isOpen,
  library,
  onClose,
}) => {
  const {
    documents,
    isLoadingDocuments,
    requestDocuments,
    deleteDocument,
    disableDocument,
    enableDocument,
    showToast,
  } = useConsoleStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<KnowledgeDocument | null>(null);
  const [chunkViewOpen, setChunkViewOpen] = useState(false);
  const [documentForChunks, setDocumentForChunks] = useState<KnowledgeDocument | null>(null);

  // Load documents when dialog opens
  useEffect(() => {
    if (isOpen && library) {
      requestDocuments(library.library_id);
    }
  }, [isOpen, library, requestDocuments]);

  if (!isOpen || !library) return null;

  // Filter documents by search query
  const filteredDocuments = documents.filter(doc =>
    doc.file_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    doc.file_path.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'indexed':
        return <CheckCircle size={14} className="text-green-400" />;
      case 'pending':
        return <Clock size={14} className="text-yellow-400" />;
      case 'error':
      case 'failed':
        return <AlertCircle size={14} className="text-red-400" />;
      case 'disabled':
        return <EyeOff size={14} className="text-gray-500" />;
      case 'indexing':
        return <Loader2 size={14} className="text-yellow-400 animate-spin" />;
      default:
        return <Clock size={14} className="text-gray-400" />;
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'indexed':
        return 'Indexed';
      case 'pending':
        return 'Pending';
      case 'error':
      case 'failed':
        return 'Error';
      case 'disabled':
        return 'Disabled';
      case 'indexing':
        return 'Indexing';
      default:
        return status;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleDeleteClick = (doc: KnowledgeDocument) => {
    setDocumentToDelete(doc);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = () => {
    if (documentToDelete && library) {
      deleteDocument(library.library_id, documentToDelete.doc_id);
    }
    setDeleteDialogOpen(false);
    setDocumentToDelete(null);
  };

  const handleToggleStatus = (doc: KnowledgeDocument) => {
    if (doc.status === 'disabled') {
      enableDocument(library.library_id, doc.doc_id);
      showToast('Document enabled', 'success');
    } else if (doc.status === 'indexed') {
      disableDocument(library.library_id, doc.doc_id);
      showToast('Document disabled', 'info');
    }
  };

  const handleViewChunks = (doc: KnowledgeDocument) => {
    setDocumentForChunks(doc);
    setChunkViewOpen(true);
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
          className="bg-gray-800 border border-gray-600 rounded-lg shadow-xl w-[700px] max-h-[80vh] flex flex-col pointer-events-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <div>
              <h3 className="text-sm font-semibold text-gray-200">Documents</h3>
              <p className="text-xs text-gray-400 mt-1">
                {library.name} · {documents.length} documents
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => requestDocuments(library.library_id)}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
                title="Refresh"
              >
                <RefreshCw size={14} className={isLoadingDocuments ? 'animate-spin' : ''} />
              </button>
              <button
                onClick={onClose}
                className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Search */}
          <div className="p-3 border-b border-gray-700">
            <div className="relative">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search documents..."
                className="w-full pl-9 pr-3 py-2 bg-gray-900 border border-gray-700 rounded text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {/* Document List */}
          <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
            {isLoadingDocuments ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 size={24} className="text-gray-400 animate-spin" />
              </div>
            ) : filteredDocuments.length === 0 ? (
              <div className="text-center py-8">
                <FileText size={32} className="text-gray-600 mx-auto mb-2" />
                <p className="text-xs text-gray-500">
                  {searchQuery ? 'No matching documents' : 'No documents in this library'}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {filteredDocuments.map((doc) => (
                  <div
                    key={doc.doc_id}
                    className={`
                      bg-gray-900 border border-gray-700 rounded p-3
                      ${doc.status === 'disabled' ? 'opacity-60' : ''}
                    `}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <FileText size={14} className="text-gray-400 flex-shrink-0" />
                          <span className="text-xs font-medium text-gray-200 truncate">
                            {doc.file_name}
                          </span>
                        </div>
                        <div className="text-[10px] text-gray-500 mt-1 truncate">
                          {doc.file_path}
                        </div>
                        <div className="flex items-center gap-3 mt-2 text-[10px] text-gray-500">
                          <div className="flex items-center gap-1">
                            {getStatusIcon(doc.status)}
                            <span>{getStatusLabel(doc.status)}</span>
                          </div>
                          <span>{doc.chunk_count} chunks</span>
                          <span>{formatFileSize(doc.file_size)}</span>
                          <span>{formatDate(doc.created_at)}</span>
                        </div>
                        {(doc.error_message || doc.metadata?.error) && (
                          <div className="text-[10px] text-red-400 mt-1">
                            Error: {doc.error_message || doc.metadata?.error}
                          </div>
                        )}
                      </div>
                      <div className="flex gap-1 ml-2">
                        {doc.chunk_count > 0 && (
                          <button
                            onClick={() => handleViewChunks(doc)}
                            className="p-1.5 hover:bg-gray-700 rounded text-gray-500 hover:text-blue-400"
                            title="View chunks"
                          >
                            <Hash size={14} />
                          </button>
                        )}
                        {(doc.status === 'indexed' || doc.status === 'disabled') && (
                          <button
                            onClick={() => handleToggleStatus(doc)}
                            className={`p-1.5 hover:bg-gray-700 rounded ${
                              doc.status === 'disabled'
                                ? 'text-gray-500 hover:text-green-400'
                                : 'text-gray-500 hover:text-yellow-400'
                            }`}
                            title={doc.status === 'disabled' ? 'Enable' : 'Disable'}
                          >
                            {doc.status === 'disabled' ? <Eye size={14} /> : <EyeOff size={14} />}
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteClick(doc)}
                          className="p-1.5 hover:bg-gray-700 rounded text-gray-500 hover:text-red-400"
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-3 border-t border-gray-700 flex justify-between items-center">
            <div className="text-[10px] text-gray-500">
              {filteredDocuments.length} of {documents.length} documents
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

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        isOpen={deleteDialogOpen}
        title="Delete Document"
        message="Are you sure you want to delete this document? This will also remove all its chunks from the index."
        itemName={documentToDelete?.file_name}
        confirmText="Delete"
        variant="danger"
        onConfirm={handleConfirmDelete}
        onCancel={() => {
          setDeleteDialogOpen(false);
          setDocumentToDelete(null);
        }}
      />

      {/* Chunk View Dialog */}
      <ChunkViewDialog
        isOpen={chunkViewOpen}
        libraryId={library.library_id}
        document={documentForChunks}
        onClose={() => {
          setChunkViewOpen(false);
          setDocumentForChunks(null);
        }}
      />
    </>
  );
};
