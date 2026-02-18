/**
 * Knowledge Base store actions
 * Manages knowledge libraries and search functionality
 */

import type { KnowledgeLibrary, KnowledgeSearchResult, RAGMode, KnowledgeDocument, KnowledgeChunk, ChunkPreview, QualitySummary } from '../../types/ir';
import { logger } from '../../utils/logger';
import { addSuccessfulImport } from '../../components/LeftSidebar/ImportFilesDialog';

type StoreSet = (partial: any) => void;
type StoreGet = () => any;

export interface KnowledgeState {
  knowledgeLibraries: KnowledgeLibrary[];
  selectedLibraryId: string | null;
  knowledgeSearchResults: KnowledgeSearchResult[];
  knowledgeSearchQuery: string;
  knowledgeSearchModeUsed: string;
  knowledgeSearchStrategiesUsed: string[];
  knowledgeSearchQuality: QualitySummary | null;
  isSearchingKnowledge: boolean;
  isLoadingLibraries: boolean;
  // Ingest state
  isIngesting: boolean;
  ingestLibraryId: string | null;
  ingestLibraryName: string;  // Track library name for history
  ingestPaths: string[];      // Track paths for history
  ingestProgress: number;
  ingestCurrentFile: string;
  ingestFilesProcessed: number;
  ingestTotalFiles: number;
  // Document management state
  documents: KnowledgeDocument[];
  selectedDocumentId: string | null;
  isLoadingDocuments: boolean;
  chunks: KnowledgeChunk[];
  isLoadingChunks: boolean;
  chunkPreviews: ChunkPreview[];
  isPreviewingChunks: boolean;
}

export const initialKnowledgeState: KnowledgeState = {
  knowledgeLibraries: [],
  selectedLibraryId: null,
  knowledgeSearchResults: [],
  knowledgeSearchQuery: '',
  knowledgeSearchModeUsed: '',
  knowledgeSearchStrategiesUsed: [],
  knowledgeSearchQuality: null,
  isSearchingKnowledge: false,
  isLoadingLibraries: false,
  // Ingest state
  isIngesting: false,
  ingestLibraryId: null,
  ingestLibraryName: '',
  ingestPaths: [],
  ingestProgress: 0,
  ingestCurrentFile: '',
  ingestFilesProcessed: 0,
  ingestTotalFiles: 0,
  // Document management state
  documents: [],
  selectedDocumentId: null,
  isLoadingDocuments: false,
  chunks: [],
  isLoadingChunks: false,
  chunkPreviews: [],
  isPreviewingChunks: false,
};

export const createKnowledgeActions = (set: StoreSet, get: StoreGet) => ({
  // List knowledge libraries
  requestKnowledgeLibraries: () => {
    set({ isLoadingLibraries: true });
    const command = {
      command_type: 'list_knowledge_libraries',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
    };
    const sent = get().sendCommand(command);
    if (!sent) {
      // WebSocket not connected, reset loading state
      set({ isLoadingLibraries: false });
    }
    logger.debug('Knowledge', 'Requesting library list');
  },

  // Set libraries (called when receiving list event)
  setKnowledgeLibraries: (libraries: KnowledgeLibrary[]) => {
    set({
      knowledgeLibraries: libraries,
      isLoadingLibraries: false,
    });
    logger.debug('Knowledge', 'Updated libraries:', libraries.length);
  },

  // Create new library
  createKnowledgeLibrary: (config: {
    name: string;
    description: string;
    mode: RAGMode;
    knowledge_dir: string;
    metadata?: Record<string, any>;
   
    strategies?: string[];
    embedding_provider?: string;
    contextual_retrieval?: boolean;
  }) => {
    // Validate: check for duplicate name
    const existingLibraries = get().knowledgeLibraries || [];
    const nameExists = existingLibraries.some(
      (lib: KnowledgeLibrary) => lib.name.toLowerCase() === config.name.trim().toLowerCase()
    );
    if (nameExists) {
      get().showToast(`Library "${config.name}" already exists`, 'error');
      return;
    }

    const command = {
      command_type: 'create_knowledge_library',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      ...config,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Creating library:', config.name);
    // Note: Toast will be shown in addKnowledgeLibrary when backend confirms creation
  },

  // Delete library
  deleteKnowledgeLibrary: (libraryId: string) => {
    const command = {
      command_type: 'delete_knowledge_library',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Deleting library:', libraryId);
  },

  // Add created library (called when receiving created event)
  addKnowledgeLibrary: (library: KnowledgeLibrary) => {
    const libraries = get().knowledgeLibraries || [];
    set({
      knowledgeLibraries: [...libraries, library],
    });
    get().showToast(`Knowledge library "${library.name}" created!`, 'success');
  },

  // Remove deleted library (called when receiving deleted event)
  removeKnowledgeLibrary: (libraryId: string) => {
    const libraries = get().knowledgeLibraries || [];
    set({
      knowledgeLibraries: libraries.filter((lib: KnowledgeLibrary) => lib.library_id !== libraryId),
      selectedLibraryId: get().selectedLibraryId === libraryId ? null : get().selectedLibraryId,
    });
    get().showToast('Knowledge library deleted', 'success');
  },

  // Select a library
  selectKnowledgeLibrary: (libraryId: string | null) => {
    set({ selectedLibraryId: libraryId });
    logger.debug('Knowledge', 'Selected library:', libraryId);
  },

  // Search knowledge
  searchKnowledge: (query: string, libraryId?: string, mode?: RAGMode, topK?: number) => {
    if (!query.trim()) {
      get().showToast('Please enter a search query', 'error');
      return;
    }

    set({
      isSearchingKnowledge: true,
      knowledgeSearchQuery: query,
      knowledgeSearchResults: [],
      bottomPanelTab: 'knowledge', // Auto-switch to Knowledge tab
    });

    const command = {
      command_type: 'search_knowledge',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      query,
      library_id: libraryId || get().selectedLibraryId,
      mode,
      top_k: topK,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Searching:', query);
  },

  // Set search results (called when receiving results event)
  setKnowledgeSearchResults: (results: KnowledgeSearchResult[], query: string, modeUsed?: string, strategiesUsed?: string[], quality?: QualitySummary | null) => {
    set({
      knowledgeSearchResults: results,
      knowledgeSearchQuery: query,
      knowledgeSearchModeUsed: modeUsed || '',
      knowledgeSearchStrategiesUsed: strategiesUsed || [],
      knowledgeSearchQuality: quality || null,
      isSearchingKnowledge: false,
    });
    logger.info('Knowledge', 'Search results:', results.length, 'mode:', modeUsed, 'strategies:', strategiesUsed, 'quality:', quality ? 'yes' : 'no');
  },

  // Clear search results
  clearKnowledgeSearch: () => {
    set({
      knowledgeSearchResults: [],
      knowledgeSearchQuery: '',
      knowledgeSearchStrategiesUsed: [],
      knowledgeSearchQuality: null,
      isSearchingKnowledge: false,
    });
  },

  // Handle knowledge error
  handleKnowledgeError: (error: string, operation: string) => {
    set({ isSearchingKnowledge: false, isLoadingLibraries: false, isIngesting: false });
    get().showToast(`Knowledge ${operation} failed: ${error}`, 'error');
    logger.error('Knowledge', 'Error:', operation, error);
  },

  // Ingest files into a library
  ingestKnowledgeFiles: (libraryId: string, paths: string[]) => {
    if (!libraryId) {
      get().showToast('Please select a library first', 'error');
      return;
    }
    if (!paths || paths.length === 0) {
      get().showToast('Please specify files or directories to import', 'error');
      return;
    }

    // Find library name for history tracking
    const libraries = get().knowledgeLibraries || [];
    const library = libraries.find((lib: KnowledgeLibrary) => lib.library_id === libraryId);
    const libraryName = library?.name || '';

    set({
      isIngesting: true,
      ingestLibraryId: libraryId,
      ingestLibraryName: libraryName,
      ingestPaths: paths,
      ingestProgress: 0,
      ingestCurrentFile: '',
      ingestFilesProcessed: 0,
      ingestTotalFiles: 0,
    });

    const command = {
      command_type: 'ingest_knowledge_files',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      paths,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Starting ingest for library:', libraryId, 'paths:', paths);
  },

  // Handle ingest progress update
  handleIngestProgress: (data: {
    library_id: string;
    progress: number;
    current_file: string;
    files_processed: number;
    total_files: number;
  }) => {
    set({
      ingestProgress: data.progress,
      ingestCurrentFile: data.current_file,
      ingestFilesProcessed: data.files_processed,
      ingestTotalFiles: data.total_files,
    });
  },

  // Handle ingest complete
  handleIngestComplete: (data: {
    library_id: string;
    success: boolean;
    stats: Record<string, any>;
    message: string;
    warning?: string;
  }) => {
    const { ingestLibraryId, ingestLibraryName, ingestPaths } = get();

    set({
      isIngesting: false,
      ingestLibraryId: null,
      ingestLibraryName: '',
      ingestPaths: [],
      ingestProgress: 0,
      ingestCurrentFile: '',
    });

    const filesProcessed = data.stats?.files_processed || 0;
    const filesFailed = data.stats?.files_failed || 0;
    const chunksCreated = data.stats?.chunks_created || 0;
    const filesSkipped = data.stats?.files_skipped || 0;

    if (data.success && filesProcessed > 0) {
      // Only save to history if at least some files were successfully processed
      const libId = ingestLibraryId || data.library_id;
      if (ingestPaths.length > 0 && libId && ingestLibraryName) {
        addSuccessfulImport(ingestPaths, libId, ingestLibraryName);
      }

      // Check for degraded state (warning from backend)
      if (data.warning) {
        const message = `Imported ${filesProcessed} files — but no embedding provider found. Semantic search is unavailable. Install an embedding provider and rebuild the index.`;
        get().showToast(message, 'warning');
      } else {
        let message = `Import complete: ${filesProcessed} files, ${chunksCreated} chunks`;
        if (filesSkipped > 0) {
          message += `, ${filesSkipped} skipped`;
        }
        if (filesFailed > 0) {
          message += ` (${filesFailed} failed)`;
          get().showToast(message, 'warning');
        } else {
          get().showToast(message, 'success');
        }
      }
    } else if (data.success) {
      // Success but no files processed (e.g., incremental rebuild with no changes)
      if (data.warning) {
        get().showToast(data.warning, 'warning');
      } else {
        const message = data.message || 'No changes detected';
        get().showToast(message, 'info');
      }
    } else {
      const errorMsg = data.message || `All ${filesFailed} files failed to process`;
      get().showToast(`Import failed: ${errorMsg}`, 'error');
    }
    logger.info('Knowledge', 'Ingest complete:', data);
  },

  // Update library (called when receiving updated event)
  updateKnowledgeLibrary: (library: KnowledgeLibrary) => {
    const libraries = get().knowledgeLibraries || [];
    const index = libraries.findIndex((lib: KnowledgeLibrary) => lib.library_id === library.library_id);
    if (index >= 0) {
      const updated = [...libraries];
      updated[index] = library;
      set({ knowledgeLibraries: updated });
      logger.info('Knowledge', 'Updated library:', library.library_id);
    }
  },

  // Send update command to backend
  editKnowledgeLibrary: (libraryId: string, updates: Record<string, any>) => {
    const command = {
      command_type: 'update_knowledge_library',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      updates,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Editing library:', libraryId, updates);
  },

  // Rebuild index for a library (re-index all files in knowledge_dir)
  rebuildKnowledgeIndex: (libraryId: string, incremental: boolean = false) => {
    if (!libraryId) {
      get().showToast('Please select a library first', 'error');
      return;
    }

    // Find library info for history tracking
    const libraries = get().knowledgeLibraries || [];
    const library = libraries.find((lib: KnowledgeLibrary) => lib.library_id === libraryId);
    const libraryName = library?.name || '';
    // Get existing document paths from library for history
    const existingPaths = library?.documents
      ? Object.values(library.documents).map((doc: any) => doc.file_path)
      : [];

    set({
      isIngesting: true,
      ingestLibraryId: libraryId,
      ingestLibraryName: libraryName,
      ingestPaths: existingPaths,  // Use existing document paths
      ingestProgress: 0,
      ingestCurrentFile: '',
      ingestFilesProcessed: 0,
      ingestTotalFiles: 0,
    });

    const command = {
      command_type: 'rebuild_knowledge_index',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      incremental,
    };
    get().sendCommand(command);
    logger.info('Knowledge', incremental ? 'Incremental rebuild for:' : 'Full rebuild for:', libraryId);
  },

  // Cancel ongoing ingest operation
  cancelIngest: () => {
    const libraryId = get().ingestLibraryId;
    if (!libraryId) {
      return;
    }

    const command = {
      command_type: 'cancel_ingest',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Cancelling ingest for library:', libraryId);

    // Optimistically clear ingest state
    set({
      isIngesting: false,
      ingestLibraryId: null,
      ingestProgress: 0,
      ingestCurrentFile: '',
      ingestFilesProcessed: 0,
      ingestTotalFiles: 0,
    });
    get().showToast('Import cancelled', 'info');
  },

  // ============================================================================
  // Document Management Actions
  // ============================================================================

  // Request documents list for a library
  requestDocuments: (libraryId: string) => {
    set({ isLoadingDocuments: true, documents: [] });
    const command = {
      command_type: 'list_documents',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
    };
    const sent = get().sendCommand(command);
    if (!sent) {
      set({ isLoadingDocuments: false });
    }
    logger.debug('Knowledge', 'Requesting documents for library:', libraryId);
  },

  // Set documents (called when receiving document_list event)
  setDocuments: (documents: KnowledgeDocument[]) => {
    set({ documents, isLoadingDocuments: false });
    logger.debug('Knowledge', 'Updated documents:', documents.length);
  },

  // Select a document
  selectDocument: (docId: string | null) => {
    set({ selectedDocumentId: docId });
    logger.debug('Knowledge', 'Selected document:', docId);
  },

  // Delete a document
  deleteDocument: (libraryId: string, docId: string) => {
    const command = {
      command_type: 'delete_document',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      doc_id: docId,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Deleting document:', docId);
  },

  // Remove document from local state (called when receiving document_deleted event)
  removeDocument: (docId: string) => {
    const documents = get().documents || [];
    set({
      documents: documents.filter((doc: KnowledgeDocument) => doc.doc_id !== docId),
      selectedDocumentId: get().selectedDocumentId === docId ? null : get().selectedDocumentId,
    });
    get().showToast('Document deleted', 'success');
  },

  // Disable a document (exclude from search)
  disableDocument: (libraryId: string, docId: string) => {
    const command = {
      command_type: 'disable_document',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      doc_id: docId,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Disabling document:', docId);
  },

  // Enable a document (include in search)
  enableDocument: (libraryId: string, docId: string) => {
    const command = {
      command_type: 'enable_document',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      doc_id: docId,
    };
    get().sendCommand(command);
    logger.info('Knowledge', 'Enabling document:', docId);
  },

  // Update document status in local state (called when receiving document_status_changed event)
  updateDocumentStatus: (docId: string, status: string) => {
    const documents = get().documents || [];
    const index = documents.findIndex((doc: KnowledgeDocument) => doc.doc_id === docId);
    if (index >= 0) {
      const updated = [...documents];
      updated[index] = { ...updated[index], status };
      set({ documents: updated });
      logger.info('Knowledge', 'Document status updated:', docId, status);
    }
  },

  // ============================================================================
  // Chunk Management Actions
  // ============================================================================

  // Request chunks for a document
  requestChunks: (libraryId: string, docId: string) => {
    set({ isLoadingChunks: true, chunks: [] });
    const command = {
      command_type: 'list_chunks',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      library_id: libraryId,
      doc_id: docId,
    };
    const sent = get().sendCommand(command);
    if (!sent) {
      set({ isLoadingChunks: false });
    }
    logger.debug('Knowledge', 'Requesting chunks for document:', docId);
  },

  // Set chunks (called when receiving chunk_list event)
  setChunks: (chunks: KnowledgeChunk[]) => {
    set({ chunks, isLoadingChunks: false });
    logger.debug('Knowledge', 'Updated chunks:', chunks.length);
  },

  // Preview chunks for content with given settings
  previewChunks: (content: string, chunkSize: number, chunkOverlap: number, strategy: string) => {
    set({ isPreviewingChunks: true, chunkPreviews: [] });
    const command = {
      command_type: 'preview_chunks',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      content,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
      strategy,
    };
    const sent = get().sendCommand(command);
    if (!sent) {
      set({ isPreviewingChunks: false });
    }
    logger.debug('Knowledge', 'Previewing chunks with settings:', { chunkSize, chunkOverlap, strategy });
  },

  // Set chunk previews (called when receiving chunk_preview event)
  setChunkPreviews: (previews: ChunkPreview[]) => {
    set({ chunkPreviews: previews, isPreviewingChunks: false });
    logger.debug('Knowledge', 'Updated chunk previews:', previews.length);
  },

  // Clear chunks
  clearChunks: () => {
    set({ chunks: [], chunkPreviews: [] });
  },
});
