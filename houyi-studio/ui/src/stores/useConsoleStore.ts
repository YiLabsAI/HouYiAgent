/**
 * Zustand store for console state management
 */
import { create } from 'zustand';
import type { PlanIR, NodeIR, ExecutionIR, CheckpointIR, KnowledgeLibrary, KnowledgeSearchResult, RAGMode, KnowledgeDocument, KnowledgeChunk, ChunkPreview, QualitySummary } from '@/types/ir';
import type { AnyServerEvent } from '@/types/websocket';
import { ConsoleWebSocket } from '@/utils/websocket';
import type { ToolStatistics } from './utils/toolStats';
import { createWorkflowActions } from './storeActions/workflowActions';
import { createWsActions } from './storeActions/wsActions';
import { createNodeActions } from './storeActions/nodeActions';
import { createExecutionActions } from './storeActions/executionActions';
import { createCheckpointActions } from './storeActions/checkpointActions';
import {
  createRunSettingsActions,
  getInitialRunSettingsState,
} from './storeActions/runSettingsActions';
import { createToastActions } from './storeActions/toastActions';
import { createCommandActions } from './storeActions/commandActions';
import { createToolStatsActions } from './storeActions/toolStatsActions';
import { createSpanActions, type SpanStore } from './storeActions/spanActions';
import { createKnowledgeActions, initialKnowledgeState } from './storeActions/knowledgeActions';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
}

interface ActivityLog {
  id: string;
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  detail?: string;
}

interface WorkflowSummary {
  name: string;
  saved_at: string;
  nodes_count: number;
}

interface RunSettings {
  enable_tool_calls: boolean;
  tool_names: string[];
  tool_choice: string | null;
  max_tool_calls: number;
  temperature: number | null;
  parallel_tool_calls: boolean | null;
  web_search_provider: string | null;
  retry_policy: {
    default_retries: number;
    timeout_retries: number | null;
    rate_limit_retries: number | null;
    auth_retries: number | null;
    bad_request_retries: number | null;
    content_policy_retries: number | null;
    internal_error_retries: number | null;
  };
}

/** Editor Area mode — controlled by Title Bar (Header) */
export type PrimaryMode = 'graph' | 'chat';

/** Primary Sidebar tab — constrained by primaryMode.
 *  graph → workflow | knowledge | skills
 *  chat  → conversations | knowledge | skills
 */
export type SidebarTab = 'workflow' | 'conversations' | 'knowledge' | 'skills';

/**
 * Secondary Sidebar content mode — derived from selection context.
 *
 * Routing priority:
 *   1. Graph mode + node selected  → 'node'
 *   2. Skill selected              → 'skill'
 *   3. Knowledge library selected  → 'knowledge'
 *   4. Chat mode + conversations   → 'conversation'
 *   5. Otherwise                   → 'empty'
 */
export type SecondaryContentMode =
  | 'node'
  | 'skill'
  | 'knowledge'
  | 'conversation'
  | 'empty';

interface ConsoleState {
  primaryMode: PrimaryMode;
  sidebarTab: SidebarTab;
  /** Switch editor mode and keep sidebar tab valid. */
  setPrimaryMode: (mode: PrimaryMode) => void;
  /** Switch Primary Sidebar tab (must be valid for current primaryMode). */
  setSidebarTab: (tab: SidebarTab) => void;

  // WebSocket
  ws: ConsoleWebSocket | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';

  // Current state
  sessionId: string;
  executionId: string | null;
  currentPlan: PlanIR | null;
  currentExecution: ExecutionIR | null;
  liveExecution: ExecutionIR | null; // Store live execution when viewing checkpoint
  checkpointExecution: ExecutionIR | null;
  checkpoints: CheckpointIR[];
  lastRestoredCheckpointId: string | null;
  lastRestoredCheckpointKey: { execution_id: string | null; checkpoint_id: string } | null;

  viewMode: 'live' | 'checkpoint';
  selectedCheckpointKey: { execution_id: string; checkpoint_id: string } | null;
  getViewExecution: () => ExecutionIR | null;

  // Run settings
  runSettings: RunSettings;
  isRunSettingsOpen: boolean;
  setRunSettingsOpen: (open: boolean) => void;
  updateRunSettings: (updates: Partial<RunSettings>) => void;
  resetRunSettings: () => void;
  saveRunSettingsDefaults: () => void;

  // UI state
  toasts: Toast[];
  toastKeys: Record<string, string>;
  activityLogs: ActivityLog[];
  nodeObservations: Record<string, Record<string, Record<string, any>>>;
  spanStore: SpanStore;
  executionLineageMap: Record<string, { parentExecutionId: string; parentCheckpointId?: string; replayMode?: string }>;
  serverLogLevel: 'debug' | 'info' | 'warning' | 'error';
  loadingWorkflowName: string | null; // Track which workflow is being loaded
  workflows: WorkflowSummary[];
  isLoadingWorkflows: boolean;
  bottomPanelTab: 'observability' | 'checkpoints' | 'context' | 'logs' | 'knowledge';
  setBottomPanelTab: (tab: 'observability' | 'checkpoints' | 'context' | 'logs' | 'knowledge') => void;

  // React Flow state
  nodes: any[];
  edges: any[];
  selectedNodeId: string | null;

  // Skill selection (for Secondary Sidebar routing)
  selectedSkillId: string | null;
  selectSkill: (skillId: string | null) => void;

  /** Derive which content the Secondary Sidebar should display. */
  getSecondaryContentMode: () => SecondaryContentMode;

  // Actions
  connect: (sessionId: string) => void;
  disconnect: () => void;
  handleEvent: (event: AnyServerEvent) => void;

  // Plan actions
  setPlan: (plan: PlanIR) => void;
  updatePlan: (plan: PlanIR) => void;

  // Node actions
  addNode: (nodeType: string, position: { x: number; y: number }) => void;
  updateNode: (nodeId: string, updates: Partial<NodeIR>) => void;
  updateNodePosition: (nodeId: string, position: { x: number; y: number }) => void;
  deleteNode: (nodeId: string) => void;
  selectNode: (nodeId: string | null) => void;

  // Edge actions
  addEdge: (source: string, target: string) => void;
  deleteEdge: (edgeId: string) => void;

  // Execution actions
  setExecution: (execution: ExecutionIR) => void;
  updateNodeStatus: (nodeId: string, status: any) => void;

  // Checkpoint actions
  addCheckpoint: (checkpoint: CheckpointIR) => void;
  loadCheckpoint: (checkpointId: string, executionId?: string) => void;
  exitCheckpointView: () => void;

  // Restore helpers
  clearCurrentExecutionOutputsForFreshReplay: () => void;
  prepareRestoreFromCheckpoint: (params: {
    checkpointId: string;
    executionId?: string | null;
    planId?: string;
  }) => void;

  // Toast actions
  showToast: (message: string, type: 'success' | 'error' | 'info') => void;
  showToastOnce: (key: string, message: string, type: 'success' | 'error' | 'info') => void;
  removeToast: (id: string) => void;
  removeToastByKey: (key: string) => void;
  addActivityLog: (log: {
    level?: ActivityLog['level'];
    message: string;
    detail?: string;
    timestamp?: string;
  }) => void;

  // Workflow actions
  saveWorkflow: (workflowName: string) => void;
  loadWorkflow: (workflowName: string) => void;
  requestWorkflows: () => void;

  // Command actions
  sendCommand: (command: any) => boolean;
  sendPatchPlan: (patches: any[]) => void;

  // Tool statistics
  getToolStatistics: () => ToolStatistics;

  // Span actions for timeline
  updateSpan: (event: any) => void;
  getSpanTree: (executionId: string) => any;
  clearSpans: (executionId?: string) => void;

  // Knowledge Base actions
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
  ingestProgress: number;
  ingestCurrentFile: string;
  ingestFilesProcessed: number;
  ingestTotalFiles: number;
  requestKnowledgeLibraries: () => void;
  setKnowledgeLibraries: (libraries: KnowledgeLibrary[]) => void;
  createKnowledgeLibrary: (config: {
    name: string;
    description: string;
    mode: RAGMode;
    knowledge_dir: string;
    metadata?: Record<string, any>;
   
    strategies?: string[];
    embedding_provider?: string;
    contextual_retrieval?: boolean;
  }) => void;
  deleteKnowledgeLibrary: (libraryId: string) => void;
  addKnowledgeLibrary: (library: KnowledgeLibrary) => void;
  removeKnowledgeLibrary: (libraryId: string) => void;
  selectKnowledgeLibrary: (libraryId: string | null) => void;
  searchKnowledge: (query: string, libraryId?: string, mode?: RAGMode, topK?: number) => void;
  setKnowledgeSearchResults: (results: KnowledgeSearchResult[], query: string, modeUsed?: string, strategiesUsed?: string[], quality?: QualitySummary | null) => void;
  // Ingest actions
  ingestKnowledgeFiles: (libraryId: string, paths: string[]) => void;
  handleIngestProgress: (data: {
    library_id: string;
    progress: number;
    current_file: string;
    files_processed: number;
    total_files: number;
  }) => void;
  handleIngestComplete: (data: {
    library_id: string;
    success: boolean;
    stats: Record<string, any>;
    message: string;
  }) => void;
  updateKnowledgeLibrary: (library: KnowledgeLibrary) => void;
  editKnowledgeLibrary: (libraryId: string, updates: Record<string, any>) => void;
  rebuildKnowledgeIndex: (libraryId: string, incremental?: boolean) => void;
  cancelIngest: () => void;
  clearKnowledgeSearch: () => void;
  handleKnowledgeError: (error: string, operation: string) => void;
  // Skill event subscription API (for useSkillsLogic hook)
  _skillEventHandlers: Map<string, Set<(event: unknown) => void>>;
  registerSkillEventHandler: (eventType: string, handler: (event: unknown) => void) => () => void;

  // Document management state
  documents: KnowledgeDocument[];
  selectedDocumentId: string | null;
  isLoadingDocuments: boolean;
  chunks: KnowledgeChunk[];
  isLoadingChunks: boolean;
  chunkPreviews: ChunkPreview[];
  isPreviewingChunks: boolean;
  // Document management actions
  requestDocuments: (libraryId: string) => void;
  setDocuments: (documents: KnowledgeDocument[]) => void;
  selectDocument: (docId: string | null) => void;
  deleteDocument: (libraryId: string, docId: string) => void;
  removeDocument: (docId: string) => void;
  disableDocument: (libraryId: string, docId: string) => void;
  enableDocument: (libraryId: string, docId: string) => void;
  updateDocumentStatus: (docId: string, status: string) => void;
  requestChunks: (libraryId: string, docId: string) => void;
  setChunks: (chunks: KnowledgeChunk[]) => void;
  previewChunks: (content: string, chunkSize: number, chunkOverlap: number, strategy: string) => void;
  setChunkPreviews: (previews: ChunkPreview[]) => void;
  clearChunks: () => void;
}

export const useConsoleStore = create<ConsoleState>((set, get) => ({
  primaryMode: 'graph' as PrimaryMode,
  sidebarTab: 'workflow' as SidebarTab,

  setPrimaryMode: (mode: PrimaryMode) => {
    const prev = get().sidebarTab;
    let next = prev;
    if (mode === 'chat') {
      // workflow is graph-only → remap to conversations
      if (prev === 'workflow') next = 'conversations';
    } else {
      // conversations is chat-only → remap to workflow
      if (prev === 'conversations') next = 'workflow';
    }
    const updates: Partial<ConsoleState> = {
      primaryMode: mode,
      sidebarTab: next,
    };
    // When switching to chat conversations, clear cross-tab context so
    // the right panel can reliably show conversation settings.
    if (mode === 'chat' && next === 'conversations') {
      updates.selectedSkillId = null;
      updates.selectedLibraryId = null;
      updates.selectedNodeId = null;
    }
    set(updates as Pick<ConsoleState, 'primaryMode' | 'sidebarTab' | 'selectedSkillId' | 'selectedLibraryId' | 'selectedNodeId'>);
  },

  setSidebarTab: (tab: SidebarTab) => {
    // Guard: enforce mode-specific constraints
    const mode = get().primaryMode;
    if (mode === 'graph' && tab === 'conversations') return;
    if (mode === 'chat' && tab === 'workflow') return;
    if (tab === 'skills') {
      // Keep current skill selection for quick context continuity.
      set({ sidebarTab: tab, selectedLibraryId: null, selectedNodeId: null });
      return;
    }
    if (tab === 'knowledge') {
      // Keep current library selection; clear stale skill/node selection.
      set({ sidebarTab: tab, selectedSkillId: null, selectedNodeId: null });
      return;
    }
    // workflow / conversations should not retain skill or knowledge context.
    set({ sidebarTab: tab, selectedSkillId: null, selectedLibraryId: null });
  },

  // Initial state
  ws: null,
  connectionStatus: 'disconnected',
  sessionId: `session_${Date.now()}`,
  executionId: null,
  currentPlan: null,
  currentExecution: null,
  liveExecution: null,
  checkpointExecution: null,
  checkpoints: [],
  lastRestoredCheckpointId: null,
  lastRestoredCheckpointKey: null,

  viewMode: 'live',
  selectedCheckpointKey: null,
  getViewExecution: () => {
    const state = get();
    return state.viewMode === 'checkpoint' ? state.checkpointExecution : state.liveExecution || state.currentExecution;
  },

  ...getInitialRunSettingsState(),

  toasts: [],
  toastKeys: {},
  activityLogs: [],
  nodeObservations: {},
  spanStore: {},
  executionLineageMap: {},
  serverLogLevel: 'info',
  loadingWorkflowName: null,
  workflows: [],
  isLoadingWorkflows: false,
  bottomPanelTab: 'observability',
  setBottomPanelTab: (tab) => set({ bottomPanelTab: tab }),

  nodes: [],
  edges: [],
  selectedNodeId: null,

  selectedSkillId: null,
  selectSkill: (skillId: string | null) => set({ selectedSkillId: skillId }),

  getSecondaryContentMode: (): SecondaryContentMode => {
    const { primaryMode, selectedNodeId, selectedSkillId, selectedLibraryId, sidebarTab } = get();
    // Make routing explicit by active tab to avoid stale cross-mode context.
    if (primaryMode === 'graph' && sidebarTab === 'workflow' && selectedNodeId) return 'node';
    if (sidebarTab === 'skills' && selectedSkillId) return 'skill';
    if (sidebarTab === 'knowledge' && selectedLibraryId) return 'knowledge';
    if (primaryMode === 'chat' && sidebarTab === 'conversations') return 'conversation';
    // Default
    return 'empty';
  },

  // Skill event subscription (for useSkillsLogic)
  _skillEventHandlers: new Map(),
  registerSkillEventHandler: (eventType: string, handler: (event: unknown) => void) => {
    const handlers = get()._skillEventHandlers;
    if (!handlers.has(eventType)) {
      handlers.set(eventType, new Set());
    }
    handlers.get(eventType)!.add(handler);
    // Return unsubscribe function
    return () => {
      const set = handlers.get(eventType);
      if (set) {
        set.delete(handler);
        if (set.size === 0) handlers.delete(eventType);
      }
    };
  },

  // Knowledge Base state
  ...initialKnowledgeState,

  // WebSocket actions
  ...createWsActions(set, get),

  // Node/Edge actions
  ...createNodeActions(set, get),

  // Execution actions
  ...createExecutionActions(set, get),

  // Checkpoint actions
  ...createCheckpointActions(set, get),

  // Run settings actions
  ...createRunSettingsActions(set, get),

  // Toast/activity actions
  ...createToastActions(set, get),

  // Command actions
  ...createCommandActions(set, get),

  // Plan actions
  setPlan: (plan: PlanIR) => {
    set({ currentPlan: plan });
  },

  updatePlan: (plan: PlanIR) => {
    set({ currentPlan: plan });
  },





  // Workflow management
  ...createWorkflowActions(set, get),

  // Tool statistics
  ...createToolStatsActions(set, get),

  // Span actions for timeline
  ...createSpanActions(set, get),

  // Knowledge Base actions
  ...createKnowledgeActions(set, get),
}));
