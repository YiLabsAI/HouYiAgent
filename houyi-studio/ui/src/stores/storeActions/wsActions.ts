import type { ExecutionIR } from '@/types/ir';
import { ConsoleWebSocket } from '@/utils/websocket';
import { logger } from '@/utils/logger';

type StoreSet = (partial: any) => void;
type StoreGet = () => any;

const normalizeEventType = (eventType: any): string | null => {
  if (typeof eventType === 'string') return eventType;
  if (eventType && typeof eventType === 'object' && typeof eventType.value === 'string') {
    return eventType.value;
  }
  return null;
};

export const createWsActions = (set: StoreSet, get: StoreGet) => ({
  connect: (sessionId: string) => {
    // Guard: if already connected or connecting with the same session, skip
    // re-creation (happens during HMR re-mounts and StrictMode double-mount)
    const existing = get().ws;
    if (existing && get().sessionId === sessionId && (existing.isConnected() || existing.isConnecting())) {
      console.log('[wsActions] Already connected/connecting with session', sessionId, '— skipping');
      return;
    }
    // Clean up any stale connection before creating a new one
    if (existing) {
      existing.disconnect();
    }


    const ws = new ConsoleWebSocket(sessionId);

    set({ connectionStatus: 'connecting' });

    ws.onEvent((event) => {
      get().handleEvent(event);
    });

    // Track whether we ever successfully connected.  Only show the
    // "not connected" toast after a previously successful connection is
    // lost — not during the initial handshake (avoids a flash on page
    // refresh where the old WS fires 'close' before the new one opens).
    let everConnected = false;

    ws.onStatus((status) => {
      const toastKey = 'backend-connection';
      if (status === 'connected') {
        everConnected = true;
        set({ connectionStatus: 'connected' });
        get().removeToastByKey(toastKey);
        return;
      }

      if (status === 'disconnected') {
        set({ connectionStatus: 'disconnected' });
        // Only show warning after we were previously connected
        if (everConnected) {
          get().showToastOnce(toastKey, 'Backend not connected. Please start the server.', 'error');
        }
        return;
      }

      set({ connectionStatus: 'error' });
      if (everConnected) {
        get().showToastOnce(toastKey, 'Backend not connected. Please start the server.', 'error');
      }
    });

    ws.connect();

    set({ ws, sessionId });

    // Connect the frontend logger to bridge logs to backend terminal
    logger.connect((cmd) => get().sendCommand(cmd), sessionId);
  },

  disconnect: () => {
    const { ws } = get();
    if (ws) {
      ws.disconnect();
    }
    logger.disconnect();
    set({ ws: null, connectionStatus: 'disconnected' });
  },

  handleEvent: (event: any) => {
    const eventType = normalizeEventType(event?.event_type);
    console.log('[Store] Received event:', eventType ?? event?.event_type, event);

    switch (eventType) {
      case 'plan_created':
      case 'plan_updated':
        if (!('plan' in event)) {
          console.warn('[Store] Plan event missing plan payload:', event);
          break;
        }

        console.log('[Store] Received plan event:', eventType, event.plan);
        const previousPlanId = get().currentPlan?.plan_id;
        const nextPlanId = event.plan?.plan_id;
        const planChanged = Boolean(nextPlanId && previousPlanId && nextPlanId !== previousPlanId);
        {
          const planId = event.plan?.plan_id ? `plan_id=${event.plan.plan_id}` : undefined;
          const changes =
            eventType === 'plan_updated' && 'changes' in event && Array.isArray(event.changes) && event.changes.length
              ? `changes: ${event.changes.join(', ')}`
              : undefined;
          const detail = [planId, changes].filter(Boolean).join(' · ');
          get().addActivityLog({
            message: event.event_type === 'plan_created' ? 'Plan created' : 'Plan updated',
            detail: detail || undefined,
          });

          if (planChanged) {
            set({
              viewMode: 'live',
              selectedCheckpointKey: null,
              checkpointExecution: null,
              liveExecution: null,
              currentExecution: null,
              checkpoints: [],
              selectedNodeId: null,
            });
          }
        }
        set({ currentPlan: event.plan });
        if (event.plan) {
          const nodesList = Array.isArray(event.plan.nodes)
            ? event.plan.nodes
            : Object.values(event.plan.nodes);

          console.log('[Store] Processing nodes:', nodesList);

          const existingStatusById = new Map(
            (get().nodes || []).map((n: any) => [n.id, n?.data?.status]),
          );
          const layoutPositions = event.plan?.layout?.positions || {};
          const nodes = nodesList.map((node: any) => {
            const metadata = node.metadata ?? node.data?.metadata ?? {};
            const label = metadata.label || node.label || node.node_id;
            const previousStatus = existingStatusById.get(node.node_id);
            const layoutPosition = node.node_id ? layoutPositions[node.node_id] : null;
            return {
              id: node.node_id,
              type: (node.node_type?.value || node.node_type || 'default').toLowerCase(),
              position: layoutPosition || node.position || { x: 0, y: 0 },
              data: {
                label,
                nodeType: node.node_type,
                config: node.config || {},
                inputs: node.inputs || {},
                outputs: node.outputs || {},
                metadata,
                status: planChanged ? 'pending' : (previousStatus ?? 'pending'),
              },
            };
          });

          const backendEdges = (event.plan.edges || []).map((edge: any) => ({
            id: `${edge.source_node_id || edge.source}-${edge.target_node_id || edge.target}`,
            source: edge.source_node_id || edge.source,
            target: edge.target_node_id || edge.target,
            type: 'default',
          }));

          const currentEdges = get().edges;
          const mergedEdges = [...backendEdges];

          currentEdges.forEach((frontendEdge: any) => {
            const existsInBackend = backendEdges.some((be: any) => be.id === frontendEdge.id);
            if (!existsInBackend) {
              mergedEdges.push(frontendEdge);
            }
          });

          console.log('[Store] Restored nodes:', nodes.length, 'backend edges:', backendEdges.length, 'merged edges:', mergedEdges.length);
          set({ nodes, edges: mergedEdges });

          if (event.event_type === 'plan_updated' && get().loadingWorkflowName) {
            if (nodes.length === 0) {
              get().showToast(
                `Workflow "${get().loadingWorkflowName}" loaded but has no nodes. Check the saved workflow file.`,
                'error',
              );
            }
          }

          const { loadingWorkflowName } = get();
          if (loadingWorkflowName && nodes.length > 0) {
            get().showToastOnce(
              `workflow_loaded:${loadingWorkflowName}`,
              `Workflow "${loadingWorkflowName}" loaded successfully!`,
              'success',
            );
            set({ loadingWorkflowName: null });
          }
        }
        break;

      case 'restore_checkpoint_result':
        {
          get().addActivityLog({
            level: event.success ? 'info' : 'error',
            message: `Restore checkpoint: ${event.success ? 'succeeded' : 'failed'}`,
            detail: `checkpoint=${event.checkpoint_id}` +
              (event.execution_id ? ` · execution=${event.execution_id}` : '') +
              (event.replay_mode ? ` · mode=${event.replay_mode}` : ''),
          });

          if (!event.success) {
            const state = get();
            if (state.lastRestoredCheckpointKey?.checkpoint_id === event.checkpoint_id) {
              set({
                lastRestoredCheckpointId: null,
                lastRestoredCheckpointKey: null,
                currentExecution: state.liveExecution || null,
                executionId: state.liveExecution?.execution_id ?? null,
              });
            }
            get().showToast(
              event.message ? `Restore failed: ${event.message}` : 'Restore failed. Please check logs.',
              'error',
            );
          } else {
            get().showToast(
              event.message ? event.message : `Restored from checkpoint ${event.checkpoint_id}`,
              'success',
            );

            if (event.execution_id) {
              const state = get();
              if (state.lastRestoredCheckpointKey?.checkpoint_id === event.checkpoint_id) {
                const currentPlanId = state.currentPlan?.plan_id || '';
                const nextExecution = state.currentExecution
                  ? {
                      ...state.currentExecution,
                      execution_id: event.execution_id,
                      status: 'paused',
                    }
                  : ({
                      execution_id: event.execution_id,
                      plan_id: currentPlanId,
                      status: 'paused',
                      node_executions: {},
                      context: {},
                      started_at: null,
                      completed_at: null,
                      error: null,
                      metadata: {},
                    } as ExecutionIR);
                set({
                  lastRestoredCheckpointKey: {
                    execution_id: event.execution_id,
                    checkpoint_id: event.checkpoint_id,
                  },
                  currentExecution: nextExecution,
                  executionId: event.execution_id,
                });
              }
            }
          }
        }
        break;

      case 'node_status':
        {
          get().addActivityLog({
            level: event.status === 'failed' ? 'error' : 'info',
            message: `Node ${event.node_id}: ${event.status}`,
            detail: `execution=${event.execution_id}`,
          });

          if (event.observation) {
            set((state: any) => ({
              nodeObservations: {
                ...(state.nodeObservations || {}),
                [event.execution_id]: {
                  ...(state.nodeObservations?.[event.execution_id] || {}),
                  [event.node_id]: event.observation,
                },
              },
            }));
          }

          // Ensure we have a target execution object to attach node status updates to.
          // Otherwise early node_status events can be dropped (e.g. verify/route/logic nodes)
          // before execution_status creates currentExecution.
          {
            const { currentExecution, liveExecution, viewMode } = get();
            const inCheckpointView = viewMode === 'checkpoint';
            const target = inCheckpointView ? liveExecution : currentExecution;

            if (!target || target.execution_id !== event.execution_id) {
              const nextExecution: ExecutionIR = {
                execution_id: event.execution_id,
                plan_id: get().currentPlan?.plan_id || '',
                status: 'running',
                node_executions: {},
                context: {},
                started_at: event.timestamp ? new Date(event.timestamp).toISOString() : new Date().toISOString(),
                completed_at: null,
                error: null,
                metadata: {},
              } as ExecutionIR;

              if (inCheckpointView) {
                set({ liveExecution: nextExecution, executionId: event.execution_id });
              } else {
                set({ currentExecution: nextExecution, executionId: event.execution_id });
              }
            }
          }

          const outputs = event.outputs as any;
          const toolCalls = outputs?.tool_calls ?? [];

          // Aggregate cache-hit signals into a single Activity entry per node_status event.
          // This prevents log spam (especially for many tool calls) and keeps Summary/Activity
          // cache-hit semantics consistent across tool_calls/tool_node/llm_final.
          const toolCallCacheHitCount = toolCalls.filter((call: any) => Boolean(call?.result?.metadata?.cache_hit)).length;
          const llmFinalAnswerCacheHit = Boolean(outputs?.metadata?.llm_cache_hit);
          const toolNodeCacheHit = Boolean(
            outputs?.output?.metadata?.cache_hit
            || outputs?.output?.raw?.metadata?.cache_hit
            || outputs?.output?.raw?.metadata?.cached
          );
          const toolNodeCacheHitCount = toolNodeCacheHit ? 1 : 0;
          const llmFinalCacheHitCount = llmFinalAnswerCacheHit ? 1 : 0;
          const cacheHitTotal = toolCallCacheHitCount + toolNodeCacheHitCount + llmFinalCacheHitCount;

          if (cacheHitTotal > 0) {
            get().addActivityLog({
              level: 'info',
              message: 'cache-hit',
              detail: `hits=${cacheHitTotal} (tool_calls=${toolCallCacheHitCount}, tool_node=${toolNodeCacheHitCount}, llm_final=${llmFinalCacheHitCount}) · node=${event.node_id} · execution=${event.execution_id}`,
            });
          }
          if (event.status === 'failed' && event.error) {
            get().addActivityLog({
              level: 'error',
              message: `Node failed: ${event.node_id}`,
              detail: event.error ?? `execution=${event.execution_id}`,
            });
          }
          const eventTimestamp = event.timestamp
            ? new Date(event.timestamp).toISOString()
            : new Date().toISOString();
          const statusUpdate: any = {
            status: event.status,
            error: event.error,
          };
          if (event.execution_metadata) {
            const { currentExecution, liveExecution, viewMode } = get();
            const targetExecution = viewMode === 'checkpoint' ? liveExecution : currentExecution;
            if (targetExecution && targetExecution.execution_id === event.execution_id) {
              const updatedExecution = {
                ...targetExecution,
                metadata: event.execution_metadata,
              } as ExecutionIR;
              if (viewMode === 'checkpoint') {
                set({ liveExecution: updatedExecution, executionId: updatedExecution.execution_id });
              } else {
                set({ currentExecution: updatedExecution, executionId: updatedExecution.execution_id });
              }
            }
            // Track execution lineage from execution_metadata
            const parentExecId = event.execution_metadata?.parent_execution_id;
            if (parentExecId && event.execution_id) {
              const lineageMap = { ...get().executionLineageMap };
              if (!lineageMap[event.execution_id]) {
                lineageMap[event.execution_id] = {
                  parentExecutionId: parentExecId,
                  parentCheckpointId: event.execution_metadata?.parent_checkpoint_id,
                  replayMode: event.execution_metadata?.replay_mode,
                };
                set({ executionLineageMap: lineageMap });
              }
            }
          }
          if (event.inputs !== undefined && event.inputs !== null) {
            statusUpdate.inputs = event.inputs;
          }
          if (event.outputs !== undefined && event.outputs !== null) {
            statusUpdate.outputs = event.outputs;
          }
          if (event.status === 'running') {
            statusUpdate.started_at = eventTimestamp;
          }
          if (['completed', 'failed', 'skipped'].includes(event.status)) {
            statusUpdate.completed_at = eventTimestamp;
          }
          get().updateNodeStatus(event.node_id, statusUpdate);
        }
        break;

      case 'streaming_output':
        {
          const { currentExecution, liveExecution, viewMode } = get();
          const targetExecution = viewMode === 'checkpoint' ? liveExecution : currentExecution;
          if (!targetExecution) break;

          if (!targetExecution.node_executions[event.node_id]) {
            targetExecution.node_executions[event.node_id] = {
              node_id: event.node_id,
              status: 'running',
              started_at: null,
              completed_at: null,
              inputs: {},
              outputs: {},
              error: null,
              streaming_output: '',
              metadata: {},
            };
          }

          targetExecution.node_executions[event.node_id].streaming_output += event.chunk;

          if (viewMode === 'checkpoint') {
            set({ liveExecution: { ...targetExecution } });
          } else {
            set({ currentExecution: { ...targetExecution } });
          }
        }
        break;

      case 'checkpoint_created':
        console.log('Checkpoint created:', event.checkpoint_id);

        get().addActivityLog({
          message: 'Checkpoint created',
          detail: `execution=${event.execution_id} · #${event.sequence_number}`,
        });

        {
          const { currentExecution, liveExecution, checkpointExecution, viewMode } = get();
          const candidateExecutions = [currentExecution, liveExecution, checkpointExecution].filter(Boolean) as ExecutionIR[];
          const matchedExecution = candidateExecutions.find(
            (execution) => execution.execution_id === event.execution_id,
          );

          const deepClone = <T,>(value: T): T => {
            try {
              return JSON.parse(JSON.stringify(value)) as T;
            } catch {
              return value;
            }
          };

          if (!matchedExecution) {
            console.log('[checkpoint_created] No matching execution snapshot for checkpoint:', {
              eventExecutionId: event.execution_id,
              checkpointId: event.checkpoint_id,
              viewMode,
            });
          }

          const executionSnapshot = matchedExecution
            ? {
                execution_id: matchedExecution.execution_id,
                plan_id: matchedExecution.plan_id,
                status: matchedExecution.status,
                started_at: matchedExecution.started_at,
                completed_at: matchedExecution.completed_at,
                // IMPORTANT: deep-clone to avoid later mutations making older checkpoints look identical.
                node_executions: deepClone(matchedExecution.node_executions),
                context: deepClone(matchedExecution.context),
                error: matchedExecution.error,
                metadata: deepClone(matchedExecution.metadata),
              }
            : {};

          const newCheckpoint = {
            checkpoint_id: event.checkpoint_id,
            execution_id: event.execution_id,
            plan_id: get().currentPlan?.plan_id || '',
            sequence_number: event.sequence_number,
            trigger: event.trigger,
            created_at: new Date().toISOString(),
            execution_snapshot: executionSnapshot,
            llm_call_logs: event.llm_call_logs ?? [],
            parent_checkpoint_id: null,
            delta: null,
            metadata: event.metadata ?? {},
          };

          console.log('[checkpoint_created] Saved execution snapshot:', {
            checkpoint_id: event.checkpoint_id,
            hasSnapshot: !!matchedExecution,
            snapshotStatus: executionSnapshot.status,
            nodeCount: Object.keys(executionSnapshot.node_executions || {}).length,
          });

          set((state: any) => {
            const key = `${newCheckpoint.execution_id}:${newCheckpoint.checkpoint_id}`;
            const seen = new Set(
              (state.checkpoints || []).map((cp: any) => `${cp.execution_id}:${cp.checkpoint_id}`),
            );
            if (seen.has(key)) {
              return { checkpoints: state.checkpoints };
            }
            return { checkpoints: [...state.checkpoints, newCheckpoint] };
          });
        }
        break;

      case 'execution_status':
        {
          get().addActivityLog({
            level: event.status === 'failed' ? 'error' : 'info',
            message: `Execution status: ${event.status}`,
            detail: `execution=${event.execution_id}`,
          });
          if (event.status === 'failed') {
            get().showToast(
              event.message ? `Execution failed: ${event.message}` : 'Execution failed. Please check logs.',
              'error',
            );
          }
          const eventTimestamp = event.timestamp
            ? new Date(event.timestamp).toISOString()
            : new Date().toISOString();
          const { currentExecution, liveExecution, viewMode } = get();
          const inCheckpointView = viewMode === 'checkpoint';
          const target = inCheckpointView ? liveExecution : currentExecution;

          if (target && target.execution_id === event.execution_id) {
            const updated = { ...target, status: event.status } as ExecutionIR;
            if (event.status === 'running' && !updated.started_at) {
              updated.started_at = eventTimestamp;
            }
            if (['completed', 'failed', 'aborted'].includes(event.status)) {
              updated.completed_at = eventTimestamp;
            }
            if (event.status === 'failed' && event.message) {
              updated.error = event.message;
            }
            if (inCheckpointView) {
              set({ liveExecution: updated, executionId: updated.execution_id });
            } else {
              set({ currentExecution: updated, executionId: updated.execution_id });
            }
          } else {
            const newExecution: ExecutionIR = {
              execution_id: event.execution_id,
              plan_id: get().currentPlan?.plan_id || '',
              status: event.status,
              node_executions: {},
              context: {},
              started_at: event.status === 'running' ? eventTimestamp : new Date().toISOString(),
              completed_at: ['completed', 'failed', 'aborted'].includes(event.status)
                ? eventTimestamp
                : null,
              error: event.status === 'failed' ? event.message ?? null : null,
              metadata: {},
            } as ExecutionIR;

            if (inCheckpointView) {
              set({ liveExecution: newExecution, executionId: event.execution_id });
            } else {
              set({ currentExecution: newExecution, executionId: event.execution_id });
            }
          }
        }
        break;

      case 'retry_status':
        {
          get().addActivityLog({
            message: `Retrying node ${event.node_id}`,
            detail: `attempt ${event.attempt}/${event.max_retries}`,
          });
        }
        break;

      case 'retry_success':
        {
          get().addActivityLog({
            message: `Retry succeeded for node ${event.node_id}`,
            detail: `attempt ${event.attempt}`,
          });
        }
        break;

      case 'retry_failed':
        {
          get().addActivityLog({
            level: 'error',
            message: `Retry failed for node ${event.node_id}`,
            detail: `max_retries=${event.max_retries} · error=${event.error}`,
          });
        }
        break;

      case 'conflict':
        console.warn('Plan conflict detected:', event);
        get().addActivityLog({
          level: 'warning',
          message: 'Plan conflict detected',
          detail: `server_version=${event.server_version} · your_version=${event.your_version}`,
        });
        break;

      case 'workflow_list':
        {
          const normalized = (event.workflows || []).map((w: any) => ({
            name: w.name,
            saved_at: w.saved_at,
            nodes_count: w.nodes_count ?? w.node_count ?? 0,
            edges_count: w.edges_count ?? w.edge_count ?? 0,
          }));
          set({ workflows: normalized, isLoadingWorkflows: false });
        }
        break;

      case 'log_level':
        set({ serverLogLevel: event.level });
        break;

      case 'span_update':
        {
          // Update span store for Timeline visualization
          get().updateSpan(event);
        }
        break;

      // ====================================================================
      // Skill Events (routed to registered handlers via subscription API)
      // ====================================================================
      case 'skill_list':
      case 'skill_detail':
      case 'skill_metrics':
      case 'skill_loaded':
      case 'skill_unloaded':
      case 'skill_error':
      case 'consent_requested':
      case 'consent_result':
      case 'skill_blocked':
      case 'dry_run_result':
        {
          const handlers = get()._skillEventHandlers?.get(eventType);
          if (handlers) {
            handlers.forEach((handler: (event: unknown) => void) => {
              try {
                handler(event);
              } catch (err) {
                console.error('[Store] Skill event handler error:', err);
              }
            });
          }
        }
        break;

      // ====================================================================
      // Knowledge Base Events
      // ====================================================================
      case 'knowledge_library_list':
        {
          const libraries = event.libraries || [];
          get().setKnowledgeLibraries(libraries);
          console.log('[Store] Received knowledge libraries:', libraries.length);
        }
        break;

      case 'knowledge_library_created':
        {
          const library = event.library;
          if (library) {
            get().addKnowledgeLibrary(library);
            console.log('[Store] Knowledge library created:', library.name);
          }
        }
        break;

      case 'knowledge_library_deleted':
        {
          const libraryId = event.library_id;
          if (libraryId) {
            get().removeKnowledgeLibrary(libraryId);
            console.log('[Store] Knowledge library deleted:', libraryId);
          }
        }
        break;

      case 'knowledge_search_results':
        {
          const results = event.results || [];
          const query = event.query || '';
          const modeUsed = event.mode_used || '';
          const strategiesUsed = event.strategies_used || [];
          const quality = event.quality || null;
          get().setKnowledgeSearchResults(results, query, modeUsed, strategiesUsed, quality);
          console.log('[Store] Knowledge search results:', results.length, 'for query:', query, 'mode:', modeUsed, 'strategies:', strategiesUsed, 'quality:', quality);
        }
        break;

      case 'knowledge_library_updated':
        {
          const library = event.library;
          if (library) {
            get().updateKnowledgeLibrary(library);
            console.log('[Store] Knowledge library updated:', library.library_id);
          }
        }
        break;

      case 'knowledge_ingest_progress':
        {
          get().handleIngestProgress({
            library_id: event.library_id,
            progress: event.progress,
            current_file: event.current_file || '',
            files_processed: event.files_processed || 0,
            total_files: event.total_files || 0,
          });
          console.log('[Store] Ingest progress:', event.progress, '%');
        }
        break;

      case 'knowledge_ingest_complete':
        {
          get().handleIngestComplete({
            library_id: event.library_id,
            success: event.success,
            stats: event.stats || {},
            message: event.message || '',
          });
          console.log('[Store] Ingest complete:', event.success);
        }
        break;

      case 'knowledge_error':
        {
          const error = event.error || 'Unknown error';
          const operation = event.operation || 'operation';
          get().handleKnowledgeError(error, operation);
          console.error('[Store] Knowledge error:', operation, error);
        }
        break;

      // ====================================================================
      // Document Management Events
      // ====================================================================
      case 'document_list':
        {
          const documents = event.documents || [];
          get().setDocuments(documents);
          console.log('[Store] Received documents:', documents.length);
        }
        break;

      case 'document_detail':
        {
          // Document detail is typically handled directly by components
          console.log('[Store] Document detail received:', event.document?.doc_id);
        }
        break;

      case 'document_deleted':
        {
          const docId = event.doc_id;
          if (docId) {
            get().removeDocument(docId);
            console.log('[Store] Document deleted:', docId);
          }
        }
        break;

      case 'document_status_changed':
        {
          const docId = event.doc_id;
          const status = event.status;
          if (docId && status) {
            get().updateDocumentStatus(docId, status);
            console.log('[Store] Document status changed:', docId, status);
          }
        }
        break;

      case 'chunk_list':
        {
          if (!('chunks' in event)) {
            console.warn('[Store] chunk_list event missing chunks payload:', event);
            break;
          }
          const chunks = event.chunks || [];
          get().setChunks(chunks);
          console.log('[Store] Received chunks:', chunks.length);
        }
        break;

      case 'chunk_preview':
        {
          if (!('chunks' in event)) {
            console.warn('[Store] chunk_preview event missing chunks payload:', event);
            break;
          }
          const previews = event.chunks || [];
          get().setChunkPreviews(previews);
          console.log('[Store] Received chunk previews:', previews.length);
        }
        break;

      default:
        console.warn('Unknown event type:', eventType ?? event?.event_type);
    }
  },
});
