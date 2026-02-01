import type { ExecutionIR } from '@/types/ir';

type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

type RestoreParams = {
  checkpointId: string;
  executionId?: string | null;
  planId?: string;
};

export const createCheckpointActions = (set: StoreSet, get: StoreGet) => ({
  addCheckpoint: (checkpoint: any) => {
    set((state: any) => ({
      checkpoints: [...state.checkpoints, checkpoint],
    }));
  },

  loadCheckpoint: (checkpointId: string, executionId?: string) => {
    const checkpoint = executionId
      ? get().checkpoints.find(
          (cp: any) => cp.checkpoint_id === checkpointId && cp.execution_id === executionId,
        )
      : get().checkpoints.find((cp: any) => cp.checkpoint_id === checkpointId);
    if (!checkpoint) {
      console.error('[loadCheckpoint] Checkpoint not found:', { checkpointId, executionId });
      return;
    }

    const { currentExecution, liveExecution, viewMode } = get();
    const inCheckpointView = viewMode === 'checkpoint';

    console.log('[loadCheckpoint] Loading checkpoint:', checkpointId, {
      checkpointExecutionId: checkpoint.execution_id,
      viewMode,
      hasLiveExecution: !!liveExecution,
      hasCurrentExecution: !!currentExecution,
    });

    if (!inCheckpointView) {
      console.log('[loadCheckpoint] First time entering checkpoint view, saving live execution');
      set({
        viewMode: 'checkpoint',
        selectedCheckpointKey: {
          execution_id: checkpoint.execution_id,
          checkpoint_id: checkpointId,
        },
        liveExecution: currentExecution,
      });
    } else {
      console.log('[loadCheckpoint] Already in checkpoint view, switching checkpoint');
      set({
        viewMode: 'checkpoint',
        selectedCheckpointKey: {
          execution_id: checkpoint.execution_id,
          checkpoint_id: checkpointId,
        },
      });
    }

    const snapshot = checkpoint.execution_snapshot as ExecutionIR | undefined;
    const hasSnapshotNodeExecs = Boolean((snapshot as any)?.node_executions && Object.keys((snapshot as any).node_executions || {}).length > 0);
    const resolvedExecution = (() => {
      if (snapshot && hasSnapshotNodeExecs) return snapshot;
      const state = get();
      const candidates = [state.checkpointExecution, state.liveExecution, state.currentExecution].filter(Boolean) as ExecutionIR[];
      const match = candidates.find((ex) => ex.execution_id === checkpoint.execution_id);
      return match ?? snapshot;
    })();

    if (resolvedExecution) {
      const execution = resolvedExecution as ExecutionIR;
      const nextNodes = (get().nodes || []).map((node: any) => {
        const nodeExec = (execution as any)?.node_executions?.[node.id];
        const status = nodeExec?.status ?? 'pending';
        return {
          ...node,
          data: {
            ...node.data,
            status,
          },
        };
      });

      set({ currentExecution: execution, checkpointExecution: execution, nodes: nextNodes });
      console.log('[loadCheckpoint] Loaded execution snapshot');
    }
  },

  exitCheckpointView: () => {
    const state = get();
    if (state.viewMode !== 'checkpoint') return;
    const live = state.liveExecution;
    const nextNodes = (state.nodes || []).map((node: any) => {
      const nodeExec = (live as any)?.node_executions?.[node.id];
      const status = nodeExec?.status ?? 'pending';
      return {
        ...node,
        data: {
          ...node.data,
          status,
        },
      };
    });
    set({
      viewMode: 'live',
      selectedCheckpointKey: null,
      checkpointExecution: null,
      currentExecution: live,
      liveExecution: null,
      nodes: nextNodes,
    });
  },

  prepareRestoreFromCheckpoint: ({ checkpointId, executionId, planId }: RestoreParams) => {
    const state = get();
    const checkpoint = (state.checkpoints || []).find((cp: any) => {
      if (executionId) {
        return cp.checkpoint_id === checkpointId && cp.execution_id === executionId;
      }
      return cp.checkpoint_id === checkpointId;
    });

    if (state.viewMode === 'checkpoint') {
      state.exitCheckpointView();
    }

    const resolvedExecutionId = checkpoint?.execution_id || executionId || null;

    set({
      lastRestoredCheckpointId: checkpointId,
      lastRestoredCheckpointKey: {
        execution_id: resolvedExecutionId,
        checkpoint_id: checkpointId,
      },
      currentExecution: resolvedExecutionId
        ? {
            execution_id: resolvedExecutionId,
            plan_id: planId || state.currentPlan?.plan_id || '',
            status: 'paused',
            node_executions: {},
            context: {},
            started_at: null,
            completed_at: null,
            error: null,
            metadata: {},
          }
        : state.currentExecution,
    });

    const snapshot = checkpoint?.execution_snapshot as ExecutionIR | undefined;
    if (snapshot?.node_executions && Object.keys(snapshot.node_executions || {}).length > 0) {
      const nextNodes = (get().nodes || []).map((node: any) => {
        const nodeExec = (snapshot as any)?.node_executions?.[node.id];
        const status = nodeExec?.status ?? 'pending';
        return {
          ...node,
          data: {
            ...node.data,
            status,
          },
        };
      });
      set({ nodes: nextNodes });
    } else {
      const nextNodes = (get().nodes || []).map((node: any) => ({
        ...node,
        data: {
          ...node.data,
          status: 'pending',
        },
      }));
      set({ nodes: nextNodes });
    }
  },
});
