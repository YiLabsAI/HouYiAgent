import type { ExecutionIR } from '@/types/ir';

type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

type NodeStatusUpdate = {
  status?: string;
  error?: string | null;
  inputs?: Record<string, any> | null;
  outputs?: Record<string, any> | null;
  started_at?: string | null;
  completed_at?: string | null;
  streaming_output?: string | null;
  metadata?: Record<string, any>;
};

export const createExecutionActions = (set: StoreSet, get: StoreGet) => ({
  setExecution: (execution: ExecutionIR) => {
    set({ currentExecution: execution });
  },

  updateNodeStatus: (nodeId: string, status: NodeStatusUpdate) => {
    const { currentExecution, liveExecution, viewMode } = get();
    const inCheckpointView = viewMode === 'checkpoint';
    const targetExecution = inCheckpointView ? liveExecution : currentExecution;
    if (!targetExecution) return;

    if (!targetExecution.node_executions[nodeId]) {
      targetExecution.node_executions[nodeId] = {
        node_id: nodeId,
        status: 'pending',
        started_at: null,
        completed_at: null,
        inputs: {},
        outputs: {},
        error: null,
        streaming_output: '',
        metadata: {},
      };
    }

    Object.assign(targetExecution.node_executions[nodeId], status);

    if (inCheckpointView) {
      set({ liveExecution: { ...targetExecution } });
    } else {
      set({ currentExecution: { ...targetExecution } });
    }

    set((state: any) => ({
      nodes: state.nodes.map((node: any) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, status: status.status } }
          : node,
      ),
    }));
  },

  clearCurrentExecutionOutputsForFreshReplay: () => {
    const state = get();
    const target = state.getViewExecution();
    if (!target?.node_executions) return;

    const clearedNodeExecutions: any = { ...target.node_executions };
    Object.keys(clearedNodeExecutions).forEach((nodeId) => {
      clearedNodeExecutions[nodeId] = {
        ...clearedNodeExecutions[nodeId],
        streaming_output: '',
        outputs: {},
        error: null,
        started_at: null,
        completed_at: null,
      };
    });

    if (state.viewMode === 'checkpoint') {
      set({
        checkpointExecution: {
          ...(state.checkpointExecution as any),
          node_executions: clearedNodeExecutions,
        },
      });
    } else {
      set({
        currentExecution: {
          ...(state.currentExecution as any),
          node_executions: clearedNodeExecutions,
        },
      });
    }
  },
});
