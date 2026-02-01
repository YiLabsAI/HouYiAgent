import { createCheckpointActions } from '@/stores/storeActions/checkpointActions';

describe('checkpointActions.loadCheckpoint', () => {
  it('should update nodes status using fallback execution when checkpoint execution_snapshot is empty', () => {
    const execId = 'exec_1';
    const checkpointId = 'cp_1';

    let state: any = {
      checkpoints: [
        {
          checkpoint_id: checkpointId,
          execution_id: execId,
          execution_snapshot: { execution_id: execId, node_executions: {} },
        },
      ],
      viewMode: 'live',
      selectedCheckpointKey: null,
      currentExecution: {
        execution_id: execId,
        node_executions: {
          node_a: { status: 'completed' },
          node_b: { status: 'running' },
        },
      },
      liveExecution: null,
      checkpointExecution: null,
      nodes: [
        { id: 'node_a', data: { status: 'pending' } },
        { id: 'node_b', data: { status: 'pending' } },
        { id: 'node_c', data: { status: 'pending' } },
      ],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;

    const actions = createCheckpointActions(set, get);

    actions.loadCheckpoint(checkpointId, execId);

    expect(state.viewMode).toBe('checkpoint');
    expect(state.selectedCheckpointKey).toEqual({ execution_id: execId, checkpoint_id: checkpointId });

    const statusesById = Object.fromEntries(state.nodes.map((n: any) => [n.id, n.data.status]));
    expect(statusesById.node_a).toBe('completed');
    expect(statusesById.node_b).toBe('running');
    expect(statusesById.node_c).toBe('pending');
  });

  it('should prefer checkpoint execution_snapshot when it contains node_executions', () => {
    const execId = 'exec_2';
    const checkpointId = 'cp_2';

    let state: any = {
      checkpoints: [
        {
          checkpoint_id: checkpointId,
          execution_id: execId,
          execution_snapshot: {
            execution_id: execId,
            node_executions: {
              node_a: { status: 'failed' },
            },
          },
        },
      ],
      viewMode: 'live',
      selectedCheckpointKey: null,
      currentExecution: {
        execution_id: execId,
        node_executions: {
          node_a: { status: 'completed' },
        },
      },
      liveExecution: null,
      checkpointExecution: null,
      nodes: [{ id: 'node_a', data: { status: 'pending' } }],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;

    const actions = createCheckpointActions(set, get);

    actions.loadCheckpoint(checkpointId, execId);

    expect(state.viewMode).toBe('checkpoint');
    expect(state.selectedCheckpointKey).toEqual({ execution_id: execId, checkpoint_id: checkpointId });
    expect(state.nodes[0].data.status).toBe('failed');
  });
});

describe('checkpointActions.prepareRestoreFromCheckpoint', () => {
  it('should not clear currentExecution when preparing restore from live mode', () => {
    const execId = 'exec_live';
    const checkpointId = 'cp_1';

    let state: any = {
      checkpoints: [
        {
          checkpoint_id: checkpointId,
          execution_id: execId,
          execution_snapshot: { execution_id: execId, node_executions: {} },
        },
      ],
      viewMode: 'live',
      selectedCheckpointKey: null,
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: {
        execution_id: execId,
        plan_id: 'plan_1',
        status: 'running',
        node_executions: {},
        context: {},
        started_at: null,
        completed_at: null,
        error: null,
        metadata: {},
      },
      liveExecution: null,
      checkpointExecution: null,
      nodes: [{ id: 'node_a', data: { status: 'pending' } }],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    const actions = createCheckpointActions(set, get);

    actions.prepareRestoreFromCheckpoint({ checkpointId, executionId: execId, planId: 'plan_1' });

    expect(state.viewMode).toBe('live');
    expect(state.currentExecution?.execution_id).toBe(execId);
    expect(state.lastRestoredCheckpointKey).toEqual({ execution_id: execId, checkpoint_id: checkpointId });
  });
});
