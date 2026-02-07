import { createWsActions } from '@/stores/storeActions/wsActions';
import { createExecutionActions } from '@/stores/storeActions/executionActions';
import { createToastActions } from '@/stores/storeActions/toastActions';
import { vi } from 'vitest';

let lastWsInstance: any = null;

vi.mock('@/utils/websocket', () => {
  class MockConsoleWebSocket {
    private eventHandler: any = null;
    private statusHandler: any = null;

    constructor(_sessionId: string) {
      lastWsInstance = this;
    }

    onEvent(handler: any) {
      this.eventHandler = handler;
      return () => {
        if (this.eventHandler === handler) this.eventHandler = null;
      };
    }

    onStatus(handler: any) {
      this.statusHandler = handler;
      return () => {
        if (this.statusHandler === handler) this.statusHandler = null;
      };
    }

    connect() {}

    disconnect() {}

    emitStatus(status: any) {
      if (this.statusHandler) this.statusHandler(status);
    }

    emitEvent(event: any) {
      if (this.eventHandler) this.eventHandler(event);
    }
  }

  return { ConsoleWebSocket: MockConsoleWebSocket };
});

const makeBaseEvent = (event_type: any, overrides: any = {}) => ({
  event_type,
  event_id: 'evt_1',
  timestamp: new Date('2026-02-03T00:00:00.000Z').toISOString(),
  session_id: 's1',
  ...overrides,
});

describe('wsActions.handleEvent', () => {
  it('should create currentExecution from node_status when execution_status has not arrived yet', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: null,
      liveExecution: null,
      executionId: null,
      nodes: [{ id: 'node_a', data: { status: 'pending' } }],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('node_status', {
        execution_id: 'exec_1',
        node_id: 'node_a',
        status: 'running',
        inputs: { x: 1 },
        outputs: { y: 2 },
      }),
    );

    expect(state.currentExecution).toBeTruthy();
    expect(state.currentExecution.execution_id).toBe('exec_1');
    expect(state.currentExecution.status).toBe('running');
    expect(state.executionId).toBe('exec_1');

    expect(state.currentExecution.node_executions.node_a).toBeTruthy();
    expect(state.currentExecution.node_executions.node_a.status).toBe('running');
    expect(state.nodes[0].data.status).toBe('running');
  });

  it('should update currentExecution status and timestamps from execution_status', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: null,
      liveExecution: null,
      executionId: null,
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('execution_status', {
        execution_id: 'exec_1',
        status: 'running',
      }),
    );

    expect(state.currentExecution.execution_id).toBe('exec_1');
    expect(state.currentExecution.status).toBe('running');
    expect(state.currentExecution.started_at).toBeTruthy();
    expect(state.currentExecution.completed_at).toBeNull();

    state.handleEvent(
      makeBaseEvent('execution_status', {
        execution_id: 'exec_1',
        status: 'failed',
        message: 'boom',
      }),
    );

    expect(state.currentExecution.status).toBe('failed');
    expect(state.currentExecution.completed_at).toBeTruthy();
    expect(state.currentExecution.error).toBe('boom');
    expect(state.toasts.some((t: any) => t.type === 'error')).toBe(true);
  });

  it('should reset execution/checkpoint state on plan change (plan_id changes)', () => {
    let state: any = {
      viewMode: 'live',
      selectedCheckpointKey: { execution_id: 'exec_1', checkpoint_id: 'cp_1' },
      checkpointExecution: { execution_id: 'exec_1', status: 'paused' },
      liveExecution: { execution_id: 'exec_1', status: 'running' },
      currentExecution: { execution_id: 'exec_1', status: 'running' },
      checkpoints: [{ checkpoint_id: 'cp_1', execution_id: 'exec_1' }],
      selectedNodeId: 'node_a',
      currentPlan: { plan_id: 'plan_1', nodes: [], edges: [], entry_node_id: 'node_a' },
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
      loadingWorkflowName: null,
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('plan_created', {
        plan: {
          plan_id: 'plan_2',
          version: 1,
          nodes: [{ node_id: 'node_a', node_type: 'tool', position: { x: 0, y: 0 }, config: {}, inputs: {}, outputs: {}, metadata: {} }],
          edges: [],
          entry_node_id: 'node_a',
          layout: { positions: { node_a: { x: 0, y: 0 } } },
        },
      }),
    );

    expect(state.currentPlan.plan_id).toBe('plan_2');
    expect(state.viewMode).toBe('live');
    expect(state.selectedCheckpointKey).toBeNull();
    expect(state.checkpointExecution).toBeNull();
    expect(state.liveExecution).toBeNull();
    expect(state.currentExecution).toBeNull();
    expect(state.checkpoints).toEqual([]);
    expect(state.selectedNodeId).toBeNull();
  });

  it('should update liveExecution (not currentExecution) when viewMode is checkpoint', () => {
    let state: any = {
      viewMode: 'checkpoint',
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: { execution_id: 'exec_other', status: 'running', node_executions: {}, context: {} },
      liveExecution: { execution_id: 'exec_1', status: 'running', node_executions: {}, context: {}, started_at: null, completed_at: null, error: null, metadata: {} },
      executionId: 'exec_1',
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('execution_status', {
        execution_id: 'exec_1',
        status: 'completed',
      }),
    );

    expect(state.liveExecution.status).toBe('completed');
    expect(state.currentExecution.status).toBe('running');
  });

  it('should show load-workflow error toast when plan_updated arrives with loadingWorkflowName but no nodes', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1', nodes: [], edges: [], entry_node_id: 'node_a' },
      currentExecution: null,
      liveExecution: null,
      executionId: null,
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
      loadingWorkflowName: 'position_test',
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('plan_updated', {
        plan: {
          plan_id: 'plan_1',
          version: 1,
          nodes: [],
          edges: [],
          entry_node_id: 'node_a',
          layout: { positions: {} },
        },
        changes: [],
      }),
    );

    expect(state.toasts.some((t: any) => t.type === 'error')).toBe(true);
  });

  it('should show load-workflow success toast once and clear loadingWorkflowName when plan_updated has nodes', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1', nodes: [], edges: [], entry_node_id: 'node_a' },
      currentExecution: null,
      liveExecution: null,
      executionId: null,
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
      loadingWorkflowName: 'position_test',
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('plan_updated', {
        plan: {
          plan_id: 'plan_1',
          version: 1,
          nodes: [
            {
              node_id: 'node_a',
              node_type: 'tool',
              position: { x: 0, y: 0 },
              config: {},
              inputs: {},
              outputs: {},
              metadata: {},
            },
          ],
          edges: [],
          entry_node_id: 'node_a',
          layout: { positions: { node_a: { x: 0, y: 0 } } },
        },
        changes: [],
      }),
    );

    expect(state.loadingWorkflowName).toBeNull();
    expect(state.toasts.some((t: any) => t.type === 'success')).toBe(true);
    expect(Object.keys(state.toastKeys).some((k) => k.startsWith('workflow_loaded:'))).toBe(true);
  });

  it('should append streaming_output chunks into node execution (and create node execution if missing)', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: {
        execution_id: 'exec_1',
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
      executionId: 'exec_1',
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('streaming_output', {
        execution_id: 'exec_1',
        node_id: 'node_a',
        chunk: 'Hello',
        is_final: false,
      }),
    );
    state.handleEvent(
      makeBaseEvent('streaming_output', {
        execution_id: 'exec_1',
        node_id: 'node_a',
        chunk: ' world',
        is_final: true,
      }),
    );

    expect(state.currentExecution.node_executions.node_a).toBeTruthy();
    expect(state.currentExecution.node_executions.node_a.streaming_output).toBe('Hello world');
  });

  it('should write streaming_output into liveExecution when viewMode is checkpoint (not currentExecution)', () => {
    let state: any = {
      viewMode: 'checkpoint',
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: {
        execution_id: 'exec_current',
        plan_id: 'plan_1',
        status: 'running',
        node_executions: {
          node_a: {
            node_id: 'node_a',
            status: 'running',
            started_at: null,
            completed_at: null,
            inputs: {},
            outputs: {},
            error: null,
            streaming_output: 'current:',
            metadata: {},
          },
        },
        context: {},
        started_at: null,
        completed_at: null,
        error: null,
        metadata: {},
      },
      liveExecution: {
        execution_id: 'exec_live',
        plan_id: 'plan_1',
        status: 'running',
        node_executions: {},
        context: {},
        started_at: null,
        completed_at: null,
        error: null,
        metadata: {},
      },
      executionId: 'exec_live',
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('streaming_output', {
        execution_id: 'exec_live',
        node_id: 'node_a',
        chunk: 'live',
        is_final: false,
      }),
    );

    expect(state.liveExecution.node_executions.node_a.streaming_output).toBe('live');
    expect(state.currentExecution.node_executions.node_a.streaming_output).toBe('current:');
  });

  it('should save checkpoint with a deep-cloned execution snapshot and dedupe repeated checkpoint_created events', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: {
        execution_id: 'exec_1',
        plan_id: 'plan_1',
        status: 'running',
        node_executions: {
          node_a: {
            node_id: 'node_a',
            status: 'completed',
            started_at: null,
            completed_at: null,
            inputs: { a: 1 },
            outputs: { o: 1 },
            error: null,
            streaming_output: 'orig',
            metadata: {},
          },
        },
        context: { k: 'v' },
        started_at: '2026-02-03T00:00:00.000Z',
        completed_at: null,
        error: null,
        metadata: { m: 1 },
      },
      liveExecution: null,
      checkpointExecution: null,
      checkpoints: [],
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('checkpoint_created', {
        checkpoint_id: 'cp_1',
        execution_id: 'exec_1',
        sequence_number: 1,
        trigger: 'manual',
        llm_call_logs: [{ call_id: 'c1' }],
      }),
    );

    expect(state.checkpoints).toHaveLength(1);
    expect(state.checkpoints[0].checkpoint_id).toBe('cp_1');
    expect(state.checkpoints[0].execution_id).toBe('exec_1');
    expect(state.checkpoints[0].plan_id).toBe('plan_1');
    expect(state.checkpoints[0].llm_call_logs).toEqual([{ call_id: 'c1' }]);
    expect(state.checkpoints[0].execution_snapshot.node_executions.node_a.streaming_output).toBe('orig');

    state.currentExecution.node_executions.node_a.streaming_output = 'mutated';
    state.currentExecution.context.k = 'mutated';

    expect(state.checkpoints[0].execution_snapshot.node_executions.node_a.streaming_output).toBe('orig');
    expect(state.checkpoints[0].execution_snapshot.context.k).toBe('v');

    state.handleEvent(
      makeBaseEvent('checkpoint_created', {
        checkpoint_id: 'cp_1',
        execution_id: 'exec_1',
        sequence_number: 1,
        trigger: 'manual',
      }),
    );
    expect(state.checkpoints).toHaveLength(1);
  });

  it('should handle restore_checkpoint_result failure by clearing pending restore state and showing error toast', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1' },
      lastRestoredCheckpointId: 'cp_1',
      lastRestoredCheckpointKey: { execution_id: 'exec_old', checkpoint_id: 'cp_1' },
      currentExecution: { execution_id: 'exec_old', status: 'paused', node_executions: {}, context: {} },
      liveExecution: { execution_id: 'exec_live', status: 'running', node_executions: {}, context: {} },
      executionId: 'exec_old',
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('restore_checkpoint_result', {
        checkpoint_id: 'cp_1',
        execution_id: null,
        replay_mode: 'deterministic',
        success: false,
        message: 'nope',
      }),
    );

    expect(state.lastRestoredCheckpointId).toBeNull();
    expect(state.lastRestoredCheckpointKey).toBeNull();
    expect(state.currentExecution.execution_id).toBe('exec_live');
    expect(state.executionId).toBe('exec_live');
    expect(state.toasts.some((t: any) => t.type === 'error')).toBe(true);
    expect(state.activityLogs.some((l: any) => l.level === 'error')).toBe(true);
  });

  it('should handle restore_checkpoint_result success by updating execution_id and marking execution paused', () => {
    let state: any = {
      viewMode: 'live',
      currentPlan: { plan_id: 'plan_1' },
      lastRestoredCheckpointId: 'cp_1',
      lastRestoredCheckpointKey: { execution_id: 'exec_old', checkpoint_id: 'cp_1' },
      currentExecution: {
        execution_id: 'exec_old',
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
      executionId: 'exec_old',
      nodes: [],
      edges: [],
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      nodeObservations: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.handleEvent(
      makeBaseEvent('restore_checkpoint_result', {
        checkpoint_id: 'cp_1',
        execution_id: 'exec_new',
        replay_mode: 'fresh',
        success: true,
        message: 'ok',
      }),
    );

    expect(state.currentExecution.execution_id).toBe('exec_new');
    expect(state.currentExecution.status).toBe('paused');
    expect(state.executionId).toBe('exec_new');
    expect(state.lastRestoredCheckpointKey).toEqual({ execution_id: 'exec_new', checkpoint_id: 'cp_1' });
    expect(state.toasts.some((t: any) => t.type === 'success')).toBe(true);
    expect(state.activityLogs.some((l: any) => l.level === 'info')).toBe(true);
  });
});

describe('wsActions.connect status handling', () => {
  it('should keep execution running on transient disconnect and show toast once', () => {
    let state: any = {
      viewMode: 'live',
      connectionStatus: 'disconnected',
      sessionId: 's1',
      ws: null,
      currentPlan: { plan_id: 'plan_1' },
      currentExecution: {
        execution_id: 'exec_1',
        plan_id: 'plan_1',
        status: 'running',
        node_executions: {},
        context: {},
        started_at: '2026-02-03T00:00:00.000Z',
        completed_at: null,
        error: null,
        metadata: {},
      },
      liveExecution: null,
      executionId: 'exec_1',
      toasts: [],
      toastKeys: {},
      activityLogs: [],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };
    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createExecutionActions(set, get));
    Object.assign(state, createWsActions(set, get));

    state.connect('s1');
    expect(lastWsInstance).toBeTruthy();

    lastWsInstance.emitStatus('disconnected');

    expect(state.connectionStatus).toBe('disconnected');
    // Transient disconnect should NOT abort execution (ReconnectingWebSocket will retry)
    expect(state.currentExecution.status).toBe('running');
    expect(state.currentExecution.completed_at).toBeNull();
    expect(Object.keys(state.toastKeys)).toContain('backend-connection');

    // Duplicate disconnect should not add duplicate toast
    lastWsInstance.emitStatus('disconnected');
    expect(Object.keys(state.toastKeys).filter((k) => k === 'backend-connection')).toHaveLength(1);
  });
});
