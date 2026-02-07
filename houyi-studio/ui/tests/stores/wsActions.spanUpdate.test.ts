import { describe, expect, it, vi } from 'vitest';

import { createWsActions } from '@/stores/storeActions/wsActions';


describe('wsActions: span_update dispatch', () => {
  it('dispatches span_update even when event_type is an enum object {value}', () => {
    const updateSpan = vi.fn();

    const get = () => ({
      handleEvent: undefined,
      updateSpan,
      addActivityLog: vi.fn(),
      removeToastByKey: vi.fn(),
      showToastOnce: vi.fn(),
      showToast: vi.fn(),
      nodes: [],
      edges: [],
      currentPlan: null,
      currentExecution: null,
      liveExecution: null,
      viewMode: 'live',
      loadingWorkflowName: null,
      updateNodeStatus: vi.fn(),
      clearSpans: vi.fn(),
    });

    const set = vi.fn();
    const actions = createWsActions(set as any, get as any) as any;

    actions.handleEvent({
      event_type: { value: 'span_update' },
      execution_id: 'exec_1',
      trace_id: 'exec_1',
      span_id: 'span_1',
      parent_span_id: null,
      span_type: 'node',
      name: 'node.llm',
      status: 'ok',
      start_time: Date.now() / 1000,
      end_time: null,
      attributes: {},
    });

    expect(updateSpan).toHaveBeenCalledTimes(1);
  });
});
