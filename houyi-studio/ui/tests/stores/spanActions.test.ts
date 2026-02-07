import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

import { createSpanActions } from '@/stores/storeActions/spanActions';
import type { SpanUpdateEvent } from '@/types/websocket';

function makeBaseEvent(overrides: Partial<SpanUpdateEvent> = {}): SpanUpdateEvent {
  return {
    event_type: 'span_update',
    event_id: 'evt_1',
    timestamp: new Date('2026-02-03T00:00:00.000Z').toISOString(),
    session_id: 's1',

    execution_id: 'exec_1',
    trace_id: 'trace_1',
    span_id: 'span_1',
    parent_span_id: null,
    span_type: 'execution',
    name: 'execution',
    status: 'ok',
    start_time: 100,
    end_time: 110,

    ...overrides,
  };
}

describe('spanActions', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should write spans directly to store (no batching)', () => {
    let state: any = {
      spanStore: {},
      checkpoints: [],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;
    const actions = createSpanActions(set, get);

    actions.updateSpan(makeBaseEvent({ span_id: 's1', start_time: 1, end_time: null }));
    actions.updateSpan(makeBaseEvent({ span_id: 's2', start_time: 2, end_time: null }));

    // Spans are written immediately (no batching)
    expect(Object.keys(state.spanStore.exec_1)).toEqual(['s1', 's2']);
  });

  it('should force flush when pending buffer is large', () => {
    let state: any = {
      spanStore: {},
      checkpoints: [],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;
    const actions = createSpanActions(set, get);

    for (let i = 0; i < 500; i += 1) {
      actions.updateSpan(makeBaseEvent({ span_id: `s${i}`, start_time: i, end_time: null }));
    }

    // MAX_PENDING_SPANS should have forced a flush synchronously
    expect(state.spanStore.exec_1).toBeTruthy();
    expect(Object.keys(state.spanStore.exec_1).length).toBe(500);
  });

  it('should clear spans for a single execution', () => {
    let state: any = {
      spanStore: {
        exec_1: { a: { span_id: 'a' } },
        exec_2: { b: { span_id: 'b' } },
      },
      checkpoints: [],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;
    const actions = createSpanActions(set, get);

    actions.clearSpans('exec_1');

    expect(state.spanStore.exec_1).toBeUndefined();
    expect(state.spanStore.exec_2).toBeTruthy();
  });

  it('should build a span tree with correct parent-child relationships and ordering', () => {
    let state: any = {
      spanStore: {},
      checkpoints: [
        {
          checkpoint_id: 'cp_1',
          execution_id: 'exec_1',
          trigger: 'manual',
          created_at: '2026-02-03T00:00:10.000Z',
          metadata: { trigger_node_id: 'node_a' },
        },
      ],
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;
    const actions = createSpanActions(set, get);

    // root
    actions.updateSpan(
      makeBaseEvent({
        span_id: 'root',
        parent_span_id: null,
        span_type: 'execution',
        name: 'exec',
        start_time: 100,
        end_time: 200,
      }),
    );

    // children (out of order insertion)
    actions.updateSpan(
      makeBaseEvent({
        span_id: 'child_b',
        parent_span_id: 'root',
        span_type: 'node',
        name: 'B',
        node_id: 'node_b',
        start_time: 150,
        end_time: 180,
      }),
    );

    actions.updateSpan(
      makeBaseEvent({
        span_id: 'child_a',
        parent_span_id: 'root',
        span_type: 'node',
        name: 'A',
        node_id: 'node_a',
        start_time: 120,
        end_time: 140,
      }),
    );

    const tree = actions.getSpanTree('exec_1');
    expect(tree).toBeTruthy();
    expect(tree?.root?.span_id).toBe('root');

    // Children should be sorted by start_time
    expect(tree?.root?.children.map((c) => c.span_id)).toEqual(['child_a', 'child_b']);

    // Checkpoint markers are wired
    expect(tree?.checkpoints.length).toBe(1);
    expect(tree?.checkpoints[0].checkpoint_id).toBe('cp_1');
    expect(tree?.checkpoints[0].node_id).toBe('node_a');

    // Duration computed from min/max timestamps
    expect(tree?.startTime).toBe(100);
    expect(tree?.totalDuration).toBe(100);
  });
});
