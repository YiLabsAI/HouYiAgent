/**
 * Span store actions for timeline visualization
 *
 * v1.1 Performance features:
 * - Batch processing: collect span updates and flush periodically
 * - Throttling: avoid triggering render on every span_update
 */
import type { SpanUpdateEvent, SpanType } from '@/types/websocket';

// Throttle configuration
const _SPAN_FLUSH_INTERVAL_MS = 100; // Batch flush interval (reserved for future re-enablement)
const _MAX_PENDING_SPANS = 500; // Force flush if pending count exceeds this (reserved)

// Span data structure for store
export interface SpanData {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  span_type: SpanType;
  name: string;
  status: 'ok' | 'error';
  start_time: number;
  end_time: number | null;
  duration: number | null;

  // AI-native fields
  node_id: string | null;
  model: string | null;
  tokens_input: number | null;
  tokens_output: number | null;
  cost_usd: number | null;
  cache_hit: boolean | null;
  tool_name: string | null;

  // Checkpoint lineage
  parent_trace_id: string | null;
  restore_checkpoint_id: string | null;
  replay_mode: boolean;

  // Parallel execution fields
  group_id: string | null;
  lane_id: number | null;
  seq: number | null;

  // Generic attributes (from backend span)
  attributes: Record<string, any>;

  // Children (for tree structure)
  children: SpanData[];
}

// Span store structure: execution_id -> span_id -> SpanData
export type SpanStore = Record<string, Record<string, SpanData>>;

// Checkpoint marker for timeline
export interface CheckpointMarker {
  checkpoint_id: string;
  timestamp: number;
  node_id: string | null;
  trigger: string;
}

// Span tree for timeline rendering
export interface SpanTree {
  root: SpanData | null;
  spans: SpanData[];
  totalDuration: number;
  startTime: number;
  checkpoints: CheckpointMarker[];
}

export interface SpanActions {
  updateSpan: (event: SpanUpdateEvent) => void;
  getSpanTree: (executionId: string) => SpanTree | null;
  clearSpans: (executionId?: string) => void;
  flushPendingSpans: () => void;
}

// Pending spans buffer for batch processing
let pendingSpans: SpanUpdateEvent[] = [];
let flushTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * Convert SpanUpdateEvent to SpanData
 */
function eventToSpanData(event: SpanUpdateEvent): SpanData {
  const duration = event.end_time
    ? event.end_time - event.start_time
    : null;

  return {
    span_id: event.span_id,
    trace_id: event.trace_id,
    parent_span_id: event.parent_span_id ?? null,
    span_type: event.span_type,
    name: event.name,
    status: event.status,
    start_time: event.start_time,
    end_time: event.end_time ?? null,
    duration,
    node_id: event.node_id ?? null,
    model: event.model ?? null,
    tokens_input: event.tokens_input ?? null,
    tokens_output: event.tokens_output ?? null,
    cost_usd: event.cost_usd ?? null,
    cache_hit: event.cache_hit ?? null,
    tool_name: event.tool_name ?? null,
    parent_trace_id: event.parent_trace_id ?? null,
    restore_checkpoint_id: event.restore_checkpoint_id ?? null,
    replay_mode: event.replay_mode ?? false,
    group_id: (event.attributes?.group_id as string) ?? null,
    lane_id: (event.attributes?.lane_id as number) ?? null,
    seq: (event.attributes?.seq as number) ?? null,
    attributes: event.attributes ?? {},
    children: [],
  };
}

export function createSpanActions(
  set: (partial: Partial<any>) => void,
  get: () => any
): SpanActions {
  /**
   * Flush all pending spans to store in a single batch update
   */
  const flushPendingSpans = () => {
    if (pendingSpans.length === 0) return;

    const spansToFlush = pendingSpans;
    pendingSpans = [];

    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }

    console.log('[SpanActions] flushPendingSpans: flushing', spansToFlush.length, 'spans');
    const currentState = get();
    const newSpanStore = { ...(currentState.spanStore || {}) };

    for (const event of spansToFlush) {
      const { execution_id, span_id } = event;

      if (!newSpanStore[execution_id]) {
        newSpanStore[execution_id] = {};
      }

      newSpanStore[execution_id] = {
        ...newSpanStore[execution_id],
        [span_id]: eventToSpanData(event),
      };
    }

    set({ spanStore: newSpanStore });
  };

  /**
   * Schedule a flush if not already scheduled
   */
  // scheduleFlush reserved for future batch mode re-enablement
  const _scheduleFlush = () => {
    if (flushTimer) return;
    flushTimer = setTimeout(() => {
      flushTimer = null;
      flushPendingSpans();
    }, _SPAN_FLUSH_INTERVAL_MS);
  };
  void _scheduleFlush; void _MAX_PENDING_SPANS;

  return {
    updateSpan: (event: SpanUpdateEvent) => {
      // Write directly to spanStore (no batching) for reliability
      const execId = event.execution_id;
      const spanId = event.span_id;
      if (!execId || !spanId) {
        console.warn('[SpanActions] updateSpan: missing execution_id or span_id', event);
        return;
      }
      const current = get().spanStore || {};
      const execSpans = current[execId] || {};
      const newSpanStore = {
        ...current,
        [execId]: {
          ...execSpans,
          [spanId]: eventToSpanData(event),
        },
      };
      set({ spanStore: newSpanStore });
    },

    flushPendingSpans,

    getSpanTree: (executionId: string): SpanTree | null => {
      const state = get();
      const spans = state.spanStore[executionId];

      if (!spans || Object.keys(spans).length === 0) {
        return null;
      }

      // Build tree structure
      const spanList = Object.values(spans) as SpanData[];
      const spanMap = new Map<string, SpanData>();

      // Clone spans to avoid mutating store
      spanList.forEach((span) => {
        spanMap.set(span.span_id, { ...span, children: [] });
      });

      // Find root and build parent-child relationships
      let root: SpanData | null = null;
      let minStartTime = Infinity;
      let maxEndTime = 0;

      spanMap.forEach((span) => {
        if (span.start_time < minStartTime) {
          minStartTime = span.start_time;
        }
        if (span.end_time && span.end_time > maxEndTime) {
          maxEndTime = span.end_time;
        }

        if (!span.parent_span_id) {
          root = span;
        } else {
          const parent = spanMap.get(span.parent_span_id);
          if (parent) {
            parent.children.push(span);
          }
        }
      });

      // Sort children by start_time
      const sortChildren = (span: SpanData) => {
        span.children.sort((a, b) => a.start_time - b.start_time);
        span.children.forEach(sortChildren);
      };

      if (root) {
        sortChildren(root);
      }

      // Get checkpoints from store (if available)
      const checkpoints: CheckpointMarker[] = [];
      const storeCheckpoints = state.checkpoints || [];
      for (const cp of storeCheckpoints) {
        if (cp.execution_id === executionId) {
          checkpoints.push({
            checkpoint_id: cp.checkpoint_id,
            timestamp: new Date(cp.created_at || Date.now()).getTime() / 1000,
            node_id: cp.metadata?.trigger_node_id || null,
            trigger: cp.trigger || 'manual',
          });
        }
      }

      return {
        root,
        spans: Array.from(spanMap.values()),
        totalDuration: maxEndTime - minStartTime,
        startTime: minStartTime,
        checkpoints,
      };
    },

    clearSpans: (executionId?: string) => {
      if (executionId) {
        const newSpanStore = { ...(get().spanStore || {}) };
        delete newSpanStore[executionId];
        set({ spanStore: newSpanStore });
      } else {
        set({ spanStore: {} });
      }
    },
  };
}
