import { describe, expect, it } from 'vitest';
import { computeMetrics } from '@/components/panels/MetricsPanel';
import type { SpanData } from '@/stores/storeActions/spanActions';

function makeSpan(overrides: Partial<SpanData> = {}): SpanData {
  return {
    span_id: 'sp_1',
    trace_id: 'tr_1',
    parent_span_id: null,
    span_type: 'node',
    name: 'test',
    status: 'ok',
    start_time: 100,
    end_time: 101,
    duration: 1,
    node_id: null,
    model: null,
    tokens_input: null,
    tokens_output: null,
    cost_usd: null,
    cache_hit: null,
    tool_name: null,
    parent_trace_id: null,
    restore_checkpoint_id: null,
    replay_mode: false,
    group_id: null,
    lane_id: null,
    seq: null,
    attributes: {},
    children: [],
    ...overrides,
  };
}

describe('computeMetrics', () => {
  it('returns empty metrics for empty spans', () => {
    const { exec, llm, tool } = computeMetrics([]);
    expect(exec.duration).toBeNull();
    expect(exec.nodeCount).toBe(0);
    expect(llm.totalCalls).toBe(0);
    expect(tool.totalCalls).toBe(0);
  });

  it('computes execution summary from execution span', () => {
    const spans = [
      makeSpan({ span_type: 'execution', duration: 5.0, status: 'ok' }),
      makeSpan({ span_type: 'node', status: 'ok', span_id: 'n1' }),
      makeSpan({ span_type: 'node', status: 'ok', span_id: 'n2' }),
      makeSpan({ span_type: 'node', status: 'error', span_id: 'n3' }),
    ];
    const { exec } = computeMetrics(spans);
    expect(exec.duration).toBe(5000);
    expect(exec.status).toBe('ok');
    expect(exec.nodeCount).toBe(3);
    expect(exec.completedNodes).toBe(2);
    expect(exec.errorNodes).toBe(1);
  });

  it('aggregates LLM metrics with per-model breakdown', () => {
    const spans = [
      makeSpan({
        span_type: 'llm', span_id: 'l1', model: 'gpt-4o',
        tokens_input: 1000, tokens_output: 200, cost_usd: 0.01, duration: 1.5,
      }),
      makeSpan({
        span_type: 'llm', span_id: 'l2', model: 'gpt-4o',
        tokens_input: 500, tokens_output: 100, cost_usd: 0.005, duration: 0.8, cache_hit: true,
      }),
      makeSpan({
        span_type: 'llm', span_id: 'l3', model: 'gpt-4o-mini',
        tokens_input: 300, tokens_output: 50, cost_usd: 0.001, duration: 0.3,
      }),
    ];
    const { llm } = computeMetrics(spans);
    expect(llm.totalCalls).toBe(3);
    expect(llm.tokensIn).toBe(1800);
    expect(llm.tokensOut).toBe(350);
    expect(llm.totalCost).toBeCloseTo(0.016);
    expect(llm.cacheHits).toBe(1);
    expect(llm.latencies).toHaveLength(3);

    // Per-model breakdown
    expect(Object.keys(llm.byModel)).toHaveLength(2);
    expect(llm.byModel['gpt-4o'].calls).toBe(2);
    expect(llm.byModel['gpt-4o'].tokensIn).toBe(1500);
    expect(llm.byModel['gpt-4o-mini'].calls).toBe(1);
  });

  it('aggregates tool metrics with per-tool breakdown', () => {
    const spans = [
      makeSpan({ span_type: 'tool', span_id: 't1', tool_name: 'web_search', status: 'ok', duration: 2.0 }),
      makeSpan({ span_type: 'tool', span_id: 't2', tool_name: 'web_search', status: 'error', duration: 1.0 }),
      makeSpan({ span_type: 'tool', span_id: 't3', tool_name: 'code_exec', status: 'ok', duration: 3.0, cache_hit: true }),
      makeSpan({ span_type: 'retry', span_id: 'r1' }),
      makeSpan({ span_type: 'internal', span_id: 'i1' }),
      makeSpan({ span_type: 'internal', span_id: 'i2' }),
    ];
    const { tool } = computeMetrics(spans);
    expect(tool.totalCalls).toBe(3);
    expect(tool.successCount).toBe(2);
    expect(tool.errorCount).toBe(1);
    expect(tool.cacheHits).toBe(1);
    expect(tool.retryCount).toBe(1);
    expect(tool.internalSpans).toBe(2);

    // Per-tool breakdown
    expect(tool.byTool['web_search'].calls).toBe(2);
    expect(tool.byTool['web_search'].errors).toBe(1);
    expect(tool.byTool['code_exec'].cacheHits).toBe(1);
  });

  it('handles spans with null duration gracefully', () => {
    const spans = [
      makeSpan({ span_type: 'llm', span_id: 'l1', duration: null }),
      makeSpan({ span_type: 'tool', span_id: 't1', tool_name: 'x', duration: null }),
    ];
    const { llm, tool } = computeMetrics(spans);
    expect(llm.latencies).toHaveLength(0);
    expect(tool.latencies).toHaveLength(0);
  });
});
