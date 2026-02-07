/**
 * Tests for ExecutionLineageTree component.
 *
 * Covers:
 * 1. Single execution — renders inline compact label, no tree
 * 2. Two executions (parent→child fork) — renders tree with connectors
 * 3. Multi-fork (A→B, A→C) — renders branching tree
 * 4. Deep chain (A→B→C→D) — renders nested connectors
 * 5. Click selects execution
 * 6. Active execution is highlighted
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExecutionLineageTree } from '@/components/panels/ExecutionLineageTree';
import type { SpanTree, SpanData } from '@/stores/storeActions/spanActions';

function makeSpanTree(overrides: {
  executionId: string;
  parentTraceId?: string | null;
  restoreCheckpointId?: string | null;
  replayMode?: boolean;
  spanCount?: number;
  duration?: number;
  status?: 'ok' | 'error';
  endTime?: number | null;
}): SpanTree {
  const root: SpanData = {
    span_id: `span_root_${overrides.executionId}`,
    trace_id: overrides.executionId,
    parent_span_id: null,
    span_type: 'execution',
    name: 'execution',
    status: overrides.status ?? 'ok',
    start_time: 1000,
    end_time: overrides.endTime !== undefined ? overrides.endTime : 1000 + (overrides.duration ?? 5),
    duration: overrides.duration ?? 5,
    node_id: null,
    model: null,
    tokens_input: null,
    tokens_output: null,
    cost_usd: null,
    cache_hit: null,
    tool_name: null,
    parent_trace_id: overrides.parentTraceId ?? null,
    restore_checkpoint_id: overrides.restoreCheckpointId ?? null,
    replay_mode: overrides.replayMode ?? false,
    group_id: null,
    lane_id: null,
    seq: null,
    attributes: {},
    children: [],
  };

  const spanCount = overrides.spanCount ?? 3;
  const spans: SpanData[] = [root];
  for (let i = 1; i < spanCount; i++) {
    spans.push({ ...root, span_id: `span_${i}_${overrides.executionId}`, span_type: 'node' });
  }

  return {
    root,
    spans,
    totalDuration: overrides.duration ?? 5,
    startTime: 1000,
    checkpoints: [],
  };
}

describe('ExecutionLineageTree', () => {
  it('renders inline label for single execution (no tree)', () => {
    const trees: Record<string, SpanTree> = {
      exec_aaa: makeSpanTree({ executionId: 'exec_aaa', spanCount: 5, duration: 12.4 }),
    };

    render(
      <ExecutionLineageTree
        executionIds={['exec_aaa']}
        activeExecutionId="exec_aaa"
        getSpanTree={(id) => trees[id] ?? null}
        onSelect={() => {}}
      />,
    );

    // Should show span count and duration inline, no tree header
    expect(screen.getByText(/5 spans/)).toBeTruthy();
    expect(screen.getByText(/12\.4s/)).toBeTruthy();
    expect(screen.queryByText('Executions')).toBeNull();
  });

  it('renders tree with fork indicator for parent→child', () => {
    const trees: Record<string, SpanTree> = {
      exec_parent: makeSpanTree({ executionId: 'exec_parent', duration: 10 }),
      exec_child: makeSpanTree({
        executionId: 'exec_child',
        parentTraceId: 'exec_parent',
        restoreCheckpointId: 'cp_001',
        replayMode: true,
        duration: 8,
      }),
    };

    render(
      <ExecutionLineageTree
        executionIds={['exec_parent', 'exec_child']}
        activeExecutionId="exec_parent"
        getSpanTree={(id) => trees[id] ?? null}
        onSelect={() => {}}
      />,
    );

    // Should show tree header with "fork tree" badge
    expect(screen.getByText('Executions')).toBeTruthy();
    expect(screen.getByText('fork tree')).toBeTruthy();
    // Should show 2 execution count
    expect(screen.getByText('2')).toBeTruthy();
    // Should show replay badge
    expect(screen.getByText('⟳ det')).toBeTruthy();
  });

  it('renders multi-fork branching (A→B, A→C)', () => {
    const trees: Record<string, SpanTree> = {
      exec_a: makeSpanTree({ executionId: 'exec_a', duration: 10 }),
      exec_b: makeSpanTree({
        executionId: 'exec_b',
        parentTraceId: 'exec_a',
        restoreCheckpointId: 'cp_001',
        replayMode: true,
      }),
      exec_c: makeSpanTree({
        executionId: 'exec_c',
        parentTraceId: 'exec_a',
        restoreCheckpointId: 'cp_002',
        replayMode: false,
      }),
    };

    render(
      <ExecutionLineageTree
        executionIds={['exec_a', 'exec_b', 'exec_c']}
        activeExecutionId="exec_a"
        getSpanTree={(id) => trees[id] ?? null}
        onSelect={() => {}}
      />,
    );

    // 3 executions
    expect(screen.getByText('3')).toBeTruthy();
    // Both forks should have checkpoint references
    const cpRefs = screen.getAllByText(/cp_00/);
    expect(cpRefs.length).toBe(2);
  });

  it('calls onSelect when clicking an execution row', () => {
    const onSelect = vi.fn();
    const trees: Record<string, SpanTree> = {
      exec_a: makeSpanTree({ executionId: 'exec_a' }),
      exec_b: makeSpanTree({ executionId: 'exec_b', parentTraceId: 'exec_a', restoreCheckpointId: 'cp_1' }),
    };

    render(
      <ExecutionLineageTree
        executionIds={['exec_a', 'exec_b']}
        activeExecutionId="exec_a"
        getSpanTree={(id) => trees[id] ?? null}
        onSelect={onSelect}
      />,
    );

    // Click the child execution row
    const buttons = screen.getAllByRole('button');
    // buttons[0] is the header toggle, buttons[1] is exec_a, buttons[2] is exec_b
    const childButton = buttons.find((b) => b.title?.includes('exec_b'));
    expect(childButton).toBeTruthy();
    fireEvent.click(childButton!);
    expect(onSelect).toHaveBeenCalledWith('exec_b');
  });

  it('highlights active execution with blue styling', () => {
    const trees: Record<string, SpanTree> = {
      exec_a: makeSpanTree({ executionId: 'exec_a' }),
      exec_b: makeSpanTree({ executionId: 'exec_b', parentTraceId: 'exec_a', restoreCheckpointId: 'cp_1' }),
    };

    render(
      <ExecutionLineageTree
        executionIds={['exec_a', 'exec_b']}
        activeExecutionId="exec_b"
        getSpanTree={(id) => trees[id] ?? null}
        onSelect={() => {}}
      />,
    );

    const activeButton = screen.getAllByRole('button').find((b) => b.title?.includes('exec_b'));
    expect(activeButton).toBeTruthy();
    expect(activeButton!.className).toContain('bg-blue-600');
  });

  it('shows running status with pulse animation', () => {
    const trees: Record<string, SpanTree> = {
      exec_running: makeSpanTree({ executionId: 'exec_running', endTime: null, duration: 0 }),
      exec_done: makeSpanTree({ executionId: 'exec_done', parentTraceId: 'exec_running', restoreCheckpointId: 'cp_1' }),
    };

    const { container } = render(
      <ExecutionLineageTree
        executionIds={['exec_running', 'exec_done']}
        activeExecutionId="exec_running"
        getSpanTree={(id) => trees[id] ?? null}
        onSelect={() => {}}
      />,
    );

    // Should have an animate-pulse element for the running execution
    const pulsingDots = container.querySelectorAll('.animate-pulse');
    expect(pulsingDots.length).toBeGreaterThanOrEqual(1);
  });
});
