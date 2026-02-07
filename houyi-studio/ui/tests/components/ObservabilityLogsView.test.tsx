import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ObservabilityLogsView } from '@/components/panels/ObservabilityLogsView';

describe('ObservabilityLogsView time precision', () => {
  const baseExecution = { execution_id: 'exec_1' };

  it('displays different times for start_time and end_time when they differ by sub-second amounts', () => {
    // epoch seconds with sub-second precision (differ by 0.35s)
    const startEpoch = 1738849570.100;
    const endEpoch = 1738849570.450;

    const nodeObservations = {
      exec_1: {
        node_llm: {
          status: 'ok',
          start_time: startEpoch,
          end_time: endEpoch,
        },
      },
    };

    render(
      <ObservabilityLogsView
        viewExecution={baseExecution}
        nodeObservations={nodeObservations}
      />
    );

    // Duration should be ~350ms
    expect(screen.getByText('350ms')).toBeTruthy();
  });

  it('handles ISO string timestamps gracefully', () => {
    const nodeObservations = {
      exec_1: {
        node_tool: {
          status: 'ok',
          start_time: '2026-02-06T19:46:10.100Z',
          end_time: '2026-02-06T19:46:12.600Z',
        },
      },
    };

    render(
      <ObservabilityLogsView
        viewExecution={baseExecution}
        nodeObservations={nodeObservations}
      />
    );

    // Duration should be 2.50s (adaptive units: >=1000ms shows seconds)
    expect(screen.getByText('2.50s')).toBeTruthy();
  });

  it('shows -- for missing timestamps', () => {
    const nodeObservations = {
      exec_1: {
        node_x: {
          status: 'running',
          start_time: null,
          end_time: null,
        },
      },
    };

    render(
      <ObservabilityLogsView
        viewExecution={baseExecution}
        nodeObservations={nodeObservations}
      />
    );

    // duration shows '--', start shows 'start: --', end shows 'end: --'
    const dashes = screen.getAllByText('--');
    expect(dashes.length).toBeGreaterThanOrEqual(1); // at least duration '--'
    expect(screen.getByText(/start:.*--/)).toBeTruthy();
    expect(screen.getByText(/end:.*--/)).toBeTruthy();
  });
});
