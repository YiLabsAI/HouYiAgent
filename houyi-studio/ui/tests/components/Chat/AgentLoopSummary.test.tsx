import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AgentLoopSummary } from '@/components/Chat/AgentLoopSummary';

describe('AgentLoopSummary', () => {
  it('does not render when summary is empty', () => {
    const { container } = render(
      <AgentLoopSummary rounds={0} toolCalls={0} traceId={null} usage={null} />,
    );
    expect(container.innerHTML).toBe('');
  });

  it('renders summary and toggles expanded details', () => {
    render(
      <AgentLoopSummary
        rounds={2}
        toolCalls={3}
        traceId="trace-123"
        usage={{ prompt_tokens: 34, completion_tokens: 54, total_tokens: 88 }}
        metrics={{
          decode_tokens_per_second: 26.5,
          end_to_end_tokens_per_second: 22.25,
          finish_reason: 'stop',
          tool_loop_convergence_reason: 'needs_final_stream',
          request_adapter_class: 'SiliconFlowAdapter',
          request_adapter_strict_message_string_contract: true,
          request_message_count: 4,
          request_user_message_count: 1,
          request_assistant_message_count: 2,
          request_assistant_reasoning_message_count: 1,
          request_assistant_reasoning_only_message_count: 1,
          request_assistant_tool_call_message_count: 1,
          request_tool_message_count: 1,
          final_stream_status: 'completed',
          final_stream_empty_visible_output: false,
          final_stream_assistant_reasoning_removed_count: 2,
          final_stream_assistant_reasoning_only_removed_count: 1,
          final_stream_assistant_tool_call_carrier_count: 1,
          final_stream_tool_result_projection_count: 1,
          budget: { max_tokens_guardrail_applied: true },
        }}
      />,
    );

    expect(screen.getByText('Agent Loop')).toBeInTheDocument();
    expect(screen.getByText(/2 rounds/)).toBeInTheDocument();
    expect(screen.getByText(/3 tool calls/)).toBeInTheDocument();
    expect(screen.getByText('Final stream')).toBeInTheDocument();
    expect(screen.getByText('Trace')).toBeInTheDocument();
    expect(screen.queryByText('In 34')).not.toBeInTheDocument();
    expect(screen.queryByText('Out 54')).not.toBeInTheDocument();
    expect(screen.queryByText('Total 88')).not.toBeInTheDocument();
    expect(screen.queryByText('Decode 27/s')).not.toBeInTheDocument();
    expect(screen.queryByText('E2E 22/s')).not.toBeInTheDocument();
    expect(screen.queryByText('Guardrail')).not.toBeInTheDocument();
    expect(screen.queryByText(/Iterations:/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button'));
    expect(screen.getByText(/Iterations: 2/)).toBeInTheDocument();
    expect(screen.getByText(/Tool calls: 3/)).toBeInTheDocument();
    expect(screen.getByText(/Total tokens 88/)).toBeInTheDocument();
    expect(screen.getByText(/Finish:/)).toBeInTheDocument();
    expect(screen.getByText('stop')).toBeInTheDocument();
    expect(screen.getByText(/Convergence:/)).toBeInTheDocument();
    expect(screen.getByText('needs_final_stream')).toBeInTheDocument();
    expect(screen.getByText(/Final stream:/)).toBeInTheDocument();
    expect(screen.getAllByText('completed').length).toBeGreaterThan(0);
    expect(screen.getByText(/Adapter:/)).toBeInTheDocument();
    expect(screen.getByText('SiliconFlowAdapter')).toBeInTheDocument();
    expect(screen.getByText('(strict messages)')).toBeInTheDocument();
    expect(screen.getByText(/Request:/)).toBeInTheDocument();
    expect(screen.getByText('Messages 4')).toBeInTheDocument();
    expect(screen.getByText('Users 1')).toBeInTheDocument();
    expect(screen.getByText('Assistants 2')).toBeInTheDocument();
    expect(screen.getByText('Assistant reasoning 1')).toBeInTheDocument();
    expect(screen.getAllByText('Reasoning-only 1').length).toBeGreaterThan(0);
    expect(screen.getByText('Tool-call carriers 1')).toBeInTheDocument();
    expect(screen.getByText('Tool messages 1')).toBeInTheDocument();
    expect(screen.getByText(/Sanitized:/)).toBeInTheDocument();
    expect(screen.getByText('Reasoning removed 2')).toBeInTheDocument();
    expect(screen.getAllByText('Reasoning-only 1').length).toBeGreaterThan(1);
    expect(screen.getByText('Tool carriers 1')).toBeInTheDocument();
    expect(screen.getByText('Tool projections 1')).toBeInTheDocument();
    expect(screen.getByText('trace-123')).toBeInTheDocument();
  });

  it('surfaces replay and timeout flow labels', () => {
    render(
      <>
        <AgentLoopSummary
          rounds={1}
          toolCalls={1}
          traceId={null}
          usage={null}
          metrics={{
            tool_loop_final_stream_skipped: true,
            tool_loop_convergence_reason: 'no_tool_calls_with_replay_payload',
          }}
        />
        <AgentLoopSummary
          rounds={1}
          toolCalls={1}
          traceId={null}
          usage={null}
          metrics={{
            final_stream_status: 'error',
            final_stream_error_category: 'timeout',
          }}
        />
        <AgentLoopSummary
          rounds={1}
          toolCalls={0}
          traceId={null}
          usage={null}
          metrics={{
            final_stream_status: 'reasoning_only',
          }}
        />
      </>,
    );

    expect(screen.getByText('Replay')).toBeInTheDocument();
    expect(screen.getByText('Final stream timeout')).toBeInTheDocument();
    expect(screen.getByText('Reasoning only')).toBeInTheDocument();
  });

  it('invokes onOpenTrace from details link', () => {
    const onOpenTrace = vi.fn();
    render(
      <AgentLoopSummary
        rounds={1}
        toolCalls={1}
        traceId="trace-xyz"
        usage={null}
        onOpenTrace={onOpenTrace}
      />,
    );

    fireEvent.click(screen.getByRole('button'));
    fireEvent.click(screen.getByText('details'));
    expect(onOpenTrace).toHaveBeenCalledWith('trace-xyz');
  });
});
