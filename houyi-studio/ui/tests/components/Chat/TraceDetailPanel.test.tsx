import { render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { TraceDetailPanel } from '@/components/Chat/TraceDetailPanel';

describe('TraceDetailPanel', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('loads and renders trace payload', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ trace_id: 'trace-1', root_span: { name: 'chat.send_message' } }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-1" onClose={vi.fn()} />);

    expect(screen.getByText('Loading trace...')).toBeInTheDocument();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/chat/trace/trace-1');
      expect(screen.getByText(/chat.send_message/)).toBeInTheDocument();
    });
  });

  it('renders error when request fails', async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: false, status: 500 });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-err" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/Trace API 500/)).toBeInTheDocument();
    });
  });

  it('renders context compaction', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-context',
        total_duration_ms: 42,
        context_governance: {
          dropped_blocks: ['memory'],
          drop_reasons: {
            memory: 'boundary_excluded',
            older_summary: 'budget_exceeded',
          },
          dropped_block_details: [
            {
              candidate_id: 'memory',
              block_type: 'memory',
              source: 'memory',
              token_count: 1200,
              message_count: 1,
              pinned: false,
            },
          ],
          compaction: {
            triggered: true,
            trigger: 'context.compaction',
            messages_compacted: 1,
            tokens_before: 3200,
            tokens_after: 2400,
            saved_tokens: 800,
            pin_violation_count: 0,
          },
        },
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 42,
          children: [],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-context" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('context.compaction')).toBeInTheDocument();
    });
  });

  it('renders human-readable pre-request compaction trigger label', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-pre-request',
        total_duration_ms: 36,
        context_governance: {
          dropped_blocks: [],
          drop_reasons: {},
          compaction: {
            triggered: true,
            trigger: 'pre_request_pressure',
            messages_compacted: 1,
            tokens_before: 3200,
            tokens_after: 2400,
            saved_tokens: 800,
            pin_violation_count: 0,
          },
        },
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 36,
          children: [],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-pre-request" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Compacted conversation context for this request')).toBeInTheDocument();
      expect(screen.queryByText('pre_request_pressure')).not.toBeInTheDocument();
    });
  });

  it('renders trimmed block details even when drop reasons are empty', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-detail-only',
        total_duration_ms: 28,
        context_governance: {
          dropped_blocks: [],
          drop_reasons: {},
          dropped_block_details: [
            {
              candidate_id: 'memory-block',
              block_type: 'memory',
              source: 'memory',
              token_count: 42,
              message_count: 2,
              pinned: false,
            },
          ],
        },
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 28,
          children: [],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-detail-only" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Context governance')).toBeInTheDocument();
      expect(screen.getByText('Trimmed 1 blocks')).toBeInTheDocument();
      expect(screen.getByText(/Memory/)).toBeInTheDocument();
      expect(screen.getByText(/2 msgs · 42 tokens/)).toBeInTheDocument();
    });
  });

  it('shows empty aggregate timings', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-empty-agg',
        total_duration_ms: 12,
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 12,
          children: [{ name: 'chat.prepare', span_type: 'internal', duration_ms: 12 }],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-empty-agg" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('LLM 0x · —')).toBeInTheDocument();
      expect(screen.getByText('Tool 0x · —')).toBeInTheDocument();
      expect(screen.getByText('Orchestration 0x · —')).toBeInTheDocument();
    });
  });

  it('shows tool loop split', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-split',
        total_duration_ms: 250,
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 250,
          children: [
            {
              name: 'chat.tool_loop',
              span_type: 'internal',
              duration_ms: 200,
              children: [
                { name: 'tool.exec', span_type: 'execution', duration_ms: 20 },
                { name: 'llm.call', span_type: 'llm', duration_ms: 40 },
                { name: 'tool.search', span_type: 'tool', duration_ms: 140 },
              ],
            },
          ],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-split" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByTestId('trace-runtime-breakdown')).toBeInTheDocument();
      expect(screen.getByText('LLM 1x · 40ms')).toBeInTheDocument();
      expect(screen.getByText('Tool 1x · 140ms')).toBeInTheDocument();
      expect(screen.getByText('Orchestration 1x · 20ms')).toBeInTheDocument();
      expect(screen.getByText('Tool loop runtime')).toBeInTheDocument();
      expect(screen.getByText('LLM 20% · 40ms')).toBeInTheDocument();
      expect(screen.getByText('Tool 70% · 140ms')).toBeInTheDocument();
      expect(screen.getByText('Execution overhead 10% · 20ms')).toBeInTheDocument();
      expect(screen.getByText('Execution overhead = tool loop total - LLM - tool')).toBeInTheDocument();
    });
  });

  it('shows disabled by request badge', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-tool-loop-disabled',
        total_duration_ms: 20,
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 20,
          children: [
            {
              name: 'chat.tool_loop',
              span_type: 'internal',
              duration_ms: 1,
              attributes: { 'chat.tool_loop.mode': 'disabled_by_request' },
            },
          ],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-tool-loop-disabled" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Tool loop disabled by request')).toBeInTheDocument();
    });
  });

  it('renders token breakdown', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-token-breakdown',
        total_duration_ms: 55,
        total_tokens: {
          prompt_tokens: 120,
          completion_tokens: 30,
          total_tokens: 150,
          llm_spans: 3,
          llm_spans_with_usage: 2,
          is_partial: true,
        },
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 55,
          children: [],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-token-breakdown" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Trace summary')).toBeInTheDocument();
      expect(screen.getByText('Usage')).toBeInTheDocument();
      expect(screen.getByText('150 total')).toBeInTheDocument();
      expect(screen.getByText('Partial usage')).toBeInTheDocument();
      expect(screen.getByText('Input 120 · Output 30')).toBeInTheDocument();
      expect(screen.getByText('2/3 LLM calls reported usage')).toBeInTheDocument();
    });
  });

  it('shows zero token values for partial usage', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-token-zero-partial',
        total_duration_ms: 88,
        total_tokens: {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
          llm_spans: 2,
          llm_spans_with_usage: 0,
          is_partial: true,
        },
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 88,
          children: [],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-token-zero-partial" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Trace summary')).toBeInTheDocument();
      expect(screen.getByText('Usage')).toBeInTheDocument();
      expect(screen.getByText('0 total')).toBeInTheDocument();
      expect(screen.getByText('Input 0 · Output 0')).toBeInTheDocument();
      expect(screen.getByText('0/2 LLM calls reported usage')).toBeInTheDocument();
      expect(screen.getByText('Partial usage')).toBeInTheDocument();
    });
  });

  it('shows context governance summary', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-context',
        total_duration_ms: 42,
        request_context: {
          request_id: 'req-1',
          conversation_id: 'conv-1',
          model: 'deepseek-chat',
          max_context_tokens: 8192,
          llm_messages_count: 14,
        },
        context_plan: {
          used_tokens: 1520,
          planned_prompt_tokens: 1520,
          reserved_output_tokens: 1024,
          available_input_tokens: 5648,
          block_breakdown: {
            recent: 1200,
            pinned: 300,
            current_turn: 180,
          },
        },
        context_governance: {
          dropped_blocks: ['memory'],
          drop_reasons: {
            memory: 'boundary_excluded',
            older_summary: 'budget_exceeded',
          },
          dropped_block_details: [
            {
              candidate_id: 'memory',
              block_type: 'memory',
              source: 'memory',
              token_count: 42,
              message_count: 2,
              pinned: true,
            },
            {
              candidate_id: 'older_summary',
              block_type: 'summary',
              source: 'summary',
              token_count: 120,
              message_count: null,
              pinned: false,
            },
          ],
          compaction: {
            triggered: true,
            trigger: 'repo_intent_trim',
            messages_compacted: 4,
            tokens_before: 4800,
            tokens_after: 2600,
            saved_tokens: 2200,
            pin_violation_count: 0,
          },
        },
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 42,
          attributes: {
            'chat.request_id': 'req-1',
          },
          children: [],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-context" onClose={vi.fn()} />);

    await waitFor(() => {
      const requestSection = screen.getByTestId('trace-request-context');
      expect(within(requestSection).getByText('Request context')).toBeInTheDocument();
      expect(within(requestSection).getByText('req-1')).toBeInTheDocument();
      expect(within(requestSection).getByText('deepseek-chat')).toBeInTheDocument();
      expect(within(requestSection).getByText('Planned prompt')).toBeInTheDocument();
      expect(screen.getByText('Context governance')).toBeInTheDocument();
      expect(screen.getByText('recent 1,200')).toBeInTheDocument();
      expect(screen.getByText('pinned 300')).toBeInTheDocument();
      expect(screen.getByText('current_turn 180')).toBeInTheDocument();
      expect(screen.getByText('Context trimmed')).toBeInTheDocument();
      expect(screen.getByText('Trimmed 2 blocks')).toBeInTheDocument();
      expect(screen.getByText('Excluded by planning boundary 1')).toBeInTheDocument();
      expect(screen.getByText('Trimmed to fit request budget 1')).toBeInTheDocument();
      expect(screen.getByText(/Memory/)).toBeInTheDocument();
      expect(screen.getByText(/2 msgs · 42 tokens · pinned/)).toBeInTheDocument();
      expect(screen.getByText(/Summary/)).toBeInTheDocument();
      expect(screen.getByText(/120 tokens/)).toBeInTheDocument();
      expect(screen.getByText('Trimmed request context before send')).toBeInTheDocument();
      expect(screen.getByText('Messages 4')).toBeInTheDocument();
      expect(screen.getByText('Tokens 4,800 → 2,600')).toBeInTheDocument();
      expect(screen.getByText('Saved 2,200')).toBeInTheDocument();
      expect(screen.getByText('Pins protected')).toBeInTheDocument();
    });
  });

  it('shows compaction stage in pipeline', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        trace_id: 'trace-compaction-stage',
        total_duration_ms: 48,
        root_span: {
          name: 'chat.request',
          span_type: 'node',
          duration_ms: 48,
          children: [
            { name: 'chat.prepare', span_type: 'internal', duration_ms: 8 },
            { name: 'context.compaction', span_type: 'internal', duration_ms: 18 },
            { name: 'chat.stream.llm', span_type: 'internal', duration_ms: 20 },
          ],
        },
      }),
    });
    global.fetch = mockFetch as any;

    render(<TraceDetailPanel traceId="trace-compaction-stage" onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Pipeline stages')).toBeInTheDocument();
      expect(screen.getByText('context.compaction (1x)')).toBeInTheDocument();
      expect(screen.queryByText('Context governance')).not.toBeInTheDocument();
    });
  });
});
