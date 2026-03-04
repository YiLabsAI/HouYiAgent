import { render, screen, waitFor } from '@testing-library/react';
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

  it('shows dash for aggregate timings when no llm/tool/execution spans exist', async () => {
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
      expect(screen.getByText('Orchestration (advanced): 0x · —')).toBeInTheDocument();
    });
  });

  it('shows tool loop split percentages from nested execution/llm/tool spans', async () => {
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
      expect(screen.getByText('LLM 1x · 40ms')).toBeInTheDocument();
      expect(screen.getByText('Tool 1x · 140ms')).toBeInTheDocument();
      expect(screen.getByText('Orchestration (advanced): 1x · 20ms')).toBeInTheDocument();
      expect(screen.getByText('ToolLoop Split')).toBeInTheDocument();
      expect(screen.getByText('LLM 20% · 40ms')).toBeInTheDocument();
      expect(screen.getByText('Tool 70% · 140ms')).toBeInTheDocument();
      expect(screen.getByText('Overhead 10% · 20ms')).toBeInTheDocument();
      expect(screen.getByText('Overhead = ToolLoop total - LLM - Tool')).toBeInTheDocument();
    });
  });

  it('shows disabled-by-request badge when tool loop mode is disabled_by_request', async () => {
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
      expect(screen.getByText('Tool Loop: disabled by request')).toBeInTheDocument();
    });
  });

  it('renders token input/output breakdown and partial usage hint', async () => {
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
      expect(screen.getByText('Tokens 150')).toBeInTheDocument();
      expect(screen.getByText('Partial usage')).toBeInTheDocument();
      expect(screen.getByText('Total Tokens 150')).toBeInTheDocument();
      expect(screen.getByText('Input 120')).toBeInTheDocument();
      expect(screen.getByText('Output 30')).toBeInTheDocument();
      expect(screen.getByText('2/3 LLM calls reported usage')).toBeInTheDocument();
      expect(screen.getByText('Some provider calls did not report usage; token totals are partial real usage.')).toBeInTheDocument();
    });
  });

  it('shows zero token values when usage is partial but all reported totals are zero', async () => {
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
      expect(screen.getByText('Tokens 0')).toBeInTheDocument();
      expect(screen.getByText('Total Tokens 0')).toBeInTheDocument();
      expect(screen.getByText('Input 0')).toBeInTheDocument();
      expect(screen.getByText('Output 0')).toBeInTheDocument();
      expect(screen.getByText('0/2 LLM calls reported usage')).toBeInTheDocument();
      expect(screen.getByText('Partial usage')).toBeInTheDocument();
    });
  });
});
