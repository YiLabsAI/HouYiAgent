import React from 'react';
import { X } from 'lucide-react';

type TraceSpan = {
  name?: string;
  span_type?: string;
  status?: string;
  duration_ms?: number;
  start_time_ms?: number;
  attributes?: Record<string, unknown>;
  children?: TraceSpan[];
};

type TracePayload = {
  trace_id?: string;
  total_duration_ms?: number;
  total_tokens?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    llm_spans?: number;
    llm_spans_with_usage?: number;
    is_partial?: boolean;
  };
  root_span?: TraceSpan;
};

const formatDuration = (value: unknown): string => {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms < 0) return '—';
  if (ms > 0 && ms < 1) return '<1ms';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

const collectSpans = (span: TraceSpan | undefined): TraceSpan[] => {
  if (!span) return [];
  const children = Array.isArray(span.children) ? span.children : [];
  return [span, ...children.flatMap((child) => collectSpans(child))];
};

const collectSpansByName = (span: TraceSpan | undefined, name: string): TraceSpan[] => {
  if (!span) return [];
  const children = Array.isArray(span.children) ? span.children : [];
  const matched = span.name === name ? [span] : [];
  return [...matched, ...children.flatMap((child) => collectSpansByName(child, name))];
};

const StatusBadge: React.FC<{ status?: string }> = ({ status }) => {
  const normalized = String(status || 'unknown').toLowerCase();
  const klass = normalized === 'ok'
    ? 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30'
    : normalized === 'error'
      ? 'text-red-300 bg-red-500/10 border-red-500/30'
      : 'text-amber-300 bg-amber-500/10 border-amber-500/30';
  return <span className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${klass}`}>{normalized}</span>;
};

const SpanTreeNode: React.FC<{ span: TraceSpan; depth?: number }> = ({ span, depth = 0 }) => {
  const [expanded, setExpanded] = React.useState(depth < 1);
  const children = Array.isArray(span.children) ? span.children : [];
  const hasChildren = children.length > 0;
  const attrEntries = Object.entries(span.attributes || {}).filter(([, v]) => v !== null && v !== undefined).slice(0, 6);

  return (
    <div className="space-y-1">
      <button
        type="button"
        className="flex w-full items-center justify-between rounded border border-gray-700 bg-gray-900/60 px-2 py-1.5 text-left hover:bg-gray-800/70"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[12px] text-gray-200">
            <span className="truncate font-medium">{span.name || 'span'}</span>
            {span.span_type && <span className="text-[10px] text-gray-500">{span.span_type}</span>}
          </div>
          <div className="mt-0.5 text-[10px] text-gray-500">Duration {formatDuration(span.duration_ms)}</div>
        </div>
        <div className="ml-2 flex items-center gap-2">
          <StatusBadge status={span.status} />
          {hasChildren && <span className="text-[10px] text-gray-500">{expanded ? 'Hide' : 'Show'} {children.length}</span>}
        </div>
      </button>

      {expanded && attrEntries.length > 0 && (
        <div className="rounded border border-gray-800 bg-gray-950/70 px-2 py-1.5 text-[10px] text-gray-400">
          {attrEntries.map(([key, value]) => (
            <div key={key} className="truncate">
              <span className="text-gray-500">{key}:</span> {String(value)}
            </div>
          ))}
        </div>
      )}

      {expanded && hasChildren && (
        <div className="ml-3 space-y-2 border-l border-gray-800 pl-2">
          {children.map((child, index) => (
            <SpanTreeNode key={`${child.name || 'span'}-${index}`} span={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
};

interface TraceDetailPanelProps {
  traceId: string;
  onClose: () => void;
}

type MetricKey = 'llm' | 'tool';

export const TraceDetailPanel: React.FC<TraceDetailPanelProps> = ({ traceId, onClose }) => {
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [payload, setPayload] = React.useState<TracePayload | null>(null);
  const [viewMode, setViewMode] = React.useState<'tree' | 'raw'>('tree');
  const [selectedMetric, setSelectedMetric] = React.useState<MetricKey | null>(null);
  const promptTokens = Number(payload?.total_tokens?.prompt_tokens || 0);
  const completionTokens = Number(payload?.total_tokens?.completion_tokens || 0);
  const totalTokens = Number(payload?.total_tokens?.total_tokens || 0);
  const tokenUsagePartial = Boolean(payload?.total_tokens?.is_partial);
  const tokenCoverageLabel = (() => {
    const withUsage = Number(payload?.total_tokens?.llm_spans_with_usage || 0);
    const total = Number(payload?.total_tokens?.llm_spans || 0);
    if (total <= 0) return null;
    return `${withUsage}/${total} LLM calls reported usage`;
  })();
  const hasTokenUsage = totalTokens > 0 || promptTokens > 0 || completionTokens > 0;
  const shouldRenderTokenSection = hasTokenUsage || tokenUsagePartial || Boolean(tokenCoverageLabel);

  React.useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);

    fetch(`/api/chat/trace/${traceId}`)
      .then(async (resp) => {
        if (!resp.ok) {
          let detail = '';
          try {
            const errBody = await resp.json();
            detail = errBody?.detail ? ` - ${String(errBody.detail)}` : '';
          } catch {
            // ignore parse failures, keep status-only message
          }
          throw new Error(`Trace API ${resp.status}${detail}`);
        }
        return resp.json();
      })
      .then((data) => {
        if (!mounted) return;
        setPayload(data);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (!mounted) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [traceId]);

  const hasTraceData = Boolean(payload && typeof payload === 'object' && payload.root_span);
  const stageTiming = React.useMemo(() => {
    const spans = collectSpans(payload?.root_span);
    const aggregate = (type: string) => {
      const typed = spans.filter((span) => span.span_type === type);
      const totalMs = typed.reduce((sum, span) => sum + (Number(span.duration_ms) || 0), 0);
      return { count: typed.length, totalMs, spans: typed };
    };
    return {
      llm: aggregate('llm'),
      tool: aggregate('tool'),
      execution: aggregate('execution'),
    };
  }, [payload]);

  const metricBreakdown = React.useMemo(() => {
    if (!selectedMetric) return [] as Array<{ name: string; count: number; totalMs: number }>;
    const source = stageTiming[selectedMetric].spans;
    const grouped = new Map<string, { name: string; count: number; totalMs: number }>();
    for (const span of source) {
      const name = span.name || '(unnamed)';
      const next = grouped.get(name) || { name, count: 0, totalMs: 0 };
      next.count += 1;
      next.totalMs += Number(span.duration_ms) || 0;
      grouped.set(name, next);
    }
    return Array.from(grouped.values()).sort((a, b) => b.totalMs - a.totalMs);
  }, [selectedMetric, stageTiming]);

  const toolLoopMode = React.useMemo(() => {
    const toolLoopSpans = collectSpansByName(payload?.root_span, 'chat.tool_loop');
    for (const span of toolLoopSpans) {
      const mode = span.attributes?.['chat.tool_loop.mode'];
      if (typeof mode === 'string' && mode) {
        return mode;
      }
    }
    return null;
  }, [payload]);

  const toolLoopDecision = React.useMemo(() => {
    const toolLoopSpans = collectSpansByName(payload?.root_span, 'chat.tool_loop');
    for (const span of toolLoopSpans) {
      const strategy = span.attributes?.['chat.tool_loop.strategy'];
      const reason = span.attributes?.['chat.tool_loop.gating_reason'];
      return {
        strategy: typeof strategy === 'string' && strategy ? strategy : null,
        reason: typeof reason === 'string' && reason ? reason : null,
      };
    }
    return { strategy: null, reason: null };
  }, [payload]);

  const rootPipelineStages = React.useMemo(() => {
    const children = Array.isArray(payload?.root_span?.children) ? payload?.root_span?.children : [];
    const grouped = new Map<string, { name: string; count: number; totalMs: number }>();
    for (const child of children) {
      const name = child?.name || '(unnamed)';
      const next = grouped.get(name) || { name, count: 0, totalMs: 0 };
      next.count += 1;
      next.totalMs += Number(child?.duration_ms) || 0;
      grouped.set(name, next);
    }
    return Array.from(grouped.values()).sort((a, b) => b.totalMs - a.totalMs);
  }, [payload]);

  const toolLoopBreakdown = React.useMemo(() => {
    const toolLoopSpans = collectSpansByName(payload?.root_span, 'chat.tool_loop');
    if (toolLoopSpans.length === 0) {
      return null;
    }

    const nestedSpans = toolLoopSpans.flatMap((span) => collectSpans(span).slice(1));
    const toolLoopTotalMs = toolLoopSpans.reduce((sum, span) => sum + (Number(span.duration_ms) || 0), 0);
    const aggregate = (type: string) => {
      const typed = nestedSpans.filter((span) => span.span_type === type);
      const totalMs = typed.reduce((sum, span) => sum + (Number(span.duration_ms) || 0), 0);
      return { count: typed.length, totalMs };
    };

    const llm = aggregate('llm');
    const tool = aggregate('tool');
    const execution = aggregate('execution');
    const executionOverheadMs = Math.max(0, toolLoopTotalMs - llm.totalMs - tool.totalMs);
    const totalMs = toolLoopTotalMs;
    if (totalMs <= 0) {
      return null;
    }

    const toPercent = (value: number) => `${((value / totalMs) * 100).toFixed(0)}%`;
    return {
      llm: { ...llm, percent: toPercent(llm.totalMs) },
      tool: { ...tool, percent: toPercent(tool.totalMs) },
      execution: {
        ...execution,
        totalMs: executionOverheadMs,
        percent: toPercent(executionOverheadMs),
      },
      totalMs,
    };
  }, [payload]);

  return (
    <div className="fixed inset-y-0 right-0 z-40 w-[420px] max-w-[90vw] border-l border-gray-700 bg-gray-900 shadow-2xl">
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
        <div className="min-w-0">
          <div className="text-[12px] text-gray-300">Trace Detail</div>
          <code className="block truncate text-[10px] text-gray-500">{traceId}</code>
        </div>
        <button type="button" onClick={onClose} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300">
          <X size={14} />
        </button>
      </div>

      <div className="h-[calc(100%-56px)] overflow-auto p-3">
        {!loading && !error && hasTraceData && (
          <div className="mb-2 space-y-1 rounded border border-gray-800 bg-gray-900/60 px-2 py-1.5 text-[10px] text-gray-400">
            <div className="flex items-center justify-between">
              <div className="flex flex-wrap items-center gap-3">
                <span>Total {formatDuration(payload?.total_duration_ms)}</span>
                {shouldRenderTokenSection && <span>Tokens {totalTokens.toLocaleString()}</span>}
                {tokenUsagePartial && (
                  <span className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-amber-300">
                    Partial usage
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  className={`rounded px-2 py-0.5 ${viewMode === 'tree' ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:text-gray-200'}`}
                  onClick={() => setViewMode('tree')}
                >
                  Tree
                </button>
                <button
                  type="button"
                  className={`rounded px-2 py-0.5 ${viewMode === 'raw' ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:text-gray-200'}`}
                  onClick={() => setViewMode('raw')}
                >
                  JSON
                </button>
              </div>
            </div>
            {shouldRenderTokenSection && (
              <div className="rounded border border-gray-800 bg-gray-950/70 px-2 py-1.5 text-[10px] text-gray-300">
                <div className="grid grid-cols-1 gap-1 sm:grid-cols-3">
                  <div>Total Tokens {totalTokens.toLocaleString()}</div>
                  <div>Input {promptTokens.toLocaleString()}</div>
                  <div>Output {completionTokens.toLocaleString()}</div>
                </div>
                {tokenCoverageLabel && (
                  <div className="mt-1 text-gray-500">{tokenCoverageLabel}</div>
                )}
                {tokenUsagePartial && (
                  <div className="mt-1 text-amber-300/80">Some provider calls did not report usage; token totals are partial real usage.</div>
                )}
              </div>
            )}
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {([
                {
                  key: 'llm' as const,
                  label: 'LLM',
                  value: stageTiming.llm,
                  className: 'border-purple-500/30 bg-purple-500/10 text-purple-300',
                },
                {
                  key: 'tool' as const,
                  label: 'Tool',
                  value: stageTiming.tool,
                  className: 'border-orange-500/30 bg-orange-500/10 text-orange-300',
                },
              ]).map((item) => (
                <button
                  key={item.key}
                  type="button"
                  className={`rounded border px-2 py-1 text-left text-[10px] transition-colors ${item.className} ${selectedMetric === item.key ? 'ring-1 ring-gray-300/50' : 'hover:border-gray-500/50'}`}
                  onClick={() => setSelectedMetric((current) => (current === item.key ? null : item.key))}
                >
                  <div className="font-medium">
                    {item.label} {item.value.count}x · {item.value.count > 0 ? formatDuration(item.value.totalMs) : '—'}
                  </div>
                  <div className="mt-0.5 text-[10px] text-gray-400">Click to view composition</div>
                </button>
              ))}
            </div>
            <div className="text-[10px] text-gray-500">
              LLM / Tool are span-type aggregates; they can overlap, so they do not sum to Total.
            </div>
            <div className="text-[10px] text-gray-500">
              Orchestration (advanced): {stageTiming.execution.count}x · {stageTiming.execution.count > 0 ? formatDuration(stageTiming.execution.totalMs) : '—'}
            </div>
            {selectedMetric && (
              <div className="rounded border border-gray-800 bg-gray-950/70 px-2 py-1.5 text-[10px] text-gray-400">
                <div className="mb-1 text-gray-300">
                  {selectedMetric.toUpperCase()} composition ({metricBreakdown.length} groups)
                </div>
                {metricBreakdown.length === 0 ? (
                  <div>No spans found.</div>
                ) : (
                  <div className="space-y-1">
                    {metricBreakdown.slice(0, 12).map((entry) => (
                      <div key={entry.name} className="flex items-center justify-between gap-2">
                        <span className="truncate">{entry.name} ({entry.count}x)</span>
                        <span>{formatDuration(entry.totalMs)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {rootPipelineStages.length > 0 && (
              <div className="rounded border border-emerald-500/20 bg-emerald-500/5 px-2 py-1.5 text-[10px] text-emerald-200">
                <div className="mb-1 text-emerald-300">Pipeline stages (root children, closest to wall time)</div>
                <div className="space-y-1">
                  {rootPipelineStages.map((stage) => (
                    <div key={stage.name} className="flex items-center justify-between gap-2">
                      <span className="truncate">{stage.name} ({stage.count}x)</span>
                      <span>{formatDuration(stage.totalMs)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {toolLoopBreakdown && (
              <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
                <span className="text-gray-400">ToolLoop Split</span>
                <span className="rounded border border-purple-500/30 bg-purple-500/10 px-1.5 py-0.5 text-purple-300">
                  LLM {toolLoopBreakdown.llm.percent} · {formatDuration(toolLoopBreakdown.llm.totalMs)}
                </span>
                <span className="rounded border border-orange-500/30 bg-orange-500/10 px-1.5 py-0.5 text-orange-300">
                  Tool {toolLoopBreakdown.tool.percent} · {formatDuration(toolLoopBreakdown.tool.totalMs)}
                </span>
                <span className="rounded border border-blue-500/30 bg-blue-500/10 px-1.5 py-0.5 text-blue-300">
                  Overhead {toolLoopBreakdown.execution.percent} · {formatDuration(toolLoopBreakdown.execution.totalMs)}
                </span>
                <span className="text-gray-500">Overhead = ToolLoop total - LLM - Tool</span>
              </div>
            )}
            {toolLoopMode === 'disabled_by_request' && (
              <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
                <span className="rounded border border-sky-500/30 bg-sky-500/10 px-1.5 py-0.5 text-sky-300">
                  Tool Loop: disabled by request
                </span>
              </div>
            )}
            {(toolLoopDecision.strategy || toolLoopDecision.reason) && (
              <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
                {toolLoopDecision.strategy && (
                  <span className="rounded border border-indigo-500/30 bg-indigo-500/10 px-1.5 py-0.5 text-indigo-300">
                    Strategy: {toolLoopDecision.strategy}
                  </span>
                )}
                {toolLoopDecision.reason && (
                  <span className="rounded border border-slate-500/30 bg-slate-500/10 px-1.5 py-0.5 text-slate-300">
                    Gate: {toolLoopDecision.reason}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
        {loading && <div className="text-[12px] text-gray-500">Loading trace...</div>}
        {error && <div className="text-[12px] text-red-400">{error}</div>}
        {!loading && !error && hasTraceData && viewMode === 'tree' && payload?.root_span && (
          <div className="space-y-2">
            <SpanTreeNode span={payload.root_span} />
          </div>
        )}
        {!loading && !error && hasTraceData && viewMode === 'raw' && (
          <pre className="whitespace-pre-wrap break-words rounded bg-gray-950 p-3 text-[11px] leading-relaxed text-gray-300">
            {JSON.stringify(payload, null, 2)}
          </pre>
        )}
        {!loading && !error && !hasTraceData && (
          <div className="rounded border border-gray-700 bg-gray-850 p-3 text-[12px] text-gray-400">
            Trace data is empty for this id.
            <div className="mt-1 text-[11px] text-gray-500">
              This usually means the trace already expired or observability storage is not retaining spans.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
