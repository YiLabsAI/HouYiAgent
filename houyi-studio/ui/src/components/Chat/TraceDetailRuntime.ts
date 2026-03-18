import type { TracePayload, TraceSpan } from '@/types/chat';

export type TraceMetricKey = 'llm' | 'tool' | 'execution';

export interface TraceMetricSummary {
  count: number;
  totalMs: number;
}

export interface TraceMetricBreakdownEntry {
  name: string;
  count: number;
  totalMs: number;
}

export interface TraceToolLoopBreakdown {
  llm: TraceMetricSummary & { percent: string };
  tool: TraceMetricSummary & { percent: string };
  execution: TraceMetricSummary & { percent: string };
}

export interface TraceToolLoopDecision {
  strategy: string | null;
  reason: string | null;
}

export interface TracePipelineStage {
  name: string;
  count: number;
  totalMs: number;
}

interface StageTimingSummary extends TraceMetricSummary {
  spans: TraceSpan[];
}

interface TraceRuntimeState {
  aggregates: Record<TraceMetricKey, TraceMetricSummary>;
  metricBreakdown: TraceMetricBreakdownEntry[];
  toolLoopMode: string | null;
  toolLoopDecision: TraceToolLoopDecision;
  pipelineStages: TracePipelineStage[];
  toolLoopBreakdown: TraceToolLoopBreakdown | null;
}

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

export const buildTraceDetailRuntime = (
  payload: TracePayload | null,
  selectedMetric: TraceMetricKey | null,
): TraceRuntimeState => {
  const spans = collectSpans(payload?.root_span);
  const aggregate = (type: string): StageTimingSummary => {
    const typed = spans.filter((span) => span.span_type === type);
    const totalMs = typed.reduce((sum, span) => sum + (Number(span.duration_ms) || 0), 0);
    return { count: typed.length, totalMs, spans: typed };
  };

  const stageTiming: Record<TraceMetricKey, StageTimingSummary> = {
    llm: aggregate('llm'),
    tool: aggregate('tool'),
    execution: aggregate('execution'),
  };

  const metricBreakdown = (() => {
    if (!selectedMetric) return [] as TraceMetricBreakdownEntry[];
    const source = stageTiming[selectedMetric].spans;
    const grouped = new Map<string, TraceMetricBreakdownEntry>();
    for (const span of source) {
      const name = span.name || '(unnamed)';
      const next = grouped.get(name) || { name, count: 0, totalMs: 0 };
      next.count += 1;
      next.totalMs += Number(span.duration_ms) || 0;
      grouped.set(name, next);
    }
    return Array.from(grouped.values()).sort((a, b) => b.totalMs - a.totalMs);
  })();

  const toolLoopSpans = collectSpansByName(payload?.root_span, 'chat.tool_loop');

  const toolLoopMode = (() => {
    for (const span of toolLoopSpans) {
      const mode = span.attributes?.['chat.tool_loop.mode'];
      if (typeof mode === 'string' && mode) return mode;
    }
    return null;
  })();

  const toolLoopDecision = (() => {
    for (const span of toolLoopSpans) {
      const strategy = span.attributes?.['chat.tool_loop.strategy'];
      const reason = span.attributes?.['chat.tool_loop.gating_reason'];
      return {
        strategy: typeof strategy === 'string' && strategy ? strategy : null,
        reason: typeof reason === 'string' && reason ? reason : null,
      };
    }
    return { strategy: null, reason: null };
  })();

  const pipelineStages = (() => {
    const children = Array.isArray(payload?.root_span?.children) ? payload.root_span.children : [];
    const grouped = new Map<string, TracePipelineStage>();
    for (const child of children) {
      const name = child?.name || '(unnamed)';
      const next = grouped.get(name) || { name, count: 0, totalMs: 0 };
      next.count += 1;
      next.totalMs += Number(child?.duration_ms) || 0;
      grouped.set(name, next);
    }
    return Array.from(grouped.values()).sort((a, b) => b.totalMs - a.totalMs);
  })();

  const toolLoopBreakdown = (() => {
    if (toolLoopSpans.length === 0) return null;
    const nestedSpans = toolLoopSpans.flatMap((span) => collectSpans(span).slice(1));
    const toolLoopTotalMs = toolLoopSpans.reduce((sum, span) => sum + (Number(span.duration_ms) || 0), 0);
    if (toolLoopTotalMs <= 0) return null;

    const nestedAggregate = (type: string): TraceMetricSummary => {
      const typed = nestedSpans.filter((span) => span.span_type === type);
      const totalMs = typed.reduce((sum, span) => sum + (Number(span.duration_ms) || 0), 0);
      return { count: typed.length, totalMs };
    };

    const llm = nestedAggregate('llm');
    const tool = nestedAggregate('tool');
    const executionOverheadMs = Math.max(0, toolLoopTotalMs - llm.totalMs - tool.totalMs);
    const toPercent = (value: number) => `${((value / toolLoopTotalMs) * 100).toFixed(0)}%`;

    return {
      llm: { ...llm, percent: toPercent(llm.totalMs) },
      tool: { ...tool, percent: toPercent(tool.totalMs) },
      execution: { count: 0, totalMs: executionOverheadMs, percent: toPercent(executionOverheadMs) },
    } satisfies TraceToolLoopBreakdown;
  })();

  return {
    aggregates: {
      llm: { count: stageTiming.llm.count, totalMs: stageTiming.llm.totalMs },
      tool: { count: stageTiming.tool.count, totalMs: stageTiming.tool.totalMs },
      execution: { count: stageTiming.execution.count, totalMs: stageTiming.execution.totalMs },
    },
    metricBreakdown,
    toolLoopMode,
    toolLoopDecision,
    pipelineStages,
    toolLoopBreakdown,
  };
};
