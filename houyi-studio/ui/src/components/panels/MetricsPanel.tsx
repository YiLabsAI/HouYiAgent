import React, { useMemo } from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import type { SpanData } from '@/stores/storeActions/spanActions';
import { BarChart3 } from 'lucide-react';

export interface MetricsPanelProps {
  executionId?: string;
}

interface LLMMetrics {
  totalCalls: number;
  tokensIn: number;
  tokensOut: number;
  totalCost: number;
  cacheHits: number;
  byModel: Record<string, { calls: number; tokensIn: number; tokensOut: number; cost: number }>;
  latencies: number[];
}

interface ToolMetrics {
  totalCalls: number;
  successCount: number;
  errorCount: number;
  cacheHits: number;
  retryCount: number;
  internalSpans: number;
  byTool: Record<string, { calls: number; errors: number; cacheHits: number }>;
  latencies: number[];
}

interface ExecutionSummary {
  duration: number | null;
  nodeCount: number;
  completedNodes: number;
  errorNodes: number;
  checkpointCount: number;
  status: string;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function formatCost(usd: number): string {
  if (usd === 0) return '$0';
  if (usd < 0.001) return `$${usd.toFixed(6)}`;
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(4)}`;
}

export function computeMetrics(spans: SpanData[]): {
  exec: ExecutionSummary;
  llm: LLMMetrics;
  tool: ToolMetrics;
} {
  const exec: ExecutionSummary = {
    duration: null,
    nodeCount: 0,
    completedNodes: 0,
    errorNodes: 0,
    checkpointCount: 0,
    status: 'unknown',
  };

  const llm: LLMMetrics = {
    totalCalls: 0,
    tokensIn: 0,
    tokensOut: 0,
    totalCost: 0,
    cacheHits: 0,
    byModel: {},
    latencies: [],
  };

  const tool: ToolMetrics = {
    totalCalls: 0,
    successCount: 0,
    errorCount: 0,
    cacheHits: 0,
    retryCount: 0,
    internalSpans: 0,
    byTool: {},
    latencies: [],
  };

  for (const s of spans) {
    switch (s.span_type) {
      case 'execution':
        exec.duration = s.duration ? s.duration * 1000 : null;
        exec.status = s.status;
        break;

      case 'node':
        exec.nodeCount++;
        if (s.status === 'ok') exec.completedNodes++;
        if (s.status === 'error') exec.errorNodes++;
        break;

      case 'llm': {
        llm.totalCalls++;
        const tIn = s.tokens_input ?? 0;
        const tOut = s.tokens_output ?? 0;
        llm.tokensIn += tIn;
        llm.tokensOut += tOut;
        if (s.cost_usd) llm.totalCost += s.cost_usd;
        if (s.cache_hit) llm.cacheHits++;
        if (s.duration) llm.latencies.push(s.duration * 1000);

        const model = s.model || 'unknown';
        if (!llm.byModel[model]) {
          llm.byModel[model] = { calls: 0, tokensIn: 0, tokensOut: 0, cost: 0 };
        }
        llm.byModel[model].calls++;
        llm.byModel[model].tokensIn += tIn;
        llm.byModel[model].tokensOut += tOut;
        if (s.cost_usd) llm.byModel[model].cost += s.cost_usd;
        break;
      }

      case 'tool': {
        tool.totalCalls++;
        if (s.status === 'ok') tool.successCount++;
        if (s.status === 'error') tool.errorCount++;
        if (s.cache_hit) tool.cacheHits++;
        if (s.duration) tool.latencies.push(s.duration * 1000);

        const toolName = s.tool_name || s.name || 'unknown';
        if (!tool.byTool[toolName]) {
          tool.byTool[toolName] = { calls: 0, errors: 0, cacheHits: 0 };
        }
        tool.byTool[toolName].calls++;
        if (s.status === 'error') tool.byTool[toolName].errors++;
        if (s.cache_hit) tool.byTool[toolName].cacheHits++;
        break;
      }

      case 'retry':
        tool.retryCount++;
        break;

      case 'internal':
        tool.internalSpans++;
        break;
    }
  }

  llm.latencies.sort((a, b) => a - b);
  tool.latencies.sort((a, b) => a - b);

  return { exec, llm, tool };
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ executionId }) => {
  const { spanStore, currentExecution, liveExecution } = useConsoleStore();

  const effectiveExecId = executionId || liveExecution?.execution_id || currentExecution?.execution_id;

  const spans = useMemo<SpanData[]>(() => {
    if (!effectiveExecId) return [];
    const store = spanStore[effectiveExecId];
    if (!store) return [];
    return Object.values(store);
  }, [spanStore, effectiveExecId]);

  const { exec, llm, tool } = useMemo(() => computeMetrics(spans), [spans]);

  if (!effectiveExecId || spans.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <BarChart3 size={32} className="mx-auto mb-2 opacity-50" />
          <div className="text-sm">No metrics available</div>
          <div className="text-xs mt-1">Metrics will appear here during execution</div>
        </div>
      </div>
    );
  }

  const llmCacheRate = llm.totalCalls > 0 ? ((llm.cacheHits / llm.totalCalls) * 100).toFixed(1) : '0';
  const toolSuccessRate = tool.totalCalls > 0 ? ((tool.successCount / tool.totalCalls) * 100).toFixed(1) : '0';
  const llmP50 = percentile(llm.latencies, 50);
  const llmP95 = percentile(llm.latencies, 95);
  const toolAvgLatency = tool.latencies.length > 0
    ? tool.latencies.reduce((a, b) => a + b, 0) / tool.latencies.length
    : 0;

  return (
    <div className="space-y-3 text-xs">
      {/* Execution Summary */}
      <div className="bg-gray-900/60 border border-gray-700 rounded p-3">
        <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-medium">Execution Summary</div>
        <div className="flex flex-wrap gap-x-6 gap-y-1">
          <div>
            <span className="text-gray-400">Duration: </span>
            <span className="text-gray-200 font-medium">{exec.duration !== null ? formatMs(exec.duration) : '--'}</span>
          </div>
          <div>
            <span className="text-gray-400">Nodes: </span>
            <span className="text-gray-200 font-medium">
              {exec.completedNodes}/{exec.nodeCount}
              {exec.errorNodes > 0 && <span className="text-red-400 ml-1">({exec.errorNodes} err)</span>}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Status: </span>
            <span className={`font-medium ${exec.status === 'ok' ? 'text-green-400' : exec.status === 'error' ? 'text-red-400' : 'text-gray-300'}`}>
              {exec.status}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Spans: </span>
            <span className="text-gray-200 font-medium">{spans.length}</span>
          </div>
        </div>
      </div>

      {/* LLM + Tool Metrics side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* LLM Metrics */}
        <div className="bg-gray-900/60 border border-gray-700 rounded p-3">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-medium">LLM Metrics</div>
          {llm.totalCalls === 0 ? (
            <div className="text-gray-500">No LLM calls</div>
          ) : (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-x-5 gap-y-1">
                <div>
                  <span className="text-gray-400">Calls: </span>
                  <span className="text-gray-200 font-medium">{llm.totalCalls}</span>
                </div>
                <div>
                  <span className="text-gray-400">Tokens: </span>
                  <span className="text-purple-300 font-medium">{formatTokens(llm.tokensIn)}</span>
                  <span className="text-gray-500"> in / </span>
                  <span className="text-purple-300 font-medium">{formatTokens(llm.tokensOut)}</span>
                  <span className="text-gray-500"> out</span>
                </div>
                {llm.totalCost > 0 && (
                  <div>
                    <span className="text-gray-400">Cost: </span>
                    <span className="text-yellow-300 font-medium">{formatCost(llm.totalCost)}</span>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-1">
                <div>
                  <span className="text-gray-400">Cache: </span>
                  <span className="text-green-400 font-medium">{llm.cacheHits}/{llm.totalCalls}</span>
                  <span className="text-gray-500"> ({llmCacheRate}%)</span>
                </div>
                <div>
                  <span className="text-gray-400">Latency p50/p95: </span>
                  <span className="text-gray-200 font-medium">{formatMs(llmP50)}</span>
                  <span className="text-gray-500"> / </span>
                  <span className="text-gray-200 font-medium">{formatMs(llmP95)}</span>
                </div>
              </div>
              {/* Per-model breakdown */}
              {Object.keys(llm.byModel).length > 1 && (
                <div className="mt-1 pt-1 border-t border-gray-700/50">
                  <div className="text-[10px] text-gray-500 mb-1">Per model:</div>
                  {Object.entries(llm.byModel)
                    .sort(([, a], [, b]) => b.tokensIn - a.tokensIn)
                    .map(([model, m]) => (
                      <div key={model} className="flex items-center gap-2 text-[11px]">
                        <span className="text-purple-400 font-mono w-28 truncate" title={model}>{model}</span>
                        <span className="text-gray-400">{m.calls}x</span>
                        <span className="text-gray-300">{formatTokens(m.tokensIn)}/{formatTokens(m.tokensOut)}</span>
                        {m.cost > 0 && <span className="text-yellow-400">{formatCost(m.cost)}</span>}
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Tool Metrics */}
        <div className="bg-gray-900/60 border border-gray-700 rounded p-3">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-medium">Tool Metrics</div>
          {tool.totalCalls === 0 ? (
            <div className="text-gray-500">No tool calls</div>
          ) : (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-x-5 gap-y-1">
                <div>
                  <span className="text-gray-400">Calls: </span>
                  <span className="text-gray-200 font-medium">{tool.totalCalls}</span>
                </div>
                <div>
                  <span className="text-gray-400">Success: </span>
                  <span className="text-green-400 font-medium">{toolSuccessRate}%</span>
                </div>
                {tool.retryCount > 0 && (
                  <div>
                    <span className="text-gray-400">Retries: </span>
                    <span className="text-amber-400 font-medium">{tool.retryCount}</span>
                  </div>
                )}
                {tool.cacheHits > 0 && (
                  <div>
                    <span className="text-gray-400">Cache hits: </span>
                    <span className="text-green-400 font-medium">{tool.cacheHits}</span>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-x-5 gap-y-1">
                <div>
                  <span className="text-gray-400">Avg latency: </span>
                  <span className="text-gray-200 font-medium">{formatMs(toolAvgLatency)}</span>
                </div>
                {tool.internalSpans > 0 && (
                  <div>
                    <span className="text-gray-400">Internal spans: </span>
                    <span className="text-teal-400 font-medium">{tool.internalSpans}</span>
                  </div>
                )}
              </div>
              {/* Per-tool breakdown */}
              {Object.keys(tool.byTool).length > 0 && (
                <div className="mt-1 pt-1 border-t border-gray-700/50">
                  <div className="text-[10px] text-gray-500 mb-1">Per tool:</div>
                  {Object.entries(tool.byTool)
                    .sort(([, a], [, b]) => b.calls - a.calls)
                    .map(([name, t]) => (
                      <div key={name} className="flex items-center gap-2 text-[11px]">
                        <span className="text-orange-400 font-mono w-28 truncate" title={name}>{name}</span>
                        <span className="text-gray-400">{t.calls}x</span>
                        {t.errors > 0 && <span className="text-red-400">{t.errors} err</span>}
                        {t.cacheHits > 0 && <span className="text-green-400">{t.cacheHits} cached</span>}
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
