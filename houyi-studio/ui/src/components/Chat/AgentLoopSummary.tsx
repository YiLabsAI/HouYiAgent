import React from 'react';
import { ChevronDown, ChevronRight, Route, Cpu, Wrench } from 'lucide-react';

interface AgentLoopSummaryProps {
  rounds: number;
  toolCalls: number;
  traceId: string | null;
  usage: Record<string, any> | null;
  metrics?: Record<string, any> | null;
  onOpenTrace?: (traceId: string) => void;
}

export const AgentLoopSummary: React.FC<AgentLoopSummaryProps> = ({
  rounds,
  toolCalls,
  traceId,
  usage,
  metrics,
  onOpenTrace,
}) => {
  const [expanded, setExpanded] = React.useState(false);
  const totalTokens = Number(usage?.total_tokens || 0);
  const finishReason = typeof metrics?.finish_reason === 'string' ? metrics.finish_reason : null;

  if (
    rounds <= 0
    && toolCalls <= 0
    && !traceId
    && totalTokens <= 0
    && !finishReason
  ) {
    return null;
  }

  return (
    <div className="shrink-0 border-b border-gray-800 bg-gray-900/80 px-4 py-2">
      <button
        type="button"
        className="flex w-full items-center justify-between text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="flex flex-wrap items-center gap-2 text-[11px] text-gray-300">
          <Route size={14} className="text-gray-400" />
          <span>Agent Loop</span>
          <span className="text-gray-500">{rounds} rounds</span>
          <span className="text-gray-500">{toolCalls} tool calls</span>
          {traceId && (
            <span className="rounded border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-[10px] text-violet-200">
              Trace
            </span>
          )}
        </span>
        {expanded ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
      </button>

      {expanded && (
        <div className="mt-2 space-y-1 text-[11px] text-gray-400">
          <div className="flex items-center gap-2">
            <Cpu size={12} />
            <span>Iterations: {rounds}</span>
          </div>
          <div className="flex items-center gap-2">
            <Wrench size={12} />
            <span>Tool calls: {toolCalls}</span>
          </div>
          {totalTokens > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Usage:</span>
              <span>Total tokens {totalTokens}</span>
            </div>
          )}
          {finishReason && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Finish:</span>
              <span>{finishReason}</span>
            </div>
          )}
          {traceId && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Trace:</span>
              <code className="rounded bg-gray-800 px-1.5 py-0.5 text-[10px] text-gray-300">{traceId}</code>
              {onOpenTrace && (
                <button
                  type="button"
                  className="text-blue-400 hover:text-blue-300 underline"
                  onClick={(e) => {
                    e.stopPropagation();
                    onOpenTrace(traceId);
                  }}
                >
                  details
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
