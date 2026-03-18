import React from 'react';
import { TraceTotalTokens } from '@/types/chat';
import { TraceSection } from './TraceSection';
import { formatDuration, formatInt } from './TraceDetailUtils';

interface TraceSummaryStatsProps {
  totalDurationMs?: number;
  totalTokens?: TraceTotalTokens;
}

export const TraceSummaryStats: React.FC<TraceSummaryStatsProps> = ({ totalDurationMs, totalTokens }) => {
  const promptTokens = Number(totalTokens?.prompt_tokens || 0);
  const completionTokens = Number(totalTokens?.completion_tokens || 0);
  const total = Number(totalTokens?.total_tokens || 0);
  const partial = Boolean(totalTokens?.is_partial);
  const reported = Number(totalTokens?.llm_spans_with_usage || 0);
  const llmSpans = Number(totalTokens?.llm_spans || 0);
  const coverageLabel = llmSpans > 0 ? `${reported}/${llmSpans} LLM calls reported usage` : null;
  const hasUsageSummary = Boolean(
    totalTokens
    && (
      total > 0
      || promptTokens > 0
      || completionTokens > 0
      || coverageLabel
      || partial
      || llmSpans > 0
    )
  );

  return (
    <TraceSection title="Trace summary">
      <div className={`grid gap-2 text-[11px] text-gray-200 ${hasUsageSummary ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
        <div className="rounded border border-gray-800 bg-gray-950/70 px-2 py-2">
          <div className="text-[10px] text-gray-500">Duration</div>
          <div className="mt-1 font-medium">{formatDuration(totalDurationMs)}</div>
        </div>
        {hasUsageSummary ? (
          <div className="rounded border border-gray-800 bg-gray-950/70 px-2 py-2">
            <div className="text-[10px] text-gray-500">Usage</div>
            <div className="mt-1 font-medium">{formatInt(total)} total</div>
            <div className="mt-1 text-[10px] text-gray-500">
              Input {formatInt(promptTokens)} · Output {formatInt(completionTokens)}
            </div>
          </div>
        ) : null}
      </div>
      {(coverageLabel || partial) && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
          {coverageLabel ? <span>{coverageLabel}</span> : null}
          {partial ? (
            <span className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-amber-300">
              Partial usage
            </span>
          ) : null}
        </div>
      )}
    </TraceSection>
  );
};
