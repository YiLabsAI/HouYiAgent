import React from 'react';
import { TraceContextPlan, TraceRequestContext as TraceRequestContextType } from '@/types/chat';
import { TraceSection } from './TraceSection';
import { formatInt, hasValue } from './TraceDetailUtils';

interface TraceRequestContextProps {
  requestContext?: TraceRequestContextType;
  contextPlan?: TraceContextPlan;
}

const Row: React.FC<{ label: string; value: React.ReactNode }> = ({ label, value }) => (
  <div className="flex items-start justify-between gap-3 text-[11px]">
    <span className="text-gray-500">{label}</span>
    <span className="truncate text-right text-gray-200">{value}</span>
  </div>
);

export const TraceRequestContext: React.FC<TraceRequestContextProps> = ({ requestContext, contextPlan }) => {
  const blocks = Object.entries(contextPlan?.block_breakdown || {}).filter(([, tokens]) => Number(tokens) > 0);
  const hasRequest = Boolean(
    requestContext
    && (
      hasValue(requestContext.request_id)
      || hasValue(requestContext.conversation_id)
      || hasValue(requestContext.model)
      || hasValue(requestContext.max_context_tokens)
      || hasValue(requestContext.llm_messages_count)
    ),
  );
  const hasPlan = Boolean(
    contextPlan
    && (
      hasValue(contextPlan.used_tokens)
      || hasValue(contextPlan.planned_prompt_tokens)
      || hasValue(contextPlan.reserved_output_tokens)
      || hasValue(contextPlan.available_input_tokens)
      || blocks.length > 0
    ),
  );

  if (!hasRequest && !hasPlan) return null;

  return (
    <TraceSection title="Request context" testId="trace-request-context">
      <div className="space-y-3">
        {hasRequest && (
          <div className="space-y-1.5">
            <Row label="Request" value={requestContext?.request_id || '—'} />
            <Row label="Conversation" value={requestContext?.conversation_id || '—'} />
            <Row label="Model" value={requestContext?.model || '—'} />
            <Row label="Max context" value={formatInt(requestContext?.max_context_tokens)} />
            <Row label="LLM messages" value={formatInt(requestContext?.llm_messages_count)} />
          </div>
        )}
        {hasPlan && (
          <div className="space-y-1.5 border-t border-gray-800 pt-3">
            <Row label="Used" value={formatInt(contextPlan?.used_tokens)} />
            <Row label="Planned prompt" value={formatInt(contextPlan?.planned_prompt_tokens)} />
            <Row label="Reserved output" value={formatInt(contextPlan?.reserved_output_tokens)} />
            <Row label="Available input" value={formatInt(contextPlan?.available_input_tokens)} />
            {blocks.length > 0 && (
              <div className="pt-1">
                <div className="mb-1 text-[10px] text-gray-500">Block breakdown</div>
                <div className="flex flex-wrap gap-2">
                  {blocks
                    .sort((a, b) => Number(b[1]) - Number(a[1]))
                    .map(([name, tokens]) => (
                      <span key={name} className="rounded border border-cyan-500/20 bg-cyan-500/5 px-1.5 py-0.5 text-[10px] text-cyan-100">
                        {name} {formatInt(tokens)}
                      </span>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </TraceSection>
  );
};
