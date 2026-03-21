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
  const convergenceReason = typeof metrics?.tool_loop_convergence_reason === 'string'
    ? metrics.tool_loop_convergence_reason
    : null;
  const finalStreamStatus = typeof metrics?.final_stream_status === 'string'
    ? metrics.final_stream_status
    : null;
  const finalStreamSkipped = typeof metrics?.tool_loop_final_stream_skipped === 'boolean'
    ? metrics.tool_loop_final_stream_skipped
    : null;
  const finalStreamErrorCategory = typeof metrics?.final_stream_error_category === 'string'
    ? metrics.final_stream_error_category
    : null;
  const reasoningRemovedCount = Number(metrics?.final_stream_assistant_reasoning_removed_count || 0);
  const reasoningOnlyRemovedCount = Number(metrics?.final_stream_assistant_reasoning_only_removed_count || 0);
  const toolCarrierCount = Number(metrics?.final_stream_assistant_tool_call_carrier_count || 0);
  const toolProjectionCount = Number(metrics?.final_stream_tool_result_projection_count || 0);
  const requestAdapterClass = typeof metrics?.request_adapter_class === 'string'
    ? metrics.request_adapter_class
    : null;
  const requestAdapterStrict = typeof metrics?.request_adapter_strict_message_string_contract === 'boolean'
    ? metrics.request_adapter_strict_message_string_contract
    : null;
  const requestMessageCount = Number(metrics?.request_message_count || 0);
  const requestUserMessageCount = Number(metrics?.request_user_message_count || 0);
  const requestAssistantMessageCount = Number(metrics?.request_assistant_message_count || 0);
  const requestAssistantReasoningCount = Number(metrics?.request_assistant_reasoning_message_count || 0);
  const requestAssistantReasoningOnlyCount = Number(metrics?.request_assistant_reasoning_only_message_count || 0);
  const requestAssistantToolCallCount = Number(metrics?.request_assistant_tool_call_message_count || 0);
  const requestToolMessageCount = Number(metrics?.request_tool_message_count || 0);
  const flowLabel = finalStreamSkipped
    ? 'Replay'
    : finalStreamStatus === 'reasoning_only'
      ? 'Reasoning only'
    : finalStreamStatus === 'empty_visible_output'
      ? 'Empty output'
      : finalStreamStatus === 'error'
        ? finalStreamErrorCategory ? `Final stream ${finalStreamErrorCategory}` : 'Final stream error'
        : finalStreamStatus === 'completed'
          ? 'Final stream'
          : null;

  if (
    rounds <= 0
    && toolCalls <= 0
    && !traceId
    && totalTokens <= 0
    && !finishReason
    && !flowLabel
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
          {flowLabel && <span className="text-gray-500">{flowLabel}</span>}
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
          {convergenceReason && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Convergence:</span>
              <span>{convergenceReason}</span>
            </div>
          )}
          {finalStreamStatus && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Final stream:</span>
              <span>{finalStreamStatus}</span>
              {finalStreamErrorCategory && <span className="text-gray-500">({finalStreamErrorCategory})</span>}
            </div>
          )}
          {requestAdapterClass && (
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Adapter:</span>
              <span>{requestAdapterClass}</span>
              {requestAdapterStrict !== null && (
                <span className="text-gray-500">{requestAdapterStrict ? '(strict messages)' : '(default messages)'}</span>
              )}
            </div>
          )}
          {(requestMessageCount > 0 || requestAssistantReasoningCount > 0 || requestToolMessageCount > 0) && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-gray-500">Request:</span>
              {requestMessageCount > 0 && <span>{`Messages ${requestMessageCount}`}</span>}
              {requestUserMessageCount > 0 && <span>{`Users ${requestUserMessageCount}`}</span>}
              {requestAssistantMessageCount > 0 && <span>{`Assistants ${requestAssistantMessageCount}`}</span>}
              {requestAssistantReasoningCount > 0 && <span>{`Assistant reasoning ${requestAssistantReasoningCount}`}</span>}
              {requestAssistantReasoningOnlyCount > 0 && <span>{`Reasoning-only ${requestAssistantReasoningOnlyCount}`}</span>}
              {requestAssistantToolCallCount > 0 && <span>{`Tool-call carriers ${requestAssistantToolCallCount}`}</span>}
              {requestToolMessageCount > 0 && <span>{`Tool messages ${requestToolMessageCount}`}</span>}
            </div>
          )}
          {(reasoningRemovedCount > 0 || toolCarrierCount > 0 || toolProjectionCount > 0) && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-gray-500">Sanitized:</span>
              {reasoningRemovedCount > 0 && <span>{`Reasoning removed ${reasoningRemovedCount}`}</span>}
              {reasoningOnlyRemovedCount > 0 && <span>{`Reasoning-only ${reasoningOnlyRemovedCount}`}</span>}
              {toolCarrierCount > 0 && <span>{`Tool carriers ${toolCarrierCount}`}</span>}
              {toolProjectionCount > 0 && <span>{`Tool projections ${toolProjectionCount}`}</span>}
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
