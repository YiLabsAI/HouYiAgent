import React from 'react';
import { TraceContextGovernance as TraceContextGovernanceType } from '@/types/chat';
import { TraceSection } from './TraceSection';
import { formatInt } from './TraceDetailUtils';

const dropReasonLabels: Record<string, string> = {
  truncated_to_fit: 'Trimmed to fit request budget',
  boundary_excluded: 'Excluded by planning boundary',
  lower_priority: 'Excluded as lower priority context',
  budget_exceeded: 'Trimmed to fit request budget',
  policy_excluded: 'Excluded by request policy',
  excluded_without_current_turn: 'Excluded because the current turn could not be kept',
};

const blockLabels: Record<string, string> = {
  system: 'System',
  current_turn: 'Current turn',
  recent: 'Recent',
  pinned: 'Pinned',
  tool_summary: 'Tool summary',
  summary: 'Summary',
  memory: 'Memory',
};

const compactionTriggerLabels: Record<string, string> = {
  pre_request_pressure: 'Compacted conversation context for this request',
  overflow_recovery: 'Recovered rolling context capacity',
  manual: 'Compacted conversation context',
  post_turn_background: 'Compacted conversation context in background',
  repo_intent_trim: 'Trimmed request context before send',
};

function getTrimmedDetailLabel(blockType?: string | null, source?: string | null): string {
  if (source && blockLabels[source]) return blockLabels[source];
  if (blockType && blockLabels[blockType]) return blockLabels[blockType];
  return 'Omitted context block';
}

function formatTrimmedDetailSummary(detail: {
  token_count?: number;
  message_count?: number | null;
  pinned?: boolean;
}): string {
  const parts: string[] = [];
  if (typeof detail.message_count === 'number' && detail.message_count > 0) {
    parts.push(`${formatInt(detail.message_count)} msg${detail.message_count > 1 ? 's' : ''}`);
  }
  if (typeof detail.token_count === 'number' && detail.token_count > 0) {
    parts.push(`${formatInt(detail.token_count)} tokens`);
  }
  if (detail.pinned) {
    parts.push('pinned');
  }
  return parts.join(' · ');
}

interface TraceContextGovernanceProps {
  governance?: TraceContextGovernanceType;
}

export const TraceContextGovernance: React.FC<TraceContextGovernanceProps> = ({ governance }) => {
  const dropReasons = Object.entries(governance?.drop_reasons || {});
  const droppedBlocks = governance?.dropped_blocks || [];
  const droppedBlockDetails = governance?.dropped_block_details || [];
  const droppedDetailById = new Map(
    droppedBlockDetails.map((detail) => [detail.candidate_id, detail]),
  );
  const compaction = governance?.compaction;
  const hasContent = droppedBlocks.length > 0 || dropReasons.length > 0 || droppedBlockDetails.length > 0 || Boolean(compaction?.triggered);
  const trimmedReasonCounts = dropReasons.reduce<Record<string, number>>((acc, [, reason]) => {
    const key = String(reason || 'unspecified');
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});

  if (!hasContent) return null;

  return (
    <TraceSection title="Context governance">
      <div className="space-y-3 text-[11px]">
        {(droppedBlocks.length > 0 || dropReasons.length > 0 || droppedBlockDetails.length > 0) && (
          <div>
            <div className="mb-1 text-[10px] text-gray-500">Context trimmed</div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded border border-gray-700 bg-gray-950/70 px-1.5 py-0.5 text-gray-300">
                Trimmed {formatInt(Math.max(droppedBlocks.length, dropReasons.length, droppedBlockDetails.length))} blocks
              </span>
              {Object.entries(trimmedReasonCounts).sort((a, b) => b[1] - a[1]).map(([reason, count]) => (
                <span key={reason} className="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-amber-200">
                  {dropReasonLabels[reason] ?? reason} {formatInt(count)}
                </span>
              ))}
            </div>
            {(dropReasons.length > 0 || droppedBlockDetails.length > 0) ? (
              <div className="mt-2 space-y-1">
                {(dropReasons.length > 0
                  ? dropReasons.map(([candidateId, reason], index) => {
                      const detail = droppedDetailById.get(candidateId);
                      const summary = detail ? formatTrimmedDetailSummary(detail) : '';
                      return (
                        <div
                          key={`${candidateId}-${reason}-${index}`}
                          className="rounded border border-gray-800 bg-gray-950/50 px-2 py-1 text-gray-300"
                        >
                          <span className="font-medium text-gray-200">
                            {detail ? getTrimmedDetailLabel(detail.block_type, detail.source) : 'Omitted context block'}
                          </span>
                          <span className="text-gray-500"> — </span>
                          <span>{dropReasonLabels[String(reason)] ?? String(reason)}</span>
                          {summary ? <span className="text-gray-500"> · {summary}</span> : null}
                        </div>
                      );
                    })
                  : droppedBlockDetails.map((detail, index) => {
                      const summary = formatTrimmedDetailSummary(detail);
                      return (
                        <div
                          key={`${detail.candidate_id}-${index}`}
                          className="rounded border border-gray-800 bg-gray-950/50 px-2 py-1 text-gray-300"
                        >
                          <span className="font-medium text-gray-200">
                            {getTrimmedDetailLabel(detail.block_type, detail.source)}
                          </span>
                          {summary ? <span className="text-gray-500"> · {summary}</span> : null}
                        </div>
                      );
                    }))}
              </div>
            ) : null}
          </div>
        )}
        {compaction?.triggered && (
          <div>
            <div className="mb-1 text-[10px] text-gray-500">Compaction</div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded border border-cyan-500/20 bg-cyan-500/5 px-1.5 py-0.5 text-cyan-100">
                {compaction.trigger
                  ? (compactionTriggerLabels[compaction.trigger] ?? compaction.trigger)
                  : 'triggered'}
              </span>
              <span className="rounded border border-gray-700 bg-gray-950/70 px-1.5 py-0.5 text-gray-300">
                Messages {formatInt(compaction.messages_compacted)}
              </span>
              <span className="rounded border border-gray-700 bg-gray-950/70 px-1.5 py-0.5 text-gray-300">
                Tokens {formatInt(compaction.tokens_before)} → {formatInt(compaction.tokens_after)}
              </span>
              <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-200">
                Saved {formatInt(compaction.saved_tokens)}
              </span>
              <span className={`rounded border px-1.5 py-0.5 ${Number(compaction.pin_violation_count || 0) > 0 ? 'border-red-500/30 bg-red-500/10 text-red-200' : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'}`}>
                {Number(compaction.pin_violation_count || 0) > 0 ? `Pin violations ${formatInt(compaction.pin_violation_count)}` : 'Pins protected'}
              </span>
            </div>
          </div>
        )}
      </div>
    </TraceSection>
  );
};
