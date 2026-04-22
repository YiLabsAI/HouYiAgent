import React from 'react';
import { ChevronDown, ChevronRight, X } from 'lucide-react';
import type {
  CompactionHistoryItem,
  CompactionRecord,
  ContextUsage,
  ConversationContextState,
  PinnedContextRecord,
  PinStatus,
  TokenUsage,
} from '@/types/chat';
import { formatInt } from './TraceDetailUtils';

interface ChatInspectorPanelProps {
  conversationContext?: ConversationContextState | null;
  contextUsage?: ContextUsage | null;
  latestCompaction?: CompactionRecord | null;
  compactionHistory?: CompactionHistoryItem[];
  isLoadingCompactions?: boolean;
  restoringCompactionId?: string | null;
  restoringBackupId?: string | null;
  restoreNotice?: {
    message: string;
    undoBackupId: string | null;
  } | null;
  activePins?: PinnedContextRecord[];
  usage?: TokenUsage | null;
  onUpdatePinnedContextStatus?: (pinId: string, status: PinStatus) => void;
  onRestoreCompaction?: (compactionId: string) => void;
  onRestoreBackup?: (backupId: string) => void;
  onClearRestoreNotice?: () => void;
  onClose: () => void;
}

const breakdownMeta: Record<string, { label: string; description: string }> = {
  system: { label: 'System', description: 'System instructions' },
  pinned: { label: 'Pinned', description: 'Protected context' },
  current_turn: { label: 'Current turn', description: 'Current user turn' },
  recent: { label: 'Recent', description: 'Recent dialogue' },
  memory: { label: 'Memory', description: 'Recalled memory' },
  summary: { label: 'Summary', description: 'Compressed history' },
  tool_summary: { label: 'Tool summary', description: 'Compressed tool results' },
};

const conversationStateMeta: Record<NonNullable<ConversationContextState['state']>, { label: string; toneClassName: string }> = {
  healthy: { label: 'Healthy', toneClassName: 'border-sky-500/30 bg-sky-500/10 text-sky-200' },
  elevated: { label: 'Growing', toneClassName: 'border-amber-500/30 bg-amber-500/10 text-amber-200' },
  near_compaction: { label: 'Near compaction', toneClassName: 'border-red-500/30 bg-red-500/10 text-red-200' },
  compacted_recently: { label: 'Compacted', toneClassName: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200' },
};

function resolveConversationStateMeta(
  conversationContext: ConversationContextState | null | undefined,
) {
  if (!conversationContext) return conversationStateMeta.healthy;
  if (conversationContext.state !== 'compacted_recently' || conversationContext.max_units <= 0) {
    return conversationStateMeta[conversationContext.state];
  }
  const ratio = conversationContext.used_units / conversationContext.max_units;
  if (ratio >= 0.9) {
    return {
      label: 'Compacted',
      toneClassName: 'border-red-500/30 bg-red-500/10 text-red-200',
    };
  }
  if (ratio >= 0.7) {
    return {
      label: 'Compacted',
      toneClassName: 'border-amber-500/30 bg-amber-500/10 text-amber-200',
    };
  }
  return conversationStateMeta.compacted_recently;
}

const compactionTriggerLabels: Record<string, string> = {
  pre_request_pressure: 'Compacted conversation context for this request',
  overflow_recovery: 'Recovered rolling context capacity',
  manual: 'Compacted conversation context',
  post_turn_background: 'Compacted conversation context in background',
  repo_intent_trim: 'Trimmed request context before send',
};

const compactionTriggerDescriptions: Record<string, string> = {
  pre_request_pressure: 'Triggered before send because rolling conversation context utilization was close to the configured compaction watermark.',
  overflow_recovery: 'Triggered after the system detected rolling conversation context had exceeded the critical compaction watermark.',
  manual: 'Triggered by an explicit user or operator compaction action.',
  post_turn_background: 'Triggered automatically after a completed assistant turn to keep rolling conversation context healthy for future requests.',
  repo_intent_trim: 'Triggered by the request planner while fitting the current send into the active model context window. This is request-scoped trimming, not a conversation compaction record.',
};

const backupTriggerDescriptions: Record<string, string> = {
  restore_point: 'A rollback snapshot created immediately before applying a restore action.',
  compaction_snapshot: 'A snapshot captured for a compaction record so the prior conversation state can be restored later.',
};

const dropReasonLabels: Record<string, string> = {
  truncated_to_fit: 'Trimmed to fit request budget',
  boundary_excluded: 'Excluded by planning boundary',
  lower_priority: 'Excluded as lower priority context',
  budget_exceeded: 'Trimmed to fit request budget',
  policy_excluded: 'Excluded by request policy',
  excluded_without_current_turn: 'Excluded because the current turn could not be kept',
};

const dropReasonDescriptions: Record<string, string> = {
  truncated_to_fit: 'The planner removed lower-priority context to stay inside the model budget.',
  boundary_excluded: 'This block was outside the active planning boundary for the current request.',
  lower_priority: 'Higher-priority blocks consumed the available budget first.',
  budget_exceeded: 'The request hit budget limits before this block could be included.',
  policy_excluded: 'This block type was disabled by the current request policy or planner selection settings.',
  excluded_without_current_turn: 'This older context was omitted because the planner could not keep the current turn within the available budget.',
};

const fallbackTrimmedItemLabel = 'Omitted context block';

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <section className="rounded border border-gray-800 bg-gray-900/60 p-3">
    <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">{title}</div>
    <div className="space-y-2">{children}</div>
  </section>
);

const Row: React.FC<{ label: string; value: React.ReactNode; description?: React.ReactNode }> = ({ label, value, description }) => (
  <div className="grid grid-cols-[minmax(0,140px)_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px]">
    <div className="text-gray-400">{label}</div>
    <div className="text-gray-200">
      <div>{value}</div>
      {description ? <div className="text-[10px] text-gray-500">{description}</div> : null}
    </div>
  </div>
);

function formatUsageConfidenceLabel(confidence?: string | null): string {
  if (confidence === 'reported') return 'Provider reported overall usage';
  if (confidence === 'fallback') return 'Estimated fallback usage';
  if (!confidence) return '—';
  return String(confidence);
}

function renderReportedTokenValue(value: number | undefined, reported?: boolean): string {
  if (!reported) return 'Not reported';
  return formatInt(value);
}

function renderCacheHitValue(usage?: TokenUsage | null): string {
  if (!usage?.cache_hit_reported) return 'Unknown';
  return usage.cache_hit ? 'Yes' : 'No';
}

function resolveTimingMetric(
  primary: unknown,
  fallback: unknown,
): number | null {
  const candidates = [primary, fallback];
  for (const candidate of candidates) {
    if (candidate === undefined || candidate === null || candidate === '') continue;
    const parsed = Number(candidate);
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed;
    }
  }
  return null;
}

function getTrimmedItemLabel(name: string): string {
  return breakdownMeta[name]?.label ?? fallbackTrimmedItemLabel;
}

function getTrimmedDetailLabel(blockType?: string | null, source?: string | null): string {
  if (source && breakdownMeta[source]) return breakdownMeta[source].label;
  if (blockType && breakdownMeta[blockType]) return breakdownMeta[blockType].label;
  return fallbackTrimmedItemLabel;
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
  return parts.join(' · ') || 'No additional planner detail reported.';
}

const formatCompactionTime = (timestamp?: number | null): string => {
  if (!timestamp) return '—';
  return new Date(timestamp * 1000).toLocaleString();
};

const formatBackupMetadataValue = (value: unknown): string => {
  if (typeof value === 'string' && value.trim()) return value.trim();
  return '—';
};

const PreviewList: React.FC<{
  items: Array<{
    message_id: string;
    role: string;
    name?: string | null;
    created_at?: number;
    preview: string;
  }>;
  emptyLabel: string;
}> = ({ items, emptyLabel }) => {
  if (items.length === 0) {
    return <div className="text-[10px] text-gray-500">{emptyLabel}</div>;
  }
  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.message_id} className="rounded border border-gray-800 bg-gray-950/60 p-2">
          <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
            <span className="rounded border border-gray-700 bg-gray-900/70 px-1.5 py-0.5 text-gray-300">
              {item.role}
            </span>
            {item.name ? <span>{item.name}</span> : null}
            <span className="font-mono text-[9px] text-gray-600">{item.message_id}</span>
            {item.created_at ? <span>{formatCompactionTime(item.created_at)}</span> : null}
          </div>
          <div className="mt-1 text-[11px] leading-5 text-gray-300 whitespace-pre-wrap break-words">
            {item.preview}
          </div>
        </div>
      ))}
    </div>
  );
};

const CollapsedPreviewList: React.FC<{
  items: Array<{
    message_id: string;
    role: string;
    name?: string | null;
    created_at?: number;
    preview: string;
  }>;
  emptyLabel: string;
}> = ({ items, emptyLabel }) => {
  if (items.length === 0) {
    return <div className="text-[10px] text-gray-500">{emptyLabel}</div>;
  }
  const visibleItems = items.slice(0, 2);
  return (
    <div className="space-y-2">
      {visibleItems.map((item) => (
        <div key={item.message_id} className="rounded border border-gray-800 bg-gray-950/60 p-2">
          <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
            <span className="rounded border border-gray-700 bg-gray-900/70 px-1.5 py-0.5 text-gray-300">
              {item.role}
            </span>
            {item.name ? <span>{item.name}</span> : null}
            <span className="font-mono text-[9px] text-gray-600">{item.message_id}</span>
          </div>
          <div className="mt-1 line-clamp-2 text-[11px] leading-5 text-gray-300 break-words">
            {item.preview}
          </div>
        </div>
      ))}
      {items.length > visibleItems.length ? (
        <div className="text-[10px] text-gray-500">
          {items.length - visibleItems.length} more preview item(s)
        </div>
      ) : null}
    </div>
  );
};

export const ChatInspectorPanel: React.FC<ChatInspectorPanelProps> = ({
  conversationContext,
  contextUsage,
  latestCompaction,
  compactionHistory = [],
  isLoadingCompactions = false,
  restoringCompactionId = null,
  restoringBackupId = null,
  restoreNotice = null,
  activePins = [],
  usage,
  onUpdatePinnedContextStatus,
  onRestoreCompaction,
  onRestoreBackup,
  onClearRestoreNotice,
  onClose,
}) => {
  const [selectedCompactionId, setSelectedCompactionId] = React.useState<string | null>(
    latestCompaction?.compaction_id ?? compactionHistory[0]?.compaction.compaction_id ?? null,
  );
  const [historyOpen, setHistoryOpen] = React.useState(false);
  const [restoreTarget, setRestoreTarget] = React.useState<{
    compactionId: string;
    label: string;
    when: string;
  } | null>(null);
  const [expandedDiffSections, setExpandedDiffSections] = React.useState({
    removed: false,
    added: false,
    advanced: false,
  });
  const [trimmedDetailsOpen, setTrimmedDetailsOpen] = React.useState(false);
  const breakdown = Object.entries(contextUsage?.block_breakdown ?? {})
    .filter(([, tokens]) => Number(tokens) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  const dropped = Object.entries(contextUsage?.drop_reasons ?? {});
  const droppedBlockDetails = contextUsage?.dropped_block_details ?? [];
  const droppedDetailById = new Map(
    droppedBlockDetails.map((detail) => [detail.candidate_id, detail]),
  );
  const droppedReasonCounts = dropped.reduce<Record<string, number>>((acc, [, reason]) => {
    const key = String(reason || 'unspecified');
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
  const metadataFirstTokenMs = resolveTimingMetric(usage?.first_token_latency_ms, usage?.first_token_ms);
  const metadataGenerationTimeMs = resolveTimingMetric(usage?.generation_time_ms, null);
  const metadataDecodeTimeMs = resolveTimingMetric(usage?.decode_time_ms, null);
  const metadataDecodeTps = resolveTimingMetric(usage?.decode_tokens_per_second, null);
  const metadataEndToEndTps = resolveTimingMetric(usage?.end_to_end_tokens_per_second, usage?.tokens_per_second);
  const hasUsageAccounting = [
    usage?.input_tokens,
    usage?.prompt_tokens,
    usage?.completion_tokens,
    usage?.total_tokens,
    usage?.reasoning_tokens,
    usage?.answer_tokens,
    usage?.cached_prompt_tokens,
    usage?.cache_hit_reported,
    usage?.cache_hit,
    usage?.usage_source,
    usage?.usage_confidence,
    metadataFirstTokenMs,
    metadataGenerationTimeMs,
    metadataDecodeTimeMs,
    metadataDecodeTps,
    metadataEndToEndTps,
  ].some((value) => value !== undefined && value !== null && value !== '');
  const pinViolationCount = Number(latestCompaction?.metrics?.pin_violation_count || 0);
  const tokensBefore = Number(latestCompaction?.metrics?.tokens_before || 0);
  const tokensAfter = Number(latestCompaction?.metrics?.tokens_after || 0);
  const tokensSaved = Math.max(0, tokensBefore - tokensAfter);
  const conversationState = resolveConversationStateMeta(conversationContext);
  const compactionTriggerLabel = latestCompaction
    ? (compactionTriggerLabels[latestCompaction.trigger] ?? 'Compacted conversation context')
    : 'Compacted conversation context';
  React.useEffect(() => {
    if (
      selectedCompactionId
      && compactionHistory.some((item) => item.compaction.compaction_id === selectedCompactionId)
    ) {
      return;
    }
    setSelectedCompactionId(
      latestCompaction?.compaction_id ?? compactionHistory[0]?.compaction.compaction_id ?? null,
    );
  }, [compactionHistory, latestCompaction?.compaction_id, selectedCompactionId]);
  React.useEffect(() => {
    setExpandedDiffSections({ removed: false, added: false, advanced: false });
  }, [selectedCompactionId]);
  React.useEffect(() => {
    setRestoreTarget(null);
  }, [selectedCompactionId]);
  React.useEffect(() => {
    setTrimmedDetailsOpen(false);
  }, [contextUsage?.timestamp]);
  const selectedCompaction = compactionHistory.find(
    (item) => item.compaction.compaction_id === selectedCompactionId,
  );
  const removedPreviewCount = selectedCompaction?.diff.source_message_previews?.length ?? 0;
  const addedPreviewCount = selectedCompaction?.diff.added_message_previews?.length ?? 0;
  const selectedCompactionRecord = selectedCompaction?.compaction ?? null;
  const selectedCompactionLabel = selectedCompactionRecord
    ? (compactionTriggerLabels[selectedCompactionRecord.trigger] ?? 'Compacted conversation context')
    : null;

  return (
    <div className="fixed inset-y-0 right-0 z-40 w-[420px] max-w-[92vw] border-l border-gray-700 bg-gray-900 shadow-2xl" data-testid="chat-inspector-panel">
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
        <div className="min-w-0">
          <div className="text-[12px] text-gray-300">Chat Inspector</div>
          <div className="text-[10px] text-gray-500">Conversation and latest request context</div>
        </div>
        <button type="button" onClick={onClose} className="rounded p-1 text-gray-500 hover:bg-gray-800 hover:text-gray-300" aria-label="Close chat inspector">
          <X size={14} />
        </button>
      </div>
      <div className="h-[calc(100%-56px)] overflow-auto p-3">
        <div className="space-y-3">
          <Section title="Conversation Context">
            {conversationContext ? (
              <>
                <Row
                  label="Health"
                  value={<span className={`inline-flex rounded border px-1.5 py-0.5 text-[10px] ${conversationState.toneClassName}`}>{conversationState.label}</span>}
                />
                <Row
                  label="Usage"
                  value={`${formatInt(conversationContext.used_units)} / ${formatInt(conversationContext.max_units)}`}
                />
                {conversationContext.last_compacted_at ? (
                  <Row label="Last compacted" value={new Date(conversationContext.last_compacted_at * 1000).toLocaleString()} />
                ) : null}
                {conversationContext.last_compaction_delta ? (
                  <Row label="Recovered" value={`${formatInt(conversationContext.last_compaction_delta)} units`} />
                ) : null}
              </>
            ) : (
              <div className="text-[11px] text-gray-500">No conversation context state available.</div>
            )}
          </Section>

          <Section title="Request Context">
            {contextUsage ? (
              <>
                <Row
                  label="Used"
                  value={`${formatInt(contextUsage.used_tokens)} / ${formatInt(contextUsage.max_context_tokens)}`}
                />
                <Row label="Reserved output" value={formatInt(contextUsage.reserved_output_tokens)} />
                <Row label="Available input" value={formatInt(contextUsage.available_input_tokens ?? contextUsage.available_tokens)} />
                {breakdown.length > 0 ? breakdown.map(([name, tokens]) => {
                  const meta = breakdownMeta[name] ?? { label: name, description: 'Context block' };
                  return (
                    <Row
                      key={name}
                      label={meta.label}
                      value={`${formatInt(tokens)} tokens`}
                    />
                  );
                }) : <div className="text-[11px] text-gray-500">No block breakdown reported.</div>}
                {(dropped.length > 0 || droppedBlockDetails.length > 0) ? (
                  <div className="border-t border-gray-800 pt-2">
                    <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">Context trimmed</div>
                    <div className="space-y-2">
                      <Row
                        label="Trimmed items"
                        value={formatInt(Math.max(dropped.length, droppedBlockDetails.length))}
                        description="Some lower-priority context was omitted from this request to stay within the model budget. The conversation itself was not deleted."
                      />
                      {Object.entries(droppedReasonCounts)
                        .sort((a, b) => b[1] - a[1])
                        .map(([reason, count]) => (
                          <Row
                            key={reason}
                            label={dropReasonLabels[reason] ?? reason}
                            value={formatInt(count)}
                            description={dropReasonDescriptions[reason]}
                          />
                        ))}
                      <div className="pt-1">
                        <button
                          type="button"
                          onClick={() => setTrimmedDetailsOpen((value) => !value)}
                          className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800"
                        >
                          {trimmedDetailsOpen ? 'Hide trimmed details' : 'Show trimmed details'}
                        </button>
                      </div>
                      {trimmedDetailsOpen ? (
                        <div className="space-y-2 rounded border border-gray-800 bg-gray-950/50 p-2">
                          {(dropped.length > 0
                            ? dropped.map(([name, reason], index) => {
                                const detail = droppedDetailById.get(name);
                                const label = detail
                                  ? getTrimmedDetailLabel(detail.block_type, detail.source)
                                  : getTrimmedItemLabel(name);
                                const reasonLabel = dropReasonLabels[String(reason)] ?? String(reason);
                                const reasonDescription = dropReasonDescriptions[String(reason)]
                                  ?? 'Planner reported this trim reason for omitted request context.';
                                const detailSummary = detail ? formatTrimmedDetailSummary(detail) : null;
                                const showReference = !detail && !breakdownMeta[name];
                                return (
                                  <Row
                                    key={`${name}-${reason}-${index}`}
                                    label={label}
                                    value={reasonLabel}
                                    description={showReference
                                      ? `${reasonDescription} Internal reference: ${name}`
                                      : detailSummary
                                        ? `${reasonDescription} ${detailSummary}`
                                        : reasonDescription}
                                  />
                                );
                              })
                            : droppedBlockDetails.map((detail, index) => {
                                const label = getTrimmedDetailLabel(detail.block_type, detail.source);
                                const detailSummary = formatTrimmedDetailSummary(detail);
                                return (
                                  <Row
                                    key={`${detail.candidate_id}-${index}`}
                                    label={label}
                                    value="Omitted from request context"
                                    description={detailSummary || 'Planner reported trimmed request context detail without a field-level reason code.'}
                                  />
                                );
                              }))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="text-[11px] text-gray-500">No request snapshot yet.</div>
            )}
          </Section>

          {hasUsageAccounting ? (
            <Section title="Token Accounting">
              <Row label="Input tokens" value={formatInt(usage?.input_tokens)} />
              <Row label="Prompt tokens" value={formatInt(usage?.prompt_tokens)} />
              <Row label="Completion tokens" value={formatInt(usage?.completion_tokens)} />
              <Row label="Total tokens" value={formatInt(usage?.total_tokens)} />
              <Row
                label="Reasoning tokens"
                value={renderReportedTokenValue(usage?.reasoning_tokens, usage?.reasoning_tokens_reported)}
                description={usage?.reasoning_tokens_reported ? 'Reported by provider usage payload.' : 'Provider did not report reasoning / thinking token counts for this turn.'}
              />
              <Row
                label="Answer tokens"
                value={renderReportedTokenValue(usage?.answer_tokens, usage?.answer_tokens_reported)}
                description={usage?.answer_tokens_reported ? 'Derived from completion minus reported reasoning tokens.' : 'Not shown as a separate reported field because reasoning token counts were not reported.'}
              />
              <Row
                label="Cached prompt tokens"
                value={renderReportedTokenValue(usage?.cached_prompt_tokens, usage?.cached_prompt_tokens_reported)}
                description={usage?.cached_prompt_tokens_reported ? 'Reported cached prompt reuse from provider usage payload.' : 'Provider did not report cached prompt token usage for this turn.'}
              />
              <Row
                label="Cache hit"
                value={renderCacheHitValue(usage)}
                description={usage?.cache_hit_reported ? 'Derived from provider-reported cache usage fields.' : 'Unknown because the provider did not report cache-hit usage fields.'}
              />
              <Row
                label="Usage source"
                value={usage?.usage_source ? String(usage.usage_source) : '—'}
              />
              <Row
                label="Usage confidence"
                value={formatUsageConfidenceLabel(usage?.usage_confidence)}
                description={usage?.usage_confidence === 'reported' ? 'Overall usage object came from the provider/adapter, but some subfields may still be omitted.' : undefined}
              />
              <Row label="First token" value={metadataFirstTokenMs != null ? `${metadataFirstTokenMs.toFixed(2)} ms` : '—'} description={metadataFirstTokenMs != null ? 'Time from request start until the first visible assistant token arrived.' : 'This provider or adapter did not report first-token latency for this turn.'} />
              <Row label="Generation time" value={metadataGenerationTimeMs != null ? `${metadataGenerationTimeMs.toFixed(2)} ms` : '—'} description={metadataGenerationTimeMs != null ? 'Total measured generation time for the assistant response.' : 'Generation time was not reported for this turn.'} />
              <Row label="Decode time" value={metadataDecodeTimeMs != null ? `${metadataDecodeTimeMs.toFixed(2)} ms` : '—'} description={metadataDecodeTimeMs != null ? 'Provider-reported decoding time, when available.' : 'Decode time was not reported for this turn.'} />
              <Row label="Decode throughput" value={metadataDecodeTps != null ? `${metadataDecodeTps.toFixed(2)} tok/s` : '—'} description={metadataDecodeTps != null ? 'Provider-reported decode throughput.' : 'Decode throughput was not reported for this turn.'} />
              <Row label="End-to-end throughput" value={metadataEndToEndTps != null ? `${metadataEndToEndTps.toFixed(2)} tok/s` : '—'} description={metadataEndToEndTps != null ? 'Observed overall throughput including non-decode overhead.' : 'End-to-end throughput was not reported for this turn.'} />
            </Section>
          ) : null}

          {latestCompaction ? (
            <Section title="Compaction">
              <Row
                label="Action"
                value={compactionTriggerLabel}
                description={compactionTriggerDescriptions[latestCompaction.trigger] ?? `System trigger: ${latestCompaction.trigger}`}
              />
              <Row label="When" value={formatCompactionTime(latestCompaction.created_at)} />
              <Row label="Summary" value={latestCompaction.summary} />
              <Row label="Before" value={formatInt(tokensBefore)} />
              <Row label="After" value={formatInt(tokensAfter)} />
              <Row label="Saved" value={formatInt(tokensSaved)} />
              <Row
                label="Protection"
                value={pinViolationCount > 0 ? `Pin violations ${formatInt(pinViolationCount)}` : 'Pins protected'}
              />
            </Section>
          ) : null}

          <Section title="Compaction history">
            {isLoadingCompactions ? (
              <div className="text-[11px] text-gray-500">Loading compaction history…</div>
            ) : compactionHistory.length > 0 ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2 rounded border border-gray-800 bg-gray-950/50 px-2 py-2">
                  <div className="text-[11px] text-gray-300">
                    {formatInt(compactionHistory.length)} snapshot{compactionHistory.length > 1 ? 's' : ''}
                  </div>
                  <button
                    type="button"
                    onClick={() => setHistoryOpen((value) => !value)}
                    className="inline-flex items-center gap-1 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800"
                  >
                    {historyOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    {historyOpen ? 'Hide history' : 'Show history'}
                  </button>
                </div>
                {historyOpen ? (
                  <div className="space-y-2" data-testid="compaction-history-list">
                    {compactionHistory.map((item) => {
                      const isSelected = item.compaction.compaction_id === selectedCompactionId;
                      const isLatest = item.compaction.compaction_id === latestCompaction?.compaction_id;
                      const savedTokens = Math.max(
                        0,
                        Number(item.compaction.metrics.tokens_before || 0) - Number(item.compaction.metrics.tokens_after || 0),
                      );
                      return (
                        <button
                          key={item.compaction.compaction_id}
                          type="button"
                          onClick={() => setSelectedCompactionId(item.compaction.compaction_id)}
                          className={`w-full rounded border px-2 py-2 text-left ${
                            isSelected
                              ? 'border-cyan-500/50 bg-cyan-500/10'
                              : 'border-gray-800 bg-gray-950/60 hover:bg-gray-900'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-[11px] text-gray-200">
                              {compactionTriggerLabels[item.compaction.trigger] ?? 'Compacted conversation context'}
                            </span>
                            {isLatest ? (
                              <span className="rounded border border-emerald-500/30 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-200">
                                Latest
                              </span>
                            ) : null}
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
                            <span>{formatCompactionTime(item.compaction.created_at)}</span>
                            <span>{`${formatInt(item.compaction.metrics.messages_compacted)} msgs`}</span>
                            {savedTokens > 0 ? <span>{`Saved ${formatInt(savedTokens)}`}</span> : null}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="text-[11px] text-gray-500">No compaction history yet.</div>
            )}
          </Section>

          {selectedCompaction && selectedCompactionRecord && selectedCompactionLabel ? (
            <Section title="Latest compaction details">
              <Row
                label="Action"
                value={selectedCompactionLabel}
                description={compactionTriggerDescriptions[selectedCompactionRecord.trigger] ?? `System trigger: ${selectedCompactionRecord.trigger}`}
              />
              <Row label="When" value={formatCompactionTime(selectedCompactionRecord.created_at)} />
              {selectedCompaction.backup ? (
                <>
                  <Row label="Snapshot" value={selectedCompaction.backup.backup_id} />
                </>
              ) : null}
              <Row
                label="Summary"
                value={selectedCompactionRecord.summary}
                description="LLM-generated replacement text for the compacted messages. Stored in conversation metadata (not a real message)."
              />
              <Row
                label="Compacted"
                value={`${formatInt(selectedCompactionRecord.metrics.messages_compacted)} message(s)`}
                description="Messages folded into the summary by this compaction."
              />
              <Row
                label="Added later"
                value={`${formatInt(selectedCompaction.diff.added_message_ids.length)} message(s)`}
                description="Messages that appeared after this snapshot was taken. Empty means the current conversation matches the snapshot timeline for this segment."
              />
              <div className="space-y-2">
                <div
                  className="rounded border border-gray-800 bg-gray-950/40 p-2"
                  data-testid="compaction-removed-preview"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-[11px] text-gray-200">Compacted preview</div>
                      <div className="text-[10px] text-gray-500">
                        {formatInt(selectedCompaction.diff.source_message_previews?.length ?? 0)} preview item(s)
                      </div>
                    </div>
                    <button
                      type="button"
                      disabled={removedPreviewCount === 0}
                      onClick={() => setExpandedDiffSections((current) => ({
                        ...current,
                        removed: !current.removed,
                      }))}
                      className="inline-flex items-center gap-1 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {expandedDiffSections.removed ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                      {expandedDiffSections.removed ? 'Hide details' : 'Show details'}
                    </button>
                  </div>
                  <div className="mt-2">
                    {expandedDiffSections.removed ? (
                      <PreviewList
                        items={selectedCompaction.diff.source_message_previews ?? []}
                        emptyLabel="No compacted message previews available."
                      />
                    ) : (
                      <CollapsedPreviewList
                        items={selectedCompaction.diff.source_message_previews ?? []}
                        emptyLabel="No compacted message previews available."
                      />
                    )}
                  </div>
                </div>
                <div
                  className="rounded border border-gray-800 bg-gray-950/40 p-2"
                  data-testid="compaction-added-preview"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-[11px] text-gray-200">Added later</div>
                      <div className="text-[10px] text-gray-500">
                        {formatInt(selectedCompaction.diff.added_message_previews?.length ?? 0)} preview item(s)
                      </div>
                    </div>
                    <button
                      type="button"
                      disabled={addedPreviewCount === 0}
                      onClick={() => setExpandedDiffSections((current) => ({
                        ...current,
                        added: !current.added,
                      }))}
                      className="inline-flex items-center gap-1 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {expandedDiffSections.added ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                      {expandedDiffSections.added ? 'Hide details' : 'Show details'}
                    </button>
                  </div>
                  <div className="mt-2">
                    {expandedDiffSections.added ? (
                      <PreviewList
                        items={selectedCompaction.diff.added_message_previews ?? []}
                        emptyLabel="No later-added message previews."
                      />
                    ) : (
                      <CollapsedPreviewList
                        items={selectedCompaction.diff.added_message_previews ?? []}
                        emptyLabel="No later-added message previews."
                      />
                    )}
                  </div>
                </div>
                <div className="rounded border border-gray-800 bg-gray-950/40 p-2" data-testid="compaction-advanced-diff">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-[11px] text-gray-200">Advanced diff</div>
                      <div className="text-[10px] text-gray-500">Snapshot metadata and raw message ids</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setExpandedDiffSections((current) => ({
                        ...current,
                        advanced: !current.advanced,
                      }))}
                      className="inline-flex items-center gap-1 rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800"
                    >
                      {expandedDiffSections.advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                      {expandedDiffSections.advanced ? 'Hide advanced' : 'Show advanced'}
                    </button>
                  </div>
                  {expandedDiffSections.advanced ? (
                    <div className="mt-2 space-y-2 rounded border border-gray-800 bg-gray-950/60 p-2">
                      {selectedCompaction.backup ? (
                        <>
                          <Row label="Snapshot time" value={formatCompactionTime(selectedCompaction.backup.created_at)} />
                          <Row label="Snapshot trigger" value={selectedCompaction.backup.trigger} description={backupTriggerDescriptions[selectedCompaction.backup.trigger] ?? 'Backend event that created this backup snapshot.'} />
                          <Row
                            label="Snapshot kind"
                            value={formatBackupMetadataValue(selectedCompaction.backup.metadata?.kind)}
                            description="High-level backup category recorded by the backend for this snapshot."
                          />
                          <Row
                            label="Snapshot reason"
                            value={formatBackupMetadataValue(selectedCompaction.backup.metadata?.reason)}
                            description="More specific reason the snapshot was created, when the backend recorded one."
                          />
                          <Row
                            label="Source backup"
                            value={formatBackupMetadataValue(selectedCompaction.backup.metadata?.source_backup_id)}
                            description="The earlier backup this restore-point snapshot was created from. Empty means this snapshot was not derived from another backup."
                          />
                          <Row
                            label="Restored compaction"
                            value={formatBackupMetadataValue(
                              selectedCompaction.backup.metadata?.restored_compaction_id,
                            )}
                            description="The compaction record this restore-point snapshot was created to restore. Empty means this backup was not created by a restore action."
                          />
                        </>
                      ) : null}
                      <Row
                        label="Messages"
                        value={`${formatInt(selectedCompaction.diff.current_message_count)} current / ${selectedCompaction.diff.backup_message_count !== null ? formatInt(selectedCompaction.diff.backup_message_count) : '—'} backup`}
                        description="Total message count now versus the message count stored inside the selected snapshot backup."
                      />
                      <Row
                        label="Visible"
                        value={`${formatInt(selectedCompaction.diff.current_visible_message_count)} current / ${selectedCompaction.diff.backup_visible_message_count !== null ? formatInt(selectedCompaction.diff.backup_visible_message_count) : '—'} backup`}
                        description="Visible non-system messages now versus the visible message count inside the selected snapshot backup."
                      />
                      <Row
                        label="Removed IDs"
                        value={selectedCompaction.diff.removed_message_ids.length > 0 ? selectedCompaction.diff.removed_message_ids.join(', ') : '—'}
                        description="Internal message ids removed from the selected snapshot source set."
                      />
                      <Row
                        label="Added later IDs"
                        value={selectedCompaction.diff.added_message_ids.length > 0 ? selectedCompaction.diff.added_message_ids.join(', ') : '—'}
                        description="Internal message ids added after the snapshot was taken."
                      />
                    </div>
                  ) : null}
                </div>
              </div>
              {selectedCompaction.backup ? (
                <div className="rounded border border-gray-800 bg-gray-950/40 p-2 text-[10px] text-gray-500">
                  Full snapshot content is preserved in backup `{selectedCompaction.backup.backup_id}` from {formatCompactionTime(selectedCompaction.backup.created_at)}. Use restore to inspect it in the conversation timeline.
                </div>
              ) : null}
              <button
                type="button"
                onClick={() => setRestoreTarget({
                  compactionId: selectedCompactionRecord.compaction_id,
                  label: selectedCompactionLabel,
                  when: formatCompactionTime(selectedCompactionRecord.created_at),
                })}
                disabled={restoringCompactionId === selectedCompactionRecord.compaction_id}
                className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-200 hover:bg-amber-500/20 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {restoringCompactionId === selectedCompactionRecord.compaction_id
                  ? 'Restoring snapshot…'
                  : 'Restore…'}
              </button>
            </Section>
          ) : null}

          {restoreNotice ? (
            <Section title="Restore status">
              <Row
                label="Latest action"
                value={restoreNotice.message}
                description="Restoring a compaction first creates a restore-point backup, then applies the selected snapshot. Undo restore re-applies that restore-point backup."
              />
              {restoreNotice.undoBackupId ? (
                <Row label="Restore-point backup" value={restoreNotice.undoBackupId} />
              ) : null}
              <div className="flex flex-wrap gap-2">
                {restoreNotice.undoBackupId ? (
                  <button
                    type="button"
                    onClick={() => onRestoreBackup?.(restoreNotice.undoBackupId as string)}
                    disabled={restoringBackupId === restoreNotice.undoBackupId}
                    className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] text-cyan-200 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {restoringBackupId === restoreNotice.undoBackupId ? 'Undoing restore…' : 'Undo restore'}
                  </button>
                ) : null}
                <button
                  type="button"
                  onClick={() => onClearRestoreNotice?.()}
                  className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800"
                >
                  Dismiss
                </button>
              </div>
            </Section>
          ) : null}

          {activePins.length > 0 ? (
            <Section title="Active pins">
              <div className="text-[10px] text-gray-500">
                Add or replace pins from a message&apos;s pin menu. Use this panel to review active pins and archive or remove them.
              </div>
              {activePins.map((pin) => (
                <div key={pin.pin_id} className="rounded border border-gray-800 bg-gray-950/60 p-2">
                  <Row label="Title" value={pin.title || 'Pinned context'} />
                  <Row label="Content" value={pin.content} />
                  <Row label="Role" value={pin.role} />
                  <Row label="Source" value={pin.source_message_id} />
                  <Row label="Tokens" value={formatInt(pin.token_count)} />
                  <Row label="Updated" value={formatCompactionTime(pin.updated_at)} />
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => onUpdatePinnedContextStatus?.(pin.pin_id, 'removed')}
                      className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-700"
                    >
                      Unpin
                    </button>
                    <button
                      type="button"
                      onClick={() => onUpdatePinnedContextStatus?.(pin.pin_id, 'archived')}
                      className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-700"
                    >
                      Archive
                    </button>
                  </div>
                </div>
              ))}
            </Section>
          ) : null}
        </div>
      </div>
      {restoreTarget ? (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/70 p-4" data-testid="restore-confirmation-dialog">
          <div className="w-full max-w-sm rounded-lg border border-gray-800 bg-gray-900 p-4 shadow-2xl">
            <div className="text-[12px] font-medium text-gray-100">Restore snapshot?</div>
            <div className="mt-2 space-y-2 text-[11px] text-gray-300">
              <p>This will replace the current conversation view with the selected snapshot.</p>
              <p className="text-gray-400">
                {restoreTarget.label} · {restoreTarget.when}
              </p>
              {selectedCompaction?.backup ? (
                <p className="text-gray-400">
                  Snapshot ID: {selectedCompaction.backup.backup_id}
                </p>
              ) : null}
              <p className="text-gray-500">
                A restore point of the current state will be created first so you can undo this action.
              </p>
            </div>
            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setRestoreTarget(null)}
                className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-gray-200 hover:bg-gray-800"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  onRestoreCompaction?.(restoreTarget.compactionId);
                  setRestoreTarget(null);
                }}
                className="rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-200 hover:bg-amber-500/20"
              >
                Restore snapshot
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
