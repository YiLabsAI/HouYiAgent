import React from 'react';
import type { CompactionRecord, ContextUsage } from '@/types/chat';
import { formatInt } from './TraceDetailUtils';

interface LastRequestPlanDetailsProps {
  contextUsage: ContextUsage;
  latestCompaction?: CompactionRecord | null;
}

const breakdownMeta: Record<string, { label: string; description: string }> = {
  system: { label: 'System', description: 'System instructions' },
  pinned: { label: 'Pinned', description: 'Protected context' },
  current_turn: { label: 'Current', description: 'Current user turn' },
  recent: { label: 'Recent', description: 'Recent dialogue' },
  memory: { label: 'Memory', description: 'Recalled memory' },
  summary: { label: 'Summary', description: 'Compressed history' },
  tool_summary: { label: 'Tool summary', description: 'Compressed tool results' },
};

const dropReasonMeta: Record<string, { label: string; description: string }> = {
  truncated_to_fit: {
    label: 'Trimmed to fit request budget',
    description: 'The planner removed lower-priority context to stay inside the model budget.',
  },
  boundary_excluded: {
    label: 'Excluded by planning boundary',
    description: 'This block was outside the active planning boundary for the current request.',
  },
  lower_priority: {
    label: 'Excluded as lower priority context',
    description: 'Higher-priority blocks consumed the available budget first.',
  },
  budget_exceeded: {
    label: 'Trimmed to fit request budget',
    description: 'The request hit budget limits before this block could be included.',
  },
  policy_excluded: {
    label: 'Excluded by request policy',
    description: 'This block type was disabled by the current request policy or planner selection settings.',
  },
  excluded_without_current_turn: {
    label: 'Excluded because the current turn could not be kept',
    description: 'This older context was omitted because the planner could not keep the current turn within the available budget.',
  },
};

const Row: React.FC<{ label: string; value: React.ReactNode; description?: React.ReactNode }> = ({ label, value, description }) => (
  <div className="grid grid-cols-[minmax(0,140px)_minmax(0,1fr)] gap-x-3 gap-y-1 text-[11px]">
    <div className="text-gray-400">{label}</div>
    <div className="text-gray-200">
      <div>{value}</div>
      {description ? <div className="text-[10px] text-gray-500">{description}</div> : null}
    </div>
  </div>
);

export const LastRequestPlanDetails: React.FC<LastRequestPlanDetailsProps> = ({ contextUsage, latestCompaction }) => {
  const breakdown = Object.entries(contextUsage.block_breakdown ?? {})
    .filter(([, tokens]) => Number(tokens) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  const dropped = Object.entries(contextUsage.drop_reasons ?? {});
  const pinViolationCount = Number(latestCompaction?.metrics?.pin_violation_count || 0);
  const tokensBefore = Number(latestCompaction?.metrics?.tokens_before || 0);
  const tokensAfter = Number(latestCompaction?.metrics?.tokens_after || 0);
  const tokensSaved = Math.max(0, tokensBefore - tokensAfter);

  return (
    <div
      className="absolute right-0 top-full z-20 mt-2 w-[min(560px,calc(100vw-3rem))] rounded-lg border border-gray-700 bg-gray-950/95 shadow-2xl backdrop-blur"
      data-testid="last-request-plan-details"
    >
      <div className="max-h-[420px] overflow-auto p-3">
        <div className="mb-3 text-[12px] font-medium text-gray-100">Last Request Plan details</div>
        <div className="space-y-3">
          <section className="rounded border border-gray-800 bg-gray-900/60 p-3">
            <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">Overview</div>
            <div className="space-y-2">
              <Row
                label="Used"
                value={`${formatInt(contextUsage.used_tokens)} / ${formatInt(contextUsage.max_context_tokens)}`}
              />
              <Row label="Reserved output" value={formatInt(contextUsage.reserved_output_tokens)} />
              <Row label="Available input" value={formatInt(contextUsage.available_input_tokens ?? contextUsage.available_tokens)} />
            </div>
          </section>

          <section className="rounded border border-gray-800 bg-gray-900/60 p-3">
            <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">Breakdown</div>
            <div className="space-y-2">
              {breakdown.length > 0 ? breakdown.map(([name, tokens]) => {
                const meta = breakdownMeta[name] ?? { label: name, description: 'Context block' };
                return (
                  <Row
                    key={name}
                    label={meta.label}
                    value={`${formatInt(tokens)} tokens`}
                    description={meta.description}
                  />
                );
              }) : <div className="text-[11px] text-gray-500">No block breakdown reported</div>}
            </div>
          </section>

          {dropped.length > 0 && (
            <section className="rounded border border-gray-800 bg-gray-900/60 p-3">
              <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">Context trimmed</div>
              <div className="space-y-2">
                {dropped.map(([name, reason]) => {
                  const meta = dropReasonMeta[String(reason)] ?? {
                    label: String(reason),
                    description: 'Planner reported this trim reason for the omitted context block.',
                  };
                  return (
                    <Row
                      key={`${name}-${reason}`}
                      label={name}
                      value={meta.label}
                      description={meta.description}
                    />
                  );
                })}
              </div>
            </section>
          )}

          {latestCompaction && (
            <section className="rounded border border-gray-800 bg-gray-900/60 p-3">
              <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">Compaction</div>
              <div className="space-y-2">
                <Row label="Before" value={formatInt(tokensBefore)} />
                <Row label="After" value={formatInt(tokensAfter)} />
                <Row label="Saved" value={formatInt(tokensSaved)} />
              </div>
            </section>
          )}

          {latestCompaction && (
            <section className="rounded border border-gray-800 bg-gray-900/60 p-3">
              <div className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-400">Protection</div>
              <div className="space-y-2">
                <Row
                  label="Pins"
                  value={pinViolationCount > 0 ? `Pin violations ${formatInt(pinViolationCount)}` : 'Pins protected'}
                />
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
};
