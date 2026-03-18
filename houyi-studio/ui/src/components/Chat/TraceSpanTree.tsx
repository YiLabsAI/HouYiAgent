import React from 'react';
import { TraceSpan } from '@/types/chat';
import { TraceSection } from './TraceSection';
import { formatDuration } from './TraceDetailUtils';

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
  const attrEntries = Object.entries(span.attributes || {}).filter(([, value]) => value !== null && value !== undefined).slice(0, 8);

  return (
    <div className="space-y-1">
      <button
        type="button"
        className="flex w-full items-center justify-between rounded border border-gray-700 bg-gray-900/60 px-2 py-1.5 text-left hover:bg-gray-800/70"
        onClick={() => setExpanded((value) => !value)}
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[12px] text-gray-200">
            <span className="truncate font-medium">{span.name || 'span'}</span>
            {span.span_type ? <span className="text-[10px] text-gray-500">{span.span_type}</span> : null}
          </div>
          <div className="mt-0.5 text-[10px] text-gray-500">Duration {formatDuration(span.duration_ms)}</div>
        </div>
        <div className="ml-2 flex items-center gap-2">
          <StatusBadge status={span.status} />
          {hasChildren ? <span className="text-[10px] text-gray-500">{expanded ? 'Hide' : 'Show'} {children.length}</span> : null}
        </div>
      </button>
      {expanded && attrEntries.length > 0 ? (
        <div className="rounded border border-gray-800 bg-gray-950/70 px-2 py-1.5 text-[10px] text-gray-400">
          {attrEntries.map(([key, value]) => (
            <div key={key} className="truncate">
              <span className="text-gray-500">{key}:</span> {String(value)}
            </div>
          ))}
        </div>
      ) : null}
      {expanded && hasChildren ? (
        <div className="ml-3 space-y-2 border-l border-gray-800 pl-2">
          {children.map((child, index) => (
            <SpanTreeNode key={`${child.name || 'span'}-${index}`} span={child} depth={depth + 1} />
          ))}
        </div>
      ) : null}
    </div>
  );
};

interface TraceSpanTreeProps {
  rootSpan: TraceSpan;
}

export const TraceSpanTree: React.FC<TraceSpanTreeProps> = ({ rootSpan }) => (
  <TraceSection title="Span tree / node attrs">
    <SpanTreeNode span={rootSpan} />
  </TraceSection>
);
