/**
 * Timeline Waterfall Component
 *
 * Displays hierarchical span tree as a waterfall visualization:
 * - execution (root span)
 *   - node spans
 *     - llm/tool sub-spans
 *
 * - Degradation: collapse llm/tool spans when total count exceeds threshold
 * - On-demand expansion: click to expand collapsed sub-spans
 */
import { useMemo, useState, useCallback, useEffect, useRef } from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import type { SpanData, SpanTree } from '@/stores/storeActions/spanActions';
import { ExecutionLineageTree } from './ExecutionLineageTree';

// Degradation thresholds
const DEGRADATION_THRESHOLD = 200; // Collapse sub-spans when total exceeds this
const WARN_THRESHOLD = 500; // Show warning when approaching capacity

interface SpanRowProps {
  span: SpanData;
  depth: number;
  startTime: number;
  totalDuration: number;
  collapsed?: boolean;
  childCount?: number;
  onToggle?: () => void;
  onSelect?: () => void;
  selected?: boolean;
  highlight?: boolean;
}

const SPAN_TYPE_COLORS: Record<string, string> = {
  execution: 'bg-blue-500',
  node: 'bg-green-500',
  llm: 'bg-purple-500',
  tool: 'bg-orange-500',
  retriever: 'bg-cyan-500',
  retry: 'bg-amber-500',
  internal: 'bg-teal-500',
};

const SPAN_TYPE_LABELS: Record<string, string> = {
  execution: 'Execution',
  node: 'Node',
  llm: 'LLM',
  tool: 'Tool',
  retriever: 'Retriever',
  retry: 'Retry',
  internal: 'Internal',
};

const ROW_HEIGHT_PX = 28;
const OVERSCAN_ROWS = 20;

function formatDuration(ms: number): string {
  if (ms < 1) {
    return `${(ms * 1000).toFixed(0)}µs`;
  }
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
}

function SpanRow({
  span,
  depth,
  startTime,
  totalDuration,
  collapsed,
  childCount,
  onToggle,
  onSelect,
  selected,
  highlight,
}: SpanRowProps) {
  const offsetPercent = totalDuration > 0
    ? ((span.start_time - startTime) / totalDuration) * 100
    : 0;

  const durationMs = span.duration ? span.duration * 1000 : 0;
  const widthPercent = totalDuration > 0 && span.duration
    ? (span.duration / totalDuration) * 100
    : 1;

  const colorClass = SPAN_TYPE_COLORS[span.span_type] || 'bg-gray-500';
  const isError = span.status === 'error';

  const hasChildren = childCount !== undefined && childCount > 0;
  const isExpandable = hasChildren || collapsed;

  const labelTitleParts: string[] = [];
  labelTitleParts.push(`${span.name}`);
  labelTitleParts.push(`type=${span.span_type}`);
  if (span.node_id) labelTitleParts.push(`node=${span.node_id}`);
  if (span.model) labelTitleParts.push(`model=${span.model}`);
  if (span.tool_name) labelTitleParts.push(`tool=${span.tool_name}`);
  if (span.cache_hit) labelTitleParts.push('cached=true');
  if (span.tokens_input !== null) labelTitleParts.push(`tokens_in=${span.tokens_input}`);
  if (span.tokens_output !== null) labelTitleParts.push(`tokens_out=${span.tokens_output}`);
  if (span.cost_usd !== null) labelTitleParts.push(`cost_usd=${span.cost_usd}`);
  if (span.duration) labelTitleParts.push(`dur=${formatDuration(durationMs)}`);
  if (isError) labelTitleParts.push('status=error');
  const labelTitle = labelTitleParts.join(' · ');

  const rowClassName = selected
    ? 'bg-blue-50 dark:bg-blue-900/20'
    : highlight
      ? 'bg-gray-50 dark:bg-gray-800'
      : '';

  return (
    <div className={`flex items-center py-1 hover:bg-gray-50 dark:hover:bg-gray-800 min-w-0 ${rowClassName}`}>
      {/* Label column */}
      <div
        className={`flex-shrink-0 pr-1 text-sm truncate ${isExpandable ? 'cursor-pointer' : ''}`}
        style={{ width: `var(--span-col-width)`, paddingLeft: `${depth * 16}px` }}
        onClick={(e) => {
          if (!isExpandable) return;
          e.stopPropagation();
          onToggle?.();
        }}
        title={labelTitle}
      >
        {isExpandable && (
          <span className="inline-block w-4 text-gray-400 mr-1">
            {collapsed ? '▶' : '▼'}
          </span>
        )}
        <span className={`inline-block w-2 h-2 rounded-full mr-2 ${colorClass}`} />
        <span className="text-gray-700 dark:text-gray-300">
          {span.span_type === 'node' && span.attributes?.['node.label']
            ? span.attributes['node.label']
            : span.name}
          {span.span_type === 'tool' && span.tool_name && (
            <span className="ml-1 text-orange-400 text-[10px]">({span.tool_name})</span>
          )}
          {span.span_type === 'llm' && span.model && (
            <span className="ml-1 text-purple-400 text-[10px]">({span.model})</span>
          )}
          {span.span_type === 'node' && span.attributes?.['node.type'] && (
            <span className="ml-1 text-gray-400 text-[10px]">[{span.attributes['node.type']}]</span>
          )}
        </span>
        {collapsed && childCount !== undefined && childCount > 0 && (
          <span className="ml-1 text-xs text-gray-400">
            ({childCount} hidden)
          </span>
        )}
        {span.group_id && span.lane_id !== null && (
          <span className="ml-1 text-xs text-blue-600 dark:text-blue-400">
            [L{span.lane_id}]
          </span>
        )}
        {span.cache_hit && (
          <span className="ml-1 text-xs text-green-600 dark:text-green-400">
            (cached)
          </span>
        )}
        {isError && (
          <span className="ml-1 text-xs text-red-600 dark:text-red-400">
            (error)
          </span>
        )}
      </div>

      {/* Timeline bar column */}
      <div
        className="flex-1 min-w-0 relative h-6 cursor-pointer group/bar"
        onClick={(e) => {
          e.stopPropagation();
          onSelect?.();
        }}
      >
        <div
          className={`absolute h-4 rounded ${colorClass} ${isError ? 'opacity-60' : 'opacity-80'}`}
          style={{
            left: `${offsetPercent}%`,
            width: `${Math.max(widthPercent, 0.5)}%`,
            top: '4px',
          }}
        />
        {/* Hover tooltip — hidden/block is safe because row wrapper has contain:strict */}
        <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1 hidden group-hover/bar:block z-[9999] pointer-events-none">
          <div className="bg-gray-900 text-gray-100 text-xs rounded shadow-lg px-3 py-2 whitespace-nowrap border border-gray-700">
            <div className="font-medium mb-1">
              {span.attributes?.['node.label'] || span.name}
            </div>
            <div className="flex flex-col gap-0.5 text-gray-400">
              {span.span_type !== 'execution' && span.name && (
                <span>Type: {span.name}</span>
              )}
              {span.duration != null && <span>Duration: {formatDuration(durationMs)}</span>}
              {span.model && <span>Model: <span className="text-purple-400">{span.model}</span></span>}
              {span.tool_name && <span>Tool: <span className="text-orange-400">{span.tool_name}</span></span>}
              {span.tokens_input != null && <span>Tokens in: {span.tokens_input.toLocaleString()}</span>}
              {span.tokens_output != null && <span>Tokens out: {span.tokens_output.toLocaleString()}</span>}
              {span.cost_usd != null && <span>Cost: ${span.cost_usd.toFixed(4)}</span>}
              {span.cache_hit && <span className="text-green-400">Cache hit ✓</span>}
              {isError && <span className="text-red-400">Status: error</span>}
              {span.node_id && <span>Node: {span.attributes?.['node.label'] || span.node_id}</span>}
            </div>
          </div>
        </div>
      </div>

      {/* Duration column */}
      <div
        className="flex-shrink-0 w-20 text-right text-xs text-gray-500 dark:text-gray-400 cursor-pointer"
        onClick={(e) => {
          e.stopPropagation();
          onSelect?.();
        }}
      >
        {span.duration ? formatDuration(durationMs) : '...'}
      </div>

      {/* AI metrics column */}
      <div
        className="flex-shrink-0 w-28 text-right text-xs text-gray-500 dark:text-gray-400 cursor-pointer leading-tight pr-1"
        onClick={(e) => {
          e.stopPropagation();
          onSelect?.();
        }}
      >
        {span.span_type === 'llm' && (
          <div className="flex flex-col items-end gap-0">
            {span.model && (
              <span className="text-purple-600 dark:text-purple-400 truncate max-w-full">
                {span.model.split('/').pop()}
              </span>
            )}
            {(span.tokens_input != null || span.tokens_output != null) && (
              <span className="text-[10px]">
                {span.tokens_input?.toLocaleString() ?? '?'}→{span.tokens_output?.toLocaleString() ?? '?'} tok
              </span>
            )}
            {span.cache_hit && (
              <span className="text-green-500 text-[10px]">cached</span>
            )}
          </div>
        )}
        {span.span_type === 'tool' && (
          <div className="flex flex-col items-end gap-0">
            {span.tool_name && (
              <span className="text-orange-600 dark:text-orange-400 truncate max-w-full">
                {span.tool_name}
              </span>
            )}
            {span.cache_hit && (
              <span className="text-green-500 text-[10px]">cached</span>
            )}
          </div>
        )}
        {span.span_type === 'node' && (
          <div className="flex flex-col items-end gap-0">
            {span.duration != null && (
              <span className="text-gray-400">{formatDuration(durationMs)}</span>
            )}
            <span className={`text-[10px] ${span.status === 'error' ? 'text-red-400' : 'text-green-400'}`}>
              {span.status === 'error' ? 'failed' : span.status === 'ok' ? 'ok' : 'running'}
            </span>
          </div>
        )}
        {span.span_type === 'execution' && (
          <div className="flex flex-col items-end gap-0">
            {span.duration != null && (
              <span className="text-blue-400">{formatDuration(durationMs)}</span>
            )}
            <span className={`text-[10px] ${span.status === 'error' ? 'text-red-400' : span.status === 'ok' ? 'text-green-400' : 'text-yellow-400'}`}>
              {span.status === 'error' ? 'failed' : span.status === 'ok' ? 'completed' : 'running'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Count total descendants of a span
 */
function countDescendants(span: SpanData): number {
  let count = span.children.length;
  for (const child of span.children) {
    count += countDescendants(child);
  }
  return count;
}

interface RenderContext {
  startTime: number;
  totalDuration: number;
  expandedNodes: Set<string>;
  degraded: boolean;
  onToggle: (spanId: string) => void;
  allowSpan: (span: SpanData) => boolean;
  forceVisibleSpanIds: Set<string>;
}

interface FlatSpanRow {
  span: SpanData;
  depth: number;
  collapsed: boolean;
  childCount?: number;
}

function buildSpanIndex(spans: SpanData[]) {
  const byId = new Map<string, SpanData>();
  const parentById = new Map<string, string | null>();
  for (const s of spans) {
    byId.set(s.span_id, s);
    parentById.set(s.span_id, s.parent_span_id ?? null);
  }
  return { byId, parentById };
}

function collectAncestors(spanId: string, parentById: Map<string, string | null>): Set<string> {
  const ancestors = new Set<string>();
  let cur: string | null | undefined = spanId;
  while (cur) {
    const p = parentById.get(cur);
    if (!p) break;
    ancestors.add(p);
    cur = p;
  }
  return ancestors;
}

function collectDescendants(span: SpanData): Set<string> {
  const out = new Set<string>();
  const stack: SpanData[] = [...span.children];
  while (stack.length) {
    const cur = stack.pop()!;
    out.add(cur.span_id);
    for (const child of cur.children) stack.push(child);
  }
  return out;
}

function hasDescendantIn(span: SpanData, ids: Set<string>): boolean {
  for (const child of span.children) {
    if (ids.has(child.span_id) || hasDescendantIn(child, ids)) return true;
  }
  return false;
}

function flattenSpanTree(span: SpanData, depth: number, ctx: RenderContext): FlatSpanRow[] {
  const rows: FlatSpanRow[] = [];
  const { expandedNodes, degraded, allowSpan, forceVisibleSpanIds } = ctx;

  const hasChildren = span.children.length > 0;

  // A span is visible if it passes the filter OR is an ancestor of a matching span
  const isForceVisible = forceVisibleSpanIds.has(span.span_id);
  const passesFilter = allowSpan(span);
  const shouldRenderThis = passesFilter || isForceVisible;

  // Collapse logic:
  // In degraded mode: node-level spans start collapsed (opt-in expand via toggle)
  // In normal mode: all spans start expanded (opt-in collapse via toggle)
  // User explicit toggles (expandedNodes) ALWAYS take priority over force-visible logic.
  const isNodeOrHigher = span.span_type === 'execution' || span.span_type === 'node';
  const isToggled = expandedNodes.has(span.span_id);
  const defaultCollapsed = degraded && isNodeOrHigher && hasChildren;
  const hasForceVisibleDescendant = hasChildren && hasDescendantIn(span, forceVisibleSpanIds);
  // If user explicitly toggled this span, respect that; otherwise use filter-based default
  const collapsed = hasChildren && (isToggled
    ? (defaultCollapsed ? false : true)  // user toggled: flip the default
    : (defaultCollapsed ? true : (hasForceVisibleDescendant ? false : false))  // not toggled: default expanded, unless degraded
  );
  const childCount = hasChildren ? countDescendants(span) : undefined;

  if (shouldRenderThis) {
    rows.push({ span, depth, collapsed, childCount });
  }

  // Recurse into children when not collapsed
  if (!collapsed) {
    for (const child of span.children) {
      rows.push(
        ...flattenSpanTree(child, depth + 1, ctx)
      );
    }
  }

  return rows;
}

interface TimelineWaterfallProps {
  executionId?: string;
}

export function TimelineWaterfall({
  executionId,
}: TimelineWaterfallProps) {
  const currentExecution = useConsoleStore((state) => state.currentExecution);
  const getSpanTree = useConsoleStore((state) => state.getSpanTree);
  const spanStore = useConsoleStore((state) => state.spanStore);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);
  const [filterSpanType, setFilterSpanType] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<'all' | 'ok' | 'error'>('all');
  const [filterNodeId] = useState<string>('');
  const [searchText, setSearchText] = useState<string>('');
  const [selectedExecId, setSelectedExecId] = useState<string | null>(null);
  const [spanColPercent, setSpanColPercent] = useState(25); // Span name column width as % of container
  const containerRef = useRef<HTMLDivElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(400);

  // Available execution IDs from span store (most recent first)
  const availableExecIds = useMemo(() => {
    return Object.keys(spanStore).sort().reverse();
  }, [spanStore]);

  const effectiveExecutionId = selectedExecId || executionId || currentExecution?.execution_id;

  const spanTree = useMemo<SpanTree | null>(() => {
    if (!effectiveExecutionId) return null;
    return getSpanTree(effectiveExecutionId);
  }, [effectiveExecutionId, getSpanTree, spanStore]);

  const handleToggle = useCallback((spanId: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(spanId)) {
        next.delete(spanId);
      } else {
        next.add(spanId);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = () => {
      setViewportHeight(el.clientHeight || 400);
    };
    update();
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('resize', update);
    };
  }, []);

  // Drag handler for resizing the Span Name column
  const handleColDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    const startX = e.clientX;
    const startPercent = spanColPercent;
    const containerWidth = container.getBoundingClientRect().width;

    const onMove = (ev: MouseEvent) => {
      const delta = ev.clientX - startX;
      const deltaPercent = (delta / containerWidth) * 100;
      setSpanColPercent(Math.max(10, Math.min(50, startPercent + deltaPercent)));
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.classList.remove('select-none');
    };
    document.body.style.cursor = 'col-resize';
    document.body.classList.add('select-none');
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [spanColPercent]);

  useEffect(() => {
    setExpandedNodes(new Set());
    setSelectedSpanId(null);
  }, [effectiveExecutionId]);

  // Reset user toggles when filter changes so auto-expansion for new filter takes effect
  useEffect(() => {
    setExpandedNodes(new Set());
  }, [filterSpanType, filterStatus]);

  const root = spanTree?.root ?? null;
  const spans = spanTree?.spans ?? [];
  const totalDuration = spanTree?.totalDuration ?? 0;
  const startTime = spanTree?.startTime ?? 0;
  const totalSpans = spans.length;
  const degraded = totalSpans > DEGRADATION_THRESHOLD;
  const nearCapacity = totalSpans > WARN_THRESHOLD;

  // Only show span types that exist in current data
  const presentSpanTypes = useMemo(() => {
    const types = new Set<string>();
    for (const s of spans) types.add(s.span_type);
    return types;
  }, [spans]);

  const { parentById, byId } = useMemo(() => buildSpanIndex(spans), [spans]);

  const normalizedNodeId = filterNodeId.trim().toLowerCase();
  const normalizedSearch = searchText.trim().toLowerCase();

  const allowSpan = useCallback((span: SpanData) => {
    if (filterSpanType !== 'all' && span.span_type !== filterSpanType) return false;
    if (filterStatus !== 'all' && span.status !== filterStatus) return false;
    if (normalizedNodeId && (span.node_id || '').toLowerCase() !== normalizedNodeId) return false;
    if (!normalizedSearch) return true;
    const haystack = `${span.name} ${span.model ?? ''} ${span.tool_name ?? ''} ${span.span_type} ${span.node_id ?? ''}`.toLowerCase();
    return haystack.includes(normalizedSearch);
  }, [filterSpanType, filterStatus, normalizedNodeId, normalizedSearch]);

  const forceVisibleSpanIds = useMemo(() => {
    const forced = new Set<string>();
    const hasActiveFilter = filterSpanType !== 'all' || filterStatus !== 'all' || normalizedNodeId || normalizedSearch;
    if (hasActiveFilter) {
      for (const s of spans) {
        if (allowSpan(s)) {
          forced.add(s.span_id);
          // Add ancestors so the tree path to this span is visible
          const ancestors = collectAncestors(s.span_id, parentById);
          for (const a of ancestors) forced.add(a);
          // Add descendants so expanding a matched span reveals its subtree
          const spanObj = byId.get(s.span_id);
          if (spanObj) {
            const descendants = collectDescendants(spanObj);
            for (const d of descendants) forced.add(d);
          }
        }
      }
    }
    if (selectedSpanId) {
      forced.add(selectedSpanId);
      const ancestors = collectAncestors(selectedSpanId, parentById);
      for (const a of ancestors) forced.add(a);
    }
    return forced;
  }, [allowSpan, byId, filterSpanType, filterStatus, normalizedNodeId, normalizedSearch, parentById, selectedSpanId, spans]);

  const selectedAncestors = useMemo(() => {
    if (!selectedSpanId) return new Set<string>();
    return collectAncestors(selectedSpanId, parentById);
  }, [parentById, selectedSpanId]);

  const selectedDescendants = useMemo(() => {
    if (!selectedSpanId) return new Set<string>();
    const s = byId.get(selectedSpanId);
    if (!s) return new Set<string>();
    return collectDescendants(s);
  }, [byId, selectedSpanId]);

  const flatRows = useMemo(() => {
    if (!root) return [] as FlatSpanRow[];
    return flattenSpanTree(root, 0, {
      startTime,
      totalDuration,
      expandedNodes,
      degraded,
      onToggle: handleToggle,
      allowSpan,
      forceVisibleSpanIds,
    });
  }, [allowSpan, degraded, expandedNodes, forceVisibleSpanIds, handleToggle, root, startTime, totalDuration]);

  const totalRows = flatRows.length;
  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT_PX) - OVERSCAN_ROWS);
  const visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT_PX) + OVERSCAN_ROWS * 2;
  const endIndex = Math.min(totalRows, startIndex + visibleCount);
  const topSpacerHeight = startIndex * ROW_HEIGHT_PX;
  const bottomSpacerHeight = Math.max(0, (totalRows - endIndex) * ROW_HEIGHT_PX);

  if (!spanTree || spans.length === 0) {
    return (
      <div className="p-4 text-center text-gray-500 dark:text-gray-400">
        <p className="text-sm">No span data available</p>
        <p className="text-xs mt-1">
          Spans will appear here during execution
        </p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex flex-col gap-1 h-full min-h-0">
      {/* Execution lineage tree (own row, above filters) */}
      <ExecutionLineageTree
        executionIds={availableExecIds}
        activeExecutionId={effectiveExecutionId || null}
        getSpanTree={getSpanTree}
        onSelect={(eid) => setSelectedExecId(eid)}
      />
      {/* Filter toolbar row */}
      <div className="flex items-center gap-2 text-xs">
        <input
          type="text"
          className="flex-1 min-w-[120px] max-w-[200px] px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          placeholder="Search spans..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
        />
        <select
          className="px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200"
          value={filterSpanType}
          onChange={(e) => setFilterSpanType(e.target.value)}
        >
          <option value="all">All types</option>
          {Object.keys(SPAN_TYPE_LABELS).filter((t) => presentSpanTypes.has(t)).map((t) => (
            <option key={t} value={t}>{SPAN_TYPE_LABELS[t]}</option>
          ))}
        </select>
        <select
          className="px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200"
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value as typeof filterStatus)}
        >
          <option value="all">All status</option>
          <option value="ok">OK</option>
          <option value="error">Error</option>
        </select>
        {selectedSpanId && (
          <button
            type="button"
            className="px-1.5 py-0.5 text-[10px] bg-gray-600 border border-gray-500 rounded text-gray-300 hover:text-white"
            onClick={() => setSelectedSpanId(null)}
          >
            ✕ clear
          </button>
        )}
        <span className="ml-auto text-gray-500 text-[10px] whitespace-nowrap">{totalRows} rows · {totalSpans} spans</span>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden timeline-scroll-container"
        style={{ '--span-col-width': `${spanColPercent}%` } as React.CSSProperties}
        onScroll={(e) => {
          const target = e.currentTarget;
          setScrollTop(target.scrollTop);
        }}
      >
      {/* Header */}
      <div className="flex items-center py-1.5 border-b border-gray-200 dark:border-gray-700 text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider min-w-0">
        <div className="flex-shrink-0 pr-1" style={{ width: `${spanColPercent}%` }}>Span</div>
        <div
          className="flex-shrink-0 w-1 cursor-col-resize hover:bg-blue-500/50 bg-gray-700/30 self-stretch"
          onMouseDown={handleColDragStart}
          title="Drag to resize"
        />
        <div className="flex-1 min-w-0 pl-1">Timeline</div>
        <div className="flex-shrink-0 w-20 text-right">Duration</div>
        <div className="flex-shrink-0 w-28 text-right pr-1">Metrics</div>
      </div>

      {/* Span rows — no divide-y here; spacer divs would get borders causing jitter */}
      <div className="relative">
        <div style={{ height: topSpacerHeight }} />
        {flatRows.slice(startIndex, endIndex).map((row) => {
          const { span, depth, collapsed, childCount } = row;
          const isSelected = selectedSpanId === span.span_id;
          const isAncestor = selectedAncestors.has(span.span_id);
          const isDescendant = selectedDescendants.has(span.span_id);
          const highlight = Boolean(selectedSpanId && (isAncestor || isDescendant));
          const onToggle = () => {
            handleToggle(span.span_id);
          };
          const onSelect = () => {
            setSelectedSpanId((prev) => (prev === span.span_id ? null : span.span_id));
          };
          return (
            <div
              key={span.span_id}
              className="border-b border-gray-100 dark:border-gray-800/50"
              style={{ height: ROW_HEIGHT_PX, willChange: 'transform' }}
            >
              <SpanRow
                span={span}
                depth={depth}
                startTime={startTime}
                totalDuration={totalDuration}
                collapsed={collapsed}
                childCount={childCount}
                onToggle={onToggle}
                onSelect={onSelect}
                selected={isSelected}
                highlight={highlight}
              />
            </div>
          );
        })}
        <div style={{ height: bottomSpacerHeight }} />
        {/* Bottom padding to prevent last row clipping */}
        <div className="h-2" />
      </div>
      </div>

      {/* Degradation notice — outside scroll container so always visible */}
      {degraded && (
        <div className="shrink-0 mt-1 px-2 py-1 text-xs bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded">
          ⚠️ Large trace ({totalSpans} spans) - sub-spans collapsed for performance. Click ▶ to expand.
        </div>
      )}
      {nearCapacity && !degraded && (
        <div className="shrink-0 mt-1 px-2 py-1 text-xs bg-yellow-50 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 rounded">
          ⚠️ Approaching capacity ({totalSpans} spans)
        </div>
      )}

      {/* Legend — outside scroll container so always visible */}
      <div className="shrink-0 mt-1 pt-1 border-t border-gray-200 dark:border-gray-700">
        <div className="flex flex-wrap gap-3 text-[10px] text-gray-500 dark:text-gray-400">
          {Object.entries(SPAN_TYPE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center">
              <span className={`inline-block w-2 h-2 rounded ${color} mr-1`} />
              <span>{SPAN_TYPE_LABELS[type]}</span>
            </div>
          ))}
          {spanTree.root?.replay_mode && (
            <span className="text-amber-500">⟳ replay</span>
          )}
          {spanTree.root?.parent_trace_id && (
            <span className="text-gray-400">fork: {spanTree.root.parent_trace_id.slice(0, 8)}...</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default TimelineWaterfall;
