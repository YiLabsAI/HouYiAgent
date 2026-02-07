/**
 * ExecutionLineageTree — Tree-based execution selector for Timeline.
 *
 * Design rationale (industry survey):
 * ─────────────────────────────────────────────────────────────────
 * - Jaeger/Tempo: flat trace list, no fork/branch visualization.
 * - LangSmith: RunTree with parent→child nesting, but single-run scope.
 * - Our scenario is unique: checkpoint/restore creates a **DAG of executions**.
 *   Multiple restores from the same checkpoint = branching (like git).
 *
 * This component renders a compact, git-graph-style tree:
 *
 *   ● #1 (root)  exec_a1b2...  3 nodes  12.4s
 *   ├─● #2 (fork)  exec_c3d4...  ⟳ deterministic  ← cp_001
 *   │ └─● #4 (fork)  exec_e5f6...  ⟳ fresh  ← cp_003
 *   └─● #3 (fork)  exec_g7h8...  ⟳ fresh  ← cp_002
 *
 * Data source: SpanStore root spans carry parent_trace_id, restore_checkpoint_id,
 * and replay_mode — sufficient to reconstruct the full lineage DAG.
 *
 * Interaction:
 * - Click a node to switch the Timeline waterfall to that execution.
 * - Current/active execution is highlighted.
 * - Hover shows full execution ID + checkpoint details in tooltip.
 * - Collapsed by default when only 1 execution; auto-expands on 2+.
 */
import { useMemo, useState } from 'react';
import type { SpanTree } from '@/stores/storeActions/spanActions';

// ─── Types ───────────────────────────────────────────────────────

interface ExecutionNode {
  executionId: string;
  parentExecutionId: string | null;
  restoreCheckpointId: string | null;
  replayMode: boolean;
  spanCount: number;
  duration: number | null; // seconds
  status: 'running' | 'ok' | 'error';
  startTime: number | null;
  children: ExecutionNode[];
}

interface ExecutionLineageTreeProps {
  /** All available execution IDs (from spanStore keys) */
  executionIds: string[];
  /** Currently viewed execution ID */
  activeExecutionId: string | null;
  /** Callback to get SpanTree for a given execution */
  getSpanTree: (executionId: string) => SpanTree | null;
  /** Called when user selects an execution */
  onSelect: (executionId: string) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────

function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds <= 0) return '—';
  if (seconds < 0.001) return `${(seconds * 1_000_000).toFixed(0)}µs`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function shortId(id: string): string {
  if (id.length <= 12) return id;
  return id.slice(0, 4) + '…' + id.slice(-4);
}

/**
 * Build a forest of ExecutionNodes from flat execution list.
 * Uses root span's parent_trace_id to establish parent→child edges.
 */
function buildLineageForest(
  executionIds: string[],
  getSpanTree: (id: string) => SpanTree | null,
): ExecutionNode[] {
  const nodeMap = new Map<string, ExecutionNode>();

  // Phase 1: create nodes
  for (const eid of executionIds) {
    const tree = getSpanTree(eid);
    const root = tree?.root;

    const node: ExecutionNode = {
      executionId: eid,
      parentExecutionId: root?.parent_trace_id ?? null,
      restoreCheckpointId: root?.restore_checkpoint_id ?? null,
      replayMode: root?.replay_mode ?? false,
      spanCount: tree?.spans.length ?? 0,
      duration: tree?.totalDuration ?? null,
      status: root?.status === 'error' ? 'error' : (root?.end_time ? 'ok' : 'running'),
      startTime: root?.start_time ?? null,
      children: [],
    };
    nodeMap.set(eid, node);
  }

  // Phase 2: link parent→child
  const roots: ExecutionNode[] = [];
  for (const node of nodeMap.values()) {
    const parentId = node.parentExecutionId;
    if (parentId && nodeMap.has(parentId)) {
      nodeMap.get(parentId)!.children.push(node);
    } else {
      roots.push(node);
    }
  }

  // Phase 3: sort children by startTime (oldest first)
  const sortChildren = (n: ExecutionNode) => {
    n.children.sort((a, b) => (a.startTime ?? 0) - (b.startTime ?? 0));
    n.children.forEach(sortChildren);
  };
  roots.sort((a, b) => (a.startTime ?? 0) - (b.startTime ?? 0));
  roots.forEach(sortChildren);

  return roots;
}

// ─── Rendering ───────────────────────────────────────────────────

const STATUS_COLORS: Record<string, string> = {
  running: 'bg-blue-400 animate-pulse',
  ok: 'bg-green-400',
  error: 'bg-red-400',
};

const REPLAY_BADGE: Record<string, { label: string; color: string }> = {
  deterministic: { label: '⟳ det', color: 'text-purple-400 border-purple-400/40' },
  fresh: { label: '⟳ fresh', color: 'text-cyan-400 border-cyan-400/40' },
};

interface TreeRowProps {
  node: ExecutionNode;
  depth: number;
  isLast: boolean;
  activeExecutionId: string | null;
  onSelect: (id: string) => void;
  /** Connector lines from ancestors: true = has more siblings below, false = last child */
  connectors: boolean[];
}

function TreeRow({ node, depth, isLast, activeExecutionId, onSelect, connectors }: TreeRowProps) {
  const isActive = node.executionId === activeExecutionId;
  const replayKey = node.replayMode ? 'deterministic' : undefined;
  const badge = node.restoreCheckpointId ? (replayKey ? REPLAY_BADGE[replayKey] : REPLAY_BADGE.fresh) : null;

  // Build the git-graph connector prefix
  const connectorChars = connectors.map((hasSibling) => hasSibling ? '│ ' : '  ').join('');
  const branchChar = depth === 0 ? '' : (isLast ? '└─' : '├─');

  const tooltip = [
    `Execution: ${node.executionId}`,
    node.parentExecutionId ? `Parent: ${node.parentExecutionId}` : null,
    node.restoreCheckpointId ? `Checkpoint: ${node.restoreCheckpointId}` : null,
    node.replayMode ? 'Mode: deterministic replay' : (node.restoreCheckpointId ? 'Mode: fresh replay' : null),
    `Spans: ${node.spanCount}`,
    `Duration: ${formatDuration(node.duration)}`,
  ].filter(Boolean).join('\n');

  return (
    <button
      type="button"
      onClick={() => onSelect(node.executionId)}
      className={`w-full text-left px-2 py-1 text-[11px] font-mono rounded transition-colors flex items-center gap-1.5 group ${
        isActive
          ? 'bg-blue-600/20 text-blue-200 border border-blue-500/40'
          : 'text-gray-300 hover:bg-gray-700/60 border border-transparent'
      }`}
      title={tooltip}
    >
      {/* Git-graph connectors */}
      {depth > 0 && (
        <span className="text-gray-600 select-none whitespace-pre">{connectorChars}{branchChar}</span>
      )}

      {/* Status dot */}
      <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${STATUS_COLORS[node.status]}`} />

      {/* Execution label */}
      <span className={`truncate ${isActive ? 'text-blue-100 font-semibold' : 'text-gray-200'}`}>
        {shortId(node.executionId)}
      </span>

      {/* Replay badge */}
      {badge && (
        <span className={`px-1 py-0 rounded text-[9px] border ${badge.color}`}>
          {badge.label}
        </span>
      )}

      {/* Checkpoint source */}
      {node.restoreCheckpointId && (
        <span className="text-[9px] text-gray-500 truncate max-w-[60px]" title={`from ${node.restoreCheckpointId}`}>
          ← {shortId(node.restoreCheckpointId)}
        </span>
      )}

      {/* Stats */}
      <span className="ml-auto shrink-0 text-[10px] text-gray-500 group-hover:text-gray-400">
        {node.spanCount}sp · {formatDuration(node.duration)}
      </span>
    </button>
  );
}

function renderTree(
  nodes: ExecutionNode[],
  depth: number,
  activeExecutionId: string | null,
  onSelect: (id: string) => void,
  connectors: boolean[],
): JSX.Element[] {
  const rows: JSX.Element[] = [];

  nodes.forEach((node, idx) => {
    const isLast = idx === nodes.length - 1;

    rows.push(
      <TreeRow
        key={node.executionId}
        node={node}
        depth={depth}
        isLast={isLast}
        activeExecutionId={activeExecutionId}
        onSelect={onSelect}
        connectors={connectors}
      />,
    );

    if (node.children.length > 0) {
      const childConnectors = [...connectors, !isLast];
      rows.push(
        ...renderTree(node.children, depth + 1, activeExecutionId, onSelect, childConnectors),
      );
    }
  });

  return rows;
}

// ─── Main Component ──────────────────────────────────────────────

export function ExecutionLineageTree({
  executionIds,
  activeExecutionId,
  getSpanTree,
  onSelect,
}: ExecutionLineageTreeProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  const forest = useMemo(
    () => buildLineageForest(executionIds, getSpanTree),
    [executionIds, getSpanTree],
  );

  // Count total nodes in forest (for badge)
  const totalExecutions = executionIds.length;
  const hasForks = useMemo(
    () => forest.some((root) => root.children.length > 0),
    [forest],
  );

  if (totalExecutions <= 1) {
    // Single execution — render inline compact label instead of tree
    const eid = executionIds[0];
    if (!eid) return null;
    const tree = getSpanTree(eid);
    return (
      <div className="flex items-center gap-1.5 text-[11px] text-gray-400 px-1">
        <span className={`inline-block w-2 h-2 rounded-full ${
          tree?.root?.status === 'error' ? 'bg-red-400' : (tree?.root?.end_time ? 'bg-green-400' : 'bg-blue-400 animate-pulse')
        }`} />
        <span className="font-mono text-gray-300">{shortId(eid)}</span>
        <span className="text-gray-500">{tree?.spans.length ?? 0} spans · {formatDuration(tree?.totalDuration ?? null)}</span>
      </div>
    );
  }

  return (
    <div className="border border-gray-700/50 rounded bg-gray-800/50">
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-2 py-1 text-[11px] text-gray-300 hover:bg-gray-700/40 transition-colors rounded-t"
      >
        <span className="flex items-center gap-1.5">
          <span className="text-gray-500">{isExpanded ? '▼' : '▶'}</span>
          <span className="font-medium">Executions</span>
          {hasForks && (
            <span className="text-[9px] text-purple-400 border border-purple-400/30 rounded px-1">
              fork tree
            </span>
          )}
        </span>
        <span className="text-gray-500">{totalExecutions}</span>
      </button>

      {/* Tree body */}
      {isExpanded && (
        <div className="px-1 pb-1 space-y-0.5 max-h-[200px] overflow-y-auto">
          {renderTree(forest, 0, activeExecutionId, onSelect, [])}
        </div>
      )}
    </div>
  );
}

export default ExecutionLineageTree;
