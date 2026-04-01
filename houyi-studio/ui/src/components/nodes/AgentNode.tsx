/**
 * AGENT Node component — distinct visual style for sub-agent delegation nodes.
 *
 * Features:
 *   - Purple accent (differentiates from LLM blue / TOOL green)
 *   - Agent name + optional progress percentage
 *   - Status colors: pending grey → running blue pulse → completed green → failed red
 *   - Handoff edge annotation rendered via `data.handoffTo`
 */

import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { useConsoleStore } from '@/stores/useConsoleStore';

export const AgentNode = memo(({ data, selected, id }: NodeProps) => {
  const viewExecution = useConsoleStore((state) =>
    state.viewMode === 'checkpoint'
      ? state.checkpointExecution
      : state.liveExecution || state.currentExecution,
  );
  const nodeExec = viewExecution?.node_executions?.[id];
  const status = nodeExec?.status || data.status || 'pending';

  const statusConfig: Record<string, { bg: string; border: string; icon: string; text: string }> = {
    pending: { bg: 'bg-gray-50', border: 'border-gray-400', icon: '⏸', text: 'text-gray-700' },
    running: { bg: 'bg-purple-100', border: 'border-purple-500', icon: '▶', text: 'text-purple-700' },
    completed: { bg: 'bg-green-100', border: 'border-green-500', icon: '✓', text: 'text-green-700' },
    failed: { bg: 'bg-red-100', border: 'border-red-500', icon: '✗', text: 'text-red-700' },
    skipped: { bg: 'bg-gray-100', border: 'border-gray-400', icon: '⊘', text: 'text-gray-500' },
  };

  const config = statusConfig[status] || statusConfig.pending;
  const progress = data.progress as number | undefined;

  return (
    <div
      className={`px-4 py-2 shadow-lg rounded-md border-2 ${config.bg} ${config.border} ${
        selected ? 'ring-2 ring-purple-500 ring-offset-2' : ''
      } ${status === 'running' ? 'animate-pulse' : ''} transition-all duration-300`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-purple-500 hover:!bg-purple-600 transition-colors"
      />

      <div className="flex items-center gap-2">
        <div className="text-lg">🤖</div>
        <div className="flex-1">
          <div className={`text-sm font-bold ${config.text}`}>
            {data.label || 'Agent'}
          </div>
          <div className="text-xs text-gray-600">
            {data.agentId ? `Agent: ${data.agentId}` : 'Agent Node'}
          </div>
        </div>
        <div className="flex items-center gap-1">
          {progress !== undefined && (
            <span className="text-xs text-purple-600 font-mono">{progress}%</span>
          )}
          <div className={`text-sm font-bold ${config.text}`}>{config.icon}</div>
        </div>
      </div>

      {data.handoffTo && (
        <div className="mt-1 text-[10px] text-purple-500 truncate">
          → handoff: {data.handoffTo}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-purple-500 hover:!bg-purple-600 transition-colors"
      />
    </div>
  );
});

AgentNode.displayName = 'AgentNode';
