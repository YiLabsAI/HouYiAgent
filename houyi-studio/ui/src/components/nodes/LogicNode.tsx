import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { useConsoleStore } from '@/stores/useConsoleStore';

export const LogicNode = memo(({ data, selected, id }: NodeProps) => {
  const viewExecution = useConsoleStore((state) =>
    state.viewMode === 'checkpoint'
      ? state.checkpointExecution
      : state.liveExecution || state.currentExecution,
  );
  const nodeExec = viewExecution?.node_executions?.[id];
  const status = nodeExec?.status || data.status || 'pending';

  const statusConfig: Record<string, { bg: string; border: string; icon: string; text: string }> = {
    pending: { bg: 'bg-gray-50', border: 'border-gray-400', icon: '⏸', text: 'text-gray-700' },
    running: { bg: 'bg-yellow-100', border: 'border-yellow-500', icon: '▶', text: 'text-yellow-700' },
    completed: { bg: 'bg-green-100', border: 'border-green-500', icon: '✓', text: 'text-green-700' },
    failed: { bg: 'bg-red-100', border: 'border-red-500', icon: '✗', text: 'text-red-700' },
    skipped: { bg: 'bg-gray-100', border: 'border-gray-400', icon: '⊘', text: 'text-gray-500' },
  };

  const config = statusConfig[status] || statusConfig.pending;

  return (
    <div className={`px-4 py-2 shadow-lg rounded-md border-2 ${config.bg} ${config.border} ${
      selected ? 'ring-2 ring-blue-500 ring-offset-2' : ''
    } ${status === 'running' ? 'animate-pulse' : ''} transition-all duration-300`}>
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-yellow-500 hover:!bg-yellow-600 transition-colors"
      />

      <div className="flex items-center gap-2">
        <div className="text-lg">⚡</div>
        <div className="flex-1">
          <div className={`text-sm font-bold ${config.text}`}>{data.label || 'Logic'}</div>
          <div className="text-xs text-gray-600">Logic Node</div>
        </div>
        <div className={`text-sm font-bold ${config.text}`}>{config.icon}</div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-yellow-500 hover:!bg-yellow-600 transition-colors"
      />
    </div>
  );
});

LogicNode.displayName = 'LogicNode';
