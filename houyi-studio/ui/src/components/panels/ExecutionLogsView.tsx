import React from 'react';
import { ScrollText } from 'lucide-react';

export interface ExecutionLogsViewProps {
  viewExecution: any;
  normalizedSearch: string;
}

const formatTime = (timestamp: string | null | undefined): string => {
  if (!timestamp) return new Date().toLocaleTimeString();
  return new Date(timestamp).toLocaleTimeString();
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'running':
      return 'text-blue-400';
    case 'completed':
      return 'text-green-400';
    case 'failed':
      return 'text-red-400';
    default:
      return 'text-gray-400';
  }
};

const getNodeDisplayTime = (nodeExec: any): string => {
  if (nodeExec.status === 'running') return formatTime(nodeExec.started_at);
  return formatTime(nodeExec.completed_at);
};

const ExecutionStartLog: React.FC<{ execution: any }> = ({ execution }) => (
  <div className="text-gray-400">
    [{formatTime(execution.started_at)}] Execution started: {execution.execution_id}
  </div>
);

const NodeExecutionLog: React.FC<{ nodeId: string; nodeExec: any }> = ({ nodeId, nodeExec }) => {
  const displayTime = getNodeDisplayTime(nodeExec);
  const statusColor = getStatusColor(nodeExec.status);
  const startedTime = nodeExec.started_at ? formatTime(nodeExec.started_at) : null;
  const completedTime = nodeExec.completed_at ? formatTime(nodeExec.completed_at) : null;
  const hasInputs = nodeExec.inputs && Object.keys(nodeExec.inputs).length > 0;
  const hasOutputs = nodeExec.outputs && Object.keys(nodeExec.outputs).length > 0;

  return (
    <div className="space-y-1">
      <div className={statusColor}>
        [{displayTime}] {nodeId}: {nodeExec.status}
      </div>
      {(startedTime || completedTime) && (
        <div className="text-[10px] text-gray-500 pl-4">
          {startedTime ? `started ${startedTime}` : 'started -'}
          {completedTime ? ` · completed ${completedTime}` : ''}
        </div>
      )}
      {hasInputs && (
        <details className="pl-4 text-xs text-gray-400">
          <summary className="cursor-pointer">Inputs</summary>
          <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-1">
            {JSON.stringify(nodeExec.inputs, null, 2)}
          </pre>
        </details>
      )}
      {hasOutputs && (
        <details className="pl-4 text-xs text-gray-400">
          <summary className="cursor-pointer">Outputs</summary>
          <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-1">
            {JSON.stringify(nodeExec.outputs, null, 2)}
          </pre>
        </details>
      )}
      {nodeExec.streaming_output && (
        <pre className="text-[11px] text-gray-200 pl-4 whitespace-pre-wrap">
          {String(nodeExec.streaming_output)}
        </pre>
      )}
      {nodeExec.error && <div className="text-red-400 pl-4">Error: {nodeExec.error}</div>}
    </div>
  );
};

export const ExecutionLogsView: React.FC<ExecutionLogsViewProps> = ({ viewExecution, normalizedSearch }) => {
  if (!viewExecution) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <ScrollText size={32} className="mx-auto mb-2 opacity-50" />
          <div className="text-sm">No execution logs available</div>
          <div className="text-xs mt-1">Start execution to see logs</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-1 font-mono text-xs max-h-full overflow-y-auto">
      <ExecutionStartLog execution={viewExecution} />
      {Object.entries(viewExecution.node_executions || {})
        .filter(([nodeId, nodeExec]: any) => {
          if (!normalizedSearch) return true;
          const haystack = `${nodeId} ${nodeExec.streaming_output || ''} ${nodeExec.error || ''}`.toLowerCase();
          return haystack.includes(normalizedSearch);
        })
        .map(([nodeId, nodeExec]: any) => (
          <NodeExecutionLog key={nodeId} nodeId={String(nodeId)} nodeExec={nodeExec} />
        ))}
      {viewExecution.status === 'completed' && (
        <div className="text-green-400">
          [{viewExecution.completed_at ? new Date(viewExecution.completed_at).toLocaleTimeString() : new Date().toLocaleTimeString()}] Execution completed
        </div>
      )}
      {viewExecution.status === 'failed' && (
        <div className="text-red-400">
          [{viewExecution.completed_at ? new Date(viewExecution.completed_at).toLocaleTimeString() : new Date().toLocaleTimeString()}] Execution failed: {viewExecution.error}
        </div>
      )}
    </div>
  );
};
