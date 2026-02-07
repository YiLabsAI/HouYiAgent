import React from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { ExecutionLogsView } from './ExecutionLogsView';
import { ActivityLogsView } from './ActivityLogsView';

export interface LogsPanelProps {
  viewExecution: any;
}

type LogsViewMode = 'execution' | 'activity';

export const LogsPanel: React.FC<LogsPanelProps> = ({ viewExecution }) => {
  const { activityLogs, checkpoints, serverLogLevel, sendCommand, sessionId } = useConsoleStore();
  const [logView, setLogView] = React.useState<LogsViewMode>('execution');
  const [logSearch, setLogSearch] = React.useState('');
  const [logLevel, setLogLevel] = React.useState<'all' | 'debug' | 'info' | 'warning' | 'error'>('all');
  const [selectedExecId, setSelectedExecId] = React.useState<string | null>(null);
  const normalizedSearch = logSearch.trim().toLowerCase();

  const filteredActivityLogs = React.useMemo(() => {
    return (activityLogs || []).filter((log: any) => {
      if (logLevel !== 'all' && log.level !== logLevel) return false;
      if (!normalizedSearch) return true;
      return `${log.message} ${log.detail ?? ''}`.toLowerCase().includes(normalizedSearch);
    });
  }, [activityLogs, logLevel, normalizedSearch]);

  const setServerLogLevel = (level: typeof serverLogLevel) => {
    sendCommand({ command_type: 'set_log_level', command_id: `cmd_${Date.now()}`, session_id: sessionId, level });
  };

  // Build list of all known executions from checkpoints + current viewExecution
  const allExecutions = React.useMemo(() => {
    const execMap = new Map<string, any>();
    // Collect from checkpoints (each has an execution_snapshot)
    for (const cp of checkpoints) {
      const execId = cp.execution_id;
      if (!execId) continue;
      const snapshot = cp.execution_snapshot as any;
      if (snapshot && !execMap.has(execId)) {
        execMap.set(execId, snapshot);
      }
      // Keep the latest snapshot per execution (higher sequence_number = more complete)
      const existing = execMap.get(execId);
      if (existing && snapshot) {
        const existingNodes = Object.keys(existing.node_executions || {}).length;
        const snapshotNodes = Object.keys(snapshot.node_executions || {}).length;
        if (snapshotNodes > existingNodes) {
          execMap.set(execId, snapshot);
        }
      }
    }
    // Always include the current viewExecution (most up-to-date)
    if (viewExecution?.execution_id) {
      execMap.set(viewExecution.execution_id, viewExecution);
    }
    return execMap;
  }, [checkpoints, viewExecution]);

  // Default to current execution
  const effectiveExecId = selectedExecId && allExecutions.has(selectedExecId)
    ? selectedExecId
    : viewExecution?.execution_id || null;
  const effectiveExecution = effectiveExecId ? allExecutions.get(effectiveExecId) : viewExecution;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center bg-gray-700 rounded">
          {(['execution', 'activity'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setLogView(mode)}
              className={`px-3 py-1 text-xs font-medium rounded ${logView === mode ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'}`}
            >
              {mode === 'execution' ? 'Execution' : 'Activity'}
            </button>
          ))}
        </div>

        {logView === 'execution' && allExecutions.size > 1 && (
          <select
            className="px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200"
            value={effectiveExecId || ''}
            onChange={(e) => setSelectedExecId(e.target.value || null)}
          >
            {Array.from(allExecutions.entries()).map(([execId, exec]) => (
              <option key={execId} value={execId}>
                {execId === viewExecution?.execution_id ? `${execId} (current)` : execId}
                {exec?.status ? ` · ${exec.status}` : ''}
              </option>
            ))}
          </select>
        )}

        <input
          type="text"
          value={logSearch}
          onChange={(e) => setLogSearch(e.target.value)}
          className="flex-1 min-w-[180px] px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          placeholder="Search logs"
        />

        {logView === 'activity' && (
          <select
            className="px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200"
            value={logLevel}
            onChange={(e) => setLogLevel(e.target.value as typeof logLevel)}
          >
            <option value="all">All levels</option>
            <option value="debug">Debug</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
        )}

        <select
          className="px-2 py-1 text-xs bg-gray-700 border border-gray-600 rounded text-gray-200"
          value={serverLogLevel}
          onChange={(e) => setServerLogLevel(e.target.value as typeof serverLogLevel)}
          title="Server log level"
        >
          <option value="debug">Server: Debug</option>
          <option value="info">Server: Info</option>
          <option value="warning">Server: Warning</option>
          <option value="error">Server: Error</option>
        </select>
      </div>

      {logView === 'execution' ? (
        <ExecutionLogsView viewExecution={effectiveExecution} normalizedSearch={normalizedSearch} />
      ) : (
        <ActivityLogsView logs={filteredActivityLogs} onSetSearch={setLogSearch} />
      )}
    </div>
   );
 };
