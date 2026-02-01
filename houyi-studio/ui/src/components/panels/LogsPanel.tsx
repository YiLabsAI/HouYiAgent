import React from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { ExecutionLogsView } from './ExecutionLogsView';
import { ActivityLogsView } from './ActivityLogsView';
import { ObservabilityLogsView } from './ObservabilityLogsView';

export interface LogsPanelProps {
  viewExecution: any;
}

type LogsViewMode = 'execution' | 'activity' | 'observability';

export const LogsPanel: React.FC<LogsPanelProps> = ({ viewExecution }) => {
  const { activityLogs, nodeObservations, serverLogLevel, sendCommand, sessionId } = useConsoleStore();
  const [logView, setLogView] = React.useState<LogsViewMode>('execution');
  const [logSearch, setLogSearch] = React.useState('');
  const [logLevel, setLogLevel] = React.useState<'all' | 'debug' | 'info' | 'warning' | 'error'>('all');
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

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center bg-gray-700 rounded">
          {(['execution', 'activity', 'observability'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setLogView(mode)}
              className={`px-3 py-1 text-xs font-medium rounded ${logView === mode ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'}`}
            >
              {mode === 'execution' ? 'Execution' : mode === 'activity' ? 'Activity' : 'Observability'}
            </button>
          ))}
        </div>

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
        <ExecutionLogsView viewExecution={viewExecution} normalizedSearch={normalizedSearch} />
      ) : logView === 'observability' ? (
        <ObservabilityLogsView viewExecution={viewExecution} nodeObservations={nodeObservations as any} />
      ) : (
        <ActivityLogsView logs={filteredActivityLogs} onSetSearch={setLogSearch} />
      )}
    </div>
   );
 };
