import React from 'react';

export interface ActivityLogsViewProps {
  logs: any[];
  onSetSearch?: (value: string) => void;
}

export const ActivityLogsView: React.FC<ActivityLogsViewProps> = ({ logs, onSetSearch }) => {
  return (
    <div className="space-y-2 text-xs">
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => onSetSearch?.('cache-hit')}
          className="px-2 py-1 text-[11px] rounded border border-amber-400/40 text-amber-200 hover:bg-amber-400/10"
        >
          cache-hit
        </button>
        <button
          type="button"
          onClick={() => onSetSearch?.('')}
          className="px-2 py-1 text-[11px] rounded border border-gray-600 text-gray-300 hover:bg-gray-700"
        >
          clear filter
        </button>
      </div>

      {logs.length === 0 ? (
        <div className="text-gray-400 text-center py-8">No activity logs yet</div>
      ) : (
        logs.map((log: any) => (
          <div key={log.id} className="flex items-start gap-3 bg-gray-900/60 border border-gray-700 rounded p-2">
            <span
              className={
                log.level === 'error'
                  ? 'text-red-400'
                  : log.level === 'warning'
                    ? 'text-yellow-400'
                    : 'text-blue-300'
              }
            >
              {String(log.level || 'info').toUpperCase()}
            </span>
            <div className="flex-1">
              <div className="text-gray-200">{log.message}</div>
              {log.detail && <div className="text-gray-500">{log.detail}</div>}
            </div>
            <div className="text-gray-500">{new Date(log.timestamp).toLocaleTimeString()}</div>
          </div>
        ))
      )}
    </div>
  );
};
