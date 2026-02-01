import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';

export const ToolStatistics: React.FC = () => {
  const stats = useConsoleStore((state) => state.getToolStatistics());

  if (stats.totalCalls === 0 && stats.totalNodes === 0) {
    return null;
  }

  const toolNames = Object.keys(stats.toolsByName).sort();
  const successRate = stats.totalCalls > 0
    ? ((stats.successfulCalls / stats.totalCalls) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="flex items-center gap-4 px-3 py-1.5 bg-gray-800 rounded-lg border border-gray-700">
      {/* Total Tool Calls */}
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-blue-500" />
        <span className="text-xs font-medium text-gray-300">
          Tools: <span className="text-white font-semibold">{stats.totalCalls}</span>
        </span>
      </div>

      {/* Success/Failure Breakdown */}
      {stats.totalCalls > 0 && (
        <>
          <div className="h-4 w-px bg-gray-700" />
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-xs text-gray-400">
                <span className="text-green-400 font-medium">{stats.successfulCalls}</span>
              </span>
            </div>
            {stats.failedCalls > 0 && (
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-xs text-gray-400">
                  <span className="text-red-400 font-medium">{stats.failedCalls}</span>
                </span>
              </div>
            )}
            <span className="text-xs text-gray-500">
              ({successRate}% success)
            </span>
          </div>
        </>
      )}

      {/* Tool Nodes Count */}
      {stats.totalNodes > 0 && (
        <>
          <div className="h-4 w-px bg-gray-700" />
          <span className="text-xs text-gray-400">
            Nodes: <span className="text-gray-300 font-medium">{stats.totalNodes}</span>
          </span>
        </>
      )}

      {/* Tool Breakdown (if multiple tools) */}
      {toolNames.length > 1 && (
        <>
          <div className="h-4 w-px bg-gray-700" />
          <div className="relative flex items-center gap-2">
            <details className="group">
              <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-300 list-none">
                <span className="flex items-center gap-1">
                  <span>Details</span>
                  <span className="text-gray-600 group-open:rotate-180 transition-transform">▼</span>
                </span>
              </summary>
              <div className="absolute right-0 top-full mt-2 p-2 bg-gray-900 border border-gray-700 rounded-lg shadow-lg z-50 min-w-[200px]">
                <div className="space-y-1.5">
                  {toolNames.map((toolName) => {
                    const toolStat = stats.toolsByName[toolName];
                    return (
                      <div key={toolName} className="flex items-center justify-between text-xs">
                        <span className="text-gray-300 truncate max-w-[120px]" title={toolName}>
                          {toolName}
                        </span>
                        <div className="flex items-center gap-2 ml-2">
                          <span className="text-green-400">{toolStat.successful}</span>
                          {toolStat.failed > 0 && (
                            <span className="text-red-400">/{toolStat.failed}</span>
                          )}
                          <span className="text-gray-500">({toolStat.count})</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </details>
          </div>
        </>
      )}
    </div>
  );
};
