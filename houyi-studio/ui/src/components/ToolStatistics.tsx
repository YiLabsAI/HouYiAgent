import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';
import { Zap, ChevronDown } from 'lucide-react';

interface RegisteredSkill {
  name: string;
  description: string;
}

/**
 * ToolStatistics — displayed in the Title Bar.
 *
 * Shows two layers of information:
 *   1. Registered skills count (from /api/tools, fetched once on mount).
 *      Each skill exposes one tool to the LLM via function calling (1:1 mapping).
 *   2. Execution-time tool call statistics (from ConsoleStore, graph mode only).
 *
 * Clicking the pill opens a dropdown listing registered skills and exec stats.
 */
export const ToolStatistics: React.FC = () => {
  const stats = useConsoleStore((state) => state.getToolStatistics());
  const [registeredSkills, setRegisteredSkills] = React.useState<RegisteredSkill[]>([]);
  const [isOpen, setIsOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  // Fetch registered skills from backend once on mount
  React.useEffect(() => {
    let cancelled = false;
    const fetchSkills = async () => {
      try {
        const res = await fetch('/api/tools');
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled && Array.isArray(data.tools)) {
          setRegisteredSkills(data.tools);
        }
      } catch {
        // silently ignore — the header remains functional
      }
    };
    fetchSkills();
    return () => { cancelled = true; };
  }, []);

  // Close dropdown on outside click
  React.useEffect(() => {
    if (!isOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [isOpen]);

  const hasExecStats = stats.totalCalls > 0 || stats.totalNodes > 0;
  const successRate =
    stats.totalCalls > 0
      ? ((stats.successfulCalls / stats.totalCalls) * 100).toFixed(0)
      : null;

  // Nothing to show at all
  if (registeredSkills.length === 0 && !hasExecStats) {
    return null;
  }

  return (
    <div ref={dropdownRef} className="relative" data-testid="tool-statistics">
      {/* ─── Compact pill ─── */}
      <button
        type="button"
        onClick={() => setIsOpen((o) => !o)}
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-gray-800 border border-gray-700 hover:border-gray-500 text-xs text-gray-300 transition-colors"
        title="Registered skills & execution statistics"
      >
        <Zap size={13} className="text-yellow-400" />

        {/* Registered skills count */}
        {registeredSkills.length > 0 && (
          <span className="font-medium text-gray-200" data-testid="tool-registered-count">
            {registeredSkills.length} skills
          </span>
        )}

        {/* Execution calls (graph mode) */}
        {hasExecStats && (
          <>
            <span className="text-gray-600 mx-0.5">|</span>
            <span className="text-green-400 font-medium" data-testid="tool-exec-calls">
              {stats.successfulCalls}
            </span>
            {stats.failedCalls > 0 && (
              <span className="text-red-400 font-medium">/{stats.failedCalls}</span>
            )}
          </>
        )}

        <ChevronDown
          size={12}
          className={`text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* ─── Dropdown ─── */}
      {isOpen && (
        <div
          className="absolute right-0 top-full mt-1 w-[320px] max-h-[400px] overflow-y-auto bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50"
          data-testid="tool-statistics-dropdown"
        >
          {/* Execution stats section (graph mode) */}
          {hasExecStats && (
            <div className="px-3 py-2 border-b border-gray-700">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                Execution Statistics
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-gray-300">
                  Calls: <span className="text-white font-semibold">{stats.totalCalls}</span>
                </span>
                <span className="text-green-400">{stats.successfulCalls} ok</span>
                {stats.failedCalls > 0 && (
                  <span className="text-red-400">{stats.failedCalls} fail</span>
                )}
                {successRate && (
                  <span className="text-gray-500">({successRate}%)</span>
                )}
              </div>
              {stats.totalNodes > 0 && (
                <div className="text-xs text-gray-400 mt-1">
                  Nodes: {stats.totalNodes} ({stats.toolNodes} tool)
                </div>
              )}

              {/* Per-tool breakdown */}
              {Object.keys(stats.toolsByName).length > 0 && (
                <div className="mt-2 space-y-1">
                  {Object.entries(stats.toolsByName)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([name, ts]) => (
                      <div key={name} className="flex items-center justify-between text-[11px]">
                        <span className="text-gray-300 truncate max-w-[180px]" title={name}>
                          {name}
                        </span>
                        <div className="flex items-center gap-1.5 ml-2 shrink-0">
                          <span className="text-green-400">{ts.successful}</span>
                          {ts.failed > 0 && <span className="text-red-400">/{ts.failed}</span>}
                          <span className="text-gray-500">({ts.count})</span>
                        </div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}

          {/* Registered skills section */}
          {registeredSkills.length > 0 && (
            <div className="px-3 py-2">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                Registered Skills ({registeredSkills.length})
              </div>
              <div className="space-y-1.5">
                {registeredSkills.map((skill) => (
                  <div key={skill.name} className="group">
                    <div className="text-[11px] text-gray-200 font-medium">{skill.name}</div>
                    {skill.description && (
                      <div className="text-[10px] text-gray-500 leading-tight line-clamp-2">
                        {skill.description}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {registeredSkills.length === 0 && !hasExecStats && (
            <div className="px-3 py-4 text-center text-xs text-gray-500">
              No skills registered
            </div>
          )}
        </div>
      )}
    </div>
  );
};
