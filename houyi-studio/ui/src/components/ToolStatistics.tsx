import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';
import { Zap, ChevronDown, Wrench, BookOpen } from 'lucide-react';

interface RegisteredSkill {
  name: string;
  description: string;
  type: 'executable' | 'instruction';
  tool_count: number;
  has_executor: boolean;
  classification_basis?: string;
  classification_signals?: string[];
}

/**
 * ToolStatistics — displayed in the Title Bar.
 *
 * Reflects the Tool → Skill → Expertise three-layer abstraction:
 *   - Tool  = atomic callable interface (LLM function calling target)
 *   - Skill = orchestration unit (SKILL.md with hooks/policy/permissions)
 *
 * The pill shows "N skills (M tools)" to clearly communicate that skills
 * contain tools.  The dropdown groups skills by type: Executable (has a
 * Python executor) vs Schema-only (pure instruction document).
 *
 * Execution-time tool call statistics are shown in graph mode only.
 */
export const ToolStatistics: React.FC = () => {
  const stats = useConsoleStore((state) => state.getToolStatistics());
  const [registeredSkills, setRegisteredSkills] = React.useState<RegisteredSkill[]>([]);
  const [skillCount, setSkillCount] = React.useState(0);
  const [toolCount, setToolCount] = React.useState(0);
  const [isOpen, setIsOpen] = React.useState(false);
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    let cancelled = false;
    const fetchSkills = async () => {
      try {
        const res = await fetch('/api/tools');
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          if (Array.isArray(data.tools)) {
            setRegisteredSkills(data.tools);
          }
          setSkillCount(data.skill_count ?? data.tools?.length ?? 0);
          setToolCount(data.tool_count ?? data.tools?.length ?? 0);
        }
      } catch {
        // silently ignore
      }
    };
    fetchSkills();
    return () => { cancelled = true; };
  }, []);

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

  if (registeredSkills.length === 0 && !hasExecStats) {
    return null;
  }

  const executableSkills = registeredSkills.filter((s) => s.type === 'executable');
  const instructionSkills = registeredSkills.filter((s) => s.type === 'instruction');

  return (
    <div ref={dropdownRef} className="relative" data-testid="tool-statistics">
      {/* Compact pill */}
      <button
        type="button"
        onClick={() => setIsOpen((o) => !o)}
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-gray-800 border border-gray-700 hover:border-gray-500 text-xs text-gray-300 transition-colors"
        title={`${skillCount} skills (${toolCount} tools): ${executableSkills.length} executable + ${instructionSkills.length} instruction`}
      >
        <Zap size={13} className="text-yellow-400" />

        {skillCount > 0 && (
          <span className="font-medium text-gray-200" data-testid="tool-registered-count">
            <span className="text-green-400">{executableSkills.length}</span>
            <span className="text-gray-500 mx-0.5">+</span>
            <span className="text-blue-400">{instructionSkills.length}</span>
            <span className="text-gray-400 font-normal ml-1">skills</span>
          </span>
        )}

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

      {/* Dropdown */}
      {isOpen && (
        <div
          className="absolute right-0 top-full mt-1 w-[340px] max-h-[400px] overflow-y-auto bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50"
          data-testid="tool-statistics-dropdown"
        >
          {/* Execution stats (graph mode) */}
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

          {/* Executable skills (have Python executor bound) */}
          {executableSkills.length > 0 && (
            <div className="px-3 py-2 border-b border-gray-700/50">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                <Wrench size={10} />
                <span>Executable ({executableSkills.length})</span>
                <span className="text-[9px] text-green-600 ml-auto">semantic executable</span>
              </div>
              <div className="space-y-1.5">
                {executableSkills.map((skill) => (
                  <div key={skill.name} className="group">
                    <div className="flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
                      <span className="text-[11px] text-gray-200 font-medium">{skill.name}</span>
                      {skill.tool_count > 1 && (
                        <span className="text-[9px] text-gray-500 bg-gray-800 px-1 rounded">
                          {skill.tool_count} tools
                        </span>
                      )}
                    </div>
                    {skill.description && (
                      <div className="text-[10px] text-gray-500 leading-tight line-clamp-2 ml-3">
                        {skill.description}
                      </div>
                    )}
                    {skill.classification_basis && (
                      <div className="text-[9px] text-emerald-400/80 ml-3">
                        Basis: {skill.classification_basis}
                      </div>
                    )}
                    {Array.isArray(skill.classification_signals) && skill.classification_signals.length > 0 && (
                      <div className="text-[9px] text-gray-600 ml-3">
                        Signals: {skill.classification_signals.join(', ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Instruction skills (SKILL.md based, no executor) */}
          {instructionSkills.length > 0 && (
            <div className="px-3 py-2">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                <BookOpen size={10} />
                <span>Instruction ({instructionSkills.length})</span>
                <span className="text-[9px] text-blue-500 ml-auto">semantic instruction</span>
              </div>
              <div className="space-y-1.5">
                {instructionSkills.map((skill) => (
                  <div key={skill.name} className="group">
                    <div className="flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 shrink-0" />
                      <span className="text-[11px] text-gray-200 font-medium">{skill.name}</span>
                    </div>
                    {skill.description && (
                      <div className="text-[10px] text-gray-500 leading-tight line-clamp-2 ml-3">
                        {skill.description}
                      </div>
                    )}
                    {skill.classification_basis && (
                      <div className="text-[9px] text-blue-400/80 ml-3">
                        Basis: {skill.classification_basis}
                      </div>
                    )}
                    {Array.isArray(skill.classification_signals) && skill.classification_signals.length > 0 && (
                      <div className="text-[9px] text-gray-600 ml-3">
                        Signals: {skill.classification_signals.join(', ')}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Fallback empty */}
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
