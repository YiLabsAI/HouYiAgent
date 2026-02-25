import React from 'react';

import type { DryRunResultData, DisclosurePhase } from '../../../LeftSidebar/useSkillsLogic';

const phaseIcons: Record<string, string> = {
  discovery: '①',
  activation: '②',
  negotiation: '③',
  execution: '④',
  tool_execution: '⑤',
};

const phaseColors: Record<string, string> = {
  discovery: 'border-cyan-700/40',
  activation: 'border-blue-700/40',
  negotiation: 'border-purple-700/40',
  execution: 'border-green-700/40',
  tool_execution: 'border-cyan-400/40',
};

export const LlmVerificationDetails: React.FC<{
  llm: NonNullable<DryRunResultData['llm_verification']>;
}> = ({ llm }) => {
  const [expandedPhases, setExpandedPhases] = React.useState<Set<string>>(new Set(['execution']));
  const [showToolDefs, setShowToolDefs] = React.useState(false);

  const togglePhase = (name: string) => {
    setExpandedPhases((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const phases = llm.phases || [];

  return (
    <div className="space-y-1.5 mt-1" data-testid="llm-verify-flow">
      {phases.length > 0 && (
        <div className="space-y-1">
          {phases.map((phase, idx) => {
            const p = phase as DisclosurePhase;
            const icon = phaseIcons[p.name] || `${idx + 1}`;
            const borderColor = phaseColors[p.name] || 'border-gray-700/40';
            const isExpanded = expandedPhases.has(p.name);
            const isFail = p.status === 'fail';
            const isWarn = p.status === 'warn';

            return (
              <div key={p.name} className={`border-l-2 ${borderColor} pl-2`}>
                <button
                  type="button"
                  onClick={() => togglePhase(p.name)}
                  className="w-full flex items-center gap-1.5 text-left group"
                >
                  <span className="text-[10px] text-gray-500 font-mono shrink-0 w-4">{icon}</span>
                  <span className={`text-[10px] font-medium ${isFail ? 'text-red-400' : isWarn ? 'text-yellow-400' : 'text-gray-300'}`}>
                    {p.label}
                  </span>
                  <span className="text-[9px] text-gray-600 font-mono">t={p.timestamp_ms}ms</span>
                  <span className={`text-[9px] px-1 rounded ${isFail ? 'bg-red-900/30 text-red-400' : isWarn ? 'bg-yellow-900/30 text-yellow-400' : 'bg-green-900/20 text-green-500'}`}>
                    {p.status}
                  </span>
                  <span className="ml-auto text-[9px] text-gray-600 group-hover:text-gray-400 transition-colors">
                    {isExpanded ? '▾' : '▸'}
                  </span>
                </button>

                {isExpanded && (
                  <div className="mt-1 mb-1.5 ml-5">
                    <PhaseData phase={p} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {llm.tool_definitions && llm.tool_definitions.length > 0 && (
        <div className="ml-0.5">
          <button
            type="button"
            onClick={() => setShowToolDefs((v) => !v)}
            className="text-[9px] text-gray-600 hover:text-gray-400 transition-colors"
          >
            {showToolDefs ? '▾ Hide tool definitions' : '▸ Show tool definitions (JSON)'}
          </button>
          {showToolDefs && (
            <pre className="mt-1 text-[9px] text-gray-500 bg-gray-900/80 border border-gray-700/40 rounded p-1.5 overflow-x-auto font-mono leading-relaxed max-h-32 overflow-y-auto">
              {JSON.stringify(llm.tool_definitions, null, 2)}
            </pre>
          )}
        </div>
      )}

      {llm.system_prompt && (
        <div className="text-[9px] text-gray-600 bg-gray-900/40 rounded px-2 py-1 italic leading-relaxed">
          <span className="text-gray-500 not-italic font-medium">system: </span>
          {llm.system_prompt}
        </div>
      )}

      <div className="border border-gray-700/40 rounded-lg p-2 bg-gray-900/40">
        <div className="text-[10px] text-blue-400/70 font-medium mb-1">LLM Tool Call</div>
        {llm.tool_call ? (
          <pre className="text-[10px] text-green-300 bg-gray-900/80 border border-green-800/30 rounded p-2 overflow-x-auto font-mono leading-relaxed">
            {JSON.stringify(llm.tool_call, null, 2)}
          </pre>
        ) : (
          <div className="text-[10px] text-red-300 bg-red-900/20 rounded px-2 py-1">No tool call in response</div>
        )}
        {llm.raw_content && (
          <div className="mt-1.5 border border-gray-700/40 rounded p-2 bg-gray-900/60">
            <div className="text-[10px] text-gray-400/70 font-medium mb-0.5">Text Response</div>
            <div className="text-[10px] text-gray-300 leading-relaxed whitespace-pre-wrap">{llm.raw_content}</div>
          </div>
        )}

        {llm.usage && Object.keys(llm.usage).length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {Object.entries(llm.usage).map(([k, v]) => (
              <span key={k} className="px-1.5 py-0.5 bg-gray-800 border border-gray-700/50 rounded text-gray-400 text-[9px] font-mono">
                {k}: {String(v)}
              </span>
            ))}
          </div>
        )}

        {llm.model_name && (
          <div className="mt-1.5 flex items-center gap-1.5">
            <span className="px-1.5 py-0.5 bg-purple-900/30 border border-purple-700/30 rounded text-purple-300 text-[9px] font-mono">
              {llm.model_name}
            </span>
            <span className={`text-[10px] font-medium ${llm.success ? 'text-green-400' : 'text-red-400'}`}>
              {llm.success ? '✓' : '✗'} {llm.message}
            </span>
          </div>
        )}
      </div>

      {llm.execution_result !== undefined && (
        <div className="border border-gray-700/40 rounded-lg p-2 bg-gray-900/40">
          <div className="text-[10px] text-cyan-400/70 font-medium mb-1">Tool Execution Result</div>
          <pre className="text-[10px] text-cyan-200 bg-gray-900/80 border border-cyan-800/30 rounded p-2 overflow-x-auto font-mono leading-relaxed max-h-60 overflow-y-auto whitespace-pre-wrap">
            {llm.execution_result}
          </pre>
        </div>
      )}
    </div>
  );
};

const PhaseData: React.FC<{ phase: DisclosurePhase }> = ({ phase }) => {
  const d = phase.data || {};

  switch (phase.name) {
    case 'discovery':
      return (
        <div className="text-[9px] text-gray-400 space-y-0.5">
          <div><span className="text-gray-600">name:</span> {String(d.skill_name || '')}</div>
          <div><span className="text-gray-600">desc:</span> {String(d.description || '').slice(0, 120)}</div>
          {!!d.version && <div><span className="text-gray-600">version:</span> {String(d.version)}</div>}
          {Array.isArray(d.hooks) && d.hooks.length > 0 && (
            <div className="flex gap-1 flex-wrap">
              <span className="text-gray-600">hooks:</span>
              {(d.hooks as string[]).map((h) => (
                <span key={h} className="px-1 py-0.5 bg-blue-900/30 border border-blue-700/30 rounded text-blue-300 text-[9px]">
                  {h}
                </span>
              ))}
            </div>
          )}
        </div>
      );

    case 'activation':
      return (
        <div className="text-[9px] text-gray-400 space-y-0.5">
          <div><span className="text-gray-600">tools:</span> {String(d.tool_count || 0)} definition(s) built</div>
          {Array.isArray(d.tool_names) && (
            <div className="flex gap-1 flex-wrap">
              {(d.tool_names as string[]).map((n) => (
                <code key={n} className="px-1 py-0.5 bg-gray-800 border border-gray-700/50 rounded text-cyan-300 text-[9px]">
                  {n}
                </code>
              ))}
            </div>
          )}
        </div>
      );

    case 'negotiation':
      return (
        <div className="text-[9px] text-gray-400 space-y-0.5">
          <div>
            <span className="text-gray-600">system prompt:</span>{' '}
            {String(d.system_prompt_length || 0)} chars
          </div>
          {!!d.user_query && (
            <div className="bg-gray-900/60 rounded px-2 py-1 italic leading-relaxed text-gray-500">
              {String(d.user_query)}
            </div>
          )}
        </div>
      );

    case 'execution':
      return (
        <div className="text-[9px] text-gray-400 space-y-0.5">
          {!!d.model && <div><span className="text-gray-600">model:</span> {String(d.model)}</div>}
          {d.latency_ms !== undefined && (
            <div><span className="text-gray-600">latency:</span> {String(d.latency_ms)}ms</div>
          )}
          {!!d.usage && typeof d.usage === 'object' && Object.keys(d.usage as object).length > 0 && (
            <div className="flex gap-1 flex-wrap">
              {Object.entries(d.usage as Record<string, unknown>).map(([k, v]) => (
                <span key={k} className="px-1 py-0.5 bg-gray-800 rounded text-[9px] font-mono text-gray-500">
                  {k}:{String(v)}
                </span>
              ))}
            </div>
          )}
          {!!d.error && (
            <div className="text-red-400 bg-red-900/20 rounded px-2 py-1">{String(d.error)}</div>
          )}
        </div>
      );

    default:
      return (
        <pre className="text-[9px] text-gray-500 overflow-x-auto">
          {JSON.stringify(d, null, 2)}
        </pre>
      );
  }
};
