import React from 'react';

import type { SkillDetail } from '../../../../types/websocket';
import type { DryRunResultData } from '../../../LeftSidebar/useSkillsLogic';
import { LlmVerificationDetails } from './LlmVerificationDetails';

export type StageStatus = 'pending' | 'running' | 'pass' | 'fail' | 'warn' | 'skip';

export interface PipelineStage {
  id: string;
  number: number;
  label: string;
  status: StageStatus;
  summary: string;
  details?: React.ReactNode;
}

export function computeStages(
  result: DryRunResultData | null,
  detail: SkillDetail,
  liveMode: boolean,
): PipelineStage[] {
  const stages: PipelineStage[] = [];
  let n = 1;

  const certBadge = (cert: string) => {
    const colors: Record<string, string> = {
      gold: 'bg-yellow-900/30 border-yellow-600/40 text-yellow-300',
      silver: 'bg-gray-700/40 border-gray-500/40 text-gray-300',
      bronze: 'bg-orange-900/30 border-orange-600/40 text-orange-300',
      unverified: 'bg-gray-800/40 border-gray-700/40 text-gray-500',
    };
    return colors[cert] || colors.unverified;
  };

  stages.push({
    id: 'registration',
    number: n++,
    label: 'Skill Registration',
    status: result ? 'pass' : 'running',
    summary: result
      ? `${detail.display_name || detail.name}${detail.version && detail.version !== '0.0.0' ? ` v${detail.version}` : ''} — ${detail.tools.length} tool(s)`
      : 'Loading skill from registry...',
    details: result ? (
      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className={`px-1.5 py-0.5 rounded border text-[10px] font-medium ${certBadge(detail.certification)}`}>
            {detail.certification}
          </span>
          {detail.side_effect && detail.side_effect !== 'none' && (
            <span className="px-1.5 py-0.5 rounded border border-orange-700/30 bg-orange-900/20 text-orange-300 text-[10px]">
              {detail.side_effect}
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-1">
          {detail.tools.map((t) => (
            <span key={t.name} className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700/50 text-[10px] text-gray-300 font-mono">
              {t.name}
            </span>
          ))}
        </div>
      </div>
    ) : undefined,
  });

  const schemaOk = result ? result.schema_errors.length === 0 : false;
  stages.push({
    id: 'schema',
    number: n++,
    label: 'Schema Validation',
    status: result ? (schemaOk ? 'pass' : 'fail') : 'pending',
    summary: result
      ? (schemaOk ? 'Input conforms to tool schema' : `${result.schema_errors.length} validation error(s)`)
      : 'Validating input against tool schema...',
    details: result ? (
      <div className="space-y-1">
        {!schemaOk && result.schema_errors.map((err, i) => (
          <div key={i} className="text-[10px] text-red-300 bg-red-900/20 rounded px-2 py-1">
            {err}
          </div>
        ))}
      </div>
    ) : undefined,
  });

  const policyStatus: StageStatus = result
    ? (result.policy_result === 'allow' ? 'pass' :
       result.policy_result === 'allow_with_consent' ? 'warn' : 'fail')
    : 'pending';
  stages.push({
    id: 'policy',
    number: n++,
    label: 'Policy Evaluation',
    status: policyStatus,
    summary: result
      ? (result.policy_result === 'allow'
          ? 'Invocation allowed — no restrictions'
        : result.policy_result === 'allow_with_consent'
          ? 'Requires user consent before execution'
        : 'Invocation denied by policy')
      : 'Evaluating invocation policy...',
    details: result && result.policy_result !== 'allow' ? (
      <div className={`text-[10px] rounded px-2 py-1.5 ${
        result.policy_result === 'allow_with_consent'
          ? 'bg-yellow-900/20 border border-yellow-700/30 text-yellow-300'
          : 'bg-red-900/20 border border-red-700/30 text-red-300'
      }`}>
        {result.policy_result === 'allow_with_consent' ? (
          <>
            <span className="font-medium">Consent required:</span> This skill is configured with <code className="px-1 py-0.5 bg-yellow-900/40 rounded text-[9px]">require_consent</code> policy.
            In production, the LLM will pause and ask for user approval before invoking this tool.
            Dry-run bypasses consent to show the full execution trace.
          </>
        ) : (
          <>
            <span className="font-medium">Blocked:</span> This skill is configured with <code className="px-1 py-0.5 bg-red-900/40 rounded text-[9px]">deny</code> policy.
            The LLM will not be able to invoke this tool until the policy is changed.
          </>
        )}
      </div>
    ) : undefined,
  });

  const hasSideEffects = result ? result.estimated_side_effects.length > 0 : false;
  stages.push({
    id: 'side-effects',
    number: n++,
    label: 'Side Effects',
    status: result ? (hasSideEffects ? 'warn' : 'pass') : 'pending',
    summary: result
      ? (hasSideEffects
          ? `Declared: ${result.estimated_side_effects.join(', ')}`
          : 'No side effects declared')
      : 'Checking declared side effects...',
    details: hasSideEffects && result ? (
      <div className="flex flex-wrap gap-1">
        {result.estimated_side_effects.map((effect) => (
          <span key={effect} className="px-1.5 py-0.5 bg-orange-900/30 border border-orange-700/30 rounded text-orange-300 text-[10px]">
            {effect}
          </span>
        ))}
      </div>
    ) : undefined,
  });

  const hasHooks = detail.hooks && detail.hooks.length > 0;
  stages.push({
    id: 'hooks',
    number: n++,
    label: 'Lifecycle Hooks',
    status: result ? (hasHooks ? 'pass' : 'skip') : 'pending',
    summary: result
      ? (hasHooks
          ? `${detail.hooks.length} hook(s) registered`
          : 'No lifecycle hooks configured')
      : 'Checking lifecycle hooks...',
    details: hasHooks && result ? (
      <div className="flex flex-wrap gap-1.5">
        {detail.hooks.map((hook) => (
          <span key={hook} className="px-1.5 py-0.5 bg-blue-900/30 border border-blue-700/30 rounded text-blue-300 text-[10px] font-mono">
            {hook}
          </span>
        ))}
      </div>
    ) : undefined,
  });

  if (result && result.capability_gaps.length > 0) {
    stages.push({
      id: 'gaps',
      number: n++,
      label: 'Capability Gaps',
      status: 'warn',
      summary: `${result.capability_gaps.length} gap(s) detected`,
      details: (
        <div className="space-y-0.5">
          {result.capability_gaps.map((gap, i) => (
            <div key={i} className="text-[10px] text-orange-300 bg-orange-900/20 rounded px-2 py-1">
              {gap}
            </div>
          ))}
        </div>
      ),
    });
  }

  if (liveMode || result?.llm_verification) {
    const llm = result?.llm_verification;
    stages.push({
      id: 'llm-verify',
      number: n++,
      label: 'LLM Verification',
      status: result
        ? (llm ? (llm.success ? 'pass' : 'fail') : 'skip')
        : 'pending',
      summary: result
        ? (llm
            ? (llm.message || (llm.success ? 'LLM produced correct tool call' : 'LLM verification failed'))
            : 'LLM verification not available')
        : 'Sending probe to LLM...',
      details: llm ? (
        <LlmVerificationDetails llm={llm} />
      ) : undefined,
    });

    const toolExecPhase = result?.llm_verification?.phases?.find(
      (p) => p.name === 'tool_execution',
    );
    if (toolExecPhase) {
      stages.push({
        id: 'tool-execution',
        number: n++,
        label: 'Tool Execution',
        status:
          toolExecPhase.status === 'pass'
            ? 'pass'
            : toolExecPhase.status === 'fail'
              ? 'fail'
              : toolExecPhase.status === 'warn'
                ? 'warn'
                : 'skip',
        summary: toolExecPhase.data?.result_preview
          ? `Result: ${(typeof toolExecPhase.data.result_preview === 'string' ? toolExecPhase.data.result_preview : JSON.stringify(toolExecPhase.data.result_preview)).substring(0, 120)}...`
          : String(toolExecPhase.data?.reason || toolExecPhase.data?.error || 'Skipped'),
      });
    }
  }

  return stages;
}
