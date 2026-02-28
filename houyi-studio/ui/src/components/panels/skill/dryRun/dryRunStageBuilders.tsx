import type { SkillDetail } from '../../../../types/websocket';
import type { DryRunResultData } from '../../../LeftSidebar/useSkillsLogic';
import { getToolDryRunPreset } from './dryRunToolRules';
import {
  EXTERNAL_EXAMPLE_STATIC_CHECKS,
  extractAction,
  isInstructionDrivenRuntime,
  parseLooseJson,
  PLANNING_EXPECTED_ACTION_BY_EXAMPLE_ID,
  subsetMatch,
} from './dryRunEvidenceRules';
import { LlmVerificationDetails } from './LlmVerificationDetails';
import type { DryRunPipelineContext, PipelineStage, StageStatus } from './dryRunPipelineTypes';

type StageDraft = Omit<PipelineStage, 'number'>;

const certificationBadgeClass = (cert: string): string => {
  const colors: Record<string, string> = {
    gold: 'bg-yellow-900/30 border-yellow-600/40 text-yellow-300',
    silver: 'bg-gray-700/40 border-gray-500/40 text-gray-300',
    bronze: 'bg-orange-900/30 border-orange-600/40 text-orange-300',
    unverified: 'bg-gray-800/40 border-gray-700/40 text-gray-500',
  };
  return colors[cert] || colors.unverified;
};

export const buildRegistrationStages = (
  result: DryRunResultData | null,
  detail: SkillDetail,
): StageDraft[] => ([
  {
    id: 'registration',
    label: 'Skill Registration',
    status: result ? 'pass' : 'running',
    summary: result
      ? `${detail.display_name || detail.name}${detail.version && detail.version !== '0.0.0' ? ` v${detail.version}` : ''} — ${detail.tools.length} tool(s)`
      : 'Loading skill from registry...',
    details: result ? (
      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className={`px-1.5 py-0.5 rounded border text-[10px] font-medium ${certificationBadgeClass(detail.certification)}`}>
            {detail.certification}
          </span>
          {detail.side_effect && detail.side_effect !== 'none' && (
            <span className="px-1.5 py-0.5 rounded border border-orange-700/30 bg-orange-900/20 text-orange-300 text-[10px]">
              {detail.side_effect}
            </span>
          )}
          {detail.is_core && (
            <span className="px-1.5 py-0.5 rounded border border-cyan-700/40 bg-cyan-900/20 text-cyan-300 text-[10px]">
              core
            </span>
          )}
          {detail.is_external_alias && (
            <span className="px-1.5 py-0.5 rounded border border-amber-700/40 bg-amber-900/20 text-amber-300 text-[10px]">
              {detail.alias_target ? `ext alias → ${detail.alias_target}` : 'ext alias'}
            </span>
          )}
          {detail.runtime_binding && detail.runtime_binding !== 'none' && (
            <span className="px-1.5 py-0.5 rounded border border-emerald-700/40 bg-emerald-900/20 text-emerald-300 text-[10px] font-mono">
              {detail.runtime_binding}
            </span>
          )}
          {(detail.instructions_length ?? 0) > 0 && (
            <span className="px-1.5 py-0.5 rounded border border-emerald-700/40 bg-emerald-900/20 text-emerald-300 text-[10px]">
              instructions {(detail.instructions_length ?? 0)} chars
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
  },
]);

export const buildExternalPlanningStages = (
  result: DryRunResultData | null,
  detail: SkillDetail,
  liveMode: boolean,
  context?: DryRunPipelineContext,
): StageDraft[] => {
  if (detail.name !== 'ext__planning-with-files') {
    return [];
  }

  const stages: StageDraft[] = [];
  const instructions = (detail.instructions || '').toLowerCase();
  const hookSpecs = detail.hook_specs || [];
  const hasStaticEvidence = instructions.trim().length > 0 || hookSpecs.length > 0;
  const selectedExampleId = context?.planningFlowId ?? null;
  const selectedExampleLabel = context?.planningFlowLabel ?? null;
  const hasCommandHook = (event: string, needle: string): boolean => hookSpecs.some((h) => (
    String(h.event || '').toLowerCase() === event.toLowerCase()
    && String(h.type || '').toLowerCase() === 'command'
    && String(h.command || '').toLowerCase().includes(needle)
  ));

  const checks = [
    {
      key: 'plan-files',
      label: 'Creates task_plan.md / findings.md / progress.md',
      ok: instructions.includes('task_plan.md') && instructions.includes('findings.md') && instructions.includes('progress.md'),
    },
    {
      key: 'pretool-refresh',
      label: 'PreToolUse refreshes plan before major actions',
      ok: hasCommandHook('PreToolUse', 'task_plan.md'),
    },
    {
      key: 'posttool-reminder',
      label: 'PostToolUse reminds status update after writes',
      ok: hasCommandHook('PostToolUse', 'update task_plan.md status'),
    },
    {
      key: 'stop-verify',
      label: 'Stop hook verifies completion before stop',
      ok: hasCommandHook('Stop', 'check-complete'),
    },
    {
      key: 'key-rules',
      label: 'Key rules loaded (Create Plan First / 2-Action / Log ALL Errors / Never Repeat Failures)',
      ok: instructions.includes('create plan first')
        && instructions.includes('2-action rule')
        && instructions.includes('log all errors')
        && instructions.includes('never repeat failures'),
    },
    {
      key: 'when-to-use',
      label: 'When-to-use guidance loaded (use for / skip for)',
      ok: instructions.includes('use for:') && instructions.includes('skip for:'),
    },
    {
      key: 'examples-reference',
      label: 'Examples/reference docs linked (examples.md / reference.md)',
      ok: instructions.includes('examples.md') && instructions.includes('reference.md'),
    },
  ];

  const passed = checks.filter((c) => c.ok).length;
  const selectedExample = selectedExampleId
    ? (detail.package_examples ?? []).find((example) => example.id === selectedExampleId) ?? null
    : null;
  const exampleFocus = Array.isArray(selectedExample?.expectedFocus)
    ? selectedExample.expectedFocus
    : [];

  const toolExecPhase = result?.llm_verification?.phases?.find((p) => p.name === 'tool_execution');
  const toolExecReason = String(toolExecPhase?.data?.reason || toolExecPhase?.data?.error || '');
  const promptNoExecutorSkip =
    toolExecPhase?.status === 'skip'
    && toolExecReason.toLowerCase() === 'no executor available'
    && isInstructionDrivenRuntime(detail.runtime_binding);

  const staticStatus: StageStatus = !result
    ? 'pending'
    : !hasStaticEvidence
      ? 'skip'
      : passed === 0
        ? 'warn'
        : 'pass';

  stages.push({
    id: 'planning-flow-static',
    label: 'Planning Example Spec Alignment',
    status: staticStatus,
    summary: !result
      ? 'Validating planning example spec alignment...'
      : !hasStaticEvidence
        ? 'Static evidence unavailable (instructions/hook specs missing)'
        : selectedExample
          ? `global ${passed}/${checks.length}, selected ${selectedExampleLabel || selectedExampleId}`
          : `global checks ${passed}/${checks.length} covered`,
    details: result ? (
      <div className="space-y-1">
        {selectedExampleLabel && (
          <div className="text-[10px] rounded px-2 py-1 bg-cyan-900/20 border border-cyan-700/30 text-cyan-300">
            Selected example: {selectedExampleLabel}
          </div>
        )}
        {selectedExample?.objective && (
          <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
            Example objective: {selectedExample.objective}
          </div>
        )}
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Proof basis: static validation from SKILL.md instructions + hook specs only. This stage does not prove runtime or file creation on disk.
        </div>
        {!hasStaticEvidence && (
          <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-400">
            SKILL.md instructions/hook specs are not present in this detail payload, so static checks are skipped for this run.
          </div>
        )}
        {checks.map((c) => (
          <div
            key={c.key}
            className={`text-[10px] rounded px-2 py-1 ${
              c.ok
                ? 'bg-green-900/20 border border-green-700/30 text-green-300'
                : 'bg-yellow-900/20 border border-yellow-700/30 text-yellow-300'
            }`}
          >
            {c.ok ? '✓' : '•'} {c.label}
          </div>
        ))}
        {exampleFocus.length > 0 && (
          <div className="pt-1">
            <div className="text-[10px] text-gray-500 mb-1">Example-specific focus (from package docs)</div>
            <div className="space-y-1">
              {exampleFocus.map((focus) => (
                <div
                  key={`example-focus-${focus}`}
                  className="text-[10px] rounded px-2 py-1 bg-cyan-900/20 border border-cyan-700/30 text-cyan-300"
                >
                  • {focus}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    ) : undefined,
  });

  if (!(liveMode || result?.llm_verification)) {
    return stages;
  }

  const llm = result?.llm_verification;
  const expectedToolName = 'ext__planning-with-files';
  const expectedAction = selectedExample
    && typeof selectedExample.input === 'object'
    && selectedExample.input
    && typeof (selectedExample.input as Record<string, unknown>).action === 'string'
    ? String((selectedExample.input as Record<string, unknown>).action)
    : (selectedExampleId ? PLANNING_EXPECTED_ACTION_BY_EXAMPLE_ID[selectedExampleId] ?? null : null);
  const observedToolCall = llm?.tool_call as Record<string, unknown> | undefined;
  const requestedInput = llm?.requested_input;
  const requestedAction = requestedInput
    && typeof requestedInput === 'object'
    && 'action' in requestedInput
    && typeof (requestedInput as Record<string, unknown>).action === 'string'
    ? (requestedInput as Record<string, string>).action
    : '';
  const observedToolName = typeof observedToolCall?.name === 'string'
    ? String(observedToolCall.name)
    : observedToolCall
      && typeof observedToolCall.function === 'object'
      && observedToolCall.function
      && typeof (observedToolCall.function as Record<string, unknown>).name === 'string'
      ? String((observedToolCall.function as Record<string, unknown>).name)
      : '';
  const observedArgs = observedToolCall?.arguments
    ?? (observedToolCall && typeof observedToolCall.function === 'object'
      ? (observedToolCall.function as Record<string, unknown>).arguments
      : undefined)
    ?? observedToolCall?.args;
  const observedAction = typeof observedToolCall?.action === 'string'
    ? observedToolCall.action
    : extractAction(observedArgs) || extractAction(llm?.raw_content);
  const toolMatch = observedToolName === expectedToolName;
  const requestedActionMatch = expectedAction ? requestedAction === expectedAction : true;
  const actionMatch = expectedAction ? observedAction === expectedAction : true;
  const missingObservedActionEvidence = Boolean(expectedAction) && !observedAction;
  const llmSuccess = Boolean(llm?.success);
  const promptNativeActionFallback = llmSuccess
    && toolMatch
    && requestedActionMatch
    && missingObservedActionEvidence
    && isInstructionDrivenRuntime(detail.runtime_binding);
  const executionPhase = llm?.phases?.find((p) => p.name === 'execution');

  stages.push({
    id: 'planning-flow-runtime',
    label: 'Planning Example LLM Routing',
    status: !result
      ? 'pending'
      : !llm
        ? 'skip'
        : llmSuccess && toolMatch && requestedActionMatch && actionMatch
          ? 'pass'
          : promptNativeActionFallback
            ? 'pass'
            : llmSuccess && toolMatch && requestedActionMatch && missingObservedActionEvidence
              ? 'warn'
              : 'fail',
    summary: !result
      ? 'Collecting LLM routing verification...'
      : !llm
        ? 'No LLM routing data captured'
        : llmSuccess && toolMatch && requestedActionMatch && actionMatch
          ? `LLM selected ${expectedToolName}${expectedAction ? ` with expected action "${expectedAction}"` : ''}`
          : promptNativeActionFallback
            ? 'LLM selected expected tool and snapshot action; instruction-driven mode uses requested_input as routing evidence'
            : llmSuccess && toolMatch && requestedActionMatch && missingObservedActionEvidence
              ? 'LLM selected expected tool and snapshot action, but observed action evidence is missing'
              : `LLM routing mismatch${expectedAction ? ` (expected action "${expectedAction}")` : ''}`,
    details: result ? (
      <div className="space-y-1">
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Routing proof basis: expected example action vs requested_input snapshot vs observed LLM tool call.
        </div>
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Expected: tool <span className="font-mono">{expectedToolName}</span>
          {expectedAction ? <> · action <span className="font-mono">{expectedAction}</span></> : null}
        </div>
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Snapshot sent to LLM: action <span className="font-mono">{requestedAction || '(missing)'}</span>
        </div>
        {executionPhase && (
          <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
            LLM execution: <span className="font-mono">{executionPhase.status}</span>
            {executionPhase.data?.model ? <> · model <span className="font-mono">{String(executionPhase.data.model)}</span></> : null}
            {executionPhase.data?.latency_ms != null ? <> · latency <span className="font-mono">{String(executionPhase.data.latency_ms)}ms</span></> : null}
          </div>
        )}
        {llm && (
          <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
            Observed: tool <span className="font-mono">{observedToolName || '(none)'}</span>
            {observedAction ? <> · action <span className="font-mono">{observedAction}</span></> : null}
          </div>
        )}
        {promptNoExecutorSkip && (
          <div className="text-[10px] rounded px-2 py-1 bg-yellow-900/20 border border-yellow-700/30 text-yellow-300">
            Executor replay skipped (no python executor). LLM routing proof is available, but this run did not physically create task_plan.md/findings.md/progress.md.
          </div>
        )}
      </div>
    ) : undefined,
  });

  return stages;
};

export const buildGenericExampleStages = (
  result: DryRunResultData | null,
  detail: SkillDetail,
  liveMode: boolean,
  context?: DryRunPipelineContext,
): StageDraft[] => {
  const selectedToolName = context?.selectedToolName || detail.tools[0]?.name || detail.name;
  const selectedExampleId = context?.selectedExampleId ?? null;
  const selectedExampleLabel = context?.selectedExampleLabel ?? null;
  const selectedExamplePreset = (
    context?.selectedExampleInput
      ? {
        id: selectedExampleId || 'selected-example',
        label: selectedExampleLabel || 'Selected Example',
        description: '',
        input: context.selectedExampleInput,
        objective: context.selectedExampleObjective || '',
      }
      : null
  ) ?? (
    selectedExampleId
      ? getToolDryRunPreset(selectedToolName, selectedExampleId)
      : null
  );

  const genericChecks = EXTERNAL_EXAMPLE_STATIC_CHECKS[detail.name] || [];
  const shouldRenderGenericExampleStages = detail.name !== 'ext__planning-with-files'
    && genericChecks.length > 0
    && Boolean(selectedExamplePreset);

  if (!shouldRenderGenericExampleStages) {
    return [];
  }

  const stages: StageDraft[] = [];
  const instructions = (detail.instructions || '').toLowerCase();
  const checks = genericChecks.map((c) => ({
    ...c,
    ok: c.needles.every((needle) => instructions.includes(needle.toLowerCase())),
  }));
  const passed = checks.filter((c) => c.ok).length;

  stages.push({
    id: 'tool-example-static',
    label: 'Skill Example Spec Alignment',
    status: !result ? 'pending' : (passed === checks.length ? 'pass' : 'warn'),
    summary: !result
      ? 'Validating selected skill example against SKILL.md...'
      : `${selectedExampleLabel || selectedExampleId} checks ${passed}/${checks.length}`,
    details: result ? (
      <div className="space-y-1">
        <div className="text-[10px] rounded px-2 py-1 bg-cyan-900/20 border border-cyan-700/30 text-cyan-300">
          Selected example: {selectedExampleLabel || selectedExampleId}
        </div>
        {selectedExamplePreset?.objective && (
          <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
            Example objective: {selectedExamplePreset.objective}
          </div>
        )}
        {checks.map((c) => (
          <div
            key={c.key}
            className={`text-[10px] rounded px-2 py-1 ${
              c.ok
                ? 'bg-green-900/20 border border-green-700/30 text-green-300'
                : 'bg-yellow-900/20 border border-yellow-700/30 text-yellow-300'
            }`}
          >
            {c.ok ? '✓' : '•'} {c.label}
          </div>
        ))}
      </div>
    ) : undefined,
  });

  if (!(liveMode || result?.llm_verification)) {
    return stages;
  }

  const llm = result?.llm_verification;
  const expectedToolName = selectedToolName;
  const expectedInputSubset = selectedExamplePreset?.input ?? {};
  const observedToolCall = llm?.tool_call as Record<string, unknown> | undefined;
  const observedToolName = typeof observedToolCall?.name === 'string'
    ? String(observedToolCall.name)
    : observedToolCall
      && typeof observedToolCall.function === 'object'
      && observedToolCall.function
      && typeof (observedToolCall.function as Record<string, unknown>).name === 'string'
      ? String((observedToolCall.function as Record<string, unknown>).name)
      : '';
  const observedArgsRaw = observedToolCall?.arguments
    ?? (observedToolCall && typeof observedToolCall.function === 'object'
      ? (observedToolCall.function as Record<string, unknown>).arguments
      : undefined)
    ?? observedToolCall?.args
    ?? llm?.raw_content;
  const observedArgs = typeof observedArgsRaw === 'string'
    ? (parseLooseJson(observedArgsRaw) ?? observedArgsRaw)
    : observedArgsRaw;
  const observedAction = typeof observedToolCall?.action === 'string'
    ? observedToolCall.action
    : extractAction(observedArgs) || extractAction(llm?.raw_content);
  const requestedInput = llm?.requested_input;

  const toolMatch = observedToolName === expectedToolName;
  const observedArgsWithAction = (
    observedAction
    && observedArgs
    && typeof observedArgs === 'object'
    && !Array.isArray(observedArgs)
  )
    ? ({ ...(observedArgs as Record<string, unknown>), action: observedAction })
    : observedArgs;
  const requestedMatch = subsetMatch(requestedInput, expectedInputSubset);
  const observedMatch = subsetMatch(observedArgsWithAction, expectedInputSubset);
  const hasObservedArgsEvidence = (() => {
    if (observedArgs == null) return false;
    if (typeof observedArgs === 'string') return observedArgs.trim().length > 0;
    if (Array.isArray(observedArgs)) return observedArgs.length > 0;
    if (typeof observedArgs === 'object') return Object.keys(observedArgs as Record<string, unknown>).length > 0;
    return true;
  })();
  const missingObservedArgsEvidence = !hasObservedArgsEvidence;
  const llmSuccess = Boolean(llm?.success);
  const promptNativeArgsFallback = llmSuccess
    && toolMatch
    && requestedMatch
    && missingObservedArgsEvidence
    && isInstructionDrivenRuntime(detail.runtime_binding);

  stages.push({
    id: 'tool-example-runtime',
    label: 'Skill Example LLM Routing',
    status: !result
      ? 'pending'
      : !llm
        ? 'skip'
        : llmSuccess && toolMatch && requestedMatch && observedMatch
          ? 'pass'
          : promptNativeArgsFallback
            ? 'pass'
            : llmSuccess && toolMatch && requestedMatch && missingObservedArgsEvidence
              ? 'warn'
              : 'fail',
    summary: !result
      ? 'Collecting skill example routing verification...'
      : !llm
        ? 'No LLM routing data captured'
        : llmSuccess && toolMatch && requestedMatch && observedMatch
          ? `LLM selected ${expectedToolName} with expected example payload subset`
          : promptNativeArgsFallback
            ? 'LLM selected expected tool and payload snapshot; instruction-driven mode uses requested_input as routing evidence'
            : llmSuccess && toolMatch && requestedMatch && missingObservedArgsEvidence
              ? 'LLM selected expected tool and snapshot payload, but observed arguments evidence is missing'
              : 'LLM routing mismatch for selected skill example',
    details: result ? (
      <div className="space-y-1">
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Routing proof basis: expected example payload subset vs requested_input snapshot vs observed LLM tool call arguments.
        </div>
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Expected: tool <span className="font-mono">{expectedToolName}</span> · subset <span className="font-mono">{JSON.stringify(expectedInputSubset)}</span>
        </div>
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Snapshot sent to LLM: <span className="font-mono">{requestedInput ? JSON.stringify(requestedInput) : '(missing)'}</span>
        </div>
        <div className="text-[10px] rounded px-2 py-1 bg-gray-800/60 border border-gray-700/50 text-gray-300">
          Observed: tool <span className="font-mono">{observedToolName || '(none)'}</span> · args <span className="font-mono">{observedArgs ? JSON.stringify(observedArgs) : '(missing)'}</span>
        </div>
      </div>
    ) : undefined,
  });

  return stages;
};

export const buildCoreVerificationStages = (
  result: DryRunResultData | null,
  detail: SkillDetail,
): StageDraft[] => {
  const stages: StageDraft[] = [];

  const integLevel = detail.capability_tier ?? 'metadata';
  const rtStatus = detail.runtime_status ?? 'unavailable';
  const isExecutable = integLevel === 'executable';
  const isDegraded = rtStatus === 'degraded';

  const runtimeReadinessStatus: StageStatus = result
    ? (isExecutable && rtStatus === 'ready'
      ? 'pass'
      : isExecutable && isDegraded
        ? 'warn'
        : !isExecutable
          ? 'fail'
          : 'warn')
    : 'pending';

  stages.push({
    id: 'runtime-readiness',
    label: 'Runtime Readiness',
    status: runtimeReadinessStatus,
    summary: result
      ? (isExecutable && rtStatus === 'ready'
        ? `Fully operational — integration: ${integLevel}, status: ${rtStatus}`
        : !isExecutable
          ? `Not executable — integration level is "${integLevel}"`
          : `Partially operational — status: ${rtStatus}`)
      : 'Checking runtime readiness...',
    details: result && runtimeReadinessStatus !== 'pass' ? (
      <div className={`text-[10px] rounded px-2 py-1.5 ${
        runtimeReadinessStatus === 'fail'
          ? 'bg-red-900/20 border border-red-700/30 text-red-300'
          : 'bg-yellow-900/20 border border-yellow-700/30 text-yellow-300'
      }`}>
        {!isExecutable ? (
          <>
            <span className="font-medium">Blocked:</span> This skill has not reached <code className="px-1 py-0.5 bg-red-900/40 rounded text-[9px]">executable</code> integration level.
            It needs a runtime adapter, prompt instructions, or core executor binding before it can run. Current level: <code className="px-1 py-0.5 bg-red-900/40 rounded text-[9px]">{integLevel}</code>.
          </>
        ) : (
          <>
            <span className="font-medium">Degraded:</span> Skill has schema validation but the executor may not be fully operational.
            Dry-run validation will proceed but live execution may fail.
          </>
        )}
      </div>
    ) : undefined,
  });

  const schemaOk = result ? result.schema_errors.length === 0 : false;
  stages.push({
    id: 'schema',
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
    ? (result.policy_result === 'allow' ? 'pass'
      : result.policy_result === 'allow_with_consent' ? 'warn' : 'fail')
    : 'pending';
  stages.push({
    id: 'policy',
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

  return stages;
};

export const buildLlmVerificationStages = (
  result: DryRunResultData | null,
  detail: SkillDetail,
  liveMode: boolean,
): StageDraft[] => {
  if (!(liveMode || result?.llm_verification)) {
    return [];
  }

  const stages: StageDraft[] = [];
  const llm = result?.llm_verification;

  stages.push({
    id: 'llm-verify',
    label: 'LLM Verification',
    status: result
      ? (llm
        ? (llm.success ? 'pass' : 'fail')
        : 'skip')
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

  const toolExecPhase = result?.llm_verification?.phases?.find((p) => p.name === 'tool_execution');
  if (!toolExecPhase) {
    return stages;
  }

  const toolExecReason = String(toolExecPhase.data?.reason || toolExecPhase.data?.error || 'Skipped');
  const isPromptNoExecutorSkip =
    toolExecPhase.status === 'skip'
    && toolExecReason.toLowerCase() === 'no executor available'
    && isInstructionDrivenRuntime(detail.runtime_binding);

  stages.push({
    id: 'tool-execution',
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
      : isPromptNoExecutorSkip
        ? 'Skipped: prompt-native skill has no bound Python executor (expected for SKILL.md instructions + hooks mode)'
        : toolExecReason,
    details: isPromptNoExecutorSkip ? (
      <div className="text-[10px] rounded px-2 py-1.5 bg-cyan-900/20 border border-cyan-700/30 text-cyan-300">
        LLM tool-call negotiation is verified. Runtime execution is handled by prompt instructions and hook commands, so direct Python executor replay is intentionally skipped.
      </div>
    ) : undefined,
  });

  return stages;
};
