/**
 * Dry-run dialog — Center Stage M.
 *
 * Redesigned as a CI/CD pipeline-style verification timeline.
 * When the user clicks Execute, the pipeline shows all stages as "pending",
 * then progressively reveals each stage's actual result.
 *
 * Pipeline stages:
 *   1. Registration  — Is the skill loaded?
 *   2. Schema        — Does the input conform?
 *   3. Policy        — Is invocation allowed?
 *   4. Side Effects  — What side effects are declared?
 *   5. Hooks         — What lifecycle hooks are registered?
 *   6. LLM Verify    — Did the LLM produce the correct tool call? (live mode)
 */
import React, { useEffect, useMemo, useCallback, useState } from 'react';
import { CenterStage } from '../../CenterStage';
import type { DryRunWorkflowCandidate, SkillDetail, SkillTool } from '../../../types/websocket';
import type { DryRunResultData } from '../../LeftSidebar/useSkillsLogic';
import type { DryRunSchemaField } from './dryRun/SkillFieldInput';
import {
  computeStages,
  type DryRunPipelineContext,
} from './dryRun/computeStages';
import { PipelineStagePanel } from './dryRun/PipelineStagePanel';
import { DryRunInputSection } from './dryRun/DryRunInputSection';
import { DryRunExecuteBar } from './dryRun/DryRunExecuteBar';
import {
  DEFAULT_WEATHER_CITY_OPTIONS,
  WEATHER_CITY_SUGGESTIONS,
} from './dryRun/inputPresets';
import {
  applyToolSpecificValidation,
  getToolDryRunPresets,
  getToolFieldRules,
  isExternalPlanningWithFilesTool,
  isPlanningWithFilesTool,
  normalizeToolInput,
  PLANNING_FLOW_PRESETS,
  shouldShowActionHint,
  type ToolDryRunPreset,
} from './dryRun/dryRunToolRules';
import {
  buildExecutionInputFromForm,
  parseSchemaFields,
  presetToFormValues,
} from './dryRun/dryRunInputModel';
import {
  getLiveDefaultsForTool,
  getProviderModels,
} from './dryRun/liveLlmOptions';
import { useDryRunDialogState } from './dryRun/useDryRunDialogState';

const DEFAULT_PLANNING_FLOW_ID = PLANNING_FLOW_PRESETS[0]?.id ?? '';

interface WorkflowArgField {
  key: string;
  label: string;
  tokenIndex: number;
  defaultValue: string;
  kind: 'flag' | 'positional' | 'script' | 'operation';
}

const splitCommandTokens = (command: string): string[] => {
  const tokens: string[] = [];
  let current = '';
  let quote: '"' | "'" | null = null;
  let escaped = false;

  for (const ch of command.trim()) {
    if (escaped) {
      current += ch;
      escaped = false;
      continue;
    }

    if (ch === '\\') {
      escaped = true;
      continue;
    }

    if (quote) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }

    if (/\s/.test(ch)) {
      if (current) {
        tokens.push(current);
        current = '';
      }
      continue;
    }

    current += ch;
  }

  if (current) {
    tokens.push(current);
  }

  return tokens.filter(Boolean);
};

// Unified-example strategy notes:
// 1) package_examples are preferred because they usually carry richer semantic intent.
// 2) available_workflows are then mapped into examples so UI has a single selector surface.
// 3) tool presets are only used as a final fallback.
// 4) all candidates go through semantic dedupe so users do not see near-duplicates.
const canonicalCommandSignature = (commandTemplate: string): string => {
  const tokens = splitCommandTokens(commandTemplate);
  if (tokens.length === 0) return '';
  const script = tokens.find((token) => token.endsWith('.py')) ?? '';
  const scriptIndex = script ? tokens.indexOf(script) : -1;
  const operation = (scriptIndex >= 0 && scriptIndex + 1 < tokens.length && !tokens[scriptIndex + 1].startsWith('--'))
    ? tokens[scriptIndex + 1]
    : '';
  const flags = tokens
    .filter((token) => token.startsWith('--'))
    .map((token) => token.slice(2).replace(/-/g, '_'))
    .sort();
  return `script:${script}|op:${operation}|flags:${flags.join(',')}`;
};

const presetSemanticSignature = (toolName: string, preset: ToolDryRunPreset): string => {
  if (preset.origin === 'package_examples' || preset.origin === 'tool_preset') {
    return `${toolName}|id:${preset.id}`;
  }

  const action = typeof preset.input.action === 'string' ? preset.input.action : '';
  if (preset.command_template) {
    return `${toolName}|${action}|${canonicalCommandSignature(preset.command_template)}`;
  }

  // For non-command examples we include a stable value signature so distinct
  // package scenarios that share the same field keys are not collapsed.
  const stableInputSignature = Object.entries(preset.input ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => {
      if (Array.isArray(v)) return `${k}:[${v.map((item) => String(item)).join('|')}]`;
      if (v && typeof v === 'object') return `${k}:${JSON.stringify(v)}`;
      return `${k}:${String(v)}`;
    })
    .join(';');
  return `${toolName}|${action}|input:${stableInputSignature}`;
};

const presetPriority = (preset: ToolDryRunPreset): number => {
  if (preset.origin === 'package_examples') return 4;
  if (preset.origin === 'frontmatter_workflow') return 3;
  if (preset.origin === 'instruction_workflow') return 2;
  return 1;
};

const presetConfidenceScore = (preset: ToolDryRunPreset): number => {
  const breakdownScore = preset.confidence_breakdown?.score;
  if (typeof breakdownScore === 'number') return breakdownScore;
  if (preset.confidence === 'high') return 1;
  if (preset.confidence === 'medium') return 0.7;
  if (preset.confidence === 'low') return 0.4;
  return 0;
};

const dedupePresets = (toolName: string, candidates: ToolDryRunPreset[]): ToolDryRunPreset[] => {
  const merged = new Map<string, ToolDryRunPreset>();
  for (const candidate of candidates) {
    const key = presetSemanticSignature(toolName, candidate);
    const existing = merged.get(key);
    if (!existing) {
      merged.set(key, candidate);
      continue;
    }

    const existingRank = presetPriority(existing) * 10 + presetConfidenceScore(existing);
    const candidateRank = presetPriority(candidate) * 10 + presetConfidenceScore(candidate);
    if (candidateRank > existingRank) {
      merged.set(key, candidate);
    }
  }
  return Array.from(merged.values());
};

const extractWorkflowArgFields = (command: string): { tokens: string[]; fields: WorkflowArgField[] } => {
  const tokens = splitCommandTokens(command);
  if (tokens.length === 0) return { tokens, fields: [] };

  const fields: WorkflowArgField[] = [];
  const runnerPattern = tokens[1]?.endsWith('scripts/run.py');
  if (runnerPattern && tokens[2]) {
    fields.push({
      key: 'script',
      label: 'script',
      tokenIndex: 2,
      defaultValue: tokens[2],
      kind: 'script',
    });
    if (tokens[3] && !tokens[3].startsWith('--')) {
      fields.push({
        key: 'operation',
        label: 'operation',
        tokenIndex: 3,
        defaultValue: tokens[3],
        kind: 'operation',
      });
    }
  }

  const scriptIndex = tokens.findIndex((token) => token.endsWith('.py'));
  const startIndex = scriptIndex >= 0 ? scriptIndex + 1 : 1;
  let positionalIndex = 0;

  for (let i = startIndex; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token.startsWith('--')) {
      const flagName = token.slice(2).replace(/-/g, '_');
      const next = tokens[i + 1];
      if (next && !next.startsWith('--')) {
        fields.push({
          key: flagName,
          label: flagName,
          tokenIndex: i + 1,
          defaultValue: next,
          kind: 'flag',
        });
        i += 1;
      }
      continue;
    }

    if (runnerPattern && (i === 2 || i === 3)) {
      continue;
    }

    positionalIndex += 1;
    fields.push({
      key: `arg_${positionalIndex}`,
      label: `arg_${positionalIndex}`,
      tokenIndex: i,
      defaultValue: token,
      kind: 'positional',
    });
  }

  const deduped: WorkflowArgField[] = [];
  const seen = new Set<string>();
  for (const field of fields) {
    const unique = `${field.key}:${field.tokenIndex}`;
    if (seen.has(unique)) continue;
    seen.add(unique);
    deduped.push(field);
  }

  return { tokens, fields: deduped };
};

const renderCustomizedWorkflowCommand = (
  tokens: string[],
  fields: WorkflowArgField[],
  values: Record<string, string>,
): string => {
  if (tokens.length === 0) return '';
  const nextTokens = [...tokens];
  for (const field of fields) {
    const value = values[field.key];
    if (value === undefined || value === '') continue;
    if (field.tokenIndex < 0 || field.tokenIndex >= nextTokens.length) continue;
    nextTokens[field.tokenIndex] = value;
  }
  return nextTokens.join(' ');
};

const PLANNING_ACTION_OPTIONS = ['create', 'update', 'complete', 'status'] as const;

const inferPlanningActionFromCommand = (command?: string): string | null => {
  if (!command || typeof command !== 'string') return null;
  const normalized = command.trim();
  if (!normalized) return null;

  const slashMatch = normalized.match(/^\/plan\s+(create|update|complete|status)\b/i);
  if (slashMatch) {
    return slashMatch[1].toLowerCase();
  }

  const tokens = splitCommandTokens(normalized).map((token) => token.toLowerCase());
  for (let index = 0; index < tokens.length - 1; index += 1) {
    if (tokens[index] !== 'plan') continue;
    const next = tokens[index + 1];
    if (PLANNING_ACTION_OPTIONS.includes(next as (typeof PLANNING_ACTION_OPTIONS)[number])) {
      return next;
    }
  }

  return null;
};

const parseMissingDependenciesFromIssue = (issue?: string): string[] => {
  if (!issue) return [];
  const match = issue.match(/Missing dependency:\s*([^()]+)/i);
  if (!match?.[1]) return [];
  return match[1]
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
};

const parseExecutionResultObject = (executionResult?: string): Record<string, unknown> | null => {
  if (!executionResult || typeof executionResult !== 'string') return null;
  try {
    const parsed = JSON.parse(executionResult);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
};

type ClientPlatform = 'macos' | 'windows' | 'linux' | 'unknown';

const detectClientPlatform = (): ClientPlatform => {
  if (typeof navigator === 'undefined') return 'unknown';

  const nav = navigator as Navigator & { userAgentData?: { platform?: string } };
  const userAgentDataPlatform = typeof nav.userAgentData?.platform === 'string'
    ? nav.userAgentData.platform
    : '';
  const platformRaw = `${userAgentDataPlatform} ${navigator.platform ?? ''} ${navigator.userAgent ?? ''}`.toLowerCase();

  if (platformRaw.includes('mac')) return 'macos';
  if (platformRaw.includes('win')) return 'windows';
  if (platformRaw.includes('linux') || platformRaw.includes('x11')) return 'linux';

  return 'unknown';
};

const installCommandsForDependency = (dep: string, platform: ClientPlatform): string[] => {
  if (dep === 'soffice') {
    if (platform === 'windows') {
      return [
        'winget install TheDocumentFoundation.LibreOffice',
        'where soffice',
      ];
    }
    if (platform === 'linux') {
      return [
        'sudo apt-get update && sudo apt-get install -y libreoffice',
        'which soffice',
      ];
    }
    if (platform === 'macos') {
      return [
        'brew install --cask libreoffice',
        'which soffice',
      ];
    }
    return [
      'Install LibreOffice: https://www.libreoffice.org/download/download-libreoffice/',
      'soffice --version',
    ];
  }

  if (platform === 'windows') {
    return [
      `# Install dependency: ${dep}`,
      `where ${dep}`,
    ];
  }

  return [
    `# Install dependency: ${dep}`,
    `which ${dep}`,
  ];
};

// ─── Props ───────────────────────────────────────────────────────

export interface DryRunDialogProps {
  isOpen: boolean;
  detail: SkillDetail;
  dryRunResult: DryRunResultData | null;
  onExecute: (
    toolName: string,
    input: Record<string, unknown>,
    live?: boolean,
    options?: { llmProvider?: string; llmModel?: string; workflowId?: string },
  ) => void;
  onClose: () => void;
  onClearResult?: () => void;
}

// ─── Main dialog component ───────────────────────────────────────

export const DryRunDialog: React.FC<DryRunDialogProps> = ({
  isOpen,
  detail,
  dryRunResult,
  onExecute,
  onClose,
  onClearResult,
}) => {
  const getToolInputSchema = useCallback((toolName: string): Record<string, unknown> | undefined => {
    const tool = detail.tools.find((item) => item.name === toolName);
    const schema = tool?.input_schema;
    if (!schema || typeof schema !== 'object' || Array.isArray(schema)) {
      return undefined;
    }
    return schema as Record<string, unknown>;
  }, [detail.tools]);

  const isPlanningTool = useCallback((toolName: string): boolean => {
    return isPlanningWithFilesTool(toolName, { inputSchema: getToolInputSchema(toolName) });
  }, [getToolInputSchema]);

  const isExternalPlanningTool = useCallback((toolName: string): boolean => {
    return isExternalPlanningWithFilesTool(toolName, { inputSchema: getToolInputSchema(toolName) });
  }, [getToolInputSchema]);

  const getInitialPresetForTool = useCallback((toolName: string): ToolDryRunPreset | null => {
    return getToolDryRunPresets(toolName)[0] ?? null;
  }, []);

  const {
    selectedTool,
    inputMode,
    jsonInput,
    formValues,
    jsonError,
    formErrors,
    isExecuting,
    liveMode,
    liveLlmProvider,
    liveLlmModel,
    inputCollapsed,
    toolDescriptionExpanded,
    selectedExampleId,
    executedContext,
    revealedCount,
    setJsonError,
    setFormErrors,
    setIsExecuting,
    setLiveLlmProvider,
    setLiveLlmModel,
    setInputCollapsed,
    setToolDescriptionExpanded,
    setSelectedExampleId,
    setFormValues,
    setJsonInput,
    setExecutedContext,
    setRevealedCount,
    handleToolChange,
    handleSelectPreset,
    handleSwitchInputMode,
    handleJsonInputChange,
    handleToggleLiveMode,
    updateFormField,
  } = useDryRunDialogState({
    isOpen,
    detailTools: detail.tools,
    dryRunResult,
    onClearResult,
    defaultPlanningFlowId: DEFAULT_PLANNING_FLOW_ID,
    getInitialPresetForTool,
    getLiveDefaultsForTool,
    isPlanningWithFilesTool: isPlanningTool,
  });

  const packagePresets = useMemo<ToolDryRunPreset[]>(() => {
    const raw = detail.package_examples ?? [];
    const canonicalized = raw.map((preset): ToolDryRunPreset => {
      const basePreset = preset as ToolDryRunPreset;
      const hasAction = typeof (basePreset.input as Record<string, unknown>).action === 'string';
      const commandTemplate = typeof (basePreset.input as Record<string, unknown>).command === 'string'
        ? String((basePreset.input as Record<string, unknown>).command)
        : basePreset.command_template;
      if (!isExternalPlanningTool(selectedTool) || hasAction) {
        return {
          ...basePreset,
          origin: 'package_examples' as const,
          command_template: commandTemplate,
        };
      }

      const canonical = PLANNING_FLOW_PRESETS.find((p) => p.id === basePreset.id);
      const canonicalAction = typeof canonical?.input?.action === 'string'
        ? String(canonical.input.action)
        : undefined;
      const inferredAction = inferPlanningActionFromCommand(commandTemplate);

      if (!canonicalAction && !inferredAction) {
        return {
          ...basePreset,
          origin: 'package_examples' as const,
          command_template: commandTemplate,
        };
      }

      return {
        ...basePreset,
        origin: 'package_examples' as const,
        command_template: commandTemplate,
        input: {
          ...(canonicalAction ? { action: canonicalAction } : {}),
          ...(inferredAction ? { action: inferredAction } : {}),
          ...basePreset.input,
        },
      };
    });
    return canonicalized;
  }, [detail.package_examples, selectedTool, isExternalPlanningTool]);

  const availableWorkflows = useMemo<DryRunWorkflowCandidate[]>(() => {
    const fromResult = dryRunResult?.available_workflows ?? [];
    if (fromResult.length > 0) return fromResult;
    return detail.available_workflows ?? [];
  }, [dryRunResult, detail.available_workflows]);

  const workflowMappedPresets = useMemo<ToolDryRunPreset[]>(() => {
    return availableWorkflows.map((workflow) => ({
      id: workflow.id,
      label: workflow.title || workflow.id,
      description: workflow.command,
      input: {
        ...(isPlanningTool(selectedTool)
          ? { action: inferPlanningActionFromCommand(workflow.command) ?? 'create' }
          : {}),
        workflow_id: workflow.id,
        command: workflow.command,
      },
      command_template: workflow.command,
      workflow_id: workflow.id,
      origin: workflow.source === 'frontmatter' ? 'frontmatter_workflow' : 'instruction_workflow',
      expectedFocus: [workflow.evidence || workflow.command],
      objective: `Validate workflow candidate: ${workflow.title || workflow.id}.`,
      source: workflow.source,
      evidence: workflow.evidence,
      confidence: workflow.confidence,
      confidence_reason: workflow.validation?.issues?.join(' | ') || undefined,
      confidence_breakdown: typeof workflow.confidence_score === 'number'
        ? { score: workflow.confidence_score }
        : undefined,
      validation_status: workflow.validation?.status,
      validation_issues: workflow.validation?.issues,
    }));
  }, [availableWorkflows, isPlanningTool, selectedTool]);

  const toolPresets = useMemo<ToolDryRunPreset[]>(
    () => getToolDryRunPresets(selectedTool).map((preset) => ({
      ...preset,
      origin: 'tool_preset',
      command_template: typeof (preset.input as Record<string, unknown>).command === 'string'
        ? String((preset.input as Record<string, unknown>).command)
        : preset.command_template,
    })),
    [selectedTool],
  );

  const availablePresets = useMemo<ToolDryRunPreset[]>(
    () => {
      const primary = [...packagePresets, ...workflowMappedPresets];
      // Keep tool presets as the last fallback only, except external
      // planning where canonical action presets must always stay available.
      const source = isExternalPlanningTool(selectedTool)
        ? [...primary, ...toolPresets]
        : (primary.length > 0 ? primary : toolPresets);
      return dedupePresets(selectedTool, source);
    },
    [selectedTool, packagePresets, workflowMappedPresets, toolPresets, isExternalPlanningTool],
  );
  const presetModeEnabled = availablePresets.length > 0;
  const activePreset = useMemo<ToolDryRunPreset | null>(
    () => availablePresets.find((p) => p.id === selectedExampleId) ?? availablePresets[0] ?? null,
    [availablePresets, selectedExampleId],
  );

  useEffect(() => {
    if (availablePresets.length === 0) return;
    if (availablePresets.some((p) => p.id === selectedExampleId)) return;
    setSelectedExampleId(availablePresets[0].id);
  }, [availablePresets, selectedExampleId, setSelectedExampleId]);

  const liveModelOptions = useMemo(() => getProviderModels(liveLlmProvider), [liveLlmProvider]);
  const [workflowArgValues, setWorkflowArgValues] = useState<Record<string, string>>({});
  const [installCommandCopyState, setInstallCommandCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const clientPlatform = useMemo<ClientPlatform>(() => detectClientPlatform(), []);

  const workflowCommandModel = useMemo(() => {
    if (!activePreset?.command_template) return { tokens: [], fields: [] as WorkflowArgField[] };
    return extractWorkflowArgFields(activePreset.command_template);
  }, [activePreset?.command_template]);

  useEffect(() => {
    if (workflowCommandModel.fields.length === 0) {
      setWorkflowArgValues({});
      return;
    }
    const initial: Record<string, string> = {};
    for (const field of workflowCommandModel.fields) {
      initial[field.key] = field.defaultValue;
    }
    setWorkflowArgValues(initial);
  }, [activePreset?.id, workflowCommandModel.fields]);

  const customizedWorkflowCommand = useMemo(() => {
    if (!activePreset?.command_template) return '';
    return renderCustomizedWorkflowCommand(
      workflowCommandModel.tokens,
      workflowCommandModel.fields,
      workflowArgValues,
    ) || activePreset.command_template;
  }, [activePreset?.command_template, workflowCommandModel, workflowArgValues]);

  useEffect(() => {
    if (!liveLlmProvider) {
      if (liveLlmModel !== '') setLiveLlmModel('');
      return;
    }
    if (liveModelOptions.length === 0) {
      if (liveLlmModel !== '') setLiveLlmModel('');
      return;
    }
    if (!liveModelOptions.includes(liveLlmModel)) {
      setLiveLlmModel(liveModelOptions[0]);
    }
  }, [liveLlmProvider, liveLlmModel, liveModelOptions]);

  useEffect(() => {
    if (!presetModeEnabled || !activePreset) return;
    const preset = activePreset;
    if (!preset) return;
    setFormValues(presetToFormValues(preset));
    setJsonInput(JSON.stringify(preset.input, null, 2));
  }, [presetModeEnabled, activePreset]);

  const currentPipelineContext = useMemo<DryRunPipelineContext>(() => ({
    planningFlowId: isExternalPlanningTool(selectedTool) ? selectedExampleId : null,
    planningFlowLabel: isExternalPlanningTool(selectedTool) ? activePreset?.label ?? null : null,
    selectedExampleId: activePreset?.id ?? null,
    selectedExampleLabel: activePreset?.label ?? null,
    selectedExampleInput: activePreset?.input ?? null,
    selectedExampleObjective: activePreset?.objective ?? null,
    selectedToolName: selectedTool,
  }), [selectedExampleId, activePreset, selectedTool, isExternalPlanningTool]);

  const stageContext = dryRunResult ? (executedContext ?? currentPipelineContext) : currentPipelineContext;

  // Progressive reveal: when result arrives, reveal stages one by one
  useEffect(() => {
    if (dryRunResult) {
      setIsExecuting(false);
      setInputCollapsed(true);
      setRevealedCount(0);
      const contextForReveal = executedContext ?? currentPipelineContext;
      const totalStages = computeStages(dryRunResult, detail, liveMode, contextForReveal).length;
      let count = 0;
      const timer = setInterval(() => {
        count += 1;
        setRevealedCount(count);
        if (count >= totalStages) clearInterval(timer);
      }, 200);
      return () => clearInterval(timer);
    } else {
      setRevealedCount(0);
    }
  }, [dryRunResult, detail, liveMode, executedContext]);

  const currentTool: SkillTool | undefined = useMemo(
    () => detail.tools.find((t) => t.name === selectedTool),
    [detail.tools, selectedTool],
  );
  const currentToolDescription = currentTool?.description?.trim() ?? '';
  const isLongToolDescription = currentToolDescription.length > 260 || currentToolDescription.split('\n').length > 5;
  const toolDescriptionPreview = isLongToolDescription
    ? `${currentToolDescription.slice(0, 220).trimEnd()}...`
    : currentToolDescription;
  const selectedToolSchema = useMemo(
    () => getToolInputSchema(selectedTool),
    [getToolInputSchema, selectedTool],
  );
  const presetModeForExternalPlanning = presetModeEnabled && isExternalPlanningTool(selectedTool);
  const isWeatherTool = currentTool?.name === 'get_weather';
  const isLocationTool = currentTool?.name === 'get_location';
  const isWebSearchTool = currentTool?.name === 'web_search';
  const isInstructionDrivenRuntime = detail.runtime_binding === 'prompt_instructions' || detail.runtime_binding === 'script_executor_compat';

  useEffect(() => {
    setToolDescriptionExpanded(false);
  }, [selectedTool, isOpen]);

  const schemaFields = useMemo<DryRunSchemaField[]>(
    () => parseSchemaFields(currentTool?.input_schema as Record<string, unknown> | undefined),
    [currentTool],
  );

  const getFieldRules = useCallback((field: DryRunSchemaField): { visible: boolean; required: boolean } => {
    return getToolFieldRules(selectedTool, field, formValues, { inputSchema: selectedToolSchema });
  }, [selectedTool, formValues, selectedToolSchema]);

  const visibleSchemaFields = useMemo(() => (
    schemaFields
      .map((field) => {
        const rules = getFieldRules(field);
        const forceHideAction = presetModeForExternalPlanning && field.name === 'action';
        return {
          ...field,
          required: rules.required,
          visible: forceHideAction ? false : rules.visible,
        };
      })
      .filter((field) => field.visible)
      .sort((a, b) => (a.required === b.required ? 0 : a.required ? -1 : 1))
  ), [schemaFields, getFieldRules, presetModeForExternalPlanning]);

  const selectedExecutionPayloadPreview = useMemo<Record<string, unknown> | null>(() => {
    if (!activePreset) return null;

    const payload: Record<string, unknown> = {
      ...activePreset.input,
    };

    if (inputMode === 'json') {
      try {
        const parsed = JSON.parse(jsonInput);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          Object.assign(payload, parsed as Record<string, unknown>);
        }
      } catch {
        // Keep preset baseline when JSON is invalid while typing.
      }
    } else {
      const previewErrors: Record<string, string> = {};
      const formInput = buildExecutionInputFromForm(visibleSchemaFields, formValues, previewErrors);
      Object.assign(payload, formInput);
    }

    if (activePreset?.workflow_id) {
      payload.workflow_id = activePreset.workflow_id;
    }
    if (activePreset?.command_template && !isPlanningTool(selectedTool)) {
      payload.command = customizedWorkflowCommand || activePreset.command_template;
    }
    return normalizeToolInput(selectedTool, payload, { inputSchema: selectedToolSchema });
  }, [
    selectedTool,
    activePreset,
    inputMode,
    jsonInput,
    visibleSchemaFields,
    formValues,
    customizedWorkflowCommand,
    selectedToolSchema,
    isPlanningTool,
  ]);

  const handleSwitchInputModeWithPresetSync = useCallback((mode: 'form' | 'json') => {
    handleSwitchInputMode(mode);
    if (mode !== 'json') return;
    if (!presetModeEnabled || !activePreset) return;

    const payloadForJson = selectedExecutionPayloadPreview ?? activePreset.input;
    setJsonInput(JSON.stringify(payloadForJson, null, 2));
  }, [
    handleSwitchInputMode,
    presetModeEnabled,
    activePreset,
    selectedExecutionPayloadPreview,
    setJsonInput,
  ]);

  const missingDependencies = useMemo<string[]>(() => {
    const deps = new Set<string>();

    const matchedWorkflow = activePreset?.workflow_id
      ? availableWorkflows.find((workflow) => workflow.id === activePreset.workflow_id)
      : null;

    for (const dep of matchedWorkflow?.validation?.missing_dependencies ?? []) {
      if (dep) deps.add(dep);
    }

    for (const issue of activePreset?.validation_issues ?? []) {
      for (const dep of parseMissingDependenciesFromIssue(issue)) {
        deps.add(dep);
      }
    }

    const executionResult = parseExecutionResultObject(dryRunResult?.llm_verification?.execution_result);
    if (executionResult?.error_code === 'missing_dependency') {
      const runtimeMissingDeps = executionResult.missing_dependencies;
      if (Array.isArray(runtimeMissingDeps)) {
        for (const dep of runtimeMissingDeps) {
          if (typeof dep === 'string' && dep.trim()) deps.add(dep.trim());
        }
      }
    }

    return Array.from(deps);
  }, [activePreset, availableWorkflows, dryRunResult]);

  const installCommands = useMemo<string[]>(() => {
    const commands = new Set<string>();
    for (const dep of missingDependencies) {
      for (const command of installCommandsForDependency(dep, clientPlatform)) {
        commands.add(command);
      }
    }
    return Array.from(commands);
  }, [clientPlatform, missingDependencies]);

  useEffect(() => {
    setInstallCommandCopyState('idle');
  }, [installCommands]);

  const handleCopyInstallCommands = useCallback(async () => {
    const payload = installCommands.join('\n');
    if (!payload) {
      setInstallCommandCopyState('failed');
      return;
    }
    try {
      if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
        throw new Error('Clipboard API unavailable');
      }
      await navigator.clipboard.writeText(payload);
      setInstallCommandCopyState('copied');
    } catch {
      setInstallCommandCopyState('failed');
    }
  }, [installCommands]);

  const handleExecute = useCallback(() => {
    let input: Record<string, unknown> = {};
    if (inputMode === 'json') {
      try {
        input = JSON.parse(jsonInput);
        setJsonError(null);
      } catch (e) {
        setJsonError(`Invalid JSON: ${(e as Error).message}`);
        return;
      }
      const errors: Record<string, string> = {};
      applyToolSpecificValidation(selectedTool, input, errors, { inputSchema: selectedToolSchema });
      if (Object.keys(errors).length > 0) {
        setFormErrors(errors);
        return;
      }
      setFormErrors({});
    } else {
      const errors: Record<string, string> = {};
      input = buildExecutionInputFromForm(visibleSchemaFields, formValues, errors);
      applyToolSpecificValidation(selectedTool, input, errors, { inputSchema: selectedToolSchema });
      if (Object.keys(errors).length > 0) {
        setFormErrors(errors);
        return;
      }
      setFormErrors({});
    }
    if (activePreset) {
      input = {
        ...activePreset.input,
        ...input,
      };
      if (activePreset.workflow_id) {
        input.workflow_id = activePreset.workflow_id;
      }
      if (activePreset.command_template && !isPlanningTool(selectedTool)) {
        input.command = customizedWorkflowCommand || activePreset.command_template;
      }
    }
    input = normalizeToolInput(selectedTool, input, { inputSchema: selectedToolSchema });
    setExecutedContext(currentPipelineContext);
    setIsExecuting(true);
    setRevealedCount(0);
    onClearResult?.();
    const provider = liveLlmProvider.trim();
    const model = liveLlmModel.trim();
    const workflowId = activePreset?.workflow_id;
    if (liveMode && (provider || model)) {
      onExecute(selectedTool, input, liveMode, {
        llmProvider: provider || undefined,
        llmModel: model || undefined,
        workflowId: workflowId || undefined,
      });
      return;
    }
    onExecute(selectedTool, input, liveMode, {
      workflowId: workflowId || undefined,
    });
  }, [
    inputMode,
    jsonInput,
    visibleSchemaFields,
    formValues,
    selectedTool,
    liveMode,
    liveLlmProvider,
    liveLlmModel,
    onExecute,
    onClearResult,
    currentPipelineContext,
    activePreset,
    customizedWorkflowCommand,
    selectedToolSchema,
    isPlanningTool,
  ]);

  const selectedCountry = (formValues.country || '').toUpperCase();
  const weatherCityOptions = useMemo(() => {
    if (!isWeatherTool) return [];
    if (selectedCountry && WEATHER_CITY_SUGGESTIONS[selectedCountry]) {
      return WEATHER_CITY_SUGGESTIONS[selectedCountry];
    }
    return DEFAULT_WEATHER_CITY_OPTIONS;
  }, [isWeatherTool, selectedCountry]);

  // ─── Compute pipeline stages ───────────────────────────────────

  const pipelineStages = useMemo(
    () => (dryRunResult || isExecuting)
      ? computeStages(dryRunResult, detail, liveMode, stageContext)
      : [],
    [dryRunResult, isExecuting, detail, liveMode, stageContext],
  );

  const showPipeline = isExecuting || !!dryRunResult;

  return (
    <CenterStage
      isOpen={isOpen}
      onClose={onClose}
      size="M"
      title={`Dry-run: ${detail.display_name || detail.name}`}
    >
      <div className="space-y-3" data-testid="dry-run-dialog">
        <DryRunInputSection
          showPipeline={showPipeline}
          inputCollapsed={inputCollapsed}
          selectedTool={selectedTool}
          liveMode={liveMode}
          detailTools={detail.tools}
          currentTool={currentTool}
          currentToolDescription={currentToolDescription}
          isLongToolDescription={isLongToolDescription}
          toolDescriptionPreview={toolDescriptionPreview}
          toolDescriptionExpanded={toolDescriptionExpanded}
          presetModeEnabled={presetModeEnabled}
          presetModeForExternalPlanning={presetModeForExternalPlanning}
          availablePresets={availablePresets}
          workflowArgFields={workflowCommandModel.fields}
          workflowArgValues={workflowArgValues}
          selectedExampleId={selectedExampleId}
          selectedExecutionPayloadPreview={selectedExecutionPayloadPreview}
          inputMode={inputMode}
          schemaFields={schemaFields}
          isInstructionDrivenRuntime={isInstructionDrivenRuntime}
          instructionsLength={detail.instructions_length}
          isPlanningWithFilesSelectedTool={isPlanningTool(selectedTool)}
          shouldShowActionHintForCurrentTool={shouldShowActionHint(selectedTool, formValues, { inputSchema: selectedToolSchema })}
          visibleSchemaFields={visibleSchemaFields}
          formValues={formValues}
          formErrors={formErrors}
          weatherCityOptions={weatherCityOptions}
          isWeatherTool={isWeatherTool}
          isLocationTool={isLocationTool}
          isWebSearchTool={isWebSearchTool}
          jsonInput={jsonInput}
          jsonError={jsonError}
          onToggleCollapsed={() => setInputCollapsed((v) => !v)}
          onToolChange={handleToolChange}
          onToggleToolDescriptionExpanded={() => setToolDescriptionExpanded((v) => !v)}
          onSelectPreset={handleSelectPreset}
          onUpdateWorkflowArg={(fieldKey: string, value: string) => {
            setWorkflowArgValues((prev) => ({ ...prev, [fieldKey]: value }));
          }}
          onSwitchInputMode={handleSwitchInputModeWithPresetSync}
          onUpdateFormField={updateFormField}
          onJsonInputChange={handleJsonInputChange}
        />

        <DryRunExecuteBar
          liveMode={liveMode}
          isExecuting={isExecuting}
          selectedTool={selectedTool}
          liveLlmProvider={liveLlmProvider}
          liveLlmModel={liveLlmModel}
          liveModelOptions={liveModelOptions}
          missingDependencies={missingDependencies}
          installCommands={installCommands}
          installCommandCopyState={installCommandCopyState}
          onToggleLiveMode={handleToggleLiveMode}
          onChangeLiveProvider={setLiveLlmProvider}
          onChangeLiveModel={setLiveLlmModel}
          onCopyInstallCommands={() => {
            void handleCopyInstallCommands();
          }}
          onExecute={handleExecute}
        />

        {/* ── Verification Pipeline ───────────────────────────── */}
        <PipelineStagePanel
          showPipeline={showPipeline}
          dryRunResult={dryRunResult}
          isExecuting={isExecuting}
          pipelineStages={pipelineStages}
          revealedCount={revealedCount}
        />
      </div>
    </CenterStage>
  );
};

