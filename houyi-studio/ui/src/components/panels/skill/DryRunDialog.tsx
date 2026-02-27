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
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { CenterStage } from '../../CenterStage';
import {
  Play, AlertCircle, CheckCircle, XCircle, Circle, Loader2,
  ChevronDown, ChevronUp, Zap, AlertTriangle,
} from 'lucide-react';
import type { SkillDetail, SkillTool } from '../../../types/websocket';
import type { DryRunResultData } from '../../LeftSidebar/useSkillsLogic';
import { SkillFieldInput, type DryRunSchemaField } from './dryRun/SkillFieldInput';
import {
  computeStages,
  type PipelineStage,
  type StageStatus,
} from './dryRun/computeStages';
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

const DEFAULT_PLANNING_FLOW_ID = PLANNING_FLOW_PRESETS[0]?.id ?? '';

type LiveLlmProviderOption = {
  value: string;
  label: string;
  models: string[];
};

const LIVE_LLM_PROVIDER_OPTIONS: LiveLlmProviderOption[] = [
  { value: '', label: 'provider: default', models: [] },
  { value: 'vertex', label: 'provider: vertex', models: ['gemini-2.5-pro', 'gemini-2.5-flash'] },
  { value: 'google_ai', label: 'provider: google_ai', models: ['gemini-2.5-pro', 'gemini-2.5-flash'] },
  { value: 'openai', label: 'provider: openai', models: ['gpt-4o', 'gpt-4.1-mini'] },
  { value: 'siliconflow', label: 'provider: siliconflow', models: ['deepseek-chat', 'deepseek-reasoner'] },
  { value: 'openai_compat', label: 'provider: openai_compat', models: ['deepseek-chat', 'qwen-plus'] },
  { value: 'deepseek', label: 'provider: deepseek', models: ['deepseek-chat', 'deepseek-reasoner'] },
];

const getProviderModels = (provider: string): string[] => (
  LIVE_LLM_PROVIDER_OPTIONS.find((opt) => opt.value === provider)?.models ?? []
);

const presetToFormValues = (preset: ToolDryRunPreset | null): Record<string, string> => {
  if (!preset) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(preset.input ?? {})) {
    if (Array.isArray(v)) {
      out[k] = v.join(', ');
    } else if (typeof v === 'boolean') {
      out[k] = v ? 'true' : 'false';
    } else if (v !== undefined && v !== null) {
      out[k] = String(v);
    }
  }
  return out;
};

// ─── Stage status icon ───────────────────────────────────────────

const StageIcon: React.FC<{ status: StageStatus }> = ({ status }) => {
  switch (status) {
    case 'pass':
      return <CheckCircle size={16} className="text-green-400" />;
    case 'fail':
      return <XCircle size={16} className="text-red-400" />;
    case 'warn':
      return <AlertTriangle size={16} className="text-yellow-400" />;
    case 'running':
      return <Loader2 size={16} className="text-blue-400 animate-spin" />;
    case 'skip':
      return <Circle size={16} className="text-gray-600" />;
    case 'pending':
    default:
      return <Circle size={16} className="text-gray-600" />;
  }
};

const statusColor: Record<StageStatus, string> = {
  pass: 'border-green-500/50 bg-green-500/5',
  fail: 'border-red-500/50 bg-red-500/5',
  warn: 'border-yellow-500/50 bg-yellow-500/5',
  running: 'border-blue-500/50 bg-blue-500/5',
  skip: 'border-gray-700/50 bg-gray-800/30',
  pending: 'border-gray-700/50 bg-gray-800/30',
};

const statusText: Record<StageStatus, string> = {
  pass: 'text-green-400',
  fail: 'text-red-400',
  warn: 'text-yellow-400',
  running: 'text-blue-400',
  skip: 'text-gray-600',
  pending: 'text-gray-600',
};

// ─── Single pipeline stage row ───────────────────────────────────

const PipelineStageRow: React.FC<{
  stage: PipelineStage;
  isLast: boolean;
  revealed: boolean;
}> = ({ stage, isLast, revealed }) => {
  const active = revealed && stage.status !== 'pending';
  return (
    <div
      className={`relative flex gap-3 ${revealed ? 'animate-stageReveal' : 'opacity-40'}`}
      data-testid={`dry-run-stage-${stage.id}`}
    >
      {/* Vertical connector line */}
      <div className="flex flex-col items-center shrink-0 w-5">
        <div className={`mt-0.5 transition-all duration-300 ${active ? '' : 'grayscale opacity-50'}`}>
          <StageIcon status={revealed ? stage.status : 'pending'} />
        </div>
        {!isLast && (
          <div className={`w-px flex-1 min-h-[16px] mt-1 transition-colors duration-300 ${
            active ? 'bg-gray-600' : 'bg-gray-700/40'
          }`} />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 pb-4 ${isLast ? 'pb-0' : ''}`}>
        <div className="flex items-center gap-2 mb-0.5">
          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
            active ? statusColor[stage.status] : 'bg-gray-800/30 border border-gray-700/50'
          } border`}>
            {stage.number}
          </span>
          <span className={`text-xs font-medium ${active ? 'text-gray-200' : 'text-gray-500'}`}>
            {stage.label}
          </span>
          {active && (
            <span className={`text-[10px] font-medium uppercase tracking-wider ${statusText[stage.status]}`}>
              {stage.status === 'pass' ? 'PASS' :
               stage.status === 'fail' ? 'FAIL' :
               stage.status === 'warn' ? 'WARN' :
               stage.status === 'running' ? 'RUNNING' :
               stage.status === 'skip' ? 'SKIPPED' : ''}
            </span>
          )}
        </div>
        <div className={`text-[11px] ${active ? 'text-gray-400' : 'text-gray-600'}`}>
          {revealed ? stage.summary : 'Waiting...'}
        </div>
        {revealed && stage.details && (
          <div className="mt-1.5">
            {stage.details}
          </div>
        )}
      </div>
    </div>
  );
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
    options?: { llmProvider?: string; llmModel?: string },
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
  const [selectedTool, setSelectedTool] = useState<string>('');
  const [inputMode, setInputMode] = useState<'form' | 'json'>('form');
  const [jsonInput, setJsonInput] = useState('{}');
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [liveLlmProvider, setLiveLlmProvider] = useState('');
  const [liveLlmModel, setLiveLlmModel] = useState('');
  const [inputCollapsed, setInputCollapsed] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState<string>(DEFAULT_PLANNING_FLOW_ID);

  // How many stages have been revealed with their actual status
  const [revealedCount, setRevealedCount] = useState(0);
  const packagePresets = useMemo<ToolDryRunPreset[]>(
    () => detail.package_examples ?? [],
    [detail.package_examples],
  );
  const availablePresets = useMemo<ToolDryRunPreset[]>(
    () => (packagePresets.length > 0 ? packagePresets : getToolDryRunPresets(selectedTool)),
    [packagePresets, selectedTool],
  );
  const presetModeEnabled = availablePresets.length > 0;
  const activePreset = useMemo<ToolDryRunPreset | null>(
    () => availablePresets.find((p) => p.id === selectedExampleId) ?? availablePresets[0] ?? null,
    [availablePresets, selectedExampleId],
  );

  const getInitialPresetForTool = useCallback((toolName: string): ToolDryRunPreset | null => {
    const packagePreset = (detail.package_examples ?? [])[0] ?? null;
    if (packagePreset) return packagePreset;
    return getToolDryRunPresets(toolName)[0] ?? null;
  }, [detail.package_examples]);

  const getLiveDefaultsForTool = useCallback((toolName: string): { provider: string; model: string } => {
    if (toolName === 'notebooklm') {
      return { provider: 'vertex', model: 'gemini-2.5-pro' };
    }
    return { provider: '', model: '' };
  }, []);

  const liveModelOptions = useMemo(() => getProviderModels(liveLlmProvider), [liveLlmProvider]);

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

  // Select first tool by default
  useEffect(() => {
    if (isOpen && detail.tools.length > 0 && !selectedTool) {
      setSelectedTool(detail.tools[0].name);
    }
  }, [isOpen, detail.tools, selectedTool]);

  useEffect(() => {
    if (!presetModeEnabled || !activePreset) return;
    const preset = activePreset;
    if (!preset) return;
    setFormValues(presetToFormValues(preset));
    setJsonInput(JSON.stringify(preset.input, null, 2));
  }, [presetModeEnabled, activePreset]);

  // Progressive reveal: when result arrives, reveal stages one by one
  useEffect(() => {
    if (dryRunResult) {
      setIsExecuting(false);
      setInputCollapsed(true);
      setRevealedCount(0);
      const isPresetMode = presetModeEnabled;
      const selectedPreset = isPresetMode ? activePreset : null;
      const totalStages = computeStages(dryRunResult, detail, liveMode, {
        planningFlowId: isExternalPlanningWithFilesTool(selectedTool) ? selectedExampleId : null,
        planningFlowLabel: isExternalPlanningWithFilesTool(selectedTool)
          ? (selectedPreset?.label ?? null)
          : null,
        selectedExampleId: selectedPreset?.id ?? null,
        selectedExampleLabel: selectedPreset?.label ?? null,
        selectedToolName: selectedTool,
      }).length;
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
  }, [dryRunResult, detail, liveMode, selectedExampleId, selectedTool]);

  // Reset state on open
  useEffect(() => {
    if (isOpen) {
      setJsonError(null);
      setFormErrors({});
      setIsExecuting(false);
      setLiveMode(false);
      setLiveLlmProvider('');
      setLiveLlmModel('');
      setInputCollapsed(false);
      setRevealedCount(0);
      onClearResult?.();
      if (detail.tools.length > 0) {
        const nextTool = detail.tools[0].name;
        const initialPreset = getInitialPresetForTool(nextTool);
        const liveDefaults = getLiveDefaultsForTool(nextTool);
        setSelectedTool(nextTool);
        setLiveLlmProvider(liveDefaults.provider);
        setLiveLlmModel(liveDefaults.model);
        setSelectedExampleId(initialPreset?.id ?? '');
        if (initialPreset) {
          setFormValues(presetToFormValues(initialPreset));
          setJsonInput(JSON.stringify(initialPreset.input, null, 2));
        } else {
          setFormValues({});
          setJsonInput('{}');
        }
      } else {
        setFormValues({});
        setJsonInput('{}');
      }
    }
  }, [isOpen, detail.tools, onClearResult, getInitialPresetForTool, getLiveDefaultsForTool]);

  const currentTool: SkillTool | undefined = useMemo(
    () => detail.tools.find((t) => t.name === selectedTool),
    [detail.tools, selectedTool],
  );
  const presetModeForExternalPlanning = presetModeEnabled && isExternalPlanningWithFilesTool(selectedTool);
  const isWeatherTool = currentTool?.name === 'get_weather';
  const isLocationTool = currentTool?.name === 'get_location';
  const isWebSearchTool = currentTool?.name === 'web_search';

  // Extract fields from input_schema, handling Pydantic's anyOf pattern
  const schemaFields = useMemo<DryRunSchemaField[]>(() => {
    if (!currentTool?.input_schema) return [];
    const schema = currentTool.input_schema;
    const props = schema.properties || {};
    const required = new Set(schema.required || []);

    const resolveRef = (ref: string): Record<string, unknown> => {
      if (!ref.startsWith('#/')) return {};
      const path = ref.slice(2).split('/');
      let cursor: unknown = schema;
      for (const segment of path) {
        if (!cursor || typeof cursor !== 'object' || !(segment in cursor)) {
          return {};
        }
        cursor = (cursor as Record<string, unknown>)[segment];
      }
      return (cursor && typeof cursor === 'object') ? (cursor as Record<string, unknown>) : {};
    };

    const resolveNode = (node: unknown): Record<string, unknown> => {
      if (!node || typeof node !== 'object') return {};
      const raw = node as Record<string, unknown>;
      const ref = typeof raw.$ref === 'string' ? resolveRef(raw.$ref) : {};
      const rest = { ...raw };
      delete rest.$ref;
      return { ...ref, ...rest };
    };

    return Object.entries(props).map(([name, def]) => {
      const d = resolveNode(def);
      let fieldType = d.type as string | undefined;
      let fieldFormat = d.format as string | undefined;
      let nullable = false;
      let enumValues = d.enum as string[] | undefined;

      if (!fieldType && Array.isArray(d.anyOf)) {
        const variants = (d.anyOf as Array<unknown>).map(resolveNode);
        const types = variants.map((t) => t.type as string).filter(Boolean);
        const nonNullTypes = types.filter((t) => t !== 'null');
        nullable = types.includes('null');
        fieldType = nonNullTypes[0] || 'string';
        if (!fieldFormat) {
          const withFormat = variants.find((v) => typeof v.format === 'string');
          fieldFormat = withFormat?.format as string | undefined;
        }
        if (!enumValues) {
          for (const v of variants) {
            if (Array.isArray(v.enum)) {
              enumValues = v.enum as string[];
              break;
            }
          }
        }
      }

      const defaultVal = d.default;
      const isRequired = required.has(name) && !nullable && defaultVal === undefined;
      const hasDefault = defaultVal !== undefined && defaultVal !== null;

      return {
        name,
        type: fieldType || 'string',
        format: fieldFormat,
        title: (d.title as string) || '',
        description: (d.description as string) || '',
        required: isRequired,
        default: hasDefault ? defaultVal : undefined,
        defaultRaw: defaultVal,
        nullable,
        enum: enumValues,
        minimum: d.minimum as number | undefined,
        maximum: d.maximum as number | undefined,
      };
    });
  }, [currentTool]);

  // Track validation errors
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const getFieldRules = useCallback((field: DryRunSchemaField): { visible: boolean; required: boolean } => {
    return getToolFieldRules(selectedTool, field, formValues);
  }, [selectedTool, formValues.action]);

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

  const updateFormField = (fieldName: string, value: string) => {
    setFormValues((prev) => ({ ...prev, [fieldName]: value }));
    if (formErrors[fieldName]) {
      setFormErrors((prev) => {
        const next = { ...prev };
        delete next[fieldName];
        return next;
      });
    }
    if (fieldName === 'action' && isPlanningWithFilesTool(selectedTool)) {
      setFormErrors({});
    }
  };

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
      applyToolSpecificValidation(selectedTool, input, errors);
      if (Object.keys(errors).length > 0) {
        setFormErrors(errors);
        return;
      }
      setFormErrors({});
    } else {
      const errors: Record<string, string> = {};
      for (const field of schemaFields) {
        const rules = getFieldRules(field);
        if (!rules.visible) continue;

        const val = formValues[field.name];
        if (rules.required && (val === undefined || val === '')) {
          errors[field.name] = 'This field is required';
        }
        if (val !== undefined && val !== '') {
          if (field.type === 'number' || field.type === 'integer') {
            input[field.name] = Number(val);
          } else if (field.type === 'boolean') {
            input[field.name] = val === 'true';
          } else {
            input[field.name] = val;
          }
        }
      }
      applyToolSpecificValidation(selectedTool, input, errors);
      if (Object.keys(errors).length > 0) {
        setFormErrors(errors);
        return;
      }
      setFormErrors({});
    }
    if (presetModeEnabled && activePreset) {
      const forced = activePreset.input;
      input = {
        ...input,
        ...forced,
      };
    }
    input = normalizeToolInput(selectedTool, input);
    setIsExecuting(true);
    setRevealedCount(0);
    onClearResult?.();
    const provider = liveLlmProvider.trim();
    const model = liveLlmModel.trim();
    if (liveMode && (provider || model)) {
      onExecute(selectedTool, input, liveMode, {
        llmProvider: provider || undefined,
        llmModel: model || undefined,
      });
      return;
    }
    onExecute(selectedTool, input, liveMode);
  }, [
    inputMode,
    jsonInput,
    schemaFields,
    formValues,
    getFieldRules,
    selectedTool,
    liveMode,
    liveLlmProvider,
    liveLlmModel,
    onExecute,
    onClearResult,
    presetModeEnabled,
    activePreset,
  ]);

  const fieldPlaceholder = (field: DryRunSchemaField): string => {
    if (field.required) return 'Required';
    if (field.default !== undefined) return `Default: ${field.default}`;
    return 'Optional';
  };

  const fieldLabel = (field: DryRunSchemaField) => field.title || field.name;
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
      ? computeStages(dryRunResult, detail, liveMode, {
        planningFlowId: presetModeForExternalPlanning
          ? selectedExampleId
          : null,
        planningFlowLabel: presetModeForExternalPlanning
          ? activePreset?.label ?? null
          : null,
        selectedExampleId: activePreset?.id ?? null,
        selectedExampleLabel: activePreset?.label ?? null,
        selectedToolName: selectedTool,
      })
      : [],
    [dryRunResult, isExecuting, detail, liveMode, selectedExampleId, presetModeForExternalPlanning, selectedTool, activePreset],
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

        {/* ── Collapsible input section ─────────────────────────── */}
        <div className={`transition-all duration-300 ${inputCollapsed ? '' : ''}`}>
          {/* Collapse header (only when results showing) */}
          {showPipeline && (
            <button
              type="button"
              onClick={() => setInputCollapsed((v) => !v)}
              className="w-full flex items-center justify-between text-[11px] text-gray-400 hover:text-gray-300 mb-2 transition-colors"
              data-testid="dry-run-input-toggle"
            >
              <span className="uppercase tracking-wide font-medium">
                Input Configuration
                {inputCollapsed && selectedTool && (
                  <span className="ml-2 text-gray-500 normal-case tracking-normal">
                    — {selectedTool}
                    {liveMode && <span className="text-amber-400 ml-1">(live)</span>}
                  </span>
                )}
              </span>
              {inputCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </button>
          )}

          {/* Input form body */}
          {!inputCollapsed && (
            <div className="space-y-3">
              {/* Tool selector */}
              {detail.tools.length > 1 && (
                <div>
                  <label className="block text-[11px] font-medium text-gray-400 mb-1 uppercase tracking-wide">
                    Select Tool
                  </label>
                  <div className="relative">
                    <select
                      value={selectedTool}
                      onChange={(e) => {
                        const nextTool = e.target.value;
                        const initialPreset = getInitialPresetForTool(nextTool);
                        const liveDefaults = getLiveDefaultsForTool(nextTool);
                        setSelectedTool(nextTool);
                        setLiveLlmProvider(liveDefaults.provider);
                        setLiveLlmModel(liveDefaults.model);
                        setSelectedExampleId(initialPreset?.id ?? '');
                        setFormValues(initialPreset ? presetToFormValues(initialPreset) : {});
                        setFormErrors({});
                        setJsonInput(initialPreset ? JSON.stringify(initialPreset.input, null, 2) : '{}');
                        setJsonError(null);
                        setInputCollapsed(false);
                        onClearResult?.();
                      }}
                      className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 appearance-none cursor-pointer focus:border-blue-500 focus:outline-none"
                      data-testid="dry-run-tool-select"
                    >
                      {detail.tools.map((tool) => (
                        <option key={tool.name} value={tool.name}>
                          {tool.name}{tool.description ? ` — ${tool.description}` : ''}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
                  </div>
                </div>
              )}

              {/* Single tool display */}
              {detail.tools.length === 1 && currentTool && (
                <div className="bg-gray-900/60 border border-gray-700 rounded-lg p-3">
                  <div className="text-sm font-medium text-gray-200">{currentTool.name}</div>
                  {currentTool.description && (
                    <div className="text-xs text-gray-500 mt-0.5">{currentTool.description}</div>
                  )}
                </div>
              )}

              {/* Input mode toggle */}
              {presetModeEnabled && (
                <div className="bg-gray-900/60 border border-gray-700 rounded-lg p-3 space-y-2" data-testid={presetModeForExternalPlanning ? 'planning-flow-presets' : 'tool-flow-presets'}>
                  <div className="text-[11px] font-medium text-cyan-300 uppercase tracking-wide">
                    {presetModeForExternalPlanning ? 'Planning Flow Verification' : 'Skill Example Verification'}
                  </div>
                  <div className="text-[11px] text-gray-400">Single-select one example and verify strict expected/requested/observed routing evidence end-to-end.</div>
                  <div className="space-y-1.5">
                    {availablePresets.map((preset) => {
                      const selected = selectedExampleId === preset.id;
                      const baseId = presetModeForExternalPlanning ? 'planning-flow' : 'tool-flow';
                      return (
                        <button
                          type="button"
                          key={preset.id}
                          onClick={() => {
                            setSelectedExampleId(preset.id);
                            setFormValues(presetToFormValues(preset));
                            setJsonInput(JSON.stringify(preset.input, null, 2));
                            setInputCollapsed(false);
                            onClearResult?.();
                          }}
                          className={`w-full text-left rounded border p-2 transition-colors ${
                            selected
                              ? 'border-cyan-600/50 bg-cyan-900/10'
                              : 'border-gray-700/50 bg-gray-800/40 hover:bg-gray-800/60'
                          }`}
                          data-testid={`${baseId}-select-${preset.id}`}
                        >
                          <div className="flex items-start gap-2">
                            <input
                              type="radio"
                              name={`${baseId}-example`}
                              checked={selected}
                              onChange={() => {
                                setSelectedExampleId(preset.id);
                                setFormValues(presetToFormValues(preset));
                                setJsonInput(JSON.stringify(preset.input, null, 2));
                              }}
                              className="mt-0.5"
                              data-testid={`${baseId}-radio-${preset.id}`}
                            />
                            <div className={`flex-1 ${selected ? 'text-cyan-300' : 'text-gray-300'}`}>
                              <div className="text-[11px] font-medium">{preset.label}</div>
                              <div className="text-[10px] text-gray-500">{preset.description}</div>
                              {preset.objective && (
                                <div className="text-[10px] text-gray-400 mt-0.5">Objective: {preset.objective}</div>
                              )}
                            </div>
                          </div>
                          {selected && (
                            <div className="mt-1 text-[10px] text-cyan-200">
                              Selected execution payload: <span className="font-mono">{JSON.stringify(preset.input)}</span>
                            </div>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="flex items-center gap-2">
                <label className="block text-[11px] font-medium text-gray-400 uppercase tracking-wide">
                  Input
                </label>
                <div className="flex bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
                  <button
                    type="button"
                    onClick={() => {
                      setInputMode('form');
                      setJsonError(null);
                      setInputCollapsed(false);
                      onClearResult?.();
                    }}
                    className={`px-3 py-1 text-[11px] transition-colors ${
                      inputMode === 'form'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                    data-testid="dry-run-mode-form"
                  >
                    Form
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setInputMode('json');
                      setFormErrors({});
                      setInputCollapsed(false);
                      onClearResult?.();
                    }}
                    className={`px-3 py-1 text-[11px] transition-colors ${
                      inputMode === 'json'
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                    data-testid="dry-run-mode-json"
                  >
                    JSON
                  </button>
                </div>
              </div>

              {/* Form mode */}
              {inputMode === 'form' && (
                <div className="space-y-3" data-testid="dry-run-form-inputs">
                  {schemaFields.length === 0 ? (
                    <div className="text-xs text-gray-500 bg-gray-900/40 rounded-lg p-3 border border-gray-700/50 space-y-1">
                      <div>No structured input schema is defined. Dry-run will run capability and runtime checks.</div>
                      {detail.runtime_binding === 'prompt_instructions' && (detail.instructions_length ?? 0) > 0 && (
                        <div className="text-[11px] text-green-300">
                          Prompt-native runtime detected: SKILL.md instructions are loaded ({detail.instructions_length} chars), and lifecycle hooks can execute without a JSON input schema.
                        </div>
                      )}
                      {isPlanningWithFilesTool(selectedTool) && detail.runtime_binding !== 'prompt_instructions' && (
                        <div className="text-[11px] text-gray-400">
                          Add <span className="font-mono">input_schema</span> (SKILL.md frontmatter or simpleskill.json)
                          to enable form fields and action-level dry-run validation for planning actions.
                        </div>
                      )}
                    </div>
                  ) : (
                    <>
                      {!presetModeForExternalPlanning && shouldShowActionHint(selectedTool, formValues) && (
                        <div className="text-[11px] text-gray-500 bg-gray-900/40 rounded-lg p-2 border border-gray-700/50">
                          Select an action to show relevant inputs.
                        </div>
                      )}
                      {visibleSchemaFields.map((field) => (
                      <div key={field.name} className={!field.required ? 'opacity-70' : ''}>
                        <label className="flex items-center gap-1.5 text-[11px] text-gray-400 mb-1">
                          <span className="font-medium text-gray-300">{field.name}</span>
                          {field.required ? (
                            <span className="text-red-400" title="Required">*</span>
                          ) : (
                            <span className="text-gray-600 text-[10px]">optional</span>
                          )}
                        </label>
                        {field.description && (
                          <div className="text-[10px] text-gray-600 mb-1">{fieldLabel(field)}</div>
                        )}
                        <SkillFieldInput
                          field={field}
                          value={formValues[field.name] ?? ''}
                          error={formErrors[field.name]}
                          isWeatherTool={isWeatherTool}
                          isLocationTool={isLocationTool}
                          isWebSearchTool={isWebSearchTool}
                          weatherCityOptions={weatherCityOptions}
                          onChange={(value) => updateFormField(field.name, value)}
                          placeholder={fieldPlaceholder(field)}
                        />
                        {(field.type === 'number' || field.type === 'integer') &&
                          (field.minimum !== undefined || field.maximum !== undefined) && (
                            <div className="mt-0.5 text-[10px] text-gray-600" data-testid={`dry-run-range-${field.name}`}>
                              Range: {field.minimum !== undefined ? field.minimum : '-inf'} to {field.maximum !== undefined ? field.maximum : '+inf'}
                            </div>
                          )}
                        {formErrors[field.name] && (
                          <div className="mt-0.5 text-[10px] text-red-400">{formErrors[field.name]}</div>
                        )}
                      </div>
                      ))}
                    </>
                  )}
                </div>
              )}

              {/* JSON mode */}
              {inputMode === 'json' && (
                <div data-testid="dry-run-json-input">
                  <textarea
                    value={jsonInput}
                    onChange={(e) => {
                      setJsonInput(e.target.value);
                      setJsonError(null);
                      setFormErrors({});
                    }}
                    rows={4}
                    className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 font-mono focus:border-blue-500 focus:outline-none resize-y"
                    placeholder='{"key": "value"}'
                    spellCheck={false}
                  />
                  {jsonError && (
                    <div className="mt-1 text-[10px] text-red-400 flex items-center gap-1">
                      <AlertCircle size={10} /> {jsonError}
                    </div>
                  )}
                  {!jsonError && Object.values(formErrors).length > 0 && (
                    <div className="mt-1 space-y-1" data-testid="dry-run-json-validation-errors">
                      {Object.values(formErrors).map((msg) => (
                        <div key={msg} className="text-[10px] text-red-400 flex items-center gap-1">
                          <AlertCircle size={10} /> {msg}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Execute bar ──────────────────────────────────────── */}
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer select-none shrink-0" title="When enabled, dry-run will send the input to an LLM to verify the skill produces a valid tool call.">
            <input
              type="checkbox"
              checked={liveMode}
              onChange={(e) => {
                setLiveMode(e.target.checked);
                if (e.target.checked && !liveLlmProvider && !liveLlmModel) {
                  const defaults = getLiveDefaultsForTool(selectedTool);
                  setLiveLlmProvider(defaults.provider);
                  setLiveLlmModel(defaults.model);
                }
                setInputCollapsed(false);
                onClearResult?.();
              }}
              className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 cursor-pointer"
              data-testid="dry-run-live-toggle"
            />
            <Zap size={11} className={liveMode ? 'text-amber-400' : 'text-gray-600'} />
            <span className={`text-[11px] ${liveMode ? 'text-amber-300' : 'text-gray-500'}`}>
              Live
            </span>
          </label>

          {liveMode && (
            <div className="flex items-center gap-2 shrink-0" data-testid="dry-run-live-llm-config">
              <select
                value={liveLlmProvider}
                onChange={(e) => setLiveLlmProvider(e.target.value)}
                className="bg-gray-900 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200"
                data-testid="dry-run-live-provider"
              >
                {LIVE_LLM_PROVIDER_OPTIONS.map((option) => (
                  <option key={`live-provider-${option.value || 'default'}`} value={option.value}>{option.label}</option>
                ))}
              </select>
              <select
                value={liveLlmModel}
                onChange={(e) => setLiveLlmModel(e.target.value)}
                className="w-44 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200"
                data-testid="dry-run-live-model"
                disabled={!liveLlmProvider || liveModelOptions.length === 0}
              >
                <option value="">model: default</option>
                {liveModelOptions.map((modelName) => (
                  <option key={`live-model-${liveLlmProvider}-${modelName}`} value={modelName}>{modelName}</option>
                ))}
              </select>
            </div>
          )}

          <button
            type="button"
            onClick={handleExecute}
            disabled={isExecuting || !selectedTool}
            className={`flex-1 min-w-[220px] whitespace-nowrap flex items-center justify-center gap-2 px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors ${
              liveMode
                ? 'bg-amber-600 hover:bg-amber-500 disabled:bg-gray-700 disabled:text-gray-500'
                : 'bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500'
            }`}
            data-testid="dry-run-execute"
          >
            {liveMode ? <Zap size={14} /> : <Play size={14} />}
            {isExecuting ? 'Verifying...' : liveMode ? 'Execute Live Dry-run' : 'Execute Dry-run'}
          </button>
        </div>

        {/* ── Verification Pipeline ───────────────────────────── */}
        {showPipeline && (
          <div data-testid="dry-run-result-panel">
            {/* Pipeline header */}
            <div className="flex items-center gap-2 mb-3 pt-1 border-t border-gray-700/50">
              <div className={`w-2 h-2 rounded-full ${
                !dryRunResult ? 'bg-blue-400 animate-pulse' :
                dryRunResult.valid ? 'bg-green-400' : 'bg-red-400'
              }`} />
              <span className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
                Verification Pipeline
              </span>
              {dryRunResult && (
                <span className={`ml-auto text-[11px] font-semibold ${
                  dryRunResult.valid ? 'text-green-400' : 'text-red-400'
                }`}>
                  {dryRunResult.valid ? 'ALL PASSED' : 'FAILED'}
                </span>
              )}
              {!dryRunResult && isExecuting && (
                <span className="ml-auto text-[11px] text-blue-400">
                  Running...
                </span>
              )}
            </div>

            {/* Pipeline stages */}
            <div className="pl-1">
              {pipelineStages.map((stage, i) => (
                <PipelineStageRow
                  key={stage.id}
                  stage={stage}
                  isLast={i === pipelineStages.length - 1}
                  revealed={i < revealedCount}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </CenterStage>
  );
};

