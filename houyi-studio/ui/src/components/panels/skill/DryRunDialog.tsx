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
import React, { useEffect, useMemo, useCallback } from 'react';
import { CenterStage } from '../../CenterStage';
import type { SkillDetail, SkillTool } from '../../../types/websocket';
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
    isPlanningWithFilesTool,
  });

  const packagePresets = useMemo<ToolDryRunPreset[]>(() => {
    const raw = detail.package_examples ?? [];
    if (!isExternalPlanningWithFilesTool(selectedTool)) {
      return raw;
    }

    return raw.map((preset) => {
      const hasAction = typeof (preset.input as Record<string, unknown>).action === 'string';
      if (hasAction) return preset;

      const canonical = PLANNING_FLOW_PRESETS.find((p) => p.id === preset.id);
      const canonicalAction = canonical?.input?.action;
      if (typeof canonicalAction !== 'string') {
        return preset;
      }

      return {
        ...preset,
        input: {
          ...(canonical?.input ?? {}),
          ...preset.input,
        },
      };
    });
  }, [detail.package_examples, selectedTool]);
  const toolPresets = useMemo<ToolDryRunPreset[]>(
    () => getToolDryRunPresets(selectedTool),
    [selectedTool],
  );
  const availablePresets = useMemo<ToolDryRunPreset[]>(
    () => (packagePresets.length > 0 ? packagePresets : toolPresets),
    [packagePresets, toolPresets],
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
    planningFlowId: isExternalPlanningWithFilesTool(selectedTool) ? selectedExampleId : null,
    planningFlowLabel: isExternalPlanningWithFilesTool(selectedTool) ? activePreset?.label ?? null : null,
    selectedExampleId: activePreset?.id ?? null,
    selectedExampleLabel: activePreset?.label ?? null,
    selectedExampleInput: activePreset?.input ?? null,
    selectedExampleObjective: activePreset?.objective ?? null,
    selectedToolName: selectedTool,
  }), [selectedExampleId, activePreset, selectedTool]);

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
  const presetModeForExternalPlanning = presetModeEnabled && isExternalPlanningWithFilesTool(selectedTool);
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
      input = buildExecutionInputFromForm(visibleSchemaFields, formValues, errors);
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
    setExecutedContext(currentPipelineContext);
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
    visibleSchemaFields,
    formValues,
    selectedTool,
    liveMode,
    liveLlmProvider,
    liveLlmModel,
    onExecute,
    onClearResult,
    currentPipelineContext,
    presetModeEnabled,
    activePreset,
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
          selectedExampleId={selectedExampleId}
          inputMode={inputMode}
          schemaFields={schemaFields}
          isInstructionDrivenRuntime={isInstructionDrivenRuntime}
          instructionsLength={detail.instructions_length}
          isPlanningWithFilesSelectedTool={isPlanningWithFilesTool(selectedTool)}
          shouldShowActionHintForCurrentTool={shouldShowActionHint(selectedTool, formValues)}
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
          onSwitchInputMode={handleSwitchInputMode}
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
          onToggleLiveMode={handleToggleLiveMode}
          onChangeLiveProvider={setLiveLlmProvider}
          onChangeLiveModel={setLiveLlmModel}
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

