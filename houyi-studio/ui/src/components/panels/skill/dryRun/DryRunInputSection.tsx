import React from 'react';
import { AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

import type { SkillTool } from '../../../../types/websocket';
import { MarkdownRenderer } from '../../../Chat/MarkdownRenderer';
import { SkillFieldInput, type DryRunSchemaField } from './SkillFieldInput';
import type { ToolDryRunPreset } from './dryRunToolRules';
import { fieldPlaceholder } from './dryRunInputModel';

const presetSourceLabel = (preset: ToolDryRunPreset): string => {
  if (preset.source) return preset.source;
  if (preset.origin === 'package_examples') return 'package_examples';
  if (preset.origin === 'frontmatter_workflow') return 'SKILL.md#workflows';
  if (preset.origin === 'instruction_workflow') return 'instructions';
  if (preset.origin === 'tool_preset') return 'built-in preset';
  return 'unknown';
};

interface DryRunInputSectionProps {
  showPipeline: boolean;
  inputCollapsed: boolean;
  selectedTool: string;
  liveMode: boolean;
  detailTools: SkillTool[];
  currentTool?: SkillTool;
  currentToolDescription: string;
  isLongToolDescription: boolean;
  toolDescriptionPreview: string;
  toolDescriptionExpanded: boolean;
  presetModeEnabled: boolean;
  presetModeForExternalPlanning: boolean;
  availablePresets: ToolDryRunPreset[];
  workflowArgFields: Array<{
    key: string;
    label: string;
    defaultValue: string;
    kind: 'flag' | 'positional' | 'script' | 'operation';
  }>;
  workflowArgValues: Record<string, string>;
  selectedExampleId: string;
  selectedExecutionPayloadPreview: Record<string, unknown> | null;
  inputMode: 'form' | 'json';
  schemaFields: DryRunSchemaField[];
  isInstructionDrivenRuntime: boolean;
  instructionsLength?: number;
  isPlanningWithFilesSelectedTool: boolean;
  shouldShowActionHintForCurrentTool: boolean;
  visibleSchemaFields: Array<DryRunSchemaField & { visible?: boolean }>;
  formValues: Record<string, string>;
  formErrors: Record<string, string>;
  weatherCityOptions: string[];
  isWeatherTool: boolean;
  isLocationTool: boolean;
  isWebSearchTool: boolean;
  jsonInput: string;
  jsonError: string | null;
  onToggleCollapsed: () => void;
  onToolChange: (toolName: string) => void;
  onToggleToolDescriptionExpanded: () => void;
  onSelectPreset: (preset: ToolDryRunPreset) => void;
  onUpdateWorkflowArg: (fieldKey: string, value: string) => void;
  onSwitchInputMode: (mode: 'form' | 'json') => void;
  onUpdateFormField: (fieldName: string, value: string) => void;
  onJsonInputChange: (value: string) => void;
}

export const DryRunInputSection: React.FC<DryRunInputSectionProps> = ({
  showPipeline,
  inputCollapsed,
  selectedTool,
  liveMode,
  detailTools,
  currentTool,
  currentToolDescription,
  isLongToolDescription,
  toolDescriptionPreview,
  toolDescriptionExpanded,
  presetModeEnabled,
  presetModeForExternalPlanning,
  availablePresets,
  workflowArgFields,
  workflowArgValues,
  selectedExampleId,
  selectedExecutionPayloadPreview,
  inputMode,
  schemaFields,
  isInstructionDrivenRuntime,
  instructionsLength,
  isPlanningWithFilesSelectedTool,
  shouldShowActionHintForCurrentTool,
  visibleSchemaFields,
  formValues,
  formErrors,
  weatherCityOptions,
  isWeatherTool,
  isLocationTool,
  isWebSearchTool,
  jsonInput,
  jsonError,
  onToggleCollapsed,
  onToolChange,
  onToggleToolDescriptionExpanded,
  onSelectPreset,
  onUpdateWorkflowArg,
  onSwitchInputMode,
  onUpdateFormField,
  onJsonInputChange,
}) => {
  const fieldLabel = (field: DryRunSchemaField): string => field.title || field.name;
  const [expandedPayloadPresetId, setExpandedPayloadPresetId] = React.useState<string | null>(null);
  const [showAllExamples, setShowAllExamples] = React.useState(false);
  const EXAMPLE_COLLAPSE_COUNT = 3;

  // One-list UX policy: always expose a single unified example list.
  // package_examples + mapped workflows are merged upstream in DryRunDialog.
  const activeExample = React.useMemo(
    () => availablePresets.find((preset) => preset.id === selectedExampleId) ?? availablePresets[0] ?? null,
    [availablePresets, selectedExampleId],
  );

  React.useEffect(() => {
    setShowAllExamples(false);
  }, [selectedTool, availablePresets]);

  const visibleExamples = React.useMemo(() => {
    if (availablePresets.length <= EXAMPLE_COLLAPSE_COUNT || showAllExamples) {
      return availablePresets;
    }

    const top = availablePresets.slice(0, EXAMPLE_COLLAPSE_COUNT);
    if (top.some((preset) => preset.id === selectedExampleId)) {
      return top;
    }

    const selected = availablePresets.find((preset) => preset.id === selectedExampleId);
    return selected ? [...top, selected] : top;
  }, [availablePresets, showAllExamples, selectedExampleId]);

  const hiddenExampleCount = Math.max(0, availablePresets.length - EXAMPLE_COLLAPSE_COUNT);
  const isBuiltInFallbackOnly = React.useMemo(
    () => availablePresets.length > 0 && availablePresets.every((preset) => preset.origin === 'tool_preset'),
    [availablePresets],
  );

  return (
    <div className={`transition-all duration-300 ${inputCollapsed ? '' : ''}`}>
      {showPipeline && (
        <button
          type="button"
          onClick={onToggleCollapsed}
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

      {!inputCollapsed && (
        <div className="space-y-3">
          {detailTools.length > 1 && (
            <div>
              <label className="block text-[11px] font-medium text-gray-400 mb-1 uppercase tracking-wide">
                Select Tool
              </label>
              <div className="relative">
                <select
                  value={selectedTool}
                  onChange={(e) => onToolChange(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 focus:border-blue-500 focus:outline-none appearance-none"
                  data-testid="dry-run-tool-select"
                >
                  {detailTools.map((tool) => (
                    <option key={tool.name} value={tool.name}>
                      {tool.name}{tool.description ? ` — ${tool.description}` : ''}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
              </div>
            </div>
          )}

          {detailTools.length === 1 && currentTool && (
            <div className="bg-gray-900/60 border border-gray-700 rounded-lg p-3">
              <div className="text-sm font-medium text-gray-200">{currentTool.name}</div>
              {currentToolDescription && (
                <div className="mt-1" data-testid="dry-run-tool-description">
                  {isLongToolDescription ? (
                    <>
                      {!toolDescriptionExpanded ? (
                        <div className="text-gray-400 text-[12px] leading-relaxed">{toolDescriptionPreview}</div>
                      ) : (
                        <div className="text-gray-300 text-[12px] leading-relaxed markdown-body">
                          <MarkdownRenderer content={currentToolDescription} />
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={onToggleToolDescriptionExpanded}
                        className="mt-1 text-[11px] text-cyan-300 hover:text-cyan-200"
                        data-testid="dry-run-tool-description-more"
                      >
                        {toolDescriptionExpanded ? 'Show less' : 'Show more'}
                      </button>
                    </>
                  ) : (
                    <div className="text-gray-300 text-[12px] leading-relaxed markdown-body">
                      <MarkdownRenderer content={currentToolDescription} />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {presetModeEnabled && (
            <div className="bg-gray-900/60 border border-gray-700 rounded-lg p-3 space-y-2" data-testid={presetModeForExternalPlanning ? 'planning-flow-presets' : 'tool-flow-presets'}>
              <div className="text-[11px] font-medium text-cyan-300 uppercase tracking-wide">
                {presetModeForExternalPlanning ? 'Planning Flow Verification' : 'Skill Example Verification'}
              </div>
              <div className="text-[11px] text-gray-400">Single-select one example and verify strict expected/requested/observed routing evidence end-to-end.</div>
              {isBuiltInFallbackOnly && (
                <div className="text-[10px] text-amber-300 bg-amber-900/15 border border-amber-700/30 rounded px-2 py-1" data-testid="dry-run-example-source-fallback">
                  Showing built-in fallback examples. Add package_examples or SKILL.md workflows/instruction command templates to get skill-authored examples.
                </div>
              )}
              <div className="space-y-1.5">
                {visibleExamples.map((preset) => {
                  const selected = selectedExampleId === preset.id;
                  const baseId = presetModeForExternalPlanning ? 'planning-flow' : 'tool-flow';
                  const optionTestId = preset.workflow_id
                    ? `dry-run-workflow-select-${preset.workflow_id}`
                    : `${baseId}-select-${preset.id}`;
                  return (
                    <button
                      type="button"
                      key={preset.id}
                      onClick={() => onSelectPreset(preset)}
                      className={`w-full text-left rounded border p-2 transition-colors ${
                        selected
                          ? 'border-cyan-600/50 bg-cyan-900/10'
                          : 'border-gray-700/50 bg-gray-800/40 hover:bg-gray-800/60'
                      }`}
                      data-testid={optionTestId}
                    >
                      <div className="flex items-start gap-2">
                        <input
                          type="radio"
                          name={`${baseId}-example`}
                          checked={selected}
                          readOnly
                          className="mt-0.5"
                          data-testid={`${baseId}-radio-${preset.id}`}
                        />
                        <div className={`flex-1 ${selected ? 'text-cyan-300' : 'text-gray-300'}`}>
                          <div className="text-[11px] font-medium">{preset.label}</div>
                          <div className="text-[10px] text-gray-500">{preset.description}</div>
                          {preset.objective && (
                            <div className="text-[10px] text-gray-400 mt-0.5">Objective: {preset.objective}</div>
                          )}
                          <div className="text-[10px] text-gray-500 mt-0.5" data-testid={`${baseId}-audit-${preset.id}`}>
                            {`Source: ${presetSourceLabel(preset)}`}
                            {preset.confidence ? ` · Confidence: ${preset.confidence}` : ''}
                            {typeof preset.confidence_breakdown?.score === 'number'
                              ? ` (${Math.round(preset.confidence_breakdown.score * 100)}%)`
                              : ''}
                            {preset.confidence_reason && (
                              <span
                                className="ml-1 text-cyan-300/90 cursor-help"
                                title={preset.confidence_reason}
                                data-testid={`${baseId}-confidence-tooltip-${preset.id}`}
                              >
                                ⓘ
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      {selected && (
                        <div className="mt-1 text-[10px] text-cyan-200 space-y-1" data-testid="dry-run-selected-execution-payload">
                          <div>Selected execution payload:</div>
                          {(() => {
                            const payload = selectedExecutionPayloadPreview ?? preset.input;
                            const payloadText = JSON.stringify(payload, null, 2);
                            const isLongPayload = payloadText.length > 320;
                            const expanded = expandedPayloadPresetId === preset.id;
                            return (
                              <>
                                <pre
                                  className={`font-mono text-[10px] leading-relaxed bg-cyan-950/20 border border-cyan-800/40 rounded p-2 overflow-auto ${
                                    isLongPayload && !expanded ? 'max-h-20' : 'max-h-52'
                                  }`}
                                >
                                  {payloadText}
                                </pre>
                                {isLongPayload && (
                                  <div
                                    role="button"
                                    tabIndex={0}
                                    className="inline-block text-[10px] text-cyan-300 hover:text-cyan-200 cursor-pointer"
                                    data-testid="dry-run-selected-payload-toggle"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setExpandedPayloadPresetId(expanded ? null : preset.id);
                                    }}
                                    onKeyDown={(e) => {
                                      if (e.key !== 'Enter' && e.key !== ' ') return;
                                      e.preventDefault();
                                      e.stopPropagation();
                                      setExpandedPayloadPresetId(expanded ? null : preset.id);
                                    }}
                                  >
                                    {expanded ? 'Show less' : 'Show more'}
                                  </div>
                                )}
                              </>
                            );
                          })()}
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>

              {hiddenExampleCount > 0 && (
                <div className="text-[10px] text-gray-500" data-testid="dry-run-workflow-collapse-state">
                  {showAllExamples
                    ? `Showing all ${availablePresets.length} examples.`
                    : `Showing ${Math.min(visibleExamples.length, EXAMPLE_COLLAPSE_COUNT)} of ${availablePresets.length} examples.`}
                  <button
                    type="button"
                    className="ml-1 text-cyan-300 hover:text-cyan-200"
                    data-testid="dry-run-workflow-toggle"
                    onClick={() => setShowAllExamples((v) => !v)}
                  >
                    {showAllExamples ? 'Show less' : `More (${hiddenExampleCount})`}
                  </button>
                </div>
              )}

              {activeExample?.command_template && !isPlanningWithFilesSelectedTool && (
                <>
                  <div className="space-y-1.5" data-testid="dry-run-workflow-metadata">
                    <div className="text-[10px] text-gray-500 font-mono break-all" data-testid="dry-run-workflow-command-preview">
                      {activeExample.command_template}
                    </div>
                    <div className="text-[10px] text-gray-400" data-testid="dry-run-workflow-audit">
                      Source: {presetSourceLabel(activeExample)}
                      {activeExample.confidence ? ` · Confidence: ${activeExample.confidence}` : ''}
                      {typeof activeExample.confidence_breakdown?.score === 'number'
                        ? ` (${Math.round(activeExample.confidence_breakdown.score * 100)}%)`
                        : ''}
                    </div>
                    {activeExample.validation_status && (
                      <div
                        className={`text-[10px] ${
                          activeExample.validation_status === 'fail'
                            ? 'text-red-300'
                            : activeExample.validation_status === 'warn'
                              ? 'text-amber-300'
                              : 'text-emerald-300'
                        }`}
                        data-testid="dry-run-workflow-validation"
                      >
                        Validation: {activeExample.validation_status.toUpperCase()}
                        {(activeExample.validation_issues?.length ?? 0) > 0
                          ? ` · ${activeExample.validation_issues?.join(' | ')}`
                          : ''}
                      </div>
                    )}
                    {(activeExample.evidence || activeExample.confidence_reason) && (
                      <div className="text-[10px] text-gray-500" data-testid="dry-run-workflow-evidence">
                        Evidence: {activeExample.evidence || activeExample.confidence_reason}
                      </div>
                    )}
                  </div>

                  {workflowArgFields.length > 0 && (
                    <div className="space-y-2 pt-1 border-t border-gray-700/70" data-testid="dry-run-workflow-arg-fields">
                      <div className="text-[11px] font-medium text-emerald-300 uppercase tracking-wide">
                        Command placeholders
                      </div>
                      <div className="text-[11px] text-gray-400">
                        Fill actual values used to compose command preview and execution payload.
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {workflowArgFields.map((field) => (
                          <div key={field.key}>
                            <label className="flex items-center gap-1.5 text-[11px] text-gray-400 mb-1">
                              <span className="font-medium text-gray-300">{field.label}</span>
                              <span className="text-gray-600 text-[10px]">{field.kind}</span>
                            </label>
                            <input
                              type="text"
                              value={workflowArgValues[field.key] ?? ''}
                              onChange={(e) => onUpdateWorkflowArg(field.key, e.target.value)}
                              placeholder={field.defaultValue}
                              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-gray-200 focus:border-emerald-500 focus:outline-none"
                              data-testid={`dry-run-workflow-arg-${field.key}`}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          <div className="flex items-center gap-2">
            <label className="block text-[11px] font-medium text-gray-400 uppercase tracking-wide">
              Input
            </label>
            <div className="flex bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
              <button
                type="button"
                onClick={() => onSwitchInputMode('form')}
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
                onClick={() => onSwitchInputMode('json')}
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

          {inputMode === 'form' && (
            <div className="space-y-3" data-testid="dry-run-form-inputs">
              {schemaFields.length === 0 ? (
                <div className="text-xs text-gray-500 bg-gray-900/40 rounded-lg p-3 border border-gray-700/50 space-y-1">
                  <div>No structured input schema is defined. Dry-run will run capability and runtime checks.</div>
                  {isInstructionDrivenRuntime && (instructionsLength ?? 0) > 0 && (
                    <div className="text-[11px] text-green-300">
                      Instruction-driven runtime detected: SKILL.md instructions are loaded ({instructionsLength} chars), and lifecycle hooks can execute without a JSON input schema.
                    </div>
                  )}
                  {isPlanningWithFilesSelectedTool && !isInstructionDrivenRuntime && (
                    <div className="text-[11px] text-gray-400">
                      Add <span className="font-mono">input_schema</span> (SKILL.md frontmatter or simpleskill.json)
                      to enable form fields and action-level dry-run validation for planning actions.
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {!presetModeForExternalPlanning && shouldShowActionHintForCurrentTool && (
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
                        onChange={(value) => onUpdateFormField(field.name, value)}
                        placeholder={fieldPlaceholder(field)}
                      />
                      {(field.type === 'number' || field.type === 'integer')
                        && (field.minimum !== undefined || field.maximum !== undefined) && (
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

          {inputMode === 'json' && (
            <div data-testid="dry-run-json-input">
              <textarea
                value={jsonInput}
                onChange={(e) => onJsonInputChange(e.target.value)}
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
  );
};
