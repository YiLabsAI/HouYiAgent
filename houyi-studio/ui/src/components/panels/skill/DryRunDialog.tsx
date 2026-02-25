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
  onExecute: (toolName: string, input: Record<string, unknown>, live?: boolean) => void;
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
  const [inputCollapsed, setInputCollapsed] = useState(false);

  // How many stages have been revealed with their actual status
  const [revealedCount, setRevealedCount] = useState(0);

  // Select first tool by default
  useEffect(() => {
    if (isOpen && detail.tools.length > 0 && !selectedTool) {
      setSelectedTool(detail.tools[0].name);
    }
  }, [isOpen, detail.tools, selectedTool]);

  // Progressive reveal: when result arrives, reveal stages one by one
  useEffect(() => {
    if (dryRunResult) {
      setIsExecuting(false);
      setInputCollapsed(true);
      setRevealedCount(0);
      const totalStages = computeStages(dryRunResult, detail, liveMode).length;
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
  }, [dryRunResult, detail, liveMode]);

  // Reset state on open
  useEffect(() => {
    if (isOpen) {
      setJsonInput('{}');
      setFormValues({});
      setJsonError(null);
      setFormErrors({});
      setIsExecuting(false);
      setLiveMode(false);
      setInputCollapsed(false);
      setRevealedCount(0);
      if (detail.tools.length > 0) {
        setSelectedTool(detail.tools[0].name);
      }
    }
  }, [isOpen, detail.tools]);

  const currentTool: SkillTool | undefined = useMemo(
    () => detail.tools.find((t) => t.name === selectedTool),
    [detail.tools, selectedTool],
  );
  const isWeatherTool = currentTool?.name === 'get_weather';
  const isLocationTool = currentTool?.name === 'get_location';

  // Extract fields from input_schema, handling Pydantic's anyOf pattern
  const schemaFields = useMemo<DryRunSchemaField[]>(() => {
    if (!currentTool?.input_schema) return [];
    const schema = currentTool.input_schema;
    const props = schema.properties || {};
    const required = new Set(schema.required || []);

    return Object.entries(props).map(([name, def]) => {
      const d = def as Record<string, unknown>;
      let fieldType = d.type as string | undefined;
      let nullable = false;
      let enumValues = d.enum as string[] | undefined;

      if (!fieldType && Array.isArray(d.anyOf)) {
        const variants = d.anyOf as Array<Record<string, unknown>>;
        const types = variants.map((t) => t.type as string).filter(Boolean);
        const nonNullTypes = types.filter((t) => t !== 'null');
        nullable = types.includes('null');
        fieldType = nonNullTypes[0] || 'string';
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

  const updateFormField = (fieldName: string, value: string) => {
    setFormValues((prev) => ({ ...prev, [fieldName]: value }));
    if (formErrors[fieldName]) {
      setFormErrors((prev) => {
        const next = { ...prev };
        delete next[fieldName];
        return next;
      });
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
    } else {
      const errors: Record<string, string> = {};
      for (const field of schemaFields) {
        const val = formValues[field.name];
        if (field.required && (val === undefined || val === '')) {
          errors[field.name] = 'This field is required';
        }
      }
      if (Object.keys(errors).length > 0) {
        setFormErrors(errors);
        return;
      }
      setFormErrors({});
      for (const field of schemaFields) {
        const val = formValues[field.name];
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
    }
    setIsExecuting(true);
    setRevealedCount(0);
    onClearResult?.();
    onExecute(selectedTool, input, liveMode);
  }, [inputMode, jsonInput, schemaFields, formValues, selectedTool, liveMode, onExecute, onClearResult]);

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
      ? computeStages(dryRunResult, detail, liveMode)
      : [],
    [dryRunResult, isExecuting, detail, liveMode],
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
                        setSelectedTool(e.target.value);
                        setFormValues({});
                        setFormErrors({});
                        setJsonInput('{}');
                        setJsonError(null);
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
              <div className="flex items-center gap-2">
                <label className="block text-[11px] font-medium text-gray-400 uppercase tracking-wide">
                  Input
                </label>
                <div className="flex bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
                  <button
                    type="button"
                    onClick={() => { setInputMode('form'); setJsonError(null); }}
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
                    onClick={() => { setInputMode('json'); setFormErrors({}); }}
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
                    <div className="text-xs text-gray-500 bg-gray-900/40 rounded-lg p-3 border border-gray-700/50">
                      No input parameters defined. Dry-run will perform an availability check.
                    </div>
                  ) : (
                    schemaFields
                      .sort((a, b) => (a.required === b.required ? 0 : a.required ? -1 : 1))
                      .map((field) => (
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
                          weatherCityOptions={weatherCityOptions}
                          onChange={(value) => updateFormField(field.name, value)}
                          placeholder={fieldPlaceholder(field)}
                        />
                        {formErrors[field.name] && (
                          <div className="mt-0.5 text-[10px] text-red-400">{formErrors[field.name]}</div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* JSON mode */}
              {inputMode === 'json' && (
                <div data-testid="dry-run-json-input">
                  <textarea
                    value={jsonInput}
                    onChange={(e) => { setJsonInput(e.target.value); setJsonError(null); }}
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
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Execute bar ──────────────────────────────────────── */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer select-none shrink-0" title="When enabled, dry-run will send the input to an LLM to verify the skill produces a valid tool call.">
            <input
              type="checkbox"
              checked={liveMode}
              onChange={(e) => setLiveMode(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-0 cursor-pointer"
              data-testid="dry-run-live-toggle"
            />
            <Zap size={11} className={liveMode ? 'text-amber-400' : 'text-gray-600'} />
            <span className={`text-[11px] ${liveMode ? 'text-amber-300' : 'text-gray-500'}`}>
              Live
            </span>
          </label>

          <button
            type="button"
            onClick={handleExecute}
            disabled={isExecuting || !selectedTool}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors ${
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

