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
import { CenterStage } from '../CenterStage';
import {
  Play, AlertCircle, CheckCircle, XCircle, Circle, Loader2,
  ChevronDown, ChevronUp, Zap, AlertTriangle,
} from 'lucide-react';
import type { SkillDetail, SkillTool } from '../../types/websocket';
import type { DryRunResultData } from '../LeftSidebar/useSkillsLogic';

// ─── Pipeline stage types ────────────────────────────────────────

type StageStatus = 'pending' | 'running' | 'pass' | 'fail' | 'warn' | 'skip';

interface PipelineStage {
  id: string;
  number: number;
  label: string;
  status: StageStatus;
  summary: string;
  details?: React.ReactNode;
}

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

  // Extract fields from input_schema, handling Pydantic's anyOf pattern
  const schemaFields = useMemo(() => {
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

  const fieldPlaceholder = (field: typeof schemaFields[0]): string => {
    if (field.required) return 'Required';
    if (field.default !== undefined) return `Default: ${field.default}`;
    return 'Optional';
  };

  const fieldLabel = (field: typeof schemaFields[0]) => field.title || field.name;

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
                        {field.type === 'boolean' ? (
                          <select
                            value={formValues[field.name] ?? (field.defaultRaw !== undefined ? String(field.defaultRaw) : 'false')}
                            onChange={(e) => updateFormField(field.name, e.target.value)}
                            className={`w-full bg-gray-900 border rounded px-2 py-1.5 text-xs text-gray-200 focus:border-blue-500 focus:outline-none ${
                              formErrors[field.name] ? 'border-red-500' : 'border-gray-700'
                            }`}
                            data-testid={`dry-run-input-${field.name}`}
                          >
                            <option value="true">true</option>
                            <option value="false">false</option>
                          </select>
                        ) : field.enum ? (
                          <select
                            value={formValues[field.name] ?? (field.default !== undefined ? String(field.default) : '')}
                            onChange={(e) => updateFormField(field.name, e.target.value)}
                            className={`w-full bg-gray-900 border rounded px-2 py-1.5 text-xs text-gray-200 focus:border-blue-500 focus:outline-none ${
                              formErrors[field.name] ? 'border-red-500' : 'border-gray-700'
                            }`}
                            data-testid={`dry-run-input-${field.name}`}
                          >
                            {!field.required && field.default === undefined && (
                              <option value="">— not set —</option>
                            )}
                            {field.enum.map((v) => (
                              <option key={v} value={v}>{v}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type={field.type === 'number' || field.type === 'integer' ? 'number' : 'text'}
                            value={formValues[field.name] ?? ''}
                            onChange={(e) => updateFormField(field.name, e.target.value)}
                            placeholder={fieldPlaceholder(field)}
                            min={field.minimum}
                            max={field.maximum}
                            className={`w-full bg-gray-900 border rounded px-2 py-1.5 text-xs text-gray-200 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none ${
                              formErrors[field.name] ? 'border-red-500' : 'border-gray-700'
                            }`}
                            data-testid={`dry-run-input-${field.name}`}
                          />
                        )}
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

// ─── Stage computation (pure function) ───────────────────────────

function computeStages(
  result: DryRunResultData | null,
  detail: SkillDetail,
  liveMode: boolean,
): PipelineStage[] {
  const stages: PipelineStage[] = [];
  let n = 1;

  // Badge helpers
  const certBadge = (cert: string) => {
    const colors: Record<string, string> = {
      gold: 'bg-yellow-900/30 border-yellow-600/40 text-yellow-300',
      silver: 'bg-gray-700/40 border-gray-500/40 text-gray-300',
      bronze: 'bg-orange-900/30 border-orange-600/40 text-orange-300',
      unverified: 'bg-gray-800/40 border-gray-700/40 text-gray-500',
    };
    return colors[cert] || colors.unverified;
  };

  // 1. Registration — show loaded tools, version, certification
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

  // 2. Schema Validation — show validated fields or errors
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

  // 3. Policy Evaluation
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
          ? 'Allowed with user consent required'
        : 'Invocation denied by policy')
      : 'Evaluating invocation policy...',
  });

  // 4. Side Effects
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

  // 5. Hooks
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

  // 6. Capability Gaps (only if present)
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

  // 7. LLM Verification — show full invocation trace: prompt → tool defs → response
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
      (p: DisclosurePhase) => p.name === 'tool_execution',
    );
    if (toolExecPhase) {
      stages.push({
        id: 'tool-execution',
        number: n++,
        label: 'Tool Execution',
        status: toolExecPhase.status === 'pass' ? 'pass' : toolExecPhase.status === 'fail' ? 'fail' : 'skip',
        summary: toolExecPhase.data?.result_preview
          ? `Result: ${String(toolExecPhase.data.result_preview).substring(0, 80)}...`
          : String(toolExecPhase.data?.reason || toolExecPhase.data?.error || 'Skipped'),
      });
    }
  }

  return stages;
}

// ─── LLM Verification Details — progressive disclosure timeline ──
//
// Shows 4 phases with wall-clock timestamps:
//   ① Discovery   → metadata loaded
//   ② Activation  → tool definitions built
//   ③ Negotiation → system prompt + user query constructed
//   ④ Execution   → real LLM API call, response parsed
//
// Each phase is collapsible and shows trigger timing.

import type { DisclosurePhase } from '../LeftSidebar/useSkillsLogic';

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

const LlmVerificationDetails: React.FC<{
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

      {/* ─── Phase Timeline ────────────────────────────── */}
      {phases.length > 0 && (
        <div className="space-y-1">
          {phases.map((phase, idx) => {
            const p = phase as DisclosurePhase;
            const icon = phaseIcons[p.name] || `${idx + 1}`;
            const borderColor = phaseColors[p.name] || 'border-gray-700/40';
            const isExpanded = expandedPhases.has(p.name);
            const isFail = p.status === 'fail';

            return (
              <div key={p.name} className={`border-l-2 ${borderColor} pl-2`}>
                <button
                  type="button"
                  onClick={() => togglePhase(p.name)}
                  className="w-full flex items-center gap-1.5 text-left group"
                >
                  <span className="text-[10px] text-gray-500 font-mono shrink-0 w-4">
                    {icon}
                  </span>
                  <span className={`text-[10px] font-medium ${isFail ? 'text-red-400' : 'text-gray-300'}`}>
                    {p.label}
                  </span>
                  <span className="text-[9px] text-gray-600 font-mono">
                    t={p.timestamp_ms}ms
                  </span>
                  <span className={`text-[9px] px-1 rounded ${
                    isFail ? 'bg-red-900/30 text-red-400' : 'bg-green-900/20 text-green-500'
                  }`}>
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

      {/* ─── Tool Definitions (collapsible) ────────────── */}
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

      {/* ─── System Prompt ─────────────────────────────── */}
      {llm.system_prompt && (
        <div className="text-[9px] text-gray-600 bg-gray-900/40 rounded px-2 py-1 italic leading-relaxed">
          <span className="text-gray-500 not-italic font-medium">system: </span>
          {llm.system_prompt}
        </div>
      )}

      {/* ─── LLM Response (always visible) ─────────────── */}
      <div className="border border-gray-700/40 rounded-lg p-2 bg-gray-900/40">
        <div className="text-[10px] text-blue-400/70 font-medium mb-1">
          LLM Response
        </div>
        {llm.tool_call ? (
          <pre className="text-[10px] text-green-300 bg-gray-900/80 border border-green-800/30 rounded p-2 overflow-x-auto font-mono leading-relaxed">
            {JSON.stringify(llm.tool_call, null, 2)}
          </pre>
        ) : (
          <div className="text-[10px] text-red-300 bg-red-900/20 rounded px-2 py-1">
            No tool call in response
          </div>
        )}
        {llm.raw_content && (
          <div className="mt-1.5 border border-gray-700/40 rounded p-2 bg-gray-900/60">
            <div className="text-[10px] text-gray-400/70 font-medium mb-0.5">Text Response</div>
            <div className="text-[10px] text-gray-300 leading-relaxed whitespace-pre-wrap">{llm.raw_content}</div>
          </div>
        )}

        {/* Usage stats — proves real API call */}
        {llm.usage && Object.keys(llm.usage).length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {Object.entries(llm.usage).map(([k, v]) => (
              <span key={k} className="px-1.5 py-0.5 bg-gray-800 border border-gray-700/50 rounded text-gray-400 text-[9px] font-mono">
                {k}: {String(v)}
              </span>
            ))}
          </div>
        )}

        {/* Model badge */}
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

      {/* ─── Tool Execution Result ────────────────── */}
      {llm.execution_result !== undefined && (
        <div className="border border-gray-700/40 rounded-lg p-2 bg-gray-900/40">
          <div className="text-[10px] text-cyan-400/70 font-medium mb-1">
            Tool Execution Result
          </div>
          <pre className="text-[10px] text-cyan-200 bg-gray-900/80 border border-cyan-800/30 rounded p-2 overflow-x-auto font-mono leading-relaxed max-h-60 overflow-y-auto whitespace-pre-wrap">
            {llm.execution_result}
          </pre>
        </div>
      )}
    </div>
  );
};


// ─── Phase data renderer ─────────────────────────────────────────

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
