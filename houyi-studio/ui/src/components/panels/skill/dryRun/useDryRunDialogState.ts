import { useCallback, useEffect, useRef, useState } from 'react';
import type { SkillTool } from '../../../../types/websocket';
import type { DryRunResultData } from '../../../LeftSidebar/useSkillsLogic';
import type { DryRunPipelineContext } from './computeStages';
import type { ToolDryRunPreset } from './dryRunToolRules';
import { presetToFormValues } from './dryRunInputModel';

interface LiveDefaults {
  provider: string;
  model: string;
}

interface UseDryRunDialogStateParams {
  isOpen: boolean;
  detailTools: SkillTool[];
  dryRunResult: DryRunResultData | null;
  onClearResult?: () => void;
  defaultPlanningFlowId: string;
  getInitialPresetForTool: (toolName: string) => ToolDryRunPreset | null;
  getLiveDefaultsForTool: (toolName: string) => LiveDefaults;
  isPlanningWithFilesTool: (toolName: string) => boolean;
}

export function useDryRunDialogState({
  isOpen,
  detailTools,
  dryRunResult,
  onClearResult,
  defaultPlanningFlowId,
  getInitialPresetForTool,
  getLiveDefaultsForTool,
  isPlanningWithFilesTool,
}: UseDryRunDialogStateParams) {
  const wasOpenRef = useRef(false);
  const [selectedTool, setSelectedTool] = useState<string>(() => detailTools[0]?.name ?? '');
  const [inputMode, setInputMode] = useState<'form' | 'json'>('form');
  const [jsonInput, setJsonInput] = useState('{}');
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [isExecuting, setIsExecuting] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [liveLlmProvider, setLiveLlmProvider] = useState('');
  const [liveLlmModel, setLiveLlmModel] = useState('');
  const [inputCollapsed, setInputCollapsed] = useState<boolean>(!!dryRunResult);
  const [toolDescriptionExpanded, setToolDescriptionExpanded] = useState(false);
  const [selectedExampleId, setSelectedExampleId] = useState<string>(defaultPlanningFlowId);
  const [executedContext, setExecutedContext] = useState<DryRunPipelineContext | null>(null);
  const [revealedCount, setRevealedCount] = useState(0);

  useEffect(() => {
    const justOpened = isOpen && !wasOpenRef.current;
    wasOpenRef.current = isOpen;
    if (!justOpened) return;

    setJsonError(null);
    setFormErrors({});
    setIsExecuting(false);
    setLiveMode(false);
    setLiveLlmProvider('');
    setLiveLlmModel('');
    setInputCollapsed(!!dryRunResult);
    setToolDescriptionExpanded(false);
    setRevealedCount(0);
    setExecutedContext(null);
    onClearResult?.();

    if (detailTools.length > 0) {
      const nextTool = detailTools.some((tool) => tool.name === selectedTool)
        ? selectedTool
        : detailTools[0].name;
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
      return;
    }

    setFormValues({});
    setJsonInput('{}');
  }, [
    isOpen,
    detailTools,
    dryRunResult,
    onClearResult,
    getInitialPresetForTool,
    getLiveDefaultsForTool,
    selectedTool,
  ]);

  const handleToolChange = useCallback((nextTool: string) => {
    const initialPreset = getInitialPresetForTool(nextTool);
    const liveDefaults = getLiveDefaultsForTool(nextTool);
    setSelectedTool(nextTool);
    setLiveLlmProvider(liveDefaults.provider);
    setLiveLlmModel(liveDefaults.model);
    setSelectedExampleId(initialPreset?.id ?? '');
    setFormValues(initialPreset ? presetToFormValues(initialPreset) : {});
    setFormErrors({});
    setJsonInput(initialPreset ? JSON.stringify(initialPreset.input, null, 2) : '{}');
    setExecutedContext(null);
    setInputCollapsed(false);
    onClearResult?.();
  }, [getInitialPresetForTool, getLiveDefaultsForTool, onClearResult]);

  const handleSelectPreset = useCallback((preset: ToolDryRunPreset) => {
    setSelectedExampleId(preset.id);
    setFormValues(presetToFormValues(preset));
    setJsonInput(JSON.stringify(preset.input, null, 2));
    setInputCollapsed(false);
    setExecutedContext(null);
    onClearResult?.();
  }, [onClearResult]);

  const handleSwitchInputMode = useCallback((mode: 'form' | 'json') => {
    setInputMode(mode);
    if (mode === 'form') {
      setJsonError(null);
    } else {
      setFormErrors({});
    }
    setExecutedContext(null);
    setInputCollapsed(false);
    onClearResult?.();
  }, [onClearResult]);

  const handleJsonInputChange = useCallback((value: string) => {
    setJsonInput(value);
    setJsonError(null);
    setFormErrors({});
  }, []);

  const handleToggleLiveMode = useCallback((enabled: boolean) => {
    setLiveMode(enabled);
    if (enabled && !liveLlmProvider && !liveLlmModel) {
      const defaults = getLiveDefaultsForTool(selectedTool);
      setLiveLlmProvider(defaults.provider);
      setLiveLlmModel(defaults.model);
    }
    setExecutedContext(null);
    setInputCollapsed(false);
    onClearResult?.();
  }, [liveLlmProvider, liveLlmModel, selectedTool, getLiveDefaultsForTool, onClearResult]);

  const updateFormField = useCallback((fieldName: string, value: string) => {
    setFormValues((prev) => ({ ...prev, [fieldName]: value }));
    setFormErrors((prev) => {
      if (!prev[fieldName]) return prev;
      const next = { ...prev };
      delete next[fieldName];
      return next;
    });
    if (fieldName === 'action' && isPlanningWithFilesTool(selectedTool)) {
      setFormErrors({});
    }
  }, [isPlanningWithFilesTool, selectedTool]);

  return {
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
  };
}
