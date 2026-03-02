import { useState, useEffect, useCallback, useRef } from 'react';
import type { SkillSummary, SkillDetail, SkillMetricsData, DryRunWorkflowCandidate } from '../../types/websocket';
import { useConsoleStore } from '../../stores/useConsoleStore';

export interface DisclosurePhase {
  name: string;   // discovery | activation | negotiation | execution
  label: string;
  timestamp_ms: number;
  status: string;  // pass | fail
  data: Record<string, unknown>;
}

export interface LlmVerificationResult {
  success: boolean;
  message?: string;
  tool_call?: Record<string, unknown>;
  requested_input?: Record<string, unknown>;
  probe_prompt?: string;
  system_prompt?: string;
  tool_definitions?: Array<Record<string, unknown>>;
  model_name?: string;
  raw_content?: string;
  usage?: Record<string, unknown>;
  phases?: DisclosurePhase[];
  execution_result?: string;
}

export interface DryRunResultData {
  valid: boolean;
  schema_errors: string[];
  policy_result: string;
  capability_gaps: string[];
  estimated_side_effects: string[];
  available_workflows?: DryRunWorkflowCandidate[];
  llm_verification?: LlmVerificationResult;
}

export interface ConsentRequestData {
  request_id: string;
  skill_name: string;
  tool_name: string;
  reason: string;
  permissions: string[];
  timeout_seconds: number;
}

export interface SkillConfigValues {
  policy_action: string;
  auto_invoke: boolean;
}

export interface DryRunLiveOptions {
  llmProvider?: string;
  llmModel?: string;
  workflowId?: string;
}

export interface LoadResultData {
  success: boolean;
  message: string;
}

export type SkillInstallStrategy = 'copy' | 'symlink';

interface UseSkillsLogicReturn {
  skills: SkillSummary[];
  selectedSkill: string | null;
  skillDetail: SkillDetail | null;
  skillMetrics: SkillMetricsData | null;
  isLoadingList: boolean;
  isLoadingDetail: boolean;
  error: string | null;
  dryRunResult: DryRunResultData | null;
  pendingConsent: ConsentRequestData | null;
  blockedMessage: string | null;
  loadResult: LoadResultData | null;
  selectSkill: (skillName: string) => void;
  refreshSkills: () => void;
  loadSkill: (source: string, installStrategy?: SkillInstallStrategy) => void;
  unloadSkill: (skillName: string) => void;
  removeSkillFromDisk: (skillName: string) => void;
  configureSkill: (skillName: string, config: SkillConfigValues) => void;
  dryRunSkill: (
    skillName: string,
    toolName: string,
    input?: Record<string, unknown>,
    live?: boolean,
    options?: DryRunLiveOptions,
  ) => void;
  respondToConsent: (requestId: string, granted: boolean, remember: boolean) => void;
  clearDryRunResult: () => void;
  clearBlockedMessage: () => void;
}

/** Safety timeout (ms) for list_skills commands. */
const LIST_TIMEOUT_MS = 10_000;
const SKILLS_TIMEOUT_ERROR = 'Skills list request timed out';
const SKILLS_UP_TO_DATE_MESSAGE = 'Skills already up to date';

type RefreshReason = 'initial' | 'manual' | 'system';

function fingerprintSkills(skills: SkillSummary[]): string {
  return skills
    .map((skill) => `${skill.name}::${skill.source ?? ''}::${skill.source_group ?? ''}::${skill.runtime_status ?? ''}`)
    .sort()
    .join('|');
}

export function useSkillsLogic(
  sessionId: string,
  sendCommand: (command: Record<string, unknown>) => void,
  registerEventHandler: (eventType: string, handler: (event: unknown) => void) => () => void
): UseSkillsLogicReturn {
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  const [skillDetail, setSkillDetail] = useState<SkillDetail | null>(null);
  const [skillMetrics, setSkillMetrics] = useState<SkillMetricsData | null>(null);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dryRunResult, setDryRunResult] = useState<DryRunResultData | null>(null);
  const [pendingConsent, setPendingConsent] = useState<ConsentRequestData | null>(null);
  const [blockedMessage, setBlockedMessage] = useState<string | null>(null);
  const [loadResult, setLoadResult] = useState<LoadResultData | null>(null);

  // Ref to track selectedSkill without re-registering handlers when it changes.
  const selectedSkillRef = useRef(selectedSkill);
  selectedSkillRef.current = selectedSkill;

  // Ref to track the safety timeout for list_skills.
  // Shared between initial load and manual refresh so it's always cleared properly.
  const listTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const listRequestInFlightRef = useRef(false);
  const refreshReasonRef = useRef<RefreshReason>('initial');
  const skillsFingerprintRef = useRef<string>('');
  const removeFromDiskPendingRef = useRef<Set<string>>(new Set());

  const clearListTimeout = useCallback(() => {
    if (listTimeoutRef.current) {
      clearTimeout(listTimeoutRef.current);
      listTimeoutRef.current = null;
    }
  }, []);

  const requestSkillsList = useCallback((reason: RefreshReason = 'manual') => {
    if (listRequestInFlightRef.current && reason === 'manual') {
      return;
    }
    refreshReasonRef.current = reason;
    listRequestInFlightRef.current = true;
    setIsLoadingList(true);
    setError(null);
    sendCommand({
      command_type: 'list_skills',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
    });

    // Safety timeout — if backend never responds, stop the spinner.
    clearListTimeout();
    listTimeoutRef.current = setTimeout(() => {
      setIsLoadingList((prev) => {
        if (prev) {
          setError(SKILLS_TIMEOUT_ERROR);
        }
        return false;
      });
      listRequestInFlightRef.current = false;
    }, LIST_TIMEOUT_MS);
  }, [sendCommand, sessionId, clearListTimeout]);

  const refreshSkills = useCallback(() => {
    requestSkillsList('manual');
  }, [requestSkillsList]);

  // Register event handlers once (no `selectedSkill` in deps — uses ref instead).
  useEffect(() => {
    const unsubscribeList = registerEventHandler('skill_list', (event: unknown) => {
      const e = event as { skills: SkillSummary[] };
      const incomingSkills = e.skills || [];
      const nextFingerprint = fingerprintSkills(incomingSkills);
      const prevFingerprint = skillsFingerprintRef.current;
      const reason = refreshReasonRef.current;

      setSkills(incomingSkills);
      skillsFingerprintRef.current = nextFingerprint;
      setIsLoadingList(false);
      listRequestInFlightRef.current = false;
      clearListTimeout();

      if (reason === 'manual') {
        if (nextFingerprint === prevFingerprint) {
          useConsoleStore.getState().showToast(SKILLS_UP_TO_DATE_MESSAGE, 'info');
        } else {
          useConsoleStore.getState().showToast(`Skills refreshed (${incomingSkills.length})`, 'success');
        }
      }
      refreshReasonRef.current = 'system';
    });

    const unsubscribeDetail = registerEventHandler('skill_detail', (event: unknown) => {
      const e = event as { skill: SkillDetail };
      setSkillDetail(e.skill);
      setIsLoadingDetail(false);
    });

    const unsubscribeMetrics = registerEventHandler('skill_metrics', (event: unknown) => {
      const e = event as { metrics: SkillMetricsData };
      setSkillMetrics(e.metrics);
    });

    const unsubscribeError = registerEventHandler('skill_error', (event: unknown) => {
      const e = event as { message: string; error_code?: string };
      if (e.error_code === 'remove_from_disk_failed' && selectedSkillRef.current) {
        removeFromDiskPendingRef.current.delete(selectedSkillRef.current);
      }
      setError(e.message);
      setIsLoadingList(false);
      setIsLoadingDetail(false);
      listRequestInFlightRef.current = false;
      clearListTimeout();
      // Propagate load-specific errors to LoadSkillDialog
      const loadCodes = new Set([
        'url_load_failed', 'url_http_error', 'url_unreachable', 'url_download_failed',
        'invalid_url', 'invalid_content', 'parse_failed', 'validation_failed',
        'file_not_found', 'invalid_file', 'no_frontmatter', 'read_failed',
        'no_skills', 'dir_load_failed', 'load_failed', 'missing_source',
      ]);
      if (e.error_code && loadCodes.has(e.error_code)) {
        setLoadResult({ success: false, message: e.message });
      }
      useConsoleStore.getState().showToast(e.message, 'error');
    });

    const unsubscribeLoaded = registerEventHandler('skill_loaded', (event: unknown) => {
      const e = event as { skill_name: string; message?: string };
      requestSkillsList('system');
      setLoadResult({ success: true, message: e.message || `Skill "${e.skill_name}" loaded` });
      useConsoleStore.getState().showToast(
        `Skill "${e.skill_name}" loaded`,
        'success',
      );
    });

    const unsubscribeUnloaded = registerEventHandler('skill_unloaded', (event: unknown) => {
      requestSkillsList('system');
      const e = event as { skill_name: string };
      const removedFromDisk = removeFromDiskPendingRef.current.has(e.skill_name);
      if (removedFromDisk) {
        removeFromDiskPendingRef.current.delete(e.skill_name);
      }
      if (selectedSkillRef.current === e.skill_name) {
        setSelectedSkill(null);
        setSkillDetail(null);
        setSkillMetrics(null);
      }
      useConsoleStore.getState().showToast(
        removedFromDisk
          ? `Skill "${e.skill_name}" removed from disk`
          : `Skill "${e.skill_name}" unloaded`,
        'info',
      );
    });

    const unsubscribeDryRun = registerEventHandler('dry_run_result', (event: unknown) => {
      const e = event as { result: DryRunResultData };
      setDryRunResult(e.result);
    });

    const unsubscribeConsent = registerEventHandler('consent_requested', (event: unknown) => {
      const e = event as ConsentRequestData;
      setPendingConsent(e);
    });

    const unsubscribeConfigured = registerEventHandler('skill_configured', (event: unknown) => {
      const e = event as { skill_name: string; policy_action?: string; auto_invoke?: boolean };

      // Optimistic update: patch skillDetail immediately so the dialog
      // shows the correct value even before the full re-fetch completes.
      setSkillDetail((prev) => {
        if (!prev || prev.name !== e.skill_name) return prev;
        const updatedPolicy = { ...prev.policy };
        if (e.policy_action != null) {
          updatedPolicy.default_action = e.policy_action;
          updatedPolicy.model_auto_invoke = e.policy_action !== 'deny';
        }
        if (e.auto_invoke != null) {
          updatedPolicy.model_auto_invoke = e.auto_invoke;
        }
        return { ...prev, policy: updatedPolicy };
      });

      // Full re-fetch for complete consistency (permissions, side_effect, etc.)
      if (selectedSkillRef.current === e.skill_name) {
        sendCommand({
          command_type: 'get_skill_detail',
          command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
          session_id: sessionId,
          skill_name: e.skill_name,
        });
      }
      requestSkillsList('system');
      useConsoleStore.getState().showToast(
        `Skill "${e.skill_name}" configuration saved`,
        'success',
      );
    });

    const unsubscribeBlocked = registerEventHandler('skill_blocked', (event: unknown) => {
      const e = event as { skill_name: string; reason: string };
      setBlockedMessage(`Skill '${e.skill_name}' blocked: ${e.reason}`);
    });

    // Initial load.
    requestSkillsList('initial');

    return () => {
      unsubscribeList();
      unsubscribeDetail();
      unsubscribeMetrics();
      unsubscribeError();
      unsubscribeLoaded();
      unsubscribeUnloaded();
      unsubscribeDryRun();
      unsubscribeConsent();
      unsubscribeConfigured();
      unsubscribeBlocked();
      clearListTimeout();
      listRequestInFlightRef.current = false;
    };
  }, [registerEventHandler, requestSkillsList, clearListTimeout, sendCommand, sessionId]);

  const selectSkill = useCallback((skillName: string) => {
    const isReselectingSameSkill = selectedSkillRef.current === skillName;
    setSelectedSkill(skillName);
    setIsLoadingDetail(!isReselectingSameSkill);
    setError(null);
    setDryRunResult(null);

    sendCommand({
      command_type: 'get_skill_detail',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
    });

    sendCommand({
      command_type: 'get_skill_metrics',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
    });
  }, [sendCommand, sessionId]);

  const removeSkillFromDisk = useCallback((skillName: string) => {
    removeFromDiskPendingRef.current.add(skillName);
    sendCommand({
      command_type: 'remove_skill_from_disk',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
    });
  }, [sendCommand, sessionId]);

  const loadSkill = useCallback((source: string, installStrategy?: SkillInstallStrategy) => {
    setLoadResult(null);
    sendCommand({
      command_type: 'load_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      source,
      install_strategy: installStrategy,
    });
  }, [sendCommand, sessionId]);

  const unloadSkill = useCallback((skillName: string) => {
    sendCommand({
      command_type: 'unload_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
    });
  }, [sendCommand, sessionId]);

  const configureSkill = useCallback((skillName: string, config: SkillConfigValues) => {
    // Synchronous optimistic update — must happen BEFORE the dialog
    // closes so that reopening it immediately shows the saved value.
    setSkillDetail((prev) => {
      if (!prev || prev.name !== skillName) return prev;
      return {
        ...prev,
        policy: {
          ...prev.policy,
          default_action: config.policy_action,
          model_auto_invoke: config.policy_action !== 'deny',
        },
      };
    });

    sendCommand({
      command_type: 'configure_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
      policy_action: config.policy_action,
      auto_invoke: config.auto_invoke,
    });
  }, [sendCommand, sessionId]);

  const dryRunSkill = useCallback((
    skillName: string,
    toolName: string,
    input?: Record<string, unknown>,
    live?: boolean,
    options?: DryRunLiveOptions,
  ) => {
    setDryRunResult(null);
    const llmProvider = options?.llmProvider?.trim();
    const llmModel = options?.llmModel?.trim();
    const workflowId = options?.workflowId?.trim();
    const payloadInput = {
      ...(input ?? {}),
      ...(workflowId ? { workflow_id: workflowId } : {}),
    };
    sendCommand({
      command_type: 'dry_run_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
      tool_name: toolName,
      input: payloadInput,
      live: live ?? false,
      llm_provider: llmProvider || undefined,
      llm_model: llmModel || undefined,
    });
  }, [sendCommand, sessionId]);

  const respondToConsent = useCallback((requestId: string, granted: boolean, remember: boolean) => {
    sendCommand({
      command_type: 'consent_response',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      request_id: requestId,
      granted,
      remember,
    });
    setPendingConsent(null);
  }, [sendCommand, sessionId]);

  const clearDryRunResult = useCallback(() => setDryRunResult(null), []);
  const clearBlockedMessage = useCallback(() => setBlockedMessage(null), []);

  return {
    skills,
    selectedSkill,
    skillDetail,
    skillMetrics,
    isLoadingList,
    isLoadingDetail,
    error,
    dryRunResult,
    pendingConsent,
    blockedMessage,
    loadResult,
    selectSkill,
    refreshSkills,
    loadSkill,
    unloadSkill,
    removeSkillFromDisk,
    configureSkill,
    dryRunSkill,
    respondToConsent,
    clearDryRunResult,
    clearBlockedMessage,
  };
}

export default useSkillsLogic;
