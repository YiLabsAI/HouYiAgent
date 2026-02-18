import { useState, useEffect, useCallback, useRef } from 'react';
import type { SkillSummary, SkillDetail, SkillMetricsData } from '../../types/websocket';
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

export interface LoadResultData {
  success: boolean;
  message: string;
}

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
  loadSkill: (path: string) => void;
  unloadSkill: (skillName: string) => void;
  configureSkill: (skillName: string, config: SkillConfigValues) => void;
  dryRunSkill: (skillName: string, toolName: string, input?: Record<string, unknown>, live?: boolean) => void;
  respondToConsent: (requestId: string, granted: boolean, remember: boolean) => void;
  clearDryRunResult: () => void;
  clearBlockedMessage: () => void;
}

/** Safety timeout (ms) for list_skills commands. */
const LIST_TIMEOUT_MS = 10_000;

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

  const clearListTimeout = useCallback(() => {
    if (listTimeoutRef.current) {
      clearTimeout(listTimeoutRef.current);
      listTimeoutRef.current = null;
    }
  }, []);

  const refreshSkills = useCallback(() => {
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
          setError('Skills list request timed out');
        }
        return false;
      });
    }, LIST_TIMEOUT_MS);
  }, [sendCommand, sessionId, clearListTimeout]);

  // Register event handlers once (no `selectedSkill` in deps — uses ref instead).
  useEffect(() => {
    const unsubscribeList = registerEventHandler('skill_list', (event: unknown) => {
      const e = event as { skills: SkillSummary[] };
      setSkills(e.skills || []);
      setIsLoadingList(false);
      clearListTimeout();
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
      setError(e.message);
      setIsLoadingList(false);
      setIsLoadingDetail(false);
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
      refreshSkills();
      setLoadResult({ success: true, message: e.message || `Skill "${e.skill_name}" loaded` });
      useConsoleStore.getState().showToast(
        `Skill "${e.skill_name}" loaded`,
        'success',
      );
    });

    const unsubscribeUnloaded = registerEventHandler('skill_unloaded', (event: unknown) => {
      refreshSkills();
      const e = event as { skill_name: string };
      if (selectedSkillRef.current === e.skill_name) {
        setSelectedSkill(null);
        setSkillDetail(null);
        setSkillMetrics(null);
      }
      useConsoleStore.getState().showToast(
        `Skill "${e.skill_name}" unloaded`,
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
      const e = event as { skill_name: string };
      // Re-fetch detail to reflect the new configuration
      if (selectedSkillRef.current === e.skill_name) {
        sendCommand({
          command_type: 'get_skill_detail',
          command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
          session_id: sessionId,
          skill_name: e.skill_name,
        });
      }
      // Also refresh the skills list since policy_action badge may have changed
      refreshSkills();
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
    refreshSkills();

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
    };
  }, [registerEventHandler, refreshSkills, clearListTimeout, sendCommand, sessionId]);

  const selectSkill = useCallback((skillName: string) => {
    setSelectedSkill(skillName);
    setIsLoadingDetail(true);
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

  const loadSkill = useCallback((source: string) => {
    setLoadResult(null);
    sendCommand({
      command_type: 'load_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      source,
      path: source,
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
    sendCommand({
      command_type: 'configure_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
      policy_action: config.policy_action,
      auto_invoke: config.auto_invoke,
    });
  }, [sendCommand, sessionId]);

  const dryRunSkill = useCallback((skillName: string, toolName: string, input?: Record<string, unknown>, live?: boolean) => {
    setDryRunResult(null);
    sendCommand({
      command_type: 'dry_run_skill',
      command_id: `cmd_${crypto.randomUUID().slice(0, 8)}`,
      session_id: sessionId,
      skill_name: skillName,
      tool_name: toolName,
      input: input ?? {},
      live: live ?? false,
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
    configureSkill,
    dryRunSkill,
    respondToConsent,
    clearDryRunResult,
    clearBlockedMessage,
  };
}

export default useSkillsLogic;
