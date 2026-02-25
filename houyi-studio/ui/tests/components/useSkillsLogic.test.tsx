/**
 * Tests for useSkillsLogic hook — Skill lifecycle management via WebSocket commands.
 *
 * Covers:
 *   - Initial state and auto-refresh on mount
 *   - Event handler registration (10 handlers) and cleanup on unmount
 *   - Command dispatching: refreshSkills, selectSkill, loadSkill, unloadSkill,
 *     dryRunSkill, respondToConsent
 *   - State transitions from server events (skill_list, skill_detail, skill_metrics,
 *     skill_error, skill_loaded, skill_unloaded, dry_run_result, consent_requested,
 *     skill_blocked)
 *   - Edge cases: unload clears selection when matching, error resets loading flags
 *   - crypto.randomUUID() command_id format
 */
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useSkillsLogic } from '@/components/LeftSidebar/useSkillsLogic';
import type { SkillSummary, SkillDetail, SkillMetricsData } from '@/types/websocket';

// --- Helpers ---

/** Map of event type → registered handler, with unsub spies */
type HandlerMap = Map<string, { handler: (event: unknown) => void; unsub: ReturnType<typeof vi.fn> }>;

function createMocks() {
  const sendCommand = vi.fn();
  const handlerMap: HandlerMap = new Map();

  const registerEventHandler = vi.fn((eventType: string, handler: (event: unknown) => void) => {
    const unsub = vi.fn();
    handlerMap.set(eventType, { handler, unsub });
    return unsub;
  });

  /** Simulate a server event arriving */
  const emitEvent = (eventType: string, payload: unknown) => {
    const entry = handlerMap.get(eventType);
    if (!entry) throw new Error(`No handler registered for '${eventType}'`);
    entry.handler(payload);
  };

  return { sendCommand, registerEventHandler, handlerMap, emitEvent };
}

const SESSION_ID = 'test-session-001';

const MOCK_SKILLS: SkillSummary[] = [
  {
    name: 'get_weather',
    display_name: 'Weather',
    tools: ['get_weather'],
    certification: 'silver',
    policy_action: 'allow',
    side_effect: 'none',
  },
  {
    name: 'web_search',
    display_name: 'Web Search',
    tools: ['search_web'],
    certification: 'gold',
    policy_action: 'allow_with_consent',
    side_effect: 'network',
  },
];

const MOCK_DETAIL: SkillDetail = {
  name: 'get_weather',
  display_name: 'Weather',
  description: 'Get weather for a location',
  version: '1.0.0',
  author: 'HouYi',
  tools: [{ name: 'get_weather', description: 'Get weather' }],
  permissions: [],
  policy: { default_action: 'allow' },
  hooks: ['PreToolUse'],
  certification: 'silver',
  side_effect: 'none',
};

const MOCK_METRICS: SkillMetricsData = {
  skill_name: 'get_weather',
  total_calls: 42,
  success_count: 40,
  failure_count: 2,
  success_rate: 0.95,
  avg_latency_ms: 120,
  p50_latency_ms: 95,
  p99_latency_ms: 380,
  last_invoked: '2026-02-03T10:00:00Z',
};

// Ensure crypto.randomUUID is available in jsdom
beforeEach(() => {
  if (!globalThis.crypto?.randomUUID) {
    Object.defineProperty(globalThis, 'crypto', {
      value: {
        ...globalThis.crypto,
        randomUUID: () => '12345678-1234-1234-1234-123456789012',
      },
      configurable: true,
    });
  }
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useSkillsLogic', () => {
  let mocks: ReturnType<typeof createMocks>;

  beforeEach(() => {
    mocks = createMocks();
  });

  // --- Initial state ---

  it('returns correct initial state', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    expect(result.current.skills).toEqual([]);
    expect(result.current.selectedSkill).toBeNull();
    expect(result.current.skillDetail).toBeNull();
    expect(result.current.skillMetrics).toBeNull();
    expect(result.current.isLoadingList).toBe(true); // initial refresh sets loading
    expect(result.current.isLoadingDetail).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.dryRunResult).toBeNull();
    expect(result.current.pendingConsent).toBeNull();
    expect(result.current.blockedMessage).toBeNull();
  });

  // --- Event handler registration ---

  it('registers all 10 event handlers on mount', () => {
    renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    const expectedEvents = [
      'skill_list', 'skill_detail', 'skill_metrics', 'skill_error',
      'skill_loaded', 'skill_unloaded', 'skill_configured', 'dry_run_result', 'consent_requested',
      'skill_blocked',
    ];

    expect(mocks.registerEventHandler).toHaveBeenCalledTimes(expectedEvents.length);
    for (const eventType of expectedEvents) {
      expect(mocks.handlerMap.has(eventType)).toBe(true);
    }
  });

  it('unsubscribes all handlers on unmount', () => {
    const { unmount } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    unmount();

    for (const [, entry] of mocks.handlerMap) {
      expect(entry.unsub).toHaveBeenCalled();
    }
  });

  // --- Initial auto-refresh ---

  it('sends list_skills command on mount', () => {
    renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command_type: 'list_skills',
        session_id: SESSION_ID,
      }),
    );
  });

  it('generates command_id with cmd_ prefix and 8-char suffix', () => {
    renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    const call = mocks.sendCommand.mock.calls[0][0];
    expect(call.command_id).toMatch(/^cmd_[a-f0-9]{8}$/);
  });

  // --- refreshSkills ---

  it('refreshSkills sends list_skills and sets loading', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.refreshSkills();
    });

    expect(result.current.isLoadingList).toBe(true);
    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({ command_type: 'list_skills', session_id: SESSION_ID }),
    );
  });

  it('manual refreshSkills clears loading when skill_list arrives', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    // Complete initial load
    act(() => {
      mocks.emitEvent('skill_list', { skills: MOCK_SKILLS });
    });
    expect(result.current.isLoadingList).toBe(false);
    expect(result.current.skills).toEqual(MOCK_SKILLS);

    // Manual refresh
    act(() => {
      result.current.refreshSkills();
    });
    expect(result.current.isLoadingList).toBe(true);

    // Simulate backend response
    act(() => {
      mocks.emitEvent('skill_list', { skills: MOCK_SKILLS });
    });
    expect(result.current.isLoadingList).toBe(false);
    expect(result.current.skills).toEqual(MOCK_SKILLS);
  });

  it('refreshSkills times out after 10s if no response', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });

    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    // Initial load — don't respond
    expect(result.current.isLoadingList).toBe(true);

    // Fast-forward past the safety timeout
    act(() => {
      vi.advanceTimersByTime(11_000);
    });

    expect(result.current.isLoadingList).toBe(false);
    expect(result.current.error).toBe('Skills list request timed out');

    vi.useRealTimers();
  });

  it('manual refreshSkills also has safety timeout', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });

    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    // Complete initial load
    act(() => {
      mocks.emitEvent('skill_list', { skills: MOCK_SKILLS });
    });

    // Manual refresh — don't respond
    act(() => {
      result.current.refreshSkills();
    });
    expect(result.current.isLoadingList).toBe(true);

    // Fast-forward past the safety timeout
    act(() => {
      vi.advanceTimersByTime(11_000);
    });

    expect(result.current.isLoadingList).toBe(false);
    expect(result.current.error).toBe('Skills list request timed out');

    vi.useRealTimers();
  });

  // --- selectSkill ---

  it('selectSkill sends detail + metrics commands and sets loading', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.selectSkill('get_weather');
    });

    expect(result.current.selectedSkill).toBe('get_weather');
    expect(result.current.isLoadingDetail).toBe(true);

    // Should have sent 2 commands: get_skill_detail + get_skill_metrics
    const calls = mocks.sendCommand.mock.calls.map(c => c[0]);
    expect(calls).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ command_type: 'get_skill_detail', skill_name: 'get_weather' }),
        expect.objectContaining({ command_type: 'get_skill_metrics', skill_name: 'get_weather' }),
      ]),
    );
  });

  // --- loadSkill ---

  it('loadSkill sends load_skill command with path', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.loadSkill('/skills/weather.md');
    });

    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command_type: 'load_skill',
        session_id: SESSION_ID,
        path: '/skills/weather.md',
      }),
    );
  });

  // --- unloadSkill ---

  it('unloadSkill sends unload_skill command', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.unloadSkill('get_weather');
    });

    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command_type: 'unload_skill',
        skill_name: 'get_weather',
      }),
    );
  });

  // --- dryRunSkill ---

  it('dryRunSkill clears previous result and sends command', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.dryRunSkill('get_weather', 'get_weather_tool');
    });

    expect(result.current.dryRunResult).toBeNull();
    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command_type: 'dry_run_skill',
        skill_name: 'get_weather',
        tool_name: 'get_weather_tool',
        input: {},
      }),
    );
  });

  // --- respondToConsent ---

  it('respondToConsent sends consent_response and clears pending', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    // Simulate a consent request arriving
    act(() => {
      mocks.emitEvent('consent_requested', {
        request_id: 'req-1',
        skill_name: 'web_search',
        tool_name: 'search',
        reason: 'Network access',
        permissions: ['network'],
        timeout_seconds: 30,
      });
    });

    expect(result.current.pendingConsent).not.toBeNull();

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.respondToConsent('req-1', true, false);
    });

    expect(result.current.pendingConsent).toBeNull();
    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command_type: 'consent_response',
        request_id: 'req-1',
        granted: true,
        remember: false,
      }),
    );
  });

  // --- clearDryRunResult / clearBlockedMessage ---

  it('clearDryRunResult resets dry run result', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    // Set a result via event
    act(() => {
      mocks.emitEvent('dry_run_result', {
        result: { valid: true, schema_errors: [], policy_result: 'allow', capability_gaps: [], estimated_side_effects: [] },
      });
    });
    expect(result.current.dryRunResult).not.toBeNull();

    act(() => {
      result.current.clearDryRunResult();
    });
    expect(result.current.dryRunResult).toBeNull();
  });

  it('clearBlockedMessage resets blocked message', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    act(() => {
      mocks.emitEvent('skill_blocked', { skill_name: 'danger', reason: 'policy deny' });
    });
    expect(result.current.blockedMessage).toBe("Skill 'danger' blocked: policy deny");

    act(() => {
      result.current.clearBlockedMessage();
    });
    expect(result.current.blockedMessage).toBeNull();
  });

  // --- Server event handling ---

  describe('server events', () => {
    it('skill_list updates skills and clears loading', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        mocks.emitEvent('skill_list', { skills: MOCK_SKILLS });
      });

      expect(result.current.skills).toEqual(MOCK_SKILLS);
      expect(result.current.isLoadingList).toBe(false);
    });

    it('skill_list with empty skills defaults to []', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        mocks.emitEvent('skill_list', { skills: undefined });
      });

      expect(result.current.skills).toEqual([]);
    });

    it('skill_detail updates detail and clears loading', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        mocks.emitEvent('skill_detail', { skill: MOCK_DETAIL });
      });

      expect(result.current.skillDetail).toEqual(MOCK_DETAIL);
      expect(result.current.isLoadingDetail).toBe(false);
    });

    it('skill_metrics updates metrics', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        mocks.emitEvent('skill_metrics', { metrics: MOCK_METRICS });
      });

      expect(result.current.skillMetrics).toEqual(MOCK_METRICS);
    });

    it('skill_error updates error and resets both loading flags', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      // Start both loading states
      act(() => {
        result.current.selectSkill('get_weather');
      });
      expect(result.current.isLoadingList).toBe(true); // from initial refresh
      expect(result.current.isLoadingDetail).toBe(true);

      act(() => {
        mocks.emitEvent('skill_error', { message: 'Skill not found' });
      });

      expect(result.current.error).toBe('Skill not found');
      expect(result.current.isLoadingList).toBe(false);
      expect(result.current.isLoadingDetail).toBe(false);
    });

    it('skill_loaded triggers refresh', () => {
      renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      mocks.sendCommand.mockClear();

      act(() => {
        mocks.emitEvent('skill_loaded', { skill_name: 'new_skill' });
      });

      expect(mocks.sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({ command_type: 'list_skills' }),
      );
    });

    it('skill_unloaded triggers refresh', () => {
      renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      mocks.sendCommand.mockClear();

      act(() => {
        mocks.emitEvent('skill_unloaded', { skill_name: 'old_skill' });
      });

      expect(mocks.sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({ command_type: 'list_skills' }),
      );
    });

    it('skill_unloaded clears selection when matching selected skill', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      // Select a skill first
      act(() => {
        result.current.selectSkill('get_weather');
      });
      expect(result.current.selectedSkill).toBe('get_weather');

      // Re-render to update the closure captured by the event handler
      // (selectedSkill is in the dependency array of the useEffect)

      // Unload the same skill
      act(() => {
        mocks.emitEvent('skill_unloaded', { skill_name: 'get_weather' });
      });

      expect(result.current.selectedSkill).toBeNull();
      expect(result.current.skillDetail).toBeNull();
      expect(result.current.skillMetrics).toBeNull();
    });

    it('skill_unloaded does NOT clear selection when different skill', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        result.current.selectSkill('get_weather');
      });

      act(() => {
        mocks.emitEvent('skill_unloaded', { skill_name: 'other_skill' });
      });

      expect(result.current.selectedSkill).toBe('get_weather');
    });

    it('dry_run_result updates dryRunResult', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      const mockResult = {
        valid: false,
        schema_errors: ['missing field: input'],
        policy_result: 'deny',
        capability_gaps: ['network'],
        estimated_side_effects: ['filesystem'],
      };

      act(() => {
        mocks.emitEvent('dry_run_result', { result: mockResult });
      });

      expect(result.current.dryRunResult).toEqual(mockResult);
    });

    it('consent_requested sets pendingConsent', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      const consentData = {
        request_id: 'req-abc',
        skill_name: 'web_search',
        tool_name: 'search',
        reason: 'Requires network',
        permissions: ['network:outbound'],
        timeout_seconds: 60,
      };

      act(() => {
        mocks.emitEvent('consent_requested', consentData);
      });

      expect(result.current.pendingConsent).toEqual(consentData);
    });

    it('skill_blocked formats message correctly', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        mocks.emitEvent('skill_blocked', { skill_name: 'risky_skill', reason: 'InvocationPolicy deny' });
      });

      expect(result.current.blockedMessage).toBe("Skill 'risky_skill' blocked: InvocationPolicy deny");
    });
  });

  // --- configureSkill optimistic update ---

  describe('configureSkill', () => {
    it('updates skillDetail.policy synchronously before WebSocket round-trip', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      // 1. Select a skill and populate detail
      act(() => {
        result.current.selectSkill('web_search');
      });
      act(() => {
        mocks.emitEvent('skill_detail', {
          skill: {
            ...MOCK_DETAIL,
            name: 'web_search',
            display_name: 'Web Search',
            policy: { default_action: 'allow_with_consent', model_auto_invoke: true },
            side_effect: 'network',
          },
        });
      });
      expect(result.current.skillDetail!.policy.default_action).toBe('allow_with_consent');

      // 2. Configure: change policy to 'allow'
      mocks.sendCommand.mockClear();
      act(() => {
        result.current.configureSkill('web_search', { policy_action: 'allow', auto_invoke: true });
      });

      // 3. Verify: detail.policy is updated IMMEDIATELY (no event needed)
      expect(result.current.skillDetail!.policy.default_action).toBe('allow');
      expect(result.current.skillDetail!.policy.model_auto_invoke).toBe(true);

      // 4. Verify: command was also sent to backend
      expect(mocks.sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'configure_skill',
          skill_name: 'web_search',
          policy_action: 'allow',
          auto_invoke: true,
        }),
      );
    });

    it('sets model_auto_invoke=false when policy is deny', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        result.current.selectSkill('web_search');
      });
      act(() => {
        mocks.emitEvent('skill_detail', {
          skill: {
            ...MOCK_DETAIL,
            name: 'web_search',
            policy: { default_action: 'allow', model_auto_invoke: true },
          },
        });
      });

      act(() => {
        result.current.configureSkill('web_search', { policy_action: 'deny', auto_invoke: false });
      });

      expect(result.current.skillDetail!.policy.default_action).toBe('deny');
      expect(result.current.skillDetail!.policy.model_auto_invoke).toBe(false);
    });

    it('does not update detail when skill name does not match', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      act(() => {
        result.current.selectSkill('get_weather');
      });
      act(() => {
        mocks.emitEvent('skill_detail', { skill: MOCK_DETAIL });
      });
      expect(result.current.skillDetail!.policy.default_action).toBe('allow');

      act(() => {
        result.current.configureSkill('other_skill', { policy_action: 'deny', auto_invoke: false });
      });

      // Should NOT change because names don't match
      expect(result.current.skillDetail!.policy.default_action).toBe('allow');
    });

    it('survives rapid save-reopen cycle without losing the update', () => {
      const { result } = renderHook(() =>
        useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
      );

      // Populate detail
      act(() => {
        result.current.selectSkill('web_search');
      });
      act(() => {
        mocks.emitEvent('skill_detail', {
          skill: {
            ...MOCK_DETAIL,
            name: 'web_search',
            policy: { default_action: 'allow_with_consent', model_auto_invoke: true },
          },
        });
      });

      // Save 'allow' — simulates the dialog onSave → configureSkill → close pattern
      act(() => {
        result.current.configureSkill('web_search', { policy_action: 'allow', auto_invoke: true });
      });
      // At this point dialog would close (setShowSkillConfig(false))
      // User immediately reopens — detail should already reflect 'allow'
      expect(result.current.skillDetail!.policy.default_action).toBe('allow');

      // Then the server event arrives (late) — should NOT revert
      act(() => {
        mocks.emitEvent('skill_configured', {
          skill_name: 'web_search',
          policy_action: 'allow',
          auto_invoke: true,
        });
      });
      expect(result.current.skillDetail!.policy.default_action).toBe('allow');
    });
  });

  // --- respondToConsent with deny ---

  it('respondToConsent with deny sends granted=false', () => {
    const { result } = renderHook(() =>
      useSkillsLogic(SESSION_ID, mocks.sendCommand, mocks.registerEventHandler),
    );

    act(() => {
      mocks.emitEvent('consent_requested', {
        request_id: 'req-2',
        skill_name: 's',
        tool_name: 't',
        reason: 'r',
        permissions: [],
        timeout_seconds: 10,
      });
    });

    mocks.sendCommand.mockClear();

    act(() => {
      result.current.respondToConsent('req-2', false, true);
    });

    expect(mocks.sendCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        command_type: 'consent_response',
        request_id: 'req-2',
        granted: false,
        remember: true,
      }),
    );
  });
});
