/**
 * Zustand store for Deep Research workspace state.
 *
 * Manages session lifecycle, plan editing, SSE event streaming with
 * reconnection support, and report retrieval.
 */
import { create } from 'zustand';
import { buildVisibleChatError, type ChatErrorPayload } from '@/utils/chatErrors';

const API_BASE = '/api/research';
const CREATE_RUN_TIMEOUT_MS = 90000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ResearchPhase = 'input' | 'planning' | 'executing' | 'report';

export interface SubQuestion {
  question_id: string;
  question: string;
  priority: number;
  search_strategy: string;
  expected_sources: number;
  depends_on: string[];
}

export interface OutlineSection {
  title: string;
  description: string;
  related_question_ids: string[];
}

export interface ResearchPlan {
  query: string;
  sub_questions: SubQuestion[];
  outline: OutlineSection[];
  version: number;
  status: string;
}

export interface ResearchProgress {
  total_steps: number;
  completed_steps: number;
  current_step: string;
  elapsed_seconds: number;
  sub_question_progress: Record<string, string>;
}

export interface ReportSection {
  title: string;
  content: string;
  citations: string[];
}

export interface SourceReference {
  reference_id?: string;
  url: string;
  title: string;
  snippet: string;
  reliability: number;
  reliability_score?: number;
}

export interface ResearchReport {
  title: string;
  sections: ReportSection[];
  references: SourceReference[];
  quality_score: { race_overall: number; fact_overall: number } | null;
}

export interface SessionSummary {
  run_id: string;
  query?: string;
  status: string;
  created_at?: string;
  started_at?: number;
  error?: string | null;
}

export interface SSEEvent {
  event_id: string;
  event_type: string;
  sequence: number;
  payload: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export interface SearchResultData {
  question_id: string;
  sources: SourceReference[];
  summary: string;
  coverage_score: number;
  error?: string | null;
}

interface ResearchState {
  phase: ResearchPhase;
  sessionId: string | null;
  plan: ResearchPlan | null;
  progress: ResearchProgress | null;
  report: ResearchReport | null;
  searchResults: SearchResultData[] | null;
  sessions: SessionSummary[];
  events: SSEEvent[];
  error: string | null;
  loading: boolean;

  lastEventId: string | null;
  lastSequence: number;
  sseAbort: AbortController | null;

  // Actions
  createSession: (query: string, settings?: Record<string, unknown>) => Promise<void>;
  editPlan: (edits: Array<Record<string, unknown>>) => Promise<void>;
  confirmAndExecute: () => Promise<void>;
  cancelSession: () => Promise<void>;
  fetchReport: () => Promise<void>;
  fetchSessions: () => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  connectSSE: (sessionId: string) => void;
  disconnectSSE: () => void;
  openSession: (sessionId: string, options?: { preserveState?: boolean }) => Promise<void>;
  reset: () => void;
}

async function researchFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

function normalizeResearchError(message: string): string {
  const raw = message.trim();
  if (!raw) return 'Research request failed. Please try again.';

  const payload = _toResearchChatErrorPayload(raw);
  const mapped = buildVisibleChatError(payload, 'generic');
  if (mapped && mapped.trim()) {
    return mapped;
  }
  return 'Research request failed. Please try again.';
}

function _toResearchChatErrorPayload(raw: string): ChatErrorPayload {
  const text = _extractErrorDetail(raw);
  const normalized = text.replace(/^LLM\/planning error:\s*/i, '').trim();
  const upper = normalized.toUpperCase();

  if (
    upper.includes('TIMED OUT') ||
    upper.includes('TIMEOUT') ||
    upper.includes('DEADLINE_EXCEEDED') ||
    upper.includes('ABORTERROR')
  ) {
    return { error_code: 'provider_timeout', error: normalized };
  }

  if (
    upper.includes('SERVER DISCONNECTED WITHOUT SENDING A RESPONSE') ||
    upper.includes('REMOTEPROTOCOLERROR') ||
    upper.includes('ECONNRESET') ||
    upper.includes('NETWORK') ||
    upper.includes('CONNECTION RESET')
  ) {
    return { error_code: 'provider_network_error', error: normalized };
  }

  if (upper.includes('429') || upper.includes('RESOURCE_EXHAUSTED') || upper.includes('RATE LIMIT')) {
    return { error_code: 'provider_rate_limited', error: normalized };
  }

  if (upper.includes('401') || upper.includes('UNAUTHENTICATED') || upper.includes('INVALID API KEY')) {
    return { error_code: 'provider_auth_failed', error: normalized };
  }

  return { error: normalized, error_code: 'provider_request_failed' };
}

function _extractErrorDetail(raw: string): string {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (typeof parsed.public_message === 'string' && parsed.public_message.trim()) {
      return parsed.public_message;
    }
    if (typeof parsed.detail === 'string' && parsed.detail.trim()) {
      return parsed.detail;
    }
    if (typeof parsed.error === 'string' && parsed.error.trim()) {
      return parsed.error;
    }
  } catch {
  }
  return raw;
}

const initialState = {
  phase: 'input' as ResearchPhase,
  sessionId: null as string | null,
  plan: null as ResearchPlan | null,
  progress: null as ResearchProgress | null,
  report: null as ResearchReport | null,
  searchResults: null as SearchResultData[] | null,
  sessions: [] as SessionSummary[],
  events: [] as SSEEvent[],
  error: null as string | null,
  loading: false,
  lastEventId: null as string | null,
  lastSequence: 0,
  sseAbort: null as AbortController | null,
};

export const useResearchStore = create<ResearchState>((set, get) => ({
  ...initialState,

  createSession: async (query, settings) => {
    set({ loading: true, error: null, phase: 'planning', plan: null, sessionId: null });
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), CREATE_RUN_TIMEOUT_MS);
    try {
      const data = await researchFetch<{
        run_id: string;
        plan: ResearchPlan;
        status: string;
      }>('/runs', {
        method: 'POST',
        body: JSON.stringify({ query, settings }),
        signal: controller.signal,
      });
      set({
        sessionId: data.run_id,
        plan: data.plan,
        phase: 'planning',
        loading: false,
      });
    } catch (e) {
      set({
        error: normalizeResearchError((e as Error).message),
        loading: false,
        phase: 'input',
      });
    } finally {
      window.clearTimeout(timeout);
    }
  },

  editPlan: async (edits) => {
    const { sessionId, plan } = get();
    if (!sessionId) return;
    set({ loading: true, error: null });
    try {
      const data = await researchFetch<{ plan: ResearchPlan }>(
        `/runs/${sessionId}/plan`,
        {
          method: 'PUT',
          body: JSON.stringify({
            edits,
            client_plan_version: plan?.version,
          }),
        },
      );
      set({ plan: data.plan, loading: false });
    } catch (e) {
      set({ error: normalizeResearchError((e as Error).message), loading: false });
    }
  },

  confirmAndExecute: async () => {
    const { sessionId, connectSSE } = get();
    if (!sessionId) return;
    set({
      loading: true,
      error: null,
      phase: 'executing',
      progress: null,
      events: [],
      report: null,
      searchResults: null,
      lastEventId: null,
      lastSequence: 0,
    });
    try {
      await researchFetch(`/runs/${sessionId}/start`, {
        method: 'POST',
        body: JSON.stringify({ resume_if_running: false }),
      });
      connectSSE(sessionId);
      set({ loading: false });
    } catch (e) {
      const msg = (e as Error).message;
      const friendly = msg.includes('max_concurrent_runs')
        ? 'Too many runs are active. Please wait for one to finish or cancel one first.'
        : msg;
      set({ error: friendly, loading: false, phase: 'planning' });
    }
  },

  cancelSession: async () => {
    const { sessionId, disconnectSSE } = get();
    if (!sessionId) return;
    try {
      await researchFetch(`/runs/${sessionId}/cancel`, {
        method: 'POST',
        body: JSON.stringify({ reason: 'user_cancelled' }),
      });
      disconnectSSE();
      set({ phase: 'input', error: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchReport: async () => {
    const { sessionId } = get();
    if (!sessionId) return;
    set({ loading: true });
    try {
      const data = await researchFetch<{ report: ResearchReport }>(
        `/runs/${sessionId}/report`,
      );
      set({ report: data.report, phase: 'report', loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchSessions: async () => {
    try {
      const data = await researchFetch<{ runs: SessionSummary[] }>('/runs');
      set({ sessions: data.runs });
    } catch {
      // silent
    }
  },

  deleteSession: async (sessionId: string) => {
    try {
      await researchFetch(`/runs/${sessionId}`, { method: 'DELETE' });
      set({ sessions: get().sessions.filter((s) => s.run_id !== sessionId) });
    } catch {
      // silent
    }
  },

  connectSSE: (sessionId: string) => {
    const { disconnectSSE, lastEventId } = get();
    disconnectSSE();

    const abort = new AbortController();
    set({ sseAbort: abort });

    const url = new URL(`${API_BASE}/runs/${sessionId}/events`, window.location.origin);
    if (lastEventId) url.searchParams.set('last_event_id', lastEventId);

    fetch(url.toString(), { signal: abort.signal, headers: { Accept: 'text/event-stream' } })
      .then(async (res) => {
        if (!res.body) return;
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;
            try {
              const evt: SSEEvent = JSON.parse(raw);
              const state = get();

              if (evt.sequence <= state.lastSequence) continue;

              const newEvents = [...state.events, evt];
              const updates: Partial<ResearchState> = {
                events: newEvents,
                lastEventId: evt.event_id,
                lastSequence: evt.sequence,
              };

              if (evt.event_type === 'research.step_started' || evt.event_type === 'research.step_completed') {
                updates.progress = {
                  ...(state.progress || { total_steps: 0, completed_steps: 0, current_step: '', elapsed_seconds: 0, sub_question_progress: {} }),
                  current_step: (evt.payload.step as string) || state.progress?.current_step || '',
                  completed_steps: (evt.payload.completed_steps as number) ?? state.progress?.completed_steps ?? 0,
                  total_steps: (evt.payload.total_steps as number) ?? state.progress?.total_steps ?? 0,
                  elapsed_seconds: (evt.payload.elapsed_seconds as number) ?? state.progress?.elapsed_seconds ?? 0,
                  sub_question_progress: (evt.payload.sub_question_progress as Record<string, string>) ?? state.progress?.sub_question_progress ?? {},
                };
              }

              if (evt.event_type === 'research.completed') {
                updates.phase = 'report';
                updates.error = null;
                get().disconnectSSE();
                set(updates);
                void get().fetchReport();
                continue;
              }

              if (evt.event_type === 'research.failed') {
                updates.error = (evt.payload.error as string) || 'Research failed';
                get().disconnectSSE();
              }

              if (evt.event_type === 'research.cancelled') {
                updates.phase = 'input';
                get().disconnectSSE();
              }

              set(updates);
            } catch {
              // skip malformed events
            }
          }
        }

        const latest = get();
        if (latest.phase === 'executing' && latest.sessionId === sessionId) {
          await latest.openSession(sessionId);
        }
      })
      .catch((e) => {
        if ((e as Error).name === 'AbortError') return;
        const state = get();
        if (state.phase === 'executing') {
          setTimeout(() => {
            const current = get();
            if (current.phase === 'executing' && current.sessionId) {
              current.connectSSE(current.sessionId);
            }
          }, 2000);
        }
      });
  },

  disconnectSSE: () => {
    const { sseAbort } = get();
    if (sseAbort) {
      sseAbort.abort();
      set({ sseAbort: null });
    }
  },

  openSession: async (sessionId: string, options) => {
    const preserveState = options?.preserveState ?? false;
    get().disconnectSSE();
    if (!preserveState) {
      set({
        loading: true,
        error: null,
        sessionId,
        events: [],
        lastEventId: null,
        lastSequence: 0,
        plan: null,
        report: null,
        searchResults: null,
        progress: null,
        phase: 'input',
      });
    } else {
      set({ loading: true, error: null, sessionId });
    }
    try {
      const data = await researchFetch<{
        run_id: string;
        status: string;
        plan: ResearchPlan | null;
        progress: ResearchProgress;
        error?: string | null;
        search_results?: SearchResultData[] | null;
      }>(`/runs/${sessionId}`);

      const statusPhaseMap: Record<string, ResearchPhase> = {
        planning: 'planning',
        plan_ready: 'planning',
        draft: 'planning',
        confirmed: 'planning',
        executing: 'executing',
        generating_report: 'executing',
        completed: 'report',
        failed: 'planning',
        cancelled: 'planning',
      };
      let phase = statusPhaseMap[data.status] || 'input';

      const hasPartialResults = data.status === 'failed' && data.search_results && data.search_results.length > 0;
      if (hasPartialResults) {
        phase = 'report';
      }

      const isPlanningWithoutPlan = phase === 'planning' && !data.plan;
      const isPendingPlanning = isPlanningWithoutPlan && !data.error;

      const resolvedPhase: ResearchPhase =
        isPlanningWithoutPlan && !isPendingPlanning ? 'input' : phase;
      const resolvedError =
        isPlanningWithoutPlan && !isPendingPlanning
          ? normalizeResearchError(
              data.error || 'This session has an incomplete plan state. Please start a new research run.',
            )
          : (data.error ? normalizeResearchError(data.error) : null);

      set({
        plan: data.plan,
        progress: data.progress,
        searchResults: data.search_results || null,
        phase: resolvedPhase,
        loading: isPendingPlanning,
        error: resolvedError,
      });

      if (isPendingPlanning) {
        window.setTimeout(() => {
          const current = get();
          if (current.sessionId === sessionId && current.phase === 'planning' && !current.plan) {
            void current.openSession(sessionId, { preserveState: true });
          }
        }, 1500);
        return;
      }

      if (data.status === 'executing' || data.status === 'generating_report') {
        get().connectSSE(sessionId);
      } else if (data.status === 'completed') {
        get().fetchReport();
      }
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  reset: () => {
    const { sessions } = get();
    get().disconnectSSE();
    set({ ...initialState, sessions });
  },
}));
