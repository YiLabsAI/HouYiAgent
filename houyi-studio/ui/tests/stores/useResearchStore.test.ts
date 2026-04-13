/**
 * Tests for useResearchStore — research session lifecycle, SSE, state transitions.
 */
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

const loadStoreFresh = async () => {
  vi.resetModules();
  return await import('@/stores/useResearchStore');
};

const mockFetch = (responses: Array<{ status: number; body?: unknown; headers?: Record<string, string> }>) => {
  let callIdx = 0;
  globalThis.fetch = vi.fn(async () => {
    const resp = responses[callIdx] || responses[responses.length - 1];
    callIdx++;
    return {
      ok: resp.status >= 200 && resp.status < 300,
      status: resp.status,
      statusText: resp.status === 204 ? 'No Content' : 'OK',
      json: async () => resp.body,
      text: async () => JSON.stringify(resp.body),
      headers: new Headers(resp.headers || {}),
      body: null,
    } as unknown as Response;
  });
};

describe('useResearchStore', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('initial state', () => {
    it('starts in input phase', async () => {
      const { useResearchStore } = await loadStoreFresh();
      const s = useResearchStore.getState();
      expect(s.phase).toBe('input');
      expect(s.sessionId).toBeNull();
      expect(s.plan).toBeNull();
      expect(s.report).toBeNull();
    });
  });

  describe('createSession', () => {
    it('transitions to planning on success', async () => {
      mockFetch([{
        status: 201,
        body: { run_id: 's1', plan: { query: 'test', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' },
      }]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('test query');
      const s = useResearchStore.getState();
      expect(s.phase).toBe('planning');
      expect(s.sessionId).toBe('s1');
      expect(s.plan?.query).toBe('test');
    });

    it('enters planning before create run resolves', async () => {
      let resolveFetch: ((value: Response) => void) | undefined;
      globalThis.fetch = vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            resolveFetch = resolve;
          }),
      ) as unknown as typeof fetch;
      const { useResearchStore } = await loadStoreFresh();
      const pending = useResearchStore.getState().createSession('test query');
      expect(useResearchStore.getState().phase).toBe('planning');
      expect(useResearchStore.getState().loading).toBe(true);
      if (!resolveFetch) throw new Error('fetch resolver missing');
      resolveFetch({
        ok: true,
        status: 201,
        statusText: 'Created',
        json: async () => ({
          run_id: 's1',
          plan: { query: 'test', sub_questions: [], outline: [], version: 1, status: 'draft' },
          status: 'planning',
        }),
        text: async () => '',
        headers: new Headers(),
        body: null,
      } as unknown as Response);
      await pending;
    });

    it('sets error on failure', async () => {
      mockFetch([{ status: 500, body: 'server error' }]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('fail');
      expect(useResearchStore.getState().error).toBeTruthy();
      expect(useResearchStore.getState().phase).toBe('input');
    });

    it('normalizes upstream disconnect message on failure', async () => {
      mockFetch([
        {
          status: 502,
          body: { detail: 'LLM/planning error: Server disconnected without sending a response.' },
        },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('fail');
      expect(useResearchStore.getState().error).toContain(
        'The connection to the model was interrupted. Please retry in a moment.',
      );
      expect(useResearchStore.getState().phase).toBe('input');
    });

    it('includes settings in POST body when provided', async () => {
      const fetchMock = vi.fn(async () => ({
        ok: true,
        status: 201,
        statusText: 'Created',
        json: async () => ({
          run_id: 's-depth',
          plan: { query: 'test', sub_questions: [], outline: [], version: 1, status: 'draft' },
          status: 'planning',
        }),
        text: async () => '',
        headers: new Headers(),
        body: null,
      })) as unknown as typeof fetch;
      globalThis.fetch = fetchMock;
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('topic', { depth: 'deep' });
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/research/runs',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ query: 'topic', settings: { depth: 'deep' } }),
        }),
      );
    });

    it('deletes previous failed session when creating a new one', async () => {
      let callIdx = 0;
      const calls: Array<{ url: string; method?: string }> = [];
      globalThis.fetch = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
        const urlStr = typeof url === 'string' ? url : url.toString();
        calls.push({ url: urlStr, method: init?.method });
        callIdx++;
        if (urlStr.endsWith('/runs') && init?.method === 'POST') {
          return {
            ok: true, status: 201, statusText: 'Created',
            json: async () => ({ run_id: 's2', plan: { query: 'q', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' }),
            text: async () => '', headers: new Headers(), body: null,
          } as unknown as Response;
        }
        if (urlStr.includes('/runs/s-failed') && init?.method === 'DELETE') {
          return { ok: true, status: 204, statusText: 'No Content', json: async () => ({}), text: async () => '', headers: new Headers(), body: null } as unknown as Response;
        }
        return { ok: true, status: 200, json: async () => ({}), text: async () => '', headers: new Headers(), body: null, statusText: 'OK' } as unknown as Response;
      }) as unknown as typeof fetch;

      const { useResearchStore } = await loadStoreFresh();
      useResearchStore.setState({
        sessionId: 's-failed',
        sessions: [{ run_id: 's-failed', status: 'failed', query: 'old query' }],
        phase: 'input',
      });
      await useResearchStore.getState().createSession('retry query');
      expect(useResearchStore.getState().sessionId).toBe('s2');
      const deleteCall = calls.find((c) => c.url.includes('/runs/s-failed') && c.method === 'DELETE');
      expect(deleteCall).toBeTruthy();
      expect(useResearchStore.getState().sessions.find((s) => s.run_id === 's-failed')).toBeUndefined();
    });
  });

  describe('editPlan', () => {
    it('updates plan version', async () => {
      mockFetch([
        { status: 201, body: { run_id: 's1', plan: { query: 'q', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' } },
        { status: 200, body: { plan: { query: 'q', sub_questions: [{ question_id: 'q1', question: 'New?', priority: 3, search_strategy: 'web', expected_sources: 5, depends_on: [] }], outline: [], version: 2, status: 'draft' } } },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('q');
      await useResearchStore.getState().editPlan([{ op: 'add', target_question: 'New?' }]);
      expect(useResearchStore.getState().plan?.version).toBe(2);
      expect(useResearchStore.getState().plan?.sub_questions).toHaveLength(1);
    });
  });

  describe('cancelSession', () => {
    it('resets to input phase', async () => {
      mockFetch([
        { status: 201, body: { run_id: 's1', plan: { query: 'q', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' } },
        { status: 200, body: { status: 'cancelled' } },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('q');
      await useResearchStore.getState().cancelSession();
      expect(useResearchStore.getState().phase).toBe('input');
    });
  });

  describe('fetchSessions', () => {
    it('populates sessions list', async () => {
      mockFetch([{
        status: 200,
        body: { runs: [{ run_id: 'a', status: 'completed' }, { run_id: 'b', status: 'executing' }] },
      }]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().fetchSessions();
      expect(useResearchStore.getState().sessions).toHaveLength(2);
    });
  });

  describe('reset', () => {
    it('clears all state', async () => {
      mockFetch([{
        status: 201,
        body: { run_id: 's1', plan: { query: 'q', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' },
      }]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('q');
      useResearchStore.getState().reset();
      const s = useResearchStore.getState();
      expect(s.phase).toBe('input');
      expect(s.sessionId).toBeNull();
      expect(s.plan).toBeNull();
    });
  });

  describe('confirmAndExecute', () => {
    it('resets sse cursors on retry', async () => {
      mockFetch([
        { status: 201, body: { run_id: 's1', plan: { query: 'q', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' } },
        { status: 202, body: { run_id: 's1', status: 'executing' } },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      useResearchStore.setState({
        sessionId: 's1',
        plan: { query: 'q', sub_questions: [], outline: [], version: 1, status: 'draft' },
        lastEventId: 'evt_old',
        lastSequence: 42,
        error: 'old failure',
        events: [{ event_id: 'evt_old', event_type: 'research.failed', sequence: 42, payload: { error: 'old failure' } }],
      });
      const connectSSE = vi.fn();
      useResearchStore.setState({ connectSSE });

      await useResearchStore.getState().confirmAndExecute();

      const s = useResearchStore.getState();
      expect(s.lastEventId).toBeNull();
      expect(s.lastSequence).toBe(0);
      expect(s.events).toEqual([]);
      expect(s.error).toBeNull();
      expect(connectSSE).toHaveBeenCalledWith('s1');
    });
  });

  describe('openSession', () => {
    /** Phase 3.5 fix: openSession must fully reset stale state before fetching */
    it('resets plan/report/phase before loading new session', async () => {
      mockFetch([
        { status: 201, body: { run_id: 's-old', plan: { query: 'old', sub_questions: [], outline: [], version: 1, status: 'draft' }, status: 'planning' } },
        { status: 200, body: { run_id: 's-new', status: 'completed', plan: { query: 'new', sub_questions: [], outline: [], version: 2, status: 'completed' }, progress: { total_steps: 5, completed_steps: 5, current_step: 'done', elapsed_seconds: 10, sub_question_progress: {} } } },
        { status: 200, body: { report: { title: 'New Report', sections: [], references: [], quality_score: null } } },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().createSession('old');
      expect(useResearchStore.getState().plan?.query).toBe('old');

      await useResearchStore.getState().openSession('s-new');
      const s = useResearchStore.getState();
      expect(s.sessionId).toBe('s-new');
      expect(s.plan?.query).toBe('new');
      expect(s.events).toEqual([]);
    });

    it('loads existing completed session', async () => {
      mockFetch([
        { status: 200, body: { run_id: 's2', status: 'completed', plan: { query: 'old', sub_questions: [], outline: [], version: 3, status: 'completed' }, progress: { total_steps: 5, completed_steps: 5, current_step: 'done', elapsed_seconds: 30, sub_question_progress: {} } } },
        { status: 200, body: { report: { title: 'Test', sections: [], references: [], quality_score: null } } },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().openSession('s2');
      const s = useResearchStore.getState();
      expect(s.sessionId).toBe('s2');
      expect(s.plan?.version).toBe(3);
    });

    it('sets error on 404', async () => {
      mockFetch([{ status: 404, body: 'not found' }]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().openSession('missing');
      expect(useResearchStore.getState().error).toBeTruthy();
    });

    /** B3-29: plan_ready / failed / cancelled must land in planning (not input). */
    it('maps plan_ready status to planning phase', async () => {
      const plan = {
        query: 'q',
        sub_questions: [],
        outline: [],
        version: 1,
        status: 'draft',
      };
      const progress = {
        total_steps: 1,
        completed_steps: 0,
        current_step: '',
        elapsed_seconds: 0,
        sub_question_progress: {},
      };
      mockFetch([
        {
          status: 200,
          body: {
            session_id: 's-plan-ready',
            run_id: 's-plan-ready',
            status: 'plan_ready',
            plan,
            progress,
          },
        },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().openSession('s-plan-ready');
      expect(useResearchStore.getState().phase).toBe('planning');
    });

    it('maps failed status to planning phase', async () => {
      const plan = {
        query: 'q',
        sub_questions: [],
        outline: [],
        version: 1,
        status: 'draft',
      };
      const progress = {
        total_steps: 1,
        completed_steps: 0,
        current_step: '',
        elapsed_seconds: 0,
        sub_question_progress: {},
      };
      mockFetch([
        {
          status: 200,
          body: {
            session_id: 's-failed',
            run_id: 's-failed',
            status: 'failed',
            plan,
            progress,
            error: 'LLM error',
          },
        },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().openSession('s-failed');
      expect(useResearchStore.getState().phase).toBe('planning');
    });

    it('maps cancelled status to planning phase', async () => {
      const plan = {
        query: 'q',
        sub_questions: [],
        outline: [],
        version: 1,
        status: 'draft',
      };
      const progress = {
        total_steps: 1,
        completed_steps: 0,
        current_step: '',
        elapsed_seconds: 0,
        sub_question_progress: {},
      };
      mockFetch([
        {
          status: 200,
          body: {
            session_id: 's-cancelled',
            run_id: 's-cancelled',
            status: 'cancelled',
            plan,
            progress,
          },
        },
      ]);
      const { useResearchStore } = await loadStoreFresh();
      await useResearchStore.getState().openSession('s-cancelled');
      expect(useResearchStore.getState().phase).toBe('planning');
    });
  });

  describe('SSE dedup', () => {
    it('lastSequence prevents duplicate processing', async () => {
      const { useResearchStore } = await loadStoreFresh();
      useResearchStore.setState({ lastSequence: 5 });
      expect(useResearchStore.getState().lastSequence).toBe(5);
    });

    it('completed event transitions to report before fetching report', async () => {
      const { useResearchStore } = await loadStoreFresh();
      const disconnectSSE = vi.fn();
      const fetchReport = vi.fn();
      useResearchStore.setState({
        phase: 'executing',
        sessionId: 's1',
        disconnectSSE,
        fetchReport,
      });

      const evt = {
        event_id: 'e1',
        event_type: 'research.completed',
        sequence: 1,
        payload: {},
      };

      disconnectSSE();
      useResearchStore.setState({
        phase: 'report',
        events: [evt],
        lastEventId: 'e1',
        lastSequence: 1,
        error: null,
      });
      fetchReport();

      expect(disconnectSSE).toHaveBeenCalled();
      expect(fetchReport).toHaveBeenCalled();
      expect(useResearchStore.getState().phase).toBe('report');
      expect(useResearchStore.getState().error).toBeNull();
    });
  });
});
