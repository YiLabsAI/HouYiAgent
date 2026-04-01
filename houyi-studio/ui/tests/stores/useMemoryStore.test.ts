/**
 * Tests for useMemoryStore — candidate CRUD, filtering, record management.
 */
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

const loadStoreFresh = async () => {
  vi.resetModules();
  return await import('@/stores/useMemoryStore');
};

const mockFetch = (responses: Array<{ status: number; body?: unknown }>) => {
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
    } as unknown as Response;
  });
};

describe('useMemoryStore', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('initial state', () => {
    it('defaults to pending filter', async () => {
      const { useMemoryStore } = await loadStoreFresh();
      const s = useMemoryStore.getState();
      expect(s.filter).toBe('pending');
      expect(s.candidates).toEqual([]);
      expect(s.records).toEqual([]);
    });
  });

  describe('fetchCandidates', () => {
    it('loads pending candidates', async () => {
      mockFetch([{
        status: 200,
        body: { candidates: [{ candidate_id: 'c1', content: 'test', source_context: 'chat', confidence: 0.8, suggested_tags: [], status: 'pending' }] },
      }]);
      const { useMemoryStore } = await loadStoreFresh();
      await useMemoryStore.getState().fetchCandidates();
      expect(useMemoryStore.getState().candidates).toHaveLength(1);
      expect(useMemoryStore.getState().candidates[0].candidate_id).toBe('c1');
    });

    it('sets error on failure', async () => {
      mockFetch([{ status: 500, body: 'error' }]);
      const { useMemoryStore } = await loadStoreFresh();
      await useMemoryStore.getState().fetchCandidates();
      expect(useMemoryStore.getState().error).toBeTruthy();
    });
  });

  describe('setFilter', () => {
    it('changes filter and refetches', async () => {
      mockFetch([{ status: 200, body: { candidates: [] } }]);
      const { useMemoryStore } = await loadStoreFresh();
      useMemoryStore.getState().setFilter('approved');
      expect(useMemoryStore.getState().filter).toBe('approved');
    });
  });

  describe('approveCandidate', () => {
    it('calls approve endpoint and refreshes records', async () => {
      mockFetch([
        { status: 200, body: { record: { key: 'k1', scope: 'user', content: 'test' } } },
        { status: 200, body: { records: [{ key: 'k1', scope: 'user', content: 'test' }] } },
      ]);
      const { useMemoryStore } = await loadStoreFresh();
      useMemoryStore.setState({
        candidates: [
          { candidate_id: 'c1', content: 'test', source_context: '', confidence: 0.9, suggested_tags: [], status: 'pending' },
          { candidate_id: 'c2', content: 'other', source_context: '', confidence: 0.8, suggested_tags: [], status: 'pending' },
        ],
      });
      await useMemoryStore.getState().approveCandidate('c1');
      expect(useMemoryStore.getState().candidates).toHaveLength(1);
      expect(useMemoryStore.getState().candidates[0].candidate_id).toBe('c2');
    });
  });

  describe('rejectCandidate', () => {
    it('removes candidate optimistically on reject', async () => {
      mockFetch([
        { status: 200, body: { status: 'rejected' } },
      ]);
      const { useMemoryStore } = await loadStoreFresh();
      useMemoryStore.setState({
        candidates: [
          { candidate_id: 'c1', content: 'test', source_context: '', confidence: 0.9, suggested_tags: [], status: 'pending' },
          { candidate_id: 'c2', content: 'other', source_context: '', confidence: 0.8, suggested_tags: [], status: 'pending' },
        ],
      });
      await useMemoryStore.getState().rejectCandidate('c1');
      expect(useMemoryStore.getState().candidates).toHaveLength(1);
      expect(useMemoryStore.getState().candidates[0].candidate_id).toBe('c2');
    });
  });

  describe('fetchRecords', () => {
    it('loads records with scope filter', async () => {
      mockFetch([{
        status: 200,
        body: { records: [{ key: 'r1', scope: 'user', content: 'data' }] },
      }]);
      const { useMemoryStore } = await loadStoreFresh();
      await useMemoryStore.getState().fetchRecords('user');
      expect(useMemoryStore.getState().records).toHaveLength(1);
    });
  });

  describe('deleteRecord', () => {
    it('calls delete then refetches', async () => {
      mockFetch([
        { status: 204 },
        { status: 200, body: { records: [] } },
      ]);
      const { useMemoryStore } = await loadStoreFresh();
      await useMemoryStore.getState().deleteRecord('r1');
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });
  });

  describe('updateCandidate', () => {
    it('sends PUT then refetches', async () => {
      mockFetch([
        { status: 200, body: { candidate: { candidate_id: 'c1', content: 'updated' } } },
        { status: 200, body: { candidates: [] } },
      ]);
      const { useMemoryStore } = await loadStoreFresh();
      await useMemoryStore.getState().updateCandidate('c1', 'updated', ['tag1']);
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    });
  });
});
