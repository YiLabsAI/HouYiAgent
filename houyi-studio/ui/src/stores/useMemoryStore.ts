/**
 * Zustand store for Memory Inbox state.
 *
 * Manages memory candidate review and approved record CRUD.
 */
import { create } from 'zustand';

const API_BASE = '/api/memory';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MemoryCandidate {
  candidate_id: string;
  content: string;
  memory_type?: string;
  source_context: string;
  confidence: number;
  suggested_tags: string[];
  status: 'pending' | 'approved' | 'rejected' | 'merged';
  created_at?: string;
}

export interface MemoryRecord {
  record_id: string;
  key: string;
  scope: string;
  content: string;
  memory_type?: string;
  tags?: string[];
  confidence?: number;
  updated_at?: string;
}

export interface MemoryConfig {
  enabled: boolean;
  auto_extract: boolean;
}

type CandidateFilter = 'all' | 'pending' | 'approved' | 'rejected';

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface MemoryState {
  candidates: MemoryCandidate[];
  records: MemoryRecord[];
  config: MemoryConfig;
  filter: CandidateFilter;
  loading: boolean;
  error: string | null;

  setFilter: (filter: CandidateFilter) => void;
  fetchCandidates: () => Promise<void>;
  fetchRecords: (scope?: string) => Promise<void>;
  fetchConfig: () => Promise<void>;
  updateConfig: (updates: Partial<MemoryConfig>) => Promise<void>;
  approveCandidate: (id: string) => Promise<void>;
  rejectCandidate: (id: string) => Promise<void>;
  updateCandidate: (id: string, content: string, tags?: string[]) => Promise<void>;
  updateRecord: (id: string, content?: string, tags?: string[]) => Promise<void>;
  deleteRecord: (id: string) => Promise<void>;
  extractFromChat: (messages: Array<{ role: string; content: string }>, sessionId?: string) => Promise<void>;
}

async function memoryFetch<T>(path: string, init?: RequestInit): Promise<T> {
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

export const useMemoryStore = create<MemoryState>((set, get) => ({
  candidates: [],
  records: [],
  config: { enabled: true, auto_extract: true },
  filter: 'pending',
  loading: false,
  error: null,

  setFilter: (filter) => {
    set({ filter });
    get().fetchCandidates();
  },

  fetchCandidates: async () => {
    set({ loading: true, error: null });
    try {
      const { filter } = get();
      const qs = filter !== 'all' ? `?status=${filter}` : '';
      const data = await memoryFetch<{ candidates: MemoryCandidate[] }>(`/candidates${qs}`);
      set({ candidates: data.candidates, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchRecords: async (scope?: string) => {
    set({ loading: true, error: null });
    try {
      const qs = scope ? `?scope=${scope}` : '';
      const data = await memoryFetch<{ records: MemoryRecord[] }>(`/records${qs}`);
      set({ records: data.records, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  approveCandidate: async (id) => {
    set({ candidates: get().candidates.filter((c) => c.candidate_id !== id) });
    try {
      await memoryFetch(`/candidates/${id}/approve`, { method: 'POST' });
      get().fetchRecords();
    } catch (e) {
      set({ error: (e as Error).message });
      get().fetchCandidates();
    }
  },

  rejectCandidate: async (id) => {
    set({ candidates: get().candidates.filter((c) => c.candidate_id !== id) });
    try {
      await memoryFetch(`/candidates/${id}/reject`, { method: 'POST' });
    } catch (e) {
      set({ error: (e as Error).message });
      get().fetchCandidates();
    }
  },

  updateCandidate: async (id, content, tags) => {
    try {
      await memoryFetch(`/candidates/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ content, suggested_tags: tags }),
      });
      get().fetchCandidates();
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  updateRecord: async (id, content, tags) => {
    try {
      await memoryFetch(`/records/${id}`, {
        method: 'PUT',
        body: JSON.stringify({ content, tags }),
      });
      get().fetchRecords();
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  deleteRecord: async (id) => {
    try {
      await memoryFetch(`/records/${id}`, { method: 'DELETE' });
      get().fetchRecords();
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchConfig: async () => {
    try {
      const data = await memoryFetch<{ config: MemoryConfig }>('/config');
      set({ config: data.config });
    } catch {
      // silent — use defaults
    }
  },

  updateConfig: async (updates) => {
    try {
      const data = await memoryFetch<{ config: MemoryConfig }>('/config', {
        method: 'PUT',
        body: JSON.stringify(updates),
      });
      set({ config: data.config });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  extractFromChat: async (messages, sessionId) => {
    try {
      const { config: freshConfig } = await memoryFetch<{ config: { enabled: boolean; auto_extract: boolean } }>('/config');
      set({ config: freshConfig });
      if (!freshConfig.enabled || !freshConfig.auto_extract) return;

      await memoryFetch('/extract', {
        method: 'POST',
        body: JSON.stringify({ messages, session_id: sessionId }),
      });
      get().fetchCandidates();
    } catch {
      // silent — extraction failure should not block chat
    }
  },
}));
