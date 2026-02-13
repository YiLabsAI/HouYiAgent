/**
 * Lightweight Zustand store for global display settings.
 *
 * Fetches display config (user/assistant name + avatar) from the backend
 * on first access.  GlobalSettingsPage calls refreshSettings() after save
 * so MessageBubble picks up changes immediately.
 */
import { create } from 'zustand';

const API_BASE = '/api/chat';

interface DisplaySettings {
  user_name: string;
  user_avatar: string | null;
  assistant_name: string;
  assistant_avatar: string | null;
}

interface SettingsStoreState {
  display: DisplaySettings;
  loaded: boolean;
  fetchSettings: () => Promise<void>;
  refreshSettings: () => Promise<void>;
}

let inFlightFetch: Promise<void> | null = null;

export const useSettingsStore = create<SettingsStoreState>((set, get) => ({
  display: {
    user_name: 'You',
    user_avatar: null,
    assistant_name: 'Assistant',
    assistant_avatar: null,
  },
  loaded: false,

  fetchSettings: async () => {
    if (get().loaded) return;

    if (inFlightFetch) {
      await inFlightFetch;
      return;
    }

    inFlightFetch = (async () => {
    try {
      const res = await fetch(`${API_BASE}/settings`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.display) {
        set({ display: data.display, loaded: true });
      }
    } catch {
      // Silently fail — defaults are fine
    } finally {
      inFlightFetch = null;
    }
    })();

    await inFlightFetch;
  },

  refreshSettings: async () => {
    try {
      const res = await fetch(`${API_BASE}/settings`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.display) {
        set({ display: data.display, loaded: true });
      }
    } catch {
      // Silently fail
    }
  },
}));
