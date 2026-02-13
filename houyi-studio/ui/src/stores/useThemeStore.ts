/**
 * Theme store: manages UI theme selection with localStorage persistence.
 *
 * Themes are applied via CSS custom properties on <html>.
 * Each theme defines a palette of semantic color tokens.
 */
import { create } from 'zustand';

export type ThemeId = 'dark' | 'light' | 'nord' | 'warm';

export interface ThemeDefinition {
  id: ThemeId;
  label: string;
  description: string;
}

export const THEMES: ThemeDefinition[] = [
  { id: 'dark', label: 'Dark', description: 'Default dark theme' },
  { id: 'light', label: 'Light', description: 'Clean light theme' },
  { id: 'nord', label: 'Nord', description: 'Cool-toned dark theme' },
  { id: 'warm', label: 'Warm', description: 'Warm-toned dark theme' },
];

interface ThemeState {
  theme: ThemeId;
  setTheme: (theme: ThemeId) => void;
}

const STORAGE_KEY = 'houyi-theme';

function getInitialTheme(): ThemeId {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && THEMES.some(t => t.id === stored)) {
      return stored as ThemeId;
    }
  } catch {
    // SSR or localStorage unavailable
  }
  return 'dark';
}

let currentHljsTheme: 'light' | 'dark' | null = null;
let hljsApplySeq = 0;
let themeSwitchTimeout: number | null = null;

async function applyHljsTheme(theme: ThemeId) {
  const target = theme === 'light' ? 'light' : 'dark';
  if (target === currentHljsTheme) return;

  const seq = ++hljsApplySeq;

  // Dynamically import the CSS — Vite will bundle it correctly
  const css = target === 'light'
    ? await import('highlight.js/styles/github.css?inline')
    : await import('highlight.js/styles/github-dark.css?inline');

  if (seq !== hljsApplySeq) return;

  const style = document.createElement('style');
  style.setAttribute('data-hljs-theme', target);
  style.textContent = css.default;
  document.head.appendChild(style);

  // Remove any previously injected hljs theme styles AFTER new one is attached.
  // This avoids a frame where no hljs theme styles are present.
  document.querySelectorAll('[data-hljs-theme]').forEach((el) => {
    if (el !== style) el.remove();
  });

  currentHljsTheme = target;
}

function applyTheme(theme: ThemeId) {
  const root = document.documentElement;
  root.classList.add('theme-switching');
  if (themeSwitchTimeout) {
    window.clearTimeout(themeSwitchTimeout);
  }
  themeSwitchTimeout = window.setTimeout(() => {
    root.classList.remove('theme-switching');
    themeSwitchTimeout = null;
  }, 200);
  // Add new theme class FIRST, then remove old ones.
  // This avoids a frame where no theme class is present (which would
  // flash the :root defaults before the new theme kicks in).
  root.classList.add(`theme-${theme}`);
  THEMES.forEach(t => { if (t.id !== theme) root.classList.remove(`theme-${t.id}`); });
  // Switch highlight.js theme (light vs dark)
  applyHljsTheme(theme);
  // Persist
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // ignore
  }
}

export const useThemeStore = create<ThemeState>((set) => {
  const initial = getInitialTheme();
  // Apply on store creation (app boot)
  if (typeof document !== 'undefined') {
    applyTheme(initial);
  }

  return {
    theme: initial,
    setTheme: (theme: ThemeId) => {
      applyTheme(theme);
      set({ theme });
    },
  };
});
