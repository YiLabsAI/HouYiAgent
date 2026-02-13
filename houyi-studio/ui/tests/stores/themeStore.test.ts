import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('highlight.js/styles/github.css?inline', () => ({
  default: '/* hljs light */',
}));
vi.mock('highlight.js/styles/github-dark.css?inline', () => ({
  default: '/* hljs dark */',
}));

const flushTasks = async (): Promise<void> => {
  // applyHljsTheme uses dynamic import() which may resolve on a later macrotask
  // in the Vitest/Vite pipeline. Flush both microtasks and macrotasks.
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((r) => setTimeout(r, 0));
  await Promise.resolve();
  await new Promise((r) => setTimeout(r, 0));
};

const waitForHljsTheme = async (target: 'light' | 'dark'): Promise<void> => {
  const deadline = Date.now() + 500;
  // Poll a few ticks to avoid flaky assertions.
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const el = document.head.querySelector('style[data-hljs-theme]');
    const cur = el?.getAttribute('data-hljs-theme');
    if (cur === target) return;
    if (Date.now() > deadline) {
      throw new Error(`Timed out waiting for hljs theme '${target}', current='${cur ?? 'null'}'`);
    }
    await flushTasks();
  }
};

const STORAGE_KEY = 'houyi-theme';

type LocalStorageLike = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
  clear: () => void;
};

const createMemoryLocalStorage = (): LocalStorageLike => {
  const map = new Map<string, string>();
  return {
    getItem: (key: string) => (map.has(key) ? map.get(key)! : null),
    setItem: (key: string, value: string) => {
      map.set(key, String(value));
    },
    removeItem: (key: string) => {
      map.delete(key);
    },
    clear: () => {
      map.clear();
    },
  };
};

let originalLocalStorage: any = undefined;
let memoryLocalStorage: LocalStorageLike | null = null;

const resetStorage = (): void => {
  const ls: any = (globalThis as any).localStorage;
  if (!ls) return;
  if (typeof ls.clear === 'function') {
    ls.clear();
    return;
  }
  if (typeof ls.removeItem === 'function') {
    ls.removeItem(STORAGE_KEY);
  }
};

const loadStoreFresh = async () => {
  vi.resetModules();
  const mod = await import('@/stores/useThemeStore');
  return mod;
};

describe('useThemeStore', () => {
  beforeEach(() => {
    if (originalLocalStorage === undefined) {
      originalLocalStorage = (globalThis as any).localStorage;
    }
    memoryLocalStorage = createMemoryLocalStorage();
    vi.stubGlobal('localStorage', memoryLocalStorage);

    document.documentElement.className = '';
    document.head.querySelectorAll('[data-hljs-theme]').forEach((el) => el.remove());
    resetStorage();
  });

  afterEach(() => {
    document.documentElement.className = '';
    document.head.querySelectorAll('[data-hljs-theme]').forEach((el) => el.remove());
    resetStorage();
    vi.stubGlobal('localStorage', originalLocalStorage);
    memoryLocalStorage = null;
    vi.restoreAllMocks();
  });

  it('defaults to dark when no localStorage value exists', async () => {
    const { useThemeStore } = await loadStoreFresh();
    expect(useThemeStore.getState().theme).toBe('dark');
    expect(document.documentElement.className).toContain('theme-dark');
  });

  it('respects stored theme from localStorage', async () => {
    (globalThis as any).localStorage.setItem(STORAGE_KEY, 'light');
    const { useThemeStore } = await loadStoreFresh();
    expect(useThemeStore.getState().theme).toBe('light');
    expect(document.documentElement.className).toContain('theme-light');
  });

  it('ignores invalid stored theme and falls back to dark', async () => {
    (globalThis as any).localStorage.setItem(STORAGE_KEY, 'invalid-theme');
    const { useThemeStore } = await loadStoreFresh();
    expect(useThemeStore.getState().theme).toBe('dark');
    expect(document.documentElement.className).toContain('theme-dark');
  });

  it('setTheme adds new theme class and removes other theme classes', async () => {
    const { useThemeStore } = await loadStoreFresh();

    useThemeStore.getState().setTheme('light');
    expect(document.documentElement.classList.contains('theme-light')).toBe(true);
    expect(document.documentElement.classList.contains('theme-dark')).toBe(false);
    expect(document.documentElement.classList.contains('theme-nord')).toBe(false);
    expect(document.documentElement.classList.contains('theme-warm')).toBe(false);

    useThemeStore.getState().setTheme('nord');
    expect(document.documentElement.classList.contains('theme-nord')).toBe(true);
    expect(document.documentElement.classList.contains('theme-light')).toBe(false);
  });

  it('persists theme changes to localStorage', async () => {
    const { useThemeStore } = await loadStoreFresh();
    useThemeStore.getState().setTheme('light');
    expect((globalThis as any).localStorage.getItem(STORAGE_KEY)).toBe('light');
  });

  it('does not throw if localStorage is unavailable (getInitialTheme)', async () => {
    const orig = (globalThis as any).localStorage;
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('localStorage blocked');
      },
      setItem: () => {
        throw new Error('localStorage blocked');
      },
      removeItem: () => {},
      clear: () => {},
    });

    const { useThemeStore } = await loadStoreFresh();
    expect(useThemeStore.getState().theme).toBe('dark');

    // Restore
    vi.stubGlobal('localStorage', orig);
  });

  it('does not throw if localStorage is unavailable (persist)', async () => {
    const orig = (globalThis as any).localStorage;
    vi.stubGlobal('localStorage', {
      getItem: () => null,
      setItem: () => {
        throw new Error('localStorage blocked');
      },
      removeItem: () => {},
      clear: () => {},
    });

    const { useThemeStore } = await loadStoreFresh();
    expect(() => useThemeStore.getState().setTheme('light')).not.toThrow();

    // Restore
    vi.stubGlobal('localStorage', orig);
  });

  it('hljs theme: switching does not synchronously remove the existing style (no gap)', async () => {
    const { useThemeStore } = await loadStoreFresh();

    // Wait for initial async injection to settle.
    await flushTasks();
    await waitForHljsTheme('dark');

    const initialCount = document.head.querySelectorAll('style[data-hljs-theme]').length;
    expect(initialCount).toBeGreaterThanOrEqual(1);

    // Trigger async switch: should NOT synchronously remove existing theme.
    useThemeStore.getState().setTheme('light');
    const immediateCount = document.head.querySelectorAll('style[data-hljs-theme]').length;
    expect(immediateCount).toBeGreaterThanOrEqual(1);

    await waitForHljsTheme('light');

    const styles = Array.from(document.head.querySelectorAll('style[data-hljs-theme]'));
    expect(styles).toHaveLength(1);
    expect(styles[0].getAttribute('data-hljs-theme')).toBe('light');
  });

  it('hljs theme: rapid switching ends with a single style matching last theme', async () => {
    const { useThemeStore } = await loadStoreFresh();
    await flushTasks();
    await waitForHljsTheme('dark');

    useThemeStore.getState().setTheme('light');
    useThemeStore.getState().setTheme('dark');
    useThemeStore.getState().setTheme('light');

    await waitForHljsTheme('light');

    const styles = Array.from(document.head.querySelectorAll('style[data-hljs-theme]'));
    expect(styles).toHaveLength(1);
    expect(styles[0].getAttribute('data-hljs-theme')).toBe('light');
  });

  it('does not reinject hljs theme when switching between dark-like themes (dark/nord/warm)', async () => {
    const { useThemeStore } = await loadStoreFresh();
    await flushTasks();
    await waitForHljsTheme('dark');

    const before = document.head.querySelectorAll('style[data-hljs-theme]').length;
    useThemeStore.getState().setTheme('nord');
    useThemeStore.getState().setTheme('warm');

    await flushTasks();

    const after = document.head.querySelectorAll('style[data-hljs-theme]').length;
    // Dark-like themes all map to hljs dark; so no need to inject multiple.
    expect(after).toBe(before);
    const style = document.head.querySelector('style[data-hljs-theme]');
    expect(style?.getAttribute('data-hljs-theme')).toBe('dark');
  });
});
