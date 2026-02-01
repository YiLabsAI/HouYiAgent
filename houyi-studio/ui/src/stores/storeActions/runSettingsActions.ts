type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

type RunSettings = {
  enable_tool_calls: boolean;
  tool_names: string[];
  tool_choice: string | null;
  max_tool_calls: number;
  temperature: number | null;
  parallel_tool_calls: boolean | null;
  web_search_provider: string | null;
  retry_policy: {
    default_retries: number;
    timeout_retries: number | null;
    rate_limit_retries: number | null;
    auth_retries: number | null;
    bad_request_retries: number | null;
    content_policy_retries: number | null;
    internal_error_retries: number | null;
  };
};

type RunSettingsInput = Omit<
  Partial<RunSettings>,
  | 'tool_names'
  | 'tool_choice'
  | 'max_tool_calls'
  | 'enable_tool_calls'
  | 'temperature'
  | 'parallel_tool_calls'
  | 'retry_policy'
> & {
  tool_names?: string[] | string | null;
  tool_choice?: string | null | Record<string, any>;
  max_tool_calls?: number | string | null;
  enable_tool_calls?: boolean | null;
  temperature?: number | string | null;
  parallel_tool_calls?: boolean | string | null;
  retry_policy?: Partial<RunSettings['retry_policy']> | null;
};

export const RUN_SETTINGS_STORAGE_KEY = 'houyi.run_settings';

export const DEFAULT_RUN_SETTINGS: RunSettings = {
  enable_tool_calls: true,
  tool_names: [],
  tool_choice: null,
  max_tool_calls: 6,
  temperature: null,
  parallel_tool_calls: null,
  web_search_provider: null,
  retry_policy: {
    default_retries: 0,
    timeout_retries: null,
    rate_limit_retries: null,
    auth_retries: null,
    bad_request_retries: null,
    content_policy_retries: null,
    internal_error_retries: null,
  },
};

const normalizeRunSettings = (value: RunSettingsInput): RunSettings => {
  const merged: RunSettingsInput = { ...DEFAULT_RUN_SETTINGS, ...value };
  const toolNamesRaw = merged.tool_names;
  const toolNames = Array.isArray(toolNamesRaw)
    ? toolNamesRaw
    : typeof toolNamesRaw === 'string'
      ? toolNamesRaw.split(',').map((entry: string) => entry.trim()).filter(Boolean)
      : [];
  const toolChoice = merged.tool_choice === '' || merged.tool_choice === undefined
    ? null
    : merged.tool_choice;
  const maxToolCalls = Number.isFinite(Number(merged.max_tool_calls))
    ? Number(merged.max_tool_calls)
    : DEFAULT_RUN_SETTINGS.max_tool_calls;
  const rawTemperature = merged.temperature;
  let temperature: number | null = DEFAULT_RUN_SETTINGS.temperature;
  if (typeof rawTemperature === 'number') {
    temperature = rawTemperature;
  } else if (typeof rawTemperature === 'string') {
    const trimmed = rawTemperature.trim();
    if (trimmed) {
      const parsed = Number(trimmed);
      temperature = Number.isFinite(parsed) ? parsed : DEFAULT_RUN_SETTINGS.temperature;
    }
  }

  const rawParallel = merged.parallel_tool_calls;
  let parallelToolCalls: boolean | null = null;
  if (typeof rawParallel === 'boolean') {
    parallelToolCalls = rawParallel;
  } else if (typeof rawParallel === 'string') {
    const normalized = rawParallel.trim().toLowerCase();
    if (['true', '1', 'yes', 'y'].includes(normalized)) parallelToolCalls = true;
    if (['false', '0', 'no', 'n'].includes(normalized)) parallelToolCalls = false;
  }

  const rawRetryPolicy = merged.retry_policy || {};
  const retryPolicy = {
    ...DEFAULT_RUN_SETTINGS.retry_policy,
    ...rawRetryPolicy,
  };

  return {
    enable_tool_calls: Boolean(merged.enable_tool_calls),
    tool_names: toolNames,
    tool_choice: toolChoice === null ? null : String(toolChoice),
    max_tool_calls: maxToolCalls,
    temperature,
    parallel_tool_calls: parallelToolCalls,
    web_search_provider: merged.web_search_provider ? String(merged.web_search_provider) : null,
    retry_policy: retryPolicy,
  };
};

const loadRunSettings = (): RunSettings => {
  if (typeof window === 'undefined') {
    return DEFAULT_RUN_SETTINGS;
  }
  try {
    const raw = window.localStorage.getItem(RUN_SETTINGS_STORAGE_KEY);
    if (!raw) {
      return DEFAULT_RUN_SETTINGS;
    }
    const parsed = JSON.parse(raw) as Partial<RunSettings>;
    return normalizeRunSettings(parsed || {});
  } catch (error) {
    console.warn('[Store] Failed to load run settings defaults:', error);
    return DEFAULT_RUN_SETTINGS;
  }
};

const mergeRunSettings = (base: RunSettings, updates: Partial<RunSettings>): RunSettings =>
  normalizeRunSettings({ ...base, ...updates });

export const createRunSettingsActions = (set: StoreSet, get: StoreGet) => ({
  loadRunSettings,
  mergeRunSettings,
  setRunSettingsOpen: (open: boolean) => {
    set({ isRunSettingsOpen: open });
  },

  updateRunSettings: (updates: Partial<RunSettings>) => {
    set((state: any) => {
      const next = mergeRunSettings(state.runSettings, updates);
      if (typeof window !== 'undefined') {
        try {
          window.localStorage.setItem(RUN_SETTINGS_STORAGE_KEY, JSON.stringify(next));
        } catch (error) {
          console.warn('[Store] Failed to persist run settings defaults:', error);
        }
      }
      return { runSettings: next };
    });
  },

  resetRunSettings: () => {
    const next = { ...DEFAULT_RUN_SETTINGS };
    set({ runSettings: next });
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem(RUN_SETTINGS_STORAGE_KEY, JSON.stringify(next));
      } catch (error) {
        console.warn('[Store] Failed to persist run settings defaults:', error);
      }
    }
  },

  saveRunSettingsDefaults: () => {
    if (typeof window === 'undefined') return;
    const state = get();
    try {
      window.localStorage.setItem(RUN_SETTINGS_STORAGE_KEY, JSON.stringify(state.runSettings));
    } catch (error) {
      console.warn('[Store] Failed to persist run settings defaults:', error);
    }
  },
});

export const getInitialRunSettingsState = () => ({
  runSettings: loadRunSettings(),
  isRunSettingsOpen: false,
});
