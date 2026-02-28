export type LiveLlmProviderOption = {
  value: string;
  label: string;
  models: string[];
};

export const LIVE_LLM_PROVIDER_OPTIONS: LiveLlmProviderOption[] = [
  { value: '', label: 'provider: default', models: [] },
  {
    value: 'vertex',
    label: 'provider: vertex',
    models: ['gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash'],
  },
  {
    value: 'google_ai',
    label: 'provider: google_ai',
    models: ['gemini-3.1-pro-preview', 'gemini-3-pro-preview', 'gemini-3-flash-preview', 'gemini-2.5-pro', 'gemini-2.5-flash'],
  },
  { value: 'openai', label: 'provider: openai', models: ['gpt-4o', 'gpt-4.1-mini'] },
  { value: 'siliconflow', label: 'provider: siliconflow', models: ['deepseek-chat', 'deepseek-reasoner'] },
  { value: 'openai_compat', label: 'provider: openai_compat', models: ['deepseek-chat', 'qwen-plus'] },
  { value: 'deepseek', label: 'provider: deepseek', models: ['deepseek-chat', 'deepseek-reasoner'] },
];

export const getProviderModels = (provider: string): string[] => (
  LIVE_LLM_PROVIDER_OPTIONS.find((opt) => opt.value === provider)?.models ?? []
);

export const getLiveDefaultsForTool = (toolName: string): { provider: string; model: string } => {
  if (toolName === 'notebooklm') {
    return { provider: 'vertex', model: 'gemini-2.5-pro' };
  }
  return { provider: '', model: '' };
};
