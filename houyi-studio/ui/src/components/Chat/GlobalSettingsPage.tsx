/**
 * GlobalSettingsPage: full-page overlay for managing global chat settings.
 *
 * Sections:
 * - LLM Providers (add/edit/remove/toggle)
 * - Defaults (model, temperature, max_tokens, system prompt)
 * - Display (user/assistant name and avatar)
 *
 */
import React from 'react';
import { X, Plus, Trash2, ChevronDown, ChevronRight, Wifi, WifiOff, RefreshCw, Eye, EyeOff, Check } from 'lucide-react';
import {
  PROVIDER_DISPLAY_NAMES,
  PROVIDER_SILICONFLOW,
  PROVIDER_OPENAI,
  PROVIDER_ANTHROPIC,
  PROVIDER_DEEPSEEK,
  PROVIDER_GOOGLE_AI,
  PROVIDER_VERTEX,
} from '@/constants/models';
import { useSettingsStore } from '@/stores/useSettingsStore';

const API_BASE = '/api/chat';

const AVATAR_OPTIONS: string[] = [
  '\u{1F9D1}\u{200D}\u{1F4BB}', '\u{1F468}\u{200D}\u{1F4BB}', '\u{1F469}\u{200D}\u{1F4BB}',
  '\u{1F916}', '\u{1F9E0}', '\u{2728}', '\u{1F680}', '\u{1F4A1}',
  '\u{1F431}', '\u{1F436}', '\u{1F98A}', '\u{1F427}',
];

interface Provider {
  id: string;
  name: string;
  api_key: string;
  base_url: string;
  models: string[];
  enabled: boolean;
}

interface Defaults {
  model: string;
  system_instructions: string;
  temperature: number;
  max_tokens: number;
  stream: boolean;
}

interface Display {
  user_name: string;
  user_avatar: string | null;
  assistant_name: string;
  assistant_avatar: string | null;
}

interface Settings {
  version: number;
  providers: Provider[];
  defaults: Defaults;
  display: Display;
  updated_at: number;
}

interface GlobalSettingsPageProps {
  isOpen: boolean;
  onClose: () => void;
}

/** Strip HTML tags from error messages (frontend safety net). */
const stripHtml = (s: string): string => s.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim().slice(0, 300);

export const GlobalSettingsPage: React.FC<GlobalSettingsPageProps> = ({ isOpen, onClose }) => {
  const refreshDisplaySettings = useSettingsStore((s) => s.refreshSettings);
  const [settings, setSettings] = React.useState<Settings | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [expandedProvider, setExpandedProvider] = React.useState<string | null>(null);
  const [testingProvider, setTestingProvider] = React.useState<string | null>(null);
  const [testResult, setTestResult] = React.useState<Record<string, { ok: boolean; message: string; latency_ms: number } | null>>({});
  const [fetchingModels, setFetchingModels] = React.useState<string | null>(null);
  const [availableModels, setAvailableModels] = React.useState<Record<string, { id: string; owned_by: string }[]>>({});
  const [saveSuccess, setSaveSuccess] = React.useState(false);
  const [showApiKey, setShowApiKey] = React.useState<Record<string, boolean>>({});
  const [originalSettings, setOriginalSettings] = React.useState<string>('');

  React.useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen]);

  const loadSettings = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/settings`);
      if (!res.ok) throw new Error(`Failed to load settings: ${res.status}`);
      const data = await res.json();
      // Normalize: ensure every provider has an explicit enabled boolean
      if (data.providers) {
        data.providers = data.providers.map((p: any) => ({
          ...p,
          enabled: p.enabled !== false,
        }));
      }
      setSettings(data);
      setOriginalSettings(JSON.stringify(data));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isDirty = settings ? JSON.stringify(settings) !== originalSettings : false;

  const handleSave = async () => {
    if (!settings || !isDirty) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`Failed to save settings: ${res.status} ${detail}`);
      }
      const data = await res.json();
      // Normalize provider enabled fields after save
      if (data.providers) {
        data.providers = data.providers.map((p: any) => ({
          ...p,
          enabled: p.enabled !== false,
        }));
      }
      setSettings(data);
      setOriginalSettings(JSON.stringify(data));
      // Refresh the global display settings store so MessageBubble picks up
      // avatar/name changes immediately without a page reload.
      await refreshDisplaySettings();
      // Show success toast
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2500);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const updateDefaults = (key: keyof Defaults, value: any) => {
    if (!settings) return;
    // Validate temperature: clamp to 0-2
    if (key === 'temperature') {
      const num = parseFloat(value);
      if (isNaN(num)) return;
      value = Math.max(0, Math.min(2, num));
    }
    // Validate max_tokens: clamp to 256-131072
    if (key === 'max_tokens') {
      const num = parseInt(value);
      if (isNaN(num) || num < 0) return;
      value = Math.max(256, Math.min(131072, num));
    }
    setSettings({ ...settings, defaults: { ...settings.defaults, [key]: value } });
  };

  const updateDisplay = (key: keyof Display, value: any) => {
    if (!settings) return;
    setSettings({ ...settings, display: { ...settings.display, [key]: value } });
  };

  // Provider presets for quick setup
  const PROVIDER_PRESETS: { id: string; name: string; base_url: string; placeholder_key: string }[] = [
    { id: PROVIDER_OPENAI, name: PROVIDER_DISPLAY_NAMES[PROVIDER_OPENAI], base_url: 'https://api.openai.com/v1', placeholder_key: 'sk-...' },
    { id: PROVIDER_ANTHROPIC, name: PROVIDER_DISPLAY_NAMES[PROVIDER_ANTHROPIC], base_url: 'https://api.anthropic.com/v1', placeholder_key: 'sk-ant-...' },
    { id: PROVIDER_DEEPSEEK, name: PROVIDER_DISPLAY_NAMES[PROVIDER_DEEPSEEK], base_url: 'https://api.deepseek.com/v1', placeholder_key: 'sk-...' },
    { id: PROVIDER_SILICONFLOW, name: PROVIDER_DISPLAY_NAMES[PROVIDER_SILICONFLOW], base_url: 'https://api.siliconflow.cn/v1', placeholder_key: 'sk-...' },
    { id: PROVIDER_GOOGLE_AI, name: PROVIDER_DISPLAY_NAMES[PROVIDER_GOOGLE_AI], base_url: 'https://generativelanguage.googleapis.com/v1beta/openai', placeholder_key: 'AIza...' },
    { id: PROVIDER_VERTEX, name: PROVIDER_DISPLAY_NAMES[PROVIDER_VERTEX], base_url: '', placeholder_key: '(GCP service account)' },
    { id: 'custom', name: 'Custom', base_url: '', placeholder_key: 'sk-...' },
  ];

  const addProvider = (presetId?: string) => {
    if (!settings) return;
    const preset = presetId ? PROVIDER_PRESETS.find(p => p.id === presetId) : undefined;
    const newProvider: Provider = {
      id: `${preset?.id || 'provider'}-${Date.now()}`,
      name: preset?.name || '',
      api_key: '',
      base_url: preset?.base_url || '',
      models: [],
      enabled: true,
    };
    setSettings({ ...settings, providers: [...settings.providers, newProvider] });
    setExpandedProvider(newProvider.id);
  };

  const testConnection = async (provider: Provider) => {
    setTestingProvider(provider.id);
    setTestResult(prev => ({ ...prev, [provider.id]: null }));
    try {
      const res = await fetch(`${API_BASE}/providers/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_url: provider.base_url, api_key: provider.api_key, provider_id: provider.id }),
      });
      const data = await res.json();
      setTestResult(prev => ({ ...prev, [provider.id]: { ...data, message: stripHtml(data.message || '') } }));
    } catch (e: any) {
      setTestResult(prev => ({ ...prev, [provider.id]: { ok: false, message: stripHtml(e.message), latency_ms: 0 } }));
    } finally {
      setTestingProvider(null);
    }
  };

  const fetchModels = async (provider: Provider) => {
    setFetchingModels(provider.id);
    try {
      const res = await fetch(`${API_BASE}/providers/fetch-models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_url: provider.base_url, api_key: provider.api_key, provider_id: provider.id }),
      });
      const data = await res.json();
      if (data.error) {
        setError(`Fetch models failed: ${stripHtml(data.error)}`);
      } else {
        setAvailableModels(prev => ({ ...prev, [provider.id]: data.models }));
      }
    } catch (e: any) {
      setError(`Fetch models failed: ${e.message}`);
    } finally {
      setFetchingModels(null);
    }
  };

  const toggleModel = (providerId: string, modelId: string) => {
    if (!settings) return;
    const provider = settings.providers.find(p => p.id === providerId);
    if (!provider) return;
    const has = provider.models.includes(modelId);
    const newModels = has ? provider.models.filter(m => m !== modelId) : [...provider.models, modelId];
    updateProvider(providerId, { models: newModels });
  };

  const removeProvider = (id: string) => {
    if (!settings) return;
    setSettings({ ...settings, providers: settings.providers.filter((p) => p.id !== id) });
  };

  const updateProvider = (id: string, updates: Partial<Provider>) => {
    if (!settings) return;
    setSettings({
      ...settings,
      providers: settings.providers.map((p) => (p.id === id ? { ...p, ...updates } : p)),
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-700">
        <h1 className="text-base font-semibold text-gray-200">Global Settings</h1>
        <button
          onClick={onClose}
          className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
          type="button"
        >
          <X size={18} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {loading && <div className="text-[12px] text-gray-500">Loading settings...</div>}
        {error && <div className="text-[12px] text-red-400 mb-3">{error}</div>}

        {settings && (
          <div className="max-w-2xl mx-auto space-y-6">
            {/* LLM Providers */}
            <section>
              <h2 className="text-[13px] font-semibold text-gray-300 mb-3">LLM Providers</h2>
              <div className="space-y-2">
                {settings.providers.map((provider) => (
                  <div key={provider.id} className="border border-gray-700 rounded-lg overflow-hidden">
                    {/* Provider header */}
                    <div
                      className="flex items-center gap-2 px-3 py-2 bg-gray-800 cursor-pointer hover:bg-gray-750"
                      onClick={() =>
                        setExpandedProvider(expandedProvider === provider.id ? null : provider.id)
                      }
                    >
                      {expandedProvider === provider.id ? (
                        <ChevronDown size={14} className="text-gray-500" />
                      ) : (
                        <ChevronRight size={14} className="text-gray-500" />
                      )}
                      <input
                        type="checkbox"
                        checked={provider.enabled}
                        onChange={(e) => {
                          e.stopPropagation();
                          updateProvider(provider.id, { enabled: e.target.checked });
                        }}
                        className="accent-blue-500"
                      />
                      <span className="text-[12px] text-gray-200 flex-1">
                        {provider.name || '(unnamed provider)'}
                      </span>
                      <span className="text-[10px] text-gray-500">
                        {provider.models.length} model{provider.models.length !== 1 ? 's' : ''}
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeProvider(provider.id);
                        }}
                        className="p-1 hover:bg-gray-700 rounded text-gray-500 hover:text-red-400 transition-colors"
                        type="button"
                        title="Remove provider"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>

                    {/* Provider details */}
                    {expandedProvider === provider.id && (
                      <div className="px-3 py-3 space-y-3 bg-gray-850">
                        {/* Vertex AI info banner */}
                        {provider.id.startsWith('vertex') && (
                          <div className="bg-blue-900/20 border border-blue-800/40 rounded-lg px-3 py-2 text-[11px] text-blue-300 space-y-1">
                            <p className="font-medium">Vertex AI uses GCP Service Account authentication</p>
                            <p className="text-[10px] text-blue-400/80">
                              Set the <code className="bg-gray-800 px-1 rounded text-[10px]">GOOGLE_APPLICATION_CREDENTIALS</code> environment
                              variable to the path of your service account JSON key file before starting the server.
                              The API Key field below is not used for Vertex AI.
                            </p>
                          </div>
                        )}

                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block text-[10px] text-gray-500 mb-0.5">Name</label>
                            <input
                              value={provider.name}
                              onChange={(e) => updateProvider(provider.id, { name: e.target.value })}
                              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                              placeholder="e.g. SiliconFlow"
                            />
                          </div>
                          {!provider.id.startsWith('vertex') && (
                          <div>
                            <label className="block text-[10px] text-gray-500 mb-0.5">API Key</label>
                            <div className="relative">
                              <input
                                type={showApiKey[provider.id] ? 'text' : 'password'}
                                value={provider.api_key}
                                onChange={(e) => updateProvider(provider.id, { api_key: e.target.value })}
                                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 pr-7 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                                placeholder="sk-..."
                              />
                              <button
                                type="button"
                                onClick={() => setShowApiKey(prev => ({ ...prev, [provider.id]: !prev[provider.id] }))}
                                className="absolute right-1.5 top-1/2 -translate-y-1/2 p-0.5 text-gray-500 hover:text-gray-300 transition-colors"
                                title={showApiKey[provider.id] ? 'Hide API key' : 'Show API key'}
                              >
                                {showApiKey[provider.id] ? <EyeOff size={12} /> : <Eye size={12} />}
                              </button>
                            </div>
                            <span className="text-[9px] text-gray-600 mt-0.5 block">Multiple keys separated by commas</span>
                          </div>
                          )}
                        </div>
                        {!provider.id.startsWith('vertex') && (
                        <div>
                          <label className="block text-[10px] text-gray-500 mb-0.5">Base URL</label>
                          <input
                            value={provider.base_url}
                            onChange={(e) => updateProvider(provider.id, { base_url: e.target.value })}
                            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                            placeholder="https://api.example.com/v1"
                          />
                        </div>
                        )}

                        {/* Action buttons: Test Connection + Fetch Models */}
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => testConnection(provider)}
                            disabled={(!provider.base_url && !provider.id.startsWith('vertex')) || testingProvider === provider.id}
                            className="flex items-center gap-1 px-2.5 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded text-[11px] text-gray-300 transition-colors"
                            type="button"
                          >
                            {testingProvider === provider.id ? (
                              <RefreshCw size={11} className="animate-spin" />
                            ) : testResult[provider.id]?.ok ? (
                              <Wifi size={11} className="text-green-400" />
                            ) : testResult[provider.id] ? (
                              <WifiOff size={11} className="text-red-400" />
                            ) : (
                              <Wifi size={11} />
                            )}
                            Test Connection
                          </button>
                          <button
                            onClick={() => fetchModels(provider)}
                            disabled={(!provider.base_url && !provider.id.startsWith('vertex')) || fetchingModels === provider.id}
                            className="flex items-center gap-1 px-2.5 py-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-[11px] text-white transition-colors"
                            type="button"
                          >
                            {fetchingModels === provider.id ? (
                              <RefreshCw size={11} className="animate-spin" />
                            ) : (
                              <RefreshCw size={11} />
                            )}
                            Fetch Models
                          </button>
                          {testResult[provider.id] && (
                            <span className={`text-[10px] ${testResult[provider.id]!.ok ? 'text-green-400' : 'text-red-400'}`}>
                              {testResult[provider.id]!.message}
                              {testResult[provider.id]!.latency_ms > 0 && ` (${testResult[provider.id]!.latency_ms}ms)`}
                            </span>
                          )}
                        </div>

                        {/* Models section */}
                        <div>
                          <label className="block text-[10px] text-gray-500 mb-1">
                            Models ({provider.models.length} selected)
                          </label>

                          {/* Fetched models as checkboxes (edit mode) */}
                          {availableModels[provider.id] && availableModels[provider.id].length > 0 ? (
                            <div>
                              <div className="max-h-48 overflow-y-auto bg-gray-800 border border-gray-700 rounded p-1.5 space-y-0.5">
                                {availableModels[provider.id].map((m) => (
                                  <label
                                    key={m.id}
                                    className="flex items-center gap-2 px-1.5 py-1 rounded hover:bg-gray-700/50 cursor-pointer"
                                  >
                                    <input
                                      type="checkbox"
                                      checked={provider.models.includes(m.id)}
                                      onChange={() => toggleModel(provider.id, m.id)}
                                      className="accent-blue-500"
                                    />
                                    <span className="text-[11px] text-gray-300 flex-1 truncate">{m.id}</span>
                                    {m.owned_by && (
                                      <span className="text-[9px] text-gray-600 shrink-0">{m.owned_by}</span>
                                    )}
                                  </label>
                                ))}
                              </div>
                              <button
                                type="button"
                                onClick={() => setAvailableModels(prev => { const next = { ...prev }; delete next[provider.id]; return next; })}
                                className="mt-1.5 px-2.5 py-1 bg-gray-700 hover:bg-gray-600 rounded text-[11px] text-gray-300 transition-colors"
                              >
                                Done
                              </button>
                            </div>
                          ) : provider.models.length > 0 ? (
                            /* Read-only selected models list (non-edit mode) */
                            <div className="bg-gray-800 border border-gray-700 rounded p-1.5 space-y-0.5">
                              {provider.models.map((modelId) => (
                                <div key={modelId} className="flex items-center gap-2 px-1.5 py-0.5">
                                  <span className="text-[11px] text-gray-300 flex-1 truncate">{modelId}</span>
                                  <button
                                    type="button"
                                    onClick={() => updateProvider(provider.id, { models: provider.models.filter(m => m !== modelId) })}
                                    className="p-0.5 text-gray-600 hover:text-red-400 transition-colors"
                                    title="Remove model"
                                  >
                                    <X size={10} />
                                  </button>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div>
                              <textarea
                                value={provider.models.join('\n')}
                                onChange={(e) =>
                                  updateProvider(provider.id, {
                                    models: e.target.value
                                      .split('\n')
                                      .map((s) => s.trim())
                                      .filter(Boolean),
                                  })
                                }
                                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[12px] text-gray-200 resize-none focus:outline-none focus:border-blue-500"
                                rows={3}
                                placeholder="Click 'Fetch Models' above, or enter manually (one per line)&#10;deepseek-ai/DeepSeek-V3&#10;deepseek-ai/DeepSeek-R1"
                              />
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Add provider with presets */}
                <div className="border border-dashed border-gray-700 rounded-lg p-2">
                  <div className="text-[10px] text-gray-500 mb-1.5 text-center">Add Provider</div>
                  <div className="flex flex-wrap gap-1 justify-center">
                    {PROVIDER_PRESETS.map((preset) => (
                      <button
                        key={preset.id}
                        onClick={() => addProvider(preset.id)}
                        className="px-2 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-[11px] text-gray-400 hover:text-gray-200 transition-colors"
                        type="button"
                      >
                        <Plus size={10} className="inline mr-0.5" />
                        {preset.name}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            {/* Defaults */}
            <section>
              <h2 className="text-[13px] font-semibold text-gray-300 mb-3">Defaults</h2>
              <div className="space-y-3">
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">Default Model</label>
                  <input
                    value={settings.defaults.model}
                    onChange={(e) => updateDefaults('model', e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                    placeholder="deepseek-ai/DeepSeek-V3"
                  />
                </div>
                <div className="flex gap-3">
                  <div className="flex-1">
                    <label className="block text-[10px] text-gray-500 mb-0.5">Temperature (0-2)</label>
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="2"
                      value={settings.defaults.temperature}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!isNaN(v) && v >= 0 && v <= 2) {
                          updateDefaults('temperature', v);
                        }
                      }}
                      onBlur={(e) => {
                        const v = parseFloat(e.target.value);
                        if (isNaN(v) || v < 0) updateDefaults('temperature', 0);
                        else if (v > 2) updateDefaults('temperature', 2);
                      }}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div className="flex-1">
                    <label className="block text-[10px] text-gray-500 mb-0.5">Max Tokens (≥256)</label>
                    <input
                      type="number"
                      step="256"
                      min="256"
                      max="131072"
                      value={settings.defaults.max_tokens}
                      onChange={(e) => {
                        const v = parseInt(e.target.value);
                        if (!isNaN(v) && v >= 0) {
                          setSettings({ ...settings, defaults: { ...settings.defaults, max_tokens: Math.min(v, 131072) } });
                        }
                      }}
                      onBlur={(e) => {
                        const v = parseInt(e.target.value);
                        if (isNaN(v) || v < 256) {
                          setSettings({ ...settings!, defaults: { ...settings!.defaults, max_tokens: 256 } });
                        } else if (v > 131072) {
                          setSettings({ ...settings!, defaults: { ...settings!.defaults, max_tokens: 131072 } });
                        }
                      }}
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">
                    Default System Prompt
                  </label>
                  <textarea
                    value={settings.defaults.system_instructions}
                    onChange={(e) => updateDefaults('system_instructions', e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-[12px] text-gray-200 resize-none focus:outline-none focus:border-blue-500"
                    rows={3}
                    placeholder="You are a helpful assistant."
                  />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <label className="block text-[10px] text-gray-500">Stream Output</label>
                    <span className="text-[10px] text-gray-600">
                      {settings.defaults.stream ? 'Tokens appear as they are generated' : 'Full response appears at once'}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => updateDefaults('stream', !settings.defaults.stream)}
                    className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                      settings.defaults.stream ? 'bg-blue-600' : 'bg-gray-600'
                    }`}
                  >
                    <span
                      className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                        settings.defaults.stream ? 'translate-x-4' : 'translate-x-0.5'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </section>

            {/* Display */}
            <section>
              <h2 className="text-[13px] font-semibold text-gray-300 mb-3">Display</h2>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">User Name</label>
                  <input
                    value={settings.display.user_name}
                    onChange={(e) => updateDisplay('user_name', e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-0.5">Assistant Name</label>
                  <input
                    value={settings.display.assistant_name}
                    onChange={(e) => updateDisplay('assistant_name', e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-1">User Avatar</label>
                  <div className="flex flex-wrap gap-1">
                    {AVATAR_OPTIONS.map((emoji) => (
                      <button
                        key={`user-${emoji}`}
                        type="button"
                        onClick={() => updateDisplay('user_avatar', emoji)}
                        className={`w-8 h-8 rounded-md text-base flex items-center justify-center transition-colors ${
                          settings.display.user_avatar === emoji
                            ? 'bg-blue-600 ring-1 ring-blue-400'
                            : 'bg-gray-800 hover:bg-gray-700'
                        }`}
                      >
                        {emoji}
                      </button>
                    ))}
                    <button
                      type="button"
                      onClick={() => updateDisplay('user_avatar', null)}
                      className={`w-8 h-8 rounded-md text-[9px] flex items-center justify-center transition-colors ${
                        !settings.display.user_avatar
                          ? 'bg-blue-600 ring-1 ring-blue-400 text-white'
                          : 'bg-gray-800 hover:bg-gray-700 text-gray-500'
                      }`}
                      title="Default icon"
                    >
                      Auto
                    </button>
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] text-gray-500 mb-1">Assistant Avatar</label>
                  <div className="flex flex-wrap gap-1">
                    {AVATAR_OPTIONS.map((emoji) => (
                      <button
                        key={`asst-${emoji}`}
                        type="button"
                        onClick={() => updateDisplay('assistant_avatar', emoji)}
                        className={`w-8 h-8 rounded-md text-base flex items-center justify-center transition-colors ${
                          settings.display.assistant_avatar === emoji
                            ? 'bg-blue-600 ring-1 ring-blue-400'
                            : 'bg-gray-800 hover:bg-gray-700'
                        }`}
                      >
                        {emoji}
                      </button>
                    ))}
                    <button
                      type="button"
                      onClick={() => updateDisplay('assistant_avatar', null)}
                      className={`w-8 h-8 rounded-md text-[9px] flex items-center justify-center transition-colors ${
                        !settings.display.assistant_avatar
                          ? 'bg-blue-600 ring-1 ring-blue-400 text-white'
                          : 'bg-gray-800 hover:bg-gray-700 text-gray-500'
                      }`}
                      title="Default icon"
                    >
                      Auto
                    </button>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-gray-700 flex justify-end gap-2">
        <button
          onClick={onClose}
          className="px-4 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-md text-[12px] text-gray-300 transition-colors"
          type="button"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={saving || !settings || !isDirty}
          className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-md text-[12px] text-white transition-colors"
          type="button"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        {saveSuccess && (
          <span className="flex items-center gap-1 text-[11px] text-green-400 animate-fade-in">
            <Check size={13} /> Settings saved
          </span>
        )}
      </div>
    </div>
  );
};
