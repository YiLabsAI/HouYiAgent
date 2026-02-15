/**
 * ConversationSettingsDrawer: slide-out panel for conversation-level settings.
 *
 * Allows overriding model, system prompt, temperature, max tokens at the
 * conversation level. Shows inherited values from global settings in gray.
 *
 */
import React from 'react';
import { RotateCcw } from 'lucide-react';
import { useChatStore } from '@/stores/useChatStore';
import { DEFAULT_MODEL } from '@/constants/models';
import { useAvailableModels } from '@/hooks/useAvailableModels';
import { CenterStage } from '@/components/CenterStage';

interface ConversationSettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenGlobalSettings: () => void;
}

export const ConversationSettingsDrawer: React.FC<ConversationSettingsDrawerProps> = ({
  isOpen,
  onClose,
  onOpenGlobalSettings,
}) => {
  const activeConversation = useChatStore((s) => s.activeConversation);
  const updateConversation = useChatStore((s) => s.updateConversation);

  const [title, setTitle] = React.useState('');
  const [model, setModel] = React.useState('');
  const [systemInstructions, setSystemInstructions] = React.useState('');
  const [temperature, setTemperature] = React.useState<string>('');
  const [maxTokens, setMaxTokens] = React.useState<string>('');
  const [topP, setTopP] = React.useState<string>('');
  const [stream, setStream] = React.useState<boolean | null>(null);
  const [isDirty, setIsDirty] = React.useState(false);
  const { models: availableModelsWithProvider } = useAvailableModels();
  const [globalDefaults, setGlobalDefaults] = React.useState<{ temperature: number; max_tokens: number; top_p: number; stream: boolean } | null>(null);

  // Fetch global defaults for temperature/max_tokens display
  React.useEffect(() => {
    if (isOpen) {
      fetch('/api/chat/settings')
        .then((r) => r.json())
        .then((settings) => {
          setGlobalDefaults({
            temperature: settings.defaults?.temperature ?? 0.7,
            max_tokens: settings.defaults?.max_tokens ?? 4096,
            top_p: settings.defaults?.top_p ?? 1.0,
            stream: settings.defaults?.stream ?? true,
          });
        })
        .catch(() => {});
    }
  }, [isOpen]);

  // Sync form state when conversation changes OR drawer opens.
  // Re-syncing on isOpen ensures unsaved edits are discarded when the
  // drawer is closed and reopened (the form reverts to persisted values).
  React.useEffect(() => {
    if (activeConversation) {
      setTitle(activeConversation.title || '');
      setModel(activeConversation.model || '');
      setSystemInstructions(activeConversation.system_instructions || '');
      setTemperature(activeConversation.temperature != null ? String(activeConversation.temperature) : '');
      setMaxTokens(activeConversation.max_tokens != null ? String(activeConversation.max_tokens) : '');
      setTopP(activeConversation.top_p != null ? String(activeConversation.top_p) : '');
      setStream(activeConversation.stream);
      setIsDirty(false);
    }
  }, [activeConversation?.conversation_id, isOpen]);

  const handleSave = async () => {
    if (!activeConversation) return;
    const parsedMaxTokens = maxTokens ? Math.min(parseInt(maxTokens), 131072) : undefined;
    await updateConversation(activeConversation.conversation_id, {
      title: title || undefined,
      model: model || undefined,
      system_instructions: systemInstructions || undefined,
      temperature: temperature ? parseFloat(temperature) : undefined,
      max_tokens: parsedMaxTokens,
      top_p: topP ? parseFloat(topP) : undefined,
      stream: stream,
    });
    setIsDirty(false);
  };

  const handleReset = async () => {
    if (!activeConversation) return;
    await updateConversation(activeConversation.conversation_id, {
      model: '',
      system_instructions: '',
      temperature: null,
      max_tokens: null,
      top_p: null,
      stream: null,
    });
    setModel('');
    setSystemInstructions('');
    setTemperature('');
    setMaxTokens('');
    setTopP('');
    setStream(null);
    setIsDirty(false);
  };

  return (
    <CenterStage isOpen={isOpen} onClose={onClose} size="M" title="Conversation Settings">
      <div className="space-y-4">
        {!activeConversation ? (
          <div className="text-[12px] text-gray-500">No active conversation</div>
        ) : (
            <>
              {/* Title */}
              <div>
                <label className="block text-[11px] font-medium text-gray-400 mb-1">Title</label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => { setTitle(e.target.value); setIsDirty(true); }}
                  placeholder="New Chat"
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                />
              </div>

              {/* Model */}
              <div>
                <label className="block text-[11px] font-medium text-gray-400 mb-1">Model</label>
                <select
                  value={model}
                  onChange={(e) => { setModel(e.target.value); setIsDirty(true); }}
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                >
                  <option value="">Default ({DEFAULT_MODEL})</option>
                  {availableModelsWithProvider.map((m) => (
                    <option key={`${m.provider}-${m.model}`} value={m.model}>
                      {m.model} ({m.provider})
                    </option>
                  ))}
                </select>
                {!model && (
                  <span className="text-[10px] text-gray-600 italic">(inherited from global)</span>
                )}
              </div>

              {/* System Prompt */}
              <div>
                <label className="block text-[11px] font-medium text-gray-400 mb-1">System Prompt</label>
                <textarea
                  value={systemInstructions}
                  onChange={(e) => { setSystemInstructions(e.target.value); setIsDirty(true); }}
                  placeholder="You are a helpful assistant..."
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-[12px] text-gray-200 resize-none focus:outline-none focus:border-blue-500"
                  rows={4}
                />
                {!systemInstructions && (
                  <span className="text-[10px] text-gray-600 italic">(inherited from global)</span>
                )}
              </div>

              {/* Temperature + Max Tokens */}
              <div className="flex gap-3">
                <div className="flex-1">
                  <label className="block text-[11px] font-medium text-gray-400 mb-1">Temperature</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="2"
                    value={temperature}
                    onChange={(e) => {
                      const v = e.target.value;
                      if (v === '' || (parseFloat(v) >= 0 && parseFloat(v) <= 2)) {
                        setTemperature(v);
                        setIsDirty(true);
                      }
                    }}
                    placeholder={globalDefaults ? `${globalDefaults.temperature}` : '0.7'}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                  />
                  {!temperature && (
                    <span className="text-[10px] text-gray-600 italic">(inherited from global)</span>
                  )}
                </div>
                <div className="flex-1">
                  <label className="block text-[11px] font-medium text-gray-400 mb-1">Max Tokens</label>
                  <input
                    type="number"
                    step="256"
                    min="1"
                    max="131072"
                    value={maxTokens}
                    onChange={(e) => {
                      const v = e.target.value;
                      const num = parseInt(v);
                      if (v === '' || (num >= 0 && num <= 131072)) {
                        setMaxTokens(v);
                        setIsDirty(true);
                      }
                    }}
                    placeholder={globalDefaults ? `${globalDefaults.max_tokens}` : '4096'}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500"
                  />
                  {!maxTokens && (
                    <span className="text-[10px] text-gray-600 italic">(inherited from global)</span>
                  )}
                </div>
              </div>

              {/* Top-P */}
              <div>
                <label className="block text-[11px] font-medium text-gray-400 mb-1">Top-P</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={topP}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (v === '' || (parseFloat(v) >= 0 && parseFloat(v) <= 1)) {
                      setTopP(v);
                      setIsDirty(true);
                    }
                  }}
                  placeholder={globalDefaults ? `${globalDefaults.top_p}` : '1.0'}
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-[12px] text-gray-200 focus:outline-none focus:border-blue-500 max-w-[180px]"
                />
                {!topP && (
                  <span className="text-[10px] text-gray-600 italic">(inherited from global)</span>
                )}
              </div>

              {/* Stream Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-[11px] font-medium text-gray-400">Stream Output</label>
                  <span className="text-[10px] text-gray-600">
                    {stream == null ? '(inherited from global)' : stream ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const effectiveStream = stream ?? (globalDefaults?.stream ?? true);
                    setStream(!effectiveStream);
                    setIsDirty(true);
                  }}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    (stream ?? (globalDefaults?.stream ?? true))
                      ? 'bg-blue-600'
                      : 'bg-gray-600'
                  }`}
                >
                  <span
                    className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
                      (stream ?? (globalDefaults?.stream ?? true))
                        ? 'translate-x-4'
                        : 'translate-x-0.5'
                    }`}
                  />
                </button>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={handleSave}
                  disabled={!isDirty}
                  className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-md text-[12px] text-white transition-colors"
                  type="button"
                >
                  Save
                </button>
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-md text-[12px] text-gray-300 transition-colors"
                  type="button"
                >
                  <RotateCcw size={11} /> Reset to Global
                </button>
              </div>

              {/* Divider */}
              <hr className="border-gray-700/50" />

              {/* Global Settings link */}
              <button
                onClick={() => { onClose(); onOpenGlobalSettings(); }}
                className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded-md text-[12px] text-gray-400 hover:text-gray-200 transition-colors"
                type="button"
              >
                Global Settings...
              </button>
            </>
        )}
      </div>
    </CenterStage>
  );
};
