import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';

interface RightSidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

type TabType = 'config' | 'prompt' | 'inputs' | 'outputs' | 'verification' | 'metadata';

type PromptVariable = {
  id: string;
  key: string;
  value: string;
};

export const RightSidebar: React.FC<RightSidebarProps> = ({ isCollapsed, onToggleCollapse }) => {
  const { selectedNodeId, nodes, updateNode } = useConsoleStore();
  const [activeTab, setActiveTab] = React.useState<TabType>('config');
  const [saveStatus, setSaveStatus] = React.useState<'idle' | 'saving' | 'saved'>('idle');
  const [maxTokensDraft, setMaxTokensDraft] = React.useState<string>('');
  const [thinkingBudgetDraft, setThinkingBudgetDraft] = React.useState<string>('');
  const [systemPromptDraft, setSystemPromptDraft] = React.useState<string>('');
  const [userPromptDraft, setUserPromptDraft] = React.useState<string>('');
  const [labelDraft, setLabelDraft] = React.useState<string>('');
  const [inputsDraft, setInputsDraft] = React.useState<string>('');
  const [inputsError, setInputsError] = React.useState<string | null>(null);
  const [promptVariables, setPromptVariables] = React.useState<PromptVariable[]>([]);

  const selectedNode = nodes.find(n => n.id === selectedNodeId);
  const viewExecution = useConsoleStore((state) =>
    state.viewMode === 'checkpoint'
      ? state.checkpointExecution
      : state.liveExecution || state.currentExecution,
  );
  const nodeExecution = selectedNode ? viewExecution?.node_executions?.[selectedNode.id] : null;
  const outputsPayload = nodeExecution?.outputs ?? selectedNode?.data?.outputs ?? {};

  // Debounce save status
  const saveTimerRef = React.useRef<number | null>(null);

  const handleSave = React.useCallback(() => {
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }

    setSaveStatus('saved');
    saveTimerRef.current = window.setTimeout(() => setSaveStatus('idle'), 2000);
  }, []);

  React.useEffect(() => {
    if (!selectedNode || selectedNode.type !== 'llm') return;
    const maxTokens = selectedNode.data?.config?.max_tokens;
    const nextMaxTokens = maxTokens === undefined || maxTokens === null || Number.isNaN(maxTokens)
      ? '2000'
      : String(maxTokens);
    setMaxTokensDraft(nextMaxTokens);

    const thinkingBudget = selectedNode.data?.config?.thinking_budget;
    const nextThinkingBudget = thinkingBudget === undefined || thinkingBudget === null || Number.isNaN(thinkingBudget)
      ? '1024'
      : String(thinkingBudget);
    setThinkingBudgetDraft(nextThinkingBudget);

  }, [selectedNode, selectedNodeId, selectedNode?.type, selectedNode?.data?.config?.max_tokens, selectedNode?.data?.config?.thinking_budget]);

  React.useEffect(() => {
    if (!selectedNode || selectedNode.type !== 'llm') return;
    const config = selectedNode.data?.config || {};
    const nextSystemPrompt = (config.system_prompt ?? '') as string;
    const nextUserPrompt = (config.user_prompt ?? (config.system_prompt ? '' : config.prompt ?? '')) as string;

    const rawVariables = config.prompt_variables ?? {};
    const normalizedVariables: PromptVariable[] = Array.isArray(rawVariables)
      ? rawVariables.map((entry: any) => ({
          id: entry.id ?? `var_${Math.random().toString(36).slice(2, 8)}`,
          key: entry.key ?? '',
          value: entry.value ?? '',
        }))
      : Object.entries(rawVariables).map(([key, value]) => ({
          id: `var_${Math.random().toString(36).slice(2, 8)}`,
          key,
          value: String(value ?? ''),
        }));

    setSystemPromptDraft(nextSystemPrompt);
    setUserPromptDraft(nextUserPrompt);
    setPromptVariables(normalizedVariables);
  }, [
    selectedNode,
    selectedNodeId,
    selectedNode?.type,
    selectedNode?.data?.config?.system_prompt,
    selectedNode?.data?.config?.user_prompt,
    selectedNode?.data?.config?.prompt,
    selectedNode?.data?.config?.prompt_variables,
  ]);

  React.useEffect(() => {
    if (!selectedNode) return;
    const inputsPayload = nodeExecution?.inputs ?? selectedNode?.data?.inputs ?? {};
    setInputsDraft(JSON.stringify(inputsPayload ?? {}, null, 2));
    setInputsError(null);
  }, [selectedNode, selectedNodeId, nodeExecution?.inputs, selectedNode?.data?.inputs]);

  React.useEffect(() => {
    if (!selectedNode) return;
    setLabelDraft(selectedNode.data?.label ?? '');
  }, [selectedNode, selectedNodeId, selectedNode?.data?.label]);

  const commitNumericConfig = React.useCallback((field: 'max_tokens' | 'thinking_budget', value: string) => {
    if (!selectedNodeId || !selectedNode?.data) return;
    const trimmed = value.trim();
    if (!trimmed) return;
    const parsed = Number(trimmed);
    if (Number.isNaN(parsed)) return;

    const updatedData = {
      ...selectedNode.data,
      config: {
        ...selectedNode.data.config,
        [field]: parsed,
      },
    };
    updateNode(selectedNodeId, updatedData as any);
    handleSave();
  }, [selectedNodeId, selectedNode, updateNode, handleSave]);


  const handlePromptVariableChange = React.useCallback((id: string, field: 'key' | 'value', value: string) => {
    setPromptVariables((prev) =>
      prev.map((entry) => (entry.id === id ? { ...entry, [field]: value } : entry)),
    );
  }, []);

  const handleAddPromptVariable = React.useCallback(() => {
    setPromptVariables((prev) => ([
      ...prev,
      { id: `var_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`, key: '', value: '' },
    ]));
  }, []);

  const handleRemovePromptVariable = React.useCallback((id: string) => {
    setPromptVariables((prev) => prev.filter((entry) => entry.id !== id));
  }, []);

  const updatePromptConfig = React.useCallback((next: {
    systemPrompt?: string;
    userPrompt?: string;
    variables?: PromptVariable[];
  }) => {
    if (!selectedNodeId || !selectedNode?.data) return;

    const config = selectedNode.data?.config || {};
    const systemPrompt = next.systemPrompt ?? systemPromptDraft;
    const userPrompt = next.userPrompt ?? userPromptDraft;
    const variables = next.variables ?? promptVariables;
    const promptVariablesObject = variables.reduce<Record<string, string>>((acc, entry) => {
      const key = entry.key.trim();
      if (key) {
        acc[key] = entry.value;
      }
      return acc;
    }, {});
    const combinedPrompt = [systemPrompt, userPrompt].filter(Boolean).join('\n\n');

    const updatedData = {
      ...selectedNode.data,
      config: {
        ...config,
        system_prompt: systemPrompt,
        user_prompt: userPrompt,
        prompt_variables: promptVariablesObject,
        prompt: combinedPrompt,
      },
    };

    updateNode(selectedNodeId, updatedData as any);
    handleSave();
  }, [selectedNodeId, selectedNode, systemPromptDraft, userPromptDraft, promptVariables, updateNode, handleSave]);

  const tabs: { id: TabType; label: string }[] = React.useMemo(() => {
    if (!selectedNode) {
      return [
        { id: 'config', label: 'Config' },
        { id: 'inputs', label: 'Inputs' },
        { id: 'outputs', label: 'Outputs' },
        { id: 'metadata', label: 'Metadata' },
      ];
    }

    const base: { id: TabType; label: string }[] = [
      { id: 'config', label: 'Config' },
      { id: 'inputs', label: 'Inputs' },
      { id: 'outputs', label: 'Outputs' },
    ];

    if (selectedNode.type === 'llm') {
      base.splice(1, 0, { id: 'prompt', label: 'Prompt' });
    }
    if (selectedNode.type === 'verify') {
      base.push({ id: 'verification', label: 'Verification' });
    }
    base.push({ id: 'metadata', label: 'Metadata' });
    return base;
  }, [selectedNode]);

  React.useEffect(() => {
    const valid = tabs.some((t) => t.id === activeTab);
    if (!valid) {
      setActiveTab('config');
    }
  }, [tabs, activeTab]);

  return (
    <div className={`bg-gray-800 border-l border-gray-700 flex flex-col transition-all duration-300 ease-in-out ${
      isCollapsed ? 'w-8' : 'w-80'
    }`}>
      {isCollapsed ? (
        <div className="flex items-center justify-center h-full">
          <button
            onClick={onToggleCollapse}
            className="p-1 hover:bg-gray-700 rounded text-gray-400"
            title="Expand sidebar"
          >
            ◀
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between p-3 border-b border-gray-700">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-gray-200">Properties</h2>
              {saveStatus === 'saved' && (
                <span className="text-xs text-green-400">✓ Saved</span>
              )}
            </div>
            <button
              onClick={onToggleCollapse}
              className="p-1 hover:bg-gray-700 rounded text-gray-400"
              title="Collapse sidebar"
            >
              ▶
            </button>
          </div>

          <div className="flex border-b border-gray-700">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 min-w-0 px-3 py-2 text-xs font-medium ${
              activeTab === tab.id
                ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <span className="block truncate">{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!selectedNode ? (
          <div className="text-center text-gray-400 text-sm mt-8">
            Select a node to view properties
          </div>
        ) : (
          <>
            {activeTab === 'config' && (
              <div className="space-y-3">
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Node ID</label>
                  <input
                    type="text"
                    className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200"
                    value={selectedNode.id}
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Node Type</label>
                  <input
                    type="text"
                    className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200"
                    value={selectedNode.type}
                    readOnly
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Label</label>
                  <input
                    type="text"
                    className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    value={labelDraft}
                    onChange={(e) => setLabelDraft(e.target.value)}
                    onBlur={() => {
                      if (!selectedNodeId || !selectedNode?.data) return;
                      const nextLabel = labelDraft;
                      if (nextLabel === (selectedNode.data?.label ?? '')) return;
                      const updatedData = {
                        ...selectedNode.data,
                        label: nextLabel,
                        metadata: {
                          ...(selectedNode.data as any)?.metadata,
                          label: nextLabel,
                        },
                      };
                      updateNode(selectedNodeId, updatedData as any);
                      handleSave();
                    }}
                    placeholder="Enter label"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-400 mb-1">Description</label>
                  <textarea
                    className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    rows={3}
                    value={selectedNode.data?.description || ''}
                    onChange={(e) => {
                      if (selectedNodeId && selectedNode.data) {
                        const updatedData = { ...selectedNode.data, description: e.target.value };
                        updateNode(selectedNodeId, updatedData as any);
                        handleSave();
                      }
                    }}
                    placeholder="Enter description"
                  />
                </div>

                {/* LLM-specific config */}
                {selectedNode.type === 'llm' && (
                  <>
                    <div className="border-t border-gray-700 pt-3 mt-3">
                      <h4 className="text-xs font-semibold text-gray-300 mb-2">LLM Configuration</h4>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">Model</label>
                      <select
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        value={selectedNode.data?.config?.model || 'deepseek-ai/DeepSeek-V3'}
                        onChange={(e) => {
                          if (selectedNodeId && selectedNode.data) {
                            const updatedData = {
                              ...selectedNode.data,
                              config: { ...selectedNode.data.config, model: e.target.value }
                            };
                            updateNode(selectedNodeId, updatedData as any);
                            handleSave();
                          }
                        }}
                      >
                        <option value="deepseek-ai/DeepSeek-V3">DeepSeek-V3 (Standard)</option>
                        <option value="deepseek-ai/DeepSeek-V3.1">DeepSeek-V3.1 (Latest)</option>
                        <option value="deepseek-ai/DeepSeek-R1">DeepSeek-R1 (Reasoning)</option>
                        <option value="Qwen/Qwen2.5-72B-Instruct">Qwen 2.5 72B</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">
                        Temperature: {selectedNode.data?.config?.temperature || 0.7}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        className="w-full"
                        value={selectedNode.data?.config?.temperature || 0.7}
                        onChange={(e) => {
                          if (selectedNodeId && selectedNode.data) {
                            const updatedData = {
                              ...selectedNode.data,
                              config: { ...selectedNode.data.config, temperature: parseFloat(e.target.value) }
                            };
                            updateNode(selectedNodeId, updatedData as any);
                            handleSave();
                          }
                        }}
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">Max Tokens</label>
                      <input
                        type="number"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        value={maxTokensDraft}
                        onChange={(e) => {
                          const nextValue = e.target.value;
                          setMaxTokensDraft(nextValue);
                          commitNumericConfig('max_tokens', nextValue);
                        }}
                        onBlur={() => {
                          if (!maxTokensDraft.trim()) {
                            const fallback = selectedNode.data?.config?.max_tokens ?? 2000;
                            setMaxTokensDraft(String(fallback));
                          }
                        }}
                        placeholder="2000"
                      />
                    </div>
                    {/* Only show reasoning options for R1 model */}
                    {selectedNode.data?.config?.model?.includes('R1') && (
                      <div>
                        <label className="flex items-center gap-2 text-xs font-medium text-gray-400 mb-1">
                          <input
                            type="checkbox"
                            className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                            checked={selectedNode.data?.config?.enable_reasoning || false}
                            onChange={(e) => {
                              if (selectedNodeId && selectedNode.data) {
                                const updatedData = {
                                  ...selectedNode.data,
                                  config: { ...selectedNode.data.config, enable_reasoning: e.target.checked }
                                };
                                updateNode(selectedNodeId, updatedData as any);
                                handleSave();
                              }
                            }}
                          />
                          Enable Reasoning
                        </label>
                        {selectedNode.data?.config?.enable_reasoning && (
                          <input
                            type="number"
                            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 mt-1"
                            value={thinkingBudgetDraft}
                            onChange={(e) => {
                              const nextValue = e.target.value;
                              setThinkingBudgetDraft(nextValue);
                              commitNumericConfig('thinking_budget', nextValue);
                            }}
                            onBlur={() => {
                              if (!thinkingBudgetDraft.trim()) {
                                const fallback = selectedNode.data?.config?.thinking_budget ?? 1024;
                                setThinkingBudgetDraft(String(fallback));
                              }
                            }}
                            placeholder="1024"
                            title="Thinking budget (tokens)"
                          />
                        )}
                      </div>
                    )}

                  </>
                )}

                {/* Tool-specific config */}
                {selectedNode.type === 'tool' && (
                  <>
                    <div className="border-t border-gray-700 pt-3 mt-3">
                      <h4 className="text-xs font-semibold text-gray-300 mb-2">Tool Configuration</h4>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">Tool Name</label>
                      <input
                        type="text"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        value={selectedNode.data?.config?.tool_name || ''}
                        onChange={(e) => {
                          if (selectedNodeId && selectedNode.data) {
                            const updatedData = {
                              ...selectedNode.data,
                              config: { ...selectedNode.data.config, tool_name: e.target.value }
                            };
                            updateNode(selectedNodeId, updatedData as any);
                            handleSave();
                          }
                        }}
                        placeholder="e.g., search_web, calculate"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">Timeout (seconds)</label>
                      <input
                        type="number"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        value={selectedNode.data?.config?.timeout || 30}
                        onChange={(e) => {
                          if (selectedNodeId && selectedNode.data) {
                            const updatedData = {
                              ...selectedNode.data,
                              config: { ...selectedNode.data.config, timeout: parseInt(e.target.value) }
                            };
                            updateNode(selectedNodeId, updatedData as any);
                            handleSave();
                          }
                        }}
                        placeholder="30"
                      />
                    </div>
                  </>
                )}

                {/* Verify-specific config */}
                {selectedNode.type === 'verify' && (
                  <>
                    <div className="border-t border-gray-700 pt-3 mt-3">
                      <h4 className="text-xs font-semibold text-gray-300 mb-2">Verify Configuration</h4>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">Raise on failure</label>
                      <input
                        type="text"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200"
                        value={String(Boolean((selectedNode.data?.config as any)?.raise_on_failure))}
                        readOnly
                      />
                      <div className="mt-1 text-[10px] text-gray-500">
                        When true, failed verification will abort execution.
                      </div>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">verification_rules</label>
                      <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap bg-gray-900/60 border border-gray-700 rounded p-2 max-h-48 overflow-auto">
                        {JSON.stringify((selectedNode.data?.config as any)?.verification_rules ?? [], null, 2)}
                      </pre>
                    </div>
                  </>
                )}

                {/* Route-specific config */}
                {selectedNode.type === 'route' && (
                  <>
                    <div className="border-t border-gray-700 pt-3 mt-3">
                      <h4 className="text-xs font-semibold text-gray-300 mb-2">Route Configuration</h4>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">disable_nodes_on_true</label>
                      <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap bg-gray-900/60 border border-gray-700 rounded p-2 max-h-24 overflow-auto">
                        {JSON.stringify((selectedNode.data?.config as any)?.disable_nodes_on_true ?? [], null, 2)}
                      </pre>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">disable_nodes_on_false</label>
                      <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap bg-gray-900/60 border border-gray-700 rounded p-2 max-h-24 overflow-auto">
                        {JSON.stringify((selectedNode.data?.config as any)?.disable_nodes_on_false ?? [], null, 2)}
                      </pre>
                    </div>
                  </>
                )}

                {/* Logic-specific config */}
                {selectedNode.type === 'logic' && (
                  <>
                    <div className="border-t border-gray-700 pt-3 mt-3">
                      <h4 className="text-xs font-semibold text-gray-300 mb-2">Logic Configuration</h4>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-400 mb-1">template</label>
                      <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap bg-gray-900/60 border border-gray-700 rounded p-2 max-h-48 overflow-auto">
                        {String((selectedNode.data?.config as any)?.template ?? '')}
                      </pre>
                      <div className="mt-1 text-[10px] text-gray-500">
                        Uses Python-style format_map with execution context.
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'prompt' && (
              <div className="space-y-4">
                {selectedNode.type !== 'llm' ? (
                  <div className="text-center text-gray-400 text-sm mt-4">
                    Prompt settings are only available for LLM nodes.
                  </div>
                ) : (
                  <>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-xs font-semibold text-gray-300">System Prompt</h4>
                        <span className="text-[10px] text-gray-500">Global instructions</span>
                      </div>
                      <textarea
                        className="w-full px-2 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        rows={4}
                        value={systemPromptDraft}
                        onChange={(e) => setSystemPromptDraft(e.target.value)}
                        onBlur={() => updatePromptConfig({ systemPrompt: systemPromptDraft })}
                        placeholder="Define the assistant persona, constraints, and policies..."
                      />
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-xs font-semibold text-gray-300">User Message</h4>
                        <span className="text-[10px] text-gray-500">Task instructions</span>
                      </div>
                      <textarea
                        className="w-full px-2 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        rows={5}
                        value={userPromptDraft}
                        onChange={(e) => setUserPromptDraft(e.target.value)}
                        onBlur={() => updatePromptConfig({ userPrompt: userPromptDraft })}
                        placeholder="Write the task prompt. Use variables like {{topic}}."
                      />
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-semibold text-gray-300">Template Variables</h4>
                        <button
                          type="button"
                          onClick={handleAddPromptVariable}
                          className="text-xs text-blue-400 hover:text-blue-300"
                        >
                          + Add variable
                        </button>
                      </div>
                      {promptVariables.length === 0 ? (
                        <div className="text-xs text-gray-500 bg-gray-900/60 border border-gray-700 rounded p-3">
                          No variables yet. Add variables to bind runtime inputs.
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {promptVariables.map((entry) => (
                            <div key={entry.id} className="flex items-center gap-2">
                              <input
                                type="text"
                                className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                placeholder="variable"
                                value={entry.key}
                                onChange={(e) => handlePromptVariableChange(entry.id, 'key', e.target.value)}
                                onBlur={() => updatePromptConfig({ variables: promptVariables })}
                              />
                              <input
                                type="text"
                                className="flex-[2] px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                                placeholder="default value"
                                value={entry.value}
                                onChange={(e) => handlePromptVariableChange(entry.id, 'value', e.target.value)}
                                onBlur={() => updatePromptConfig({ variables: promptVariables })}
                              />
                              <button
                                type="button"
                                onClick={() => {
                                  const nextVariables = promptVariables.filter((variable) => variable.id !== entry.id);
                                  handleRemovePromptVariable(entry.id);
                                  updatePromptConfig({ variables: nextVariables });
                                }}
                                className="px-2 py-1 text-[10px] rounded bg-gray-800 text-gray-300 hover:bg-red-500 hover:text-white"
                                title="Remove variable"
                              >
                                Remove
                              </button>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="border-t border-gray-700 pt-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-semibold text-gray-300">Evaluate Prompt</h4>
                        <span className="text-[10px] text-gray-500">Dataset-driven scoring</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <select
                          className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-xs text-gray-200"
                          defaultValue=""
                        >
                          <option value="" disabled>
                            Select dataset (coming soon)
                          </option>
                        </select>
                        <button
                          type="button"
                          className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded"
                          title="Run evaluation"
                        >
                          Run Eval
                        </button>
                      </div>
                      <div className="text-[10px] text-gray-500">
                        Eval results will surface in Logs → Activity, plus upcoming score dashboards.
                      </div>
                    </div>

                    <div className="flex items-center justify-end">
                      <button
                        type="button"
                        onClick={() => updatePromptConfig({
                          systemPrompt: systemPromptDraft,
                          userPrompt: userPromptDraft,
                          variables: promptVariables,
                        })}
                        className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 rounded"
                      >
                        Apply changes
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'inputs' && (
              <div className="space-y-3">
                <div className="bg-gray-900/70 border border-gray-700 rounded p-3">
                  <div className="text-xs font-semibold text-gray-200">Inputs (runtime)</div>
                  <div className="text-[11px] text-gray-400 mt-1">
                    Values resolved from execution context at run time.
                  </div>
                  {inputsError ? (
                    <div className="text-xs text-red-400 mt-2">{inputsError}</div>
                  ) : null}
                  <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                    {inputsDraft}
                  </pre>
                </div>

                <div className="bg-gray-900/70 border border-gray-700 rounded p-3">
                  <div className="text-xs font-semibold text-gray-200">Inputs (static mapping)</div>
                  <div className="text-[11px] text-gray-400 mt-1">
                    This is the node's configured inputs template.
                  </div>
                  <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                    {JSON.stringify(selectedNode.data?.inputs ?? {}, null, 2)}
                  </pre>
                </div>
              </div>
            )}

            {activeTab === 'outputs' && (
              <div className="space-y-3">
                <div className="bg-gray-900/70 border border-gray-700 rounded p-3">
                  <div className="text-xs font-semibold text-gray-200">Output Mapping (static)</div>
                  <div className="text-[11px] text-gray-400 mt-1">
                    This is how node outputs are saved into execution context variables.
                  </div>
                  <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                    {JSON.stringify(selectedNode.data?.outputs ?? {}, null, 2)}
                  </pre>
                </div>

                <div className="bg-gray-900/70 border border-gray-700 rounded p-3">
                  <div className="text-xs font-semibold text-gray-200">Execution Outputs (runtime)</div>
                  {!nodeExecution ? (
                    <div className="text-xs text-gray-500 mt-2">
                      No runtime outputs yet. Run the workflow or select a checkpoint.
                    </div>
                  ) : (
                    <div className="mt-2 space-y-2">
                      {selectedNode.type === 'verify' ? (
                        <div className="text-xs text-gray-300">
                          <div>
                            <span className="text-gray-500">verified:</span>{' '}
                            <span className={(nodeExecution.outputs as any)?.verified ? 'text-green-300' : 'text-red-300'}>
                              {String((nodeExecution.outputs as any)?.verified)}
                            </span>
                          </div>
                          {Array.isArray((nodeExecution.outputs as any)?.results) ? (
                            <div className="text-gray-400">
                              results: {(nodeExecution.outputs as any).results.length}
                            </div>
                          ) : null}
                          {Array.isArray((nodeExecution.outputs as any)?.errors) ? (
                            <div className="text-red-300">
                              errors: {(nodeExecution.outputs as any).errors.length}
                            </div>
                          ) : null}
                        </div>
                      ) : null}

                      {selectedNode.type === 'route' ? (
                        <div className="text-xs text-gray-300">
                          <div>
                            <span className="text-gray-500">condition:</span>{' '}
                            <span className={(nodeExecution.outputs as any)?.condition ? 'text-green-300' : 'text-gray-300'}>
                              {String((nodeExecution.outputs as any)?.condition)}
                            </span>
                          </div>
                          <div className="text-gray-400">
                            disabled_nodes: {Array.isArray((nodeExecution.outputs as any)?.disabled_nodes)
                              ? (nodeExecution.outputs as any).disabled_nodes.length
                              : 0}
                          </div>
                        </div>
                      ) : null}

                      <details className="text-xs text-gray-400">
                        <summary className="cursor-pointer">Raw JSON</summary>
                        <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                          {JSON.stringify(nodeExecution.outputs ?? {}, null, 2)}
                        </pre>
                      </details>
                    </div>
                  )}
                </div>

                {(() => {
                  const outputs = nodeExecution?.outputs;
                  const outputsAny: any = outputs as any;

                  if (!outputs) {
                    return (
                      <div className="text-xs text-gray-500 bg-gray-900/60 border border-gray-700 rounded p-3">
                        No runtime outputs yet. Run the workflow or select a checkpoint.
                      </div>
                    );
                  }

                  // The structured renderer below is meant for llm/tool rich outputs.
                  // verify/route/logic are already covered by the "Execution Outputs (runtime)" summary above.
                  if (selectedNode.type !== 'llm' && selectedNode.type !== 'tool') {
                    return null;
                  }

                  if (outputsAny?.type === 'llm_response') {
                    const toolCalls = outputsAny.tool_calls ?? [];
                    const toolErrors = outputsAny.tool_errors ?? [];
                    const cacheHitCount = toolCalls.filter((call: any) => Boolean(
                      call?.result?.metadata?.cache_hit
                      || call?.result?.raw?.metadata?.cache_hit
                      || call?.result?.raw?.metadata?.cached
                    )).length;
                    return (
                      <>
                        <div className="bg-gray-900/70 border border-gray-700 rounded p-3 space-y-2">
                          <div className="text-xs font-semibold text-gray-200">Summary</div>
                          <div className="text-xs text-gray-400">
                            <div>
                              <span className="text-gray-500">Rounds:</span> {outputsAny.tool_call_rounds ?? 0}
                              <span className="text-gray-500 ml-2">Max reached:</span> {outputsAny.max_rounds_reached ? 'yes' : 'no'}
                            </div>
                            <div>
                              <span className="text-gray-500">Tool errors:</span> {toolErrors.length}
                            </div>
                            <div>
                              <span className="text-gray-500">Cache hits:</span>{' '}
                              {cacheHitCount}
                            </div>
                          </div>
                          <div className="text-xs text-gray-200 whitespace-pre-wrap">
                            {outputsAny.content || 'No final content'}
                          </div>
                          {outputsAny.error ? (
                            <div className="text-xs text-red-400">Error: {outputsAny.error}</div>
                          ) : null}
                        </div>

                        <div className="space-y-2">
                          <div className="text-xs font-semibold text-gray-300">Tool Calls</div>
                          {toolCalls.length === 0 ? (
                            <div className="text-xs text-gray-500 bg-gray-900/60 border border-gray-700 rounded p-3">
                              No tool calls recorded.
                            </div>
                          ) : (
                            toolCalls.map((call: any, idx: number) => {
                              const toolName = call.tool_name || call.requested_tool_name || 'unknown_tool';
                              const isError = call?.result?.is_error || Boolean(call?.result?.raw?.error);
                              const cacheHit = Boolean(call?.result?.metadata?.cache_hit);
                              return (
                                <details
                                  key={`${toolName}-${call.tool_call_id ?? idx}`}
                                  className="bg-gray-900/70 border border-gray-700 rounded p-3"
                                >
                                  <summary className="cursor-pointer text-xs text-gray-200 flex items-center gap-2">
                                    <span className={isError ? 'text-red-400' : 'text-green-400'}>
                                      {isError ? '●' : '●'}
                                    </span>
                                    {toolName}
                                    {call.tool_call_id ? (
                                      <span className="text-gray-500">#{call.tool_call_id}</span>
                                    ) : null}
                                    {cacheHit ? (
                                      <span className="text-xs text-amber-300">⚡ cache hit</span>
                                    ) : (
                                      <span className="text-xs text-gray-500">cache miss</span>
                                    )}
                                  </summary>
                                  <div className="mt-2 space-y-2 text-xs text-gray-300">
                                    <div>
                                      <div className="text-gray-500">Args</div>
                                      <pre className="text-[11px] font-mono whitespace-pre-wrap">
                                        {JSON.stringify(call.args || {}, null, 2)}
                                      </pre>
                                    </div>
                                    <div>
                                      <div className="text-gray-500">Result (raw)</div>
                                      <pre className="text-[11px] font-mono whitespace-pre-wrap">
                                        {JSON.stringify(call?.result?.raw ?? {}, null, 2)}
                                      </pre>
                                    </div>
                                    {call?.result?.content ? (
                                      <div>
                                        <div className="text-gray-500">Result (content)</div>
                                        <pre className="text-[11px] font-mono whitespace-pre-wrap">
                                          {call.result.content}
                                        </pre>
                                      </div>
                                    ) : null}
                                  </div>
                                </details>
                              );
                            })
                          )}
                        </div>

                        <details className="text-xs text-gray-400">
                          <summary className="cursor-pointer">Raw JSON</summary>
                          <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                            {JSON.stringify(outputsAny, null, 2)}
                          </pre>
                        </details>
                      </>
                    );
                  }

                  if (outputsAny?.type === 'tool_result') {
                    const rawOutput = outputsAny.output ?? {};
                    const toolResult = rawOutput?.result ?? rawOutput;
                    const results = Array.isArray(toolResult?.results) ? toolResult.results : [];
                    const metaErrors = Array.isArray(toolResult?.metadata?.errors) ? toolResult.metadata.errors : [];
                    const topResults = results.slice(0, 3);
                    const topContent = results.find((item: any) => item?.content)?.content;
                    return (
                      <>
                        <div className="bg-gray-900/70 border border-gray-700 rounded p-3 space-y-2">
                          <div className="text-xs font-semibold text-gray-200">Summary</div>
                          <div className="text-xs text-gray-400">
                            <span className="text-gray-500">Tool:</span> {outputsAny?.metadata?.tool_name ?? 'unknown'}
                          </div>
                          <div className="text-xs text-gray-400">
                            <span className="text-gray-500">Provider:</span> {toolResult?.provider ?? 'unknown'}
                            {toolResult?.metadata?.request_id ? (
                              <span className="text-gray-500 ml-2">request_id: {toolResult.metadata.request_id}</span>
                            ) : null}
                          </div>
                          <div className="text-xs text-gray-400">
                            <span className="text-gray-500">Extraction:</span>{' '}
                            {toolResult?.metadata?.extraction_provider ?? 'null'}
                          </div>
                          <div className={outputsAny.is_error ? 'text-xs text-red-400' : 'text-xs text-green-300'}>
                            {outputsAny.is_error ? 'Execution failed' : 'Execution succeeded'}
                          </div>
                          <div className="text-xs text-gray-400">
                            <span className="text-gray-500">Results:</span> {results.length}
                          </div>
                          {metaErrors.length > 0 ? (
                            <div className="space-y-1">
                              <div className="text-xs text-gray-500">Provider trace</div>
                              <div className="space-y-1">
                                {metaErrors.slice(0, 6).map((err: any, idx: number) => (
                                  <div key={`meta-err-${idx}`} className="text-[11px] text-gray-300">
                                    <span className="text-gray-500">[{err?.provider ?? 'unknown'}]</span>{' '}
                                    <span className="text-gray-400">{err?.type ?? 'Error'}:</span>{' '}
                                    <span className="text-gray-300">{err?.message ?? ''}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : null}
                          {results.length > 0 ? (
                            <div className="space-y-1 text-xs text-gray-300">
                              <ul className="list-disc pl-4 space-y-1">
                                {topResults.map((item: any, idx: number) => (
                                  <li key={`${item?.url ?? 'result'}-${idx}`}>
                                    <span className="text-gray-200">{item?.title ?? item?.url ?? 'Result'}</span>
                                    {item?.url ? (
                                      <span className="text-gray-500"> — {item.url}</span>
                                    ) : null}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          ) : null}
                          {toolResult?.answer ? (
                            <div className="text-xs text-gray-300 whitespace-pre-wrap">
                              {toolResult.answer}
                            </div>
                          ) : null}
                          {topContent ? (
                            <div className="text-xs text-gray-300 whitespace-pre-wrap">
                              <div className="text-gray-500">Content (preview)</div>
                              <div>{topContent}</div>
                            </div>
                          ) : null}
                          {results.length === 0 && !toolResult?.answer ? (
                            <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap">
                              {JSON.stringify(toolResult ?? {}, null, 2)}
                            </pre>
                          ) : null}
                        </div>
                        <details className="text-xs text-gray-400">
                          <summary className="cursor-pointer">Raw JSON</summary>
                          <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                            {JSON.stringify(outputsAny, null, 2)}
                          </pre>
                        </details>
                      </>
                    );
                  }

                  return (
                    <pre className="bg-gray-900 p-2 rounded text-xs overflow-x-auto">
                      {JSON.stringify(outputsPayload, null, 2)}
                    </pre>
                  );
                })()}
              </div>
            )}

            {activeTab === 'verification' && (
              <div className="space-y-3 max-w-full overflow-x-auto">
                {selectedNode.type !== 'verify' ? (
                  <div className="text-sm text-gray-400">
                    Verification rules are configured on Verify nodes. Select a Verify node to view guardrails.
                  </div>
                ) : (
                  <>
                    {(() => {
                      const config = selectedNode.data?.config || {};
                      const sections: Array<{ label: string; value: any }> = [
                        { label: 'Verification Rules', value: (config as any).verification_rules },
                        { label: 'Guardrails', value: config.guardrails },
                        { label: 'Rules', value: config.rules },
                        { label: 'Constraints', value: config.constraints },
                        { label: 'Policies', value: config.policies },
                        { label: 'Checks', value: config.checks },
                      ].filter((item) => item.value && (
                        Array.isArray(item.value) ? item.value.length > 0 : Object.keys(item.value || {}).length > 0
                      ));

                      if (sections.length === 0) {
                        return (
                          <div className="text-xs text-gray-500 bg-gray-900/60 border border-gray-700 rounded p-3">
                            No guardrails configured yet. Add rules in the Verify node configuration.
                          </div>
                        );
                      }

                      return sections.map((section) => (
                        <div key={section.label} className="bg-gray-900/70 border border-gray-700 rounded p-3">
                          <div className="text-xs font-semibold text-gray-300 mb-2">{section.label}</div>
                          {Array.isArray(section.value) ? (
                            <ul className="list-disc pl-5 text-xs text-gray-300 space-y-1 max-w-full">
                              {section.value.map((entry: any, idx: number) => (
                                <li key={idx} className="break-words overflow-hidden">
                                  <span className="font-mono text-[11px] whitespace-pre-wrap break-words">
                                    {typeof entry === 'string' ? entry : JSON.stringify(entry)}
                                  </span>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-x-auto max-w-full">
                              {typeof section.value === 'string'
                                ? section.value
                                : JSON.stringify(section.value, null, 2)}
                            </pre>
                          )}
                        </div>
                      ));
                    })()}

                    <details className="text-xs text-gray-400">
                      <summary className="cursor-pointer">Raw verification config</summary>
                      <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2">
                        {JSON.stringify(selectedNode.data?.config || {}, null, 2)}
                      </pre>
                    </details>
                  </>
                )}
              </div>
            )}

            {activeTab === 'metadata' && (
              <div className="text-sm text-gray-300">
                <div className="space-y-2">
                  <div>
                    <span className="text-gray-400">Node ID:</span> {selectedNode.id}
                  </div>
                  <div>
                    <span className="text-gray-400">Type:</span> {selectedNode.type}
                  </div>
                  <div>
                    <span className="text-gray-400">Label:</span> {selectedNode.data?.label ?? ''}
                  </div>
                </div>

                <details className="text-xs text-gray-400 mt-3">
                  <summary className="cursor-pointer">Raw node JSON</summary>
                  <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap mt-2 overflow-x-auto max-w-full">
                    {JSON.stringify(
                      {
                        id: selectedNode.id,
                        type: selectedNode.type,
                        data: selectedNode.data,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              </div>
            )}
          </>
        )}
      </div>
        </>
      )}
    </div>
  );
};
