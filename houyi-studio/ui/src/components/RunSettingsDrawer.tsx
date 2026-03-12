import React from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';

export const RunSettingsDrawer: React.FC = () => {
  const {
    runSettings,
    isRunSettingsOpen,
    setRunSettingsOpen,
    updateRunSettings,
    resetRunSettings,
    saveRunSettingsDefaults,
  } = useConsoleStore();

  const [enableToolCalls, setEnableToolCalls] = React.useState(runSettings.enable_tool_calls);
  const [toolCallStrategyDraft, setToolCallStrategyDraft] = React.useState(runSettings.tool_call_strategy);
  const [toolNamesDraft, setToolNamesDraft] = React.useState(runSettings.tool_names.join(', '));
  const [toolChoiceDraft, setToolChoiceDraft] = React.useState(runSettings.tool_choice ?? '');
  const [maxToolCallsDraft, setMaxToolCallsDraft] = React.useState(String(runSettings.max_tool_calls));
  const [temperatureDraft, setTemperatureDraft] = React.useState(
    runSettings.temperature === null ? '' : String(runSettings.temperature),
  );
  const [parallelToolCallsDraft, setParallelToolCallsDraft] = React.useState(
    runSettings.parallel_tool_calls === null ? 'auto' : String(runSettings.parallel_tool_calls),
  );
  const [webSearchProviderDraft, setWebSearchProviderDraft] = React.useState(
    runSettings.web_search_provider ?? 'auto',
  );
  const [defaultRetriesDraft, setDefaultRetriesDraft] = React.useState(
    String(runSettings.retry_policy.default_retries ?? 0),
  );
  const [showAdvancedRetry, setShowAdvancedRetry] = React.useState(false);
  const [timeoutRetriesDraft, setTimeoutRetriesDraft] = React.useState(
    runSettings.retry_policy.timeout_retries === null
      ? ''
      : String(runSettings.retry_policy.timeout_retries),
  );
  const [rateLimitRetriesDraft, setRateLimitRetriesDraft] = React.useState(
    runSettings.retry_policy.rate_limit_retries === null
      ? ''
      : String(runSettings.retry_policy.rate_limit_retries),
  );
  const [authRetriesDraft, setAuthRetriesDraft] = React.useState(
    runSettings.retry_policy.auth_retries === null
      ? ''
      : String(runSettings.retry_policy.auth_retries),
  );
  const [badRequestRetriesDraft, setBadRequestRetriesDraft] = React.useState(
    runSettings.retry_policy.bad_request_retries === null
      ? ''
      : String(runSettings.retry_policy.bad_request_retries),
  );
  const [contentPolicyRetriesDraft, setContentPolicyRetriesDraft] = React.useState(
    runSettings.retry_policy.content_policy_retries === null
      ? ''
      : String(runSettings.retry_policy.content_policy_retries),
  );
  const [internalErrorRetriesDraft, setInternalErrorRetriesDraft] = React.useState(
    runSettings.retry_policy.internal_error_retries === null
      ? ''
      : String(runSettings.retry_policy.internal_error_retries),
  );

  React.useEffect(() => {
    if (!isRunSettingsOpen) return;
    setEnableToolCalls(runSettings.enable_tool_calls);
    setToolCallStrategyDraft(runSettings.tool_call_strategy);
    setToolNamesDraft(runSettings.tool_names.join(', '));
    setToolChoiceDraft(runSettings.tool_choice ?? '');
    setMaxToolCallsDraft(String(runSettings.max_tool_calls));
    setTemperatureDraft(runSettings.temperature === null ? '' : String(runSettings.temperature));
    setParallelToolCallsDraft(
      runSettings.parallel_tool_calls === null ? 'auto' : String(runSettings.parallel_tool_calls),
    );
    setWebSearchProviderDraft(runSettings.web_search_provider ?? 'auto');
    setDefaultRetriesDraft(String(runSettings.retry_policy.default_retries ?? 0));
    setTimeoutRetriesDraft(
      runSettings.retry_policy.timeout_retries === null
        ? ''
        : String(runSettings.retry_policy.timeout_retries),
    );
    setRateLimitRetriesDraft(
      runSettings.retry_policy.rate_limit_retries === null
        ? ''
        : String(runSettings.retry_policy.rate_limit_retries),
    );
    setAuthRetriesDraft(
      runSettings.retry_policy.auth_retries === null
        ? ''
        : String(runSettings.retry_policy.auth_retries),
    );
    setBadRequestRetriesDraft(
      runSettings.retry_policy.bad_request_retries === null
        ? ''
        : String(runSettings.retry_policy.bad_request_retries),
    );
    setContentPolicyRetriesDraft(
      runSettings.retry_policy.content_policy_retries === null
        ? ''
        : String(runSettings.retry_policy.content_policy_retries),
    );
    setInternalErrorRetriesDraft(
      runSettings.retry_policy.internal_error_retries === null
        ? ''
        : String(runSettings.retry_policy.internal_error_retries),
    );
  }, [isRunSettingsOpen, runSettings]);

  const parseRetryValue = (value: string): number | null => {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const handleApply = () => {
    const parsedToolNames = toolNamesDraft
      .split(',')
      .map((entry) => entry.trim())
      .filter(Boolean);
    const trimmedChoice = toolChoiceDraft.trim();
    const parsedMaxToolCalls = Number(maxToolCallsDraft.trim());
    const trimmedTemperature = temperatureDraft.trim();
    const parsedTemperature = trimmedTemperature ? Number(trimmedTemperature) : null;
    const parsedParallelToolCalls =
      parallelToolCallsDraft === 'auto'
        ? null
        : parallelToolCallsDraft === 'true';
    const retryPolicy = {
      default_retries: Math.max(0, Number.parseInt(defaultRetriesDraft.trim() || '0', 10)),
      timeout_retries: parseRetryValue(timeoutRetriesDraft),
      rate_limit_retries: parseRetryValue(rateLimitRetriesDraft),
      auth_retries: parseRetryValue(authRetriesDraft),
      bad_request_retries: parseRetryValue(badRequestRetriesDraft),
      content_policy_retries: parseRetryValue(contentPolicyRetriesDraft),
      internal_error_retries: parseRetryValue(internalErrorRetriesDraft),
    };

    updateRunSettings({
      enable_tool_calls: enableToolCalls,
      tool_call_strategy: toolCallStrategyDraft,
      tool_names: parsedToolNames,
      tool_choice: trimmedChoice ? trimmedChoice : null,
      max_tool_calls: Number.isNaN(parsedMaxToolCalls) ? runSettings.max_tool_calls : parsedMaxToolCalls,
      temperature: Number.isNaN(parsedTemperature as number) ? runSettings.temperature : parsedTemperature,
      parallel_tool_calls: parsedParallelToolCalls,
      web_search_provider: webSearchProviderDraft === 'auto' ? null : webSearchProviderDraft,
      retry_policy: retryPolicy,
    });
    setRunSettingsOpen(false);
  };

  const handleReset = () => {
    resetRunSettings();
  };

  return (
    <div
      className={`fixed right-0 top-14 z-40 h-[calc(100%-3.5rem)] w-[360px] border-l border-gray-700 bg-gray-800 shadow-xl transition-transform duration-300 ${
        isRunSettingsOpen ? 'translate-x-0' : 'translate-x-full pointer-events-none'
      }`}
    >
      <div className="flex h-full flex-col">
        <div className="flex items-center justify-between border-b border-gray-700 px-4 py-3">
          <div>
            <h3 className="text-sm font-semibold text-gray-100">Run Settings</h3>
            <p className="text-[11px] text-gray-400">Applies to the next execution run.</p>
          </div>
          <button
            type="button"
            onClick={() => setRunSettingsOpen(false)}
            className="rounded px-2 py-1 text-xs text-gray-300 hover:bg-gray-700"
          >
            Close
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
          <div className="rounded border border-gray-700 bg-gray-900/50 p-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-xs font-semibold text-gray-200">Tool Calls</h4>
                <p className="text-[11px] text-gray-400">Tool routing for this run.</p>
              </div>
              <label className="flex items-center gap-2 text-xs text-gray-300">
                <input
                  type="checkbox"
                  className="rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                  checked={enableToolCalls}
                  onChange={(event) => setEnableToolCalls(event.target.checked)}
                />
                Enable
              </label>
            </div>

            <div className="mt-3 space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Tool strategy</label>
                <select
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={toolCallStrategyDraft}
                  onChange={(event) => setToolCallStrategyDraft(event.target.value as 'conservative' | 'balanced' | 'aggressive')}
                >
                  <option value="conservative">Conservative (latency first)</option>
                  <option value="balanced">Balanced (default)</option>
                  <option value="aggressive">Aggressive (recall first)</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Tool names</label>
                <input
                  type="text"
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={toolNamesDraft}
                  onChange={(event) => setToolNamesDraft(event.target.value)}
                  placeholder="get_location, get_weather"
                />
                <p className="mt-1 text-[10px] text-gray-500">Leave empty to allow all client tools.</p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Tool choice</label>
                <input
                  type="text"
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={toolChoiceDraft}
                  onChange={(event) => setToolChoiceDraft(event.target.value)}
                  placeholder="auto / required / none / type=function"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Web search provider</label>
                <select
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={webSearchProviderDraft}
                  onChange={(event) => setWebSearchProviderDraft(event.target.value)}
                >
                  <option value="auto">Auto (default)</option>
                  <option value="ddg">DDG</option>
                  <option value="serper">Serper</option>
                  <option value="tavily">Tavily</option>
                </select>
                <p className="mt-1 text-[10px] text-gray-500">
                  Requires corresponding API keys or base URL when applicable.
                </p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Max tool calls</label>
                <input
                  type="number"
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={maxToolCallsDraft}
                  onChange={(event) => setMaxToolCallsDraft(event.target.value)}
                  placeholder="6"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Temperature</label>
                <input
                  type="number"
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={temperatureDraft}
                  onChange={(event) => setTemperatureDraft(event.target.value)}
                  placeholder="0.7"
                  step="0.1"
                  min="0"
                  max="2"
                />
                <p className="mt-1 text-[10px] text-gray-500">Leave empty to use model default.</p>
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Parallel tool calls</label>
                <select
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={parallelToolCallsDraft}
                  onChange={(event) => setParallelToolCallsDraft(event.target.value)}
                >
                  <option value="auto">Auto</option>
                  <option value="true">Enable</option>
                  <option value="false">Disable</option>
                </select>
              </div>
            </div>
          </div>

          <div className="rounded border border-gray-700 bg-gray-900/50 p-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-xs font-semibold text-gray-200">Retry Policy</h4>
                <p className="text-[11px] text-gray-400">Retry failed node executions.</p>
              </div>
            </div>

            <div className="mt-3 space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-gray-400">Max retries</label>
                <input
                  type="number"
                  className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-sm text-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  value={defaultRetriesDraft}
                  onChange={(event) => setDefaultRetriesDraft(event.target.value)}
                  min="0"
                />
                <p className="mt-1 text-[10px] text-gray-500">
                  Applied to all error types unless overridden below.
                </p>
              </div>

              <button
                type="button"
                onClick={() => setShowAdvancedRetry(!showAdvancedRetry)}
                className="flex items-center gap-1 text-[11px] text-gray-400 hover:text-gray-200"
              >
                <span className={`transition-transform ${showAdvancedRetry ? 'rotate-90' : ''}`}>
                  &#9654;
                </span>
                Per-error-type overrides
              </button>

              {showAdvancedRetry && (
                <div className="grid grid-cols-2 gap-3 rounded border border-gray-700/50 bg-gray-800/30 p-2">
                  {[
                    { label: 'Timeout', value: timeoutRetriesDraft, setter: setTimeoutRetriesDraft },
                    { label: 'Rate limit', value: rateLimitRetriesDraft, setter: setRateLimitRetriesDraft },
                    { label: 'Auth', value: authRetriesDraft, setter: setAuthRetriesDraft },
                    { label: 'Bad request', value: badRequestRetriesDraft, setter: setBadRequestRetriesDraft },
                    { label: 'Content policy', value: contentPolicyRetriesDraft, setter: setContentPolicyRetriesDraft },
                    { label: 'Internal error', value: internalErrorRetriesDraft, setter: setInternalErrorRetriesDraft },
                  ].map(({ label, value, setter }) => (
                    <div key={label}>
                      <label className="mb-1 block text-[11px] font-medium text-gray-500">{label}</label>
                      <input
                        type="number"
                        className="w-full rounded border border-gray-600 bg-gray-700 px-2 py-1 text-xs text-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                        value={value}
                        onChange={(event) => setter(event.target.value)}
                        min="0"
                        placeholder="inherit"
                      />
                    </div>
                  ))}
                  <p className="col-span-2 text-[10px] text-gray-500">
                    Empty fields inherit the max retries value above.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-gray-700 px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleReset}
              className="rounded bg-gray-700 px-3 py-1.5 text-xs text-gray-200 hover:bg-gray-600"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={saveRunSettingsDefaults}
              className="rounded bg-gray-700 px-3 py-1.5 text-xs text-gray-200 hover:bg-gray-600"
            >
              Save as default
            </button>
          </div>
          <button
            type="button"
            onClick={handleApply}
            className="rounded bg-blue-600 px-3 py-1.5 text-xs text-white hover:bg-blue-500"
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
};
