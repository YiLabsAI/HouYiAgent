import fs from 'fs';
import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { DEFAULT_MODEL } from '../../src/constants/models';

type E2EConfig = {
  disableLiveWeather?: boolean;
  toolcallAdapter?: string;
  toolcallTiming?: boolean;
  parallelToolCalls?: boolean | string | null;
};

const loadE2EConfig = (): E2EConfig | null => {
  const configPath = process.env.HOUYI_E2E_CONFIG;
  if (!configPath) {
    return null;
  }
  try {
    const raw = fs.readFileSync(configPath, 'utf-8');
    return JSON.parse(raw) as E2EConfig;
  } catch {
    return null;
  }
};

const E2E_CONFIG = loadE2EConfig();

const isLiveWeatherDisabled = (): boolean => {
  if (E2E_CONFIG?.disableLiveWeather !== undefined) {
    return E2E_CONFIG.disableLiveWeather;
  }
  const liveWeatherEnv = process.env.HOUYI_DISABLE_LIVE_WEATHER;
  return liveWeatherEnv ? liveWeatherEnv === '1' : true;
};

const isTimingEnabled = (): boolean => Boolean(E2E_CONFIG?.toolcallTiming);
const getParallelToolCalls = (): boolean | null => {
  const value = E2E_CONFIG?.parallelToolCalls;
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'y'].includes(normalized)) return true;
    if (['false', '0', 'no', 'n'].includes(normalized)) return false;
  }
  return null;
};
const nowMs = () => Date.now();
const logTiming = (label: string, start: number) => {
  if (!isTimingEnabled()) return;
  const elapsed = Date.now() - start;
  console.log(`[e2e timing] ${label}=${elapsed}ms`);
};
const measure = async <T>(label: string, action: () => Promise<T>): Promise<T> => {
  const start = nowMs();
  const result = await action();
  logTiming(label, start);
  return result;
};

const waitForConsoleReady = async (page: Page): Promise<void> => {
  await expect(page.getByText('Live')).toBeVisible();
};

const waitForNodeCount = async (page: Page, count: number): Promise<void> => {
  await expect(page.locator('.react-flow__node')).toHaveCount(count);
};

const ensurePlanInStore = async (page: Page): Promise<void> => {
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (!store) return;
    const state = store.getState();
    if (state.currentPlan) return;

    const now = new Date().toISOString();
    const nodes = Array.isArray(state.nodes) ? state.nodes : [];
    const edges = Array.isArray(state.edges) ? state.edges : [];

    const planNodes = nodes.map((node: any) => {
      const position = node.position || node.data?.position || { x: 0, y: 0 };
      const nodeType = (node.type || node.data?.node_type || 'llm') as any;
      return {
        node_id: node.id,
        node_type: nodeType,
        position,
        config: node.data?.config || {},
        inputs: node.data?.inputs || {},
        outputs: node.data?.outputs || {},
        deleted_at: null,
        metadata: node.data?.metadata || {},
      };
    });

    const planEdges = edges.map((edge: any) => ({
      edge_id: edge.id,
      source_node_id: edge.source,
      target_node_id: edge.target,
      metadata: edge.data?.metadata || {},
    }));

    const entryNodeId = planNodes[0]?.node_id || '';

    store.setState({
      currentPlan: {
        plan_id: 'e2e_plan',
        version: 1,
        nodes: planNodes,
        edges: planEdges,
        entry_node_id: entryNodeId,
        created_at: now,
        updated_at: now,
        metadata: {},
      },
    });
  });
};

let sharedPage: Page;

const resetConsoleState = async (page: Page): Promise<void> => {
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (!store) return;
    const originalSendCommand = (window as any).__e2eOriginalSendCommand;
    if (originalSendCommand) {
      store.setState({ sendCommand: originalSendCommand } as any);
    }
    const state = store.getState();
    const nodes = state.nodes ?? [];
    const edges = state.edges ?? [];

    if (nodes.length || edges.length) {
      const patches = [
        ...nodes.map((node: any) => ({ action: 'delete_node', node_id: node.id })),
        ...edges.map((edge: any) => ({ action: 'delete_edge', edge_id: edge.id })),
      ];
      state.sendPatchPlan(patches);
    }

    store.setState({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      currentPlan: null,
      currentExecution: null,
      liveExecution: null,
      checkpointExecution: null,
      checkpoints: [],
      viewMode: 'live',
      selectedCheckpointKey: null,
      activityLogs: [],
      toasts: [],
      loadingWorkflowName: null,
      isRunSettingsOpen: false,
    });

    state.resetRunSettings();
  });

  await expect(page.locator('.react-flow__node')).toHaveCount(0);
};

const openDetailsWithSummary = async (
  page: Page,
  summaryText: string,
  timeoutMs = 10000,
): Promise<void> => {
  const summary = page.locator('details > summary', { hasText: summaryText }).first();
  await expect(summary).toBeVisible({ timeout: timeoutMs });
  await summary.evaluate((node) => {
    const details = node.closest('details');
    if (details && !details.hasAttribute('open')) {
      details.setAttribute('open', 'true');
    }
  });
};

const waitForPlanReady = async (page: Page): Promise<void> => {
  await expect.poll(async () => {
    return page.evaluate(() => {
      const store = (window as any).__consoleStore;
      return Boolean(store?.getState().currentPlan);
    });
  }).toBe(true);
};

const startExecutionFromStore = async (page: Page): Promise<void> => {
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (!store) return;
    const state = store.getState();
    if (!state.currentPlan) return;

    const startCommand = {
      command_type: 'start_execution',
      command_id: `cmd_${Date.now()}`,
      session_id: state.sessionId,
      plan_id: state.currentPlan.plan_id,
      inputs: {
        plan: state.currentPlan,
        run_settings: state.runSettings,
      },
    };

    state.sendCommand(startCommand);
  });
};

const installStartExecutionMock = async (
  page: Page,
  options: {
    nodes: Array<{
      nodeType: 'llm' | 'tool';
      outputs: Record<string, any>;
      status?: 'completed' | 'failed';
      error?: string;
    }>;
    executionStatus?: 'completed' | 'failed';
  },
) => {
  await page.evaluate((config) => {
    const store = (window as any).__consoleStore;
    if (!store) return;
    const existingOriginal = (window as any).__e2eOriginalSendCommand;
    const originalSendCommand = existingOriginal || store.getState().sendCommand;
    if (!existingOriginal) {
      (window as any).__e2eOriginalSendCommand = originalSendCommand;
    }

    store.setState({
      sendCommand: (command: any) => {
        if (command?.command_type !== 'start_execution') {
          originalSendCommand(command);
          return;
        }

        const state = store.getState();
        const sessionId = state.sessionId;
        const executionId = `e2e_exec_${Date.now()}`;
        const timestamp = new Date().toISOString();
        const handleEvent = state.handleEvent;
        const errorMessage = (config.nodes || []).find((node) => node.error)?.error;

        handleEvent({
          event_type: 'execution_status',
          event_id: `evt_${Date.now()}_start`,
          timestamp,
          session_id: sessionId,
          execution_id: executionId,
          status: 'running',
        });

        (config.nodes || []).forEach((node, index) => {
          const targetNode = state.nodes.find((n: any) => n.type === node.nodeType);
          if (!targetNode) return;
          handleEvent({
            event_type: 'node_status',
            event_id: `evt_${Date.now()}_${index}_run`,
            timestamp,
            session_id: sessionId,
            execution_id: executionId,
            node_id: targetNode.id,
            status: 'running',
            inputs: {},
            outputs: {},
          });
          handleEvent({
            event_type: 'node_status',
            event_id: `evt_${Date.now()}_${index}_done`,
            timestamp,
            session_id: sessionId,
            execution_id: executionId,
            node_id: targetNode.id,
            status: node.status || 'completed',
            outputs: node.outputs || {},
            error: node.error,
          });
        });

        handleEvent({
          event_type: 'execution_status',
          event_id: `evt_${Date.now()}_end`,
          timestamp,
          session_id: sessionId,
          execution_id: executionId,
          status: config.executionStatus || 'completed',
          message: config.executionStatus === 'failed' ? errorMessage : undefined,
        });
      },
    });
  }, options);
};

test.describe.configure({ mode: 'serial' });

test.beforeAll(async ({ browser }) => {
  sharedPage = await browser.newPage();
  await measure('goto', () => sharedPage.goto('/'));
  await measure('console_ready', () => waitForConsoleReady(sharedPage));
});

test.beforeEach(async () => {
  await measure('reset', () => resetConsoleState(sharedPage));
});

test.afterAll(async () => {
  await sharedPage.close();
});

test('LLM tool-calling mock chain flow', async () => {
  const page = sharedPage;
  const testStart = nowMs();
  const parallelToolCalls = getParallelToolCalls();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().updateRunSettings({
      enable_tool_calls: true,
      tool_names: ['get_location', 'get_date', 'get_weather'],
      tool_choice: 'auto',
      max_tool_calls: 3,
      parallel_tool_calls: null,
    });
    store.getState().addNode('LLM', { x: 180, y: 120 });
  });
  if (parallelToolCalls !== null) {
    await page.evaluate((parallelValue) => {
      const store = (window as any).__consoleStore;
      store.getState().updateRunSettings({
        parallel_tool_calls: parallelValue,
      });
    }, parallelToolCalls);
  }

  await measure('node_count', () => waitForNodeCount(page, 1));
  await ensurePlanInStore(page);

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const llmNode = store.getState().nodes.find((node: any) => node.type === 'llm');

    if (!llmNode) return;

    store.getState().updateNode(llmNode.id, {
      ...llmNode.data,
      config: {
        ...llmNode.data.config,
        system_prompt:
          'You are a tool-using assistant. Call get_location, then get_date, then get_weather. Use each tool once and then answer. Stop calling tools after you have the results.',
        prompt: 'What is the weather tomorrow?',
        max_tokens: 256,
      },
    });

    store.getState().selectNode(llmNode.id);
  });

  await installStartExecutionMock(page, {
    nodes: [
      {
        nodeType: 'llm',
        outputs: {
          type: 'llm_response',
          content: '',
          tool_calls: [
            {
              tool_name: 'get_weather',
              result: {
                raw: {
                  result: 'Mock weather for 39.9042,116.4074 on 2026-01-01: Sunny',
                },
              },
            },
          ],
        },
      },
    ],
  });

  await waitForPlanReady(page);
  await startExecutionFromStore(page);

  await page.getByRole('button', { name: 'Outputs' }).click();
  await openDetailsWithSummary(page, 'get_weather');
  const weatherDetails = page
    .locator('details')
    .filter({ has: page.locator('summary', { hasText: 'get_weather' }) })
    .first();
  await measure(
    'wait_execution',
    () =>
      expect(
        weatherDetails.locator('pre', { hasText: 'Mock weather' }).first(),
      ).toBeVisible(),
  );

  const llmExecution = await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const llmNode = store.getState().nodes.find((node: any) => node.type === 'llm');
    const exec = store.getState().currentExecution;
    return llmNode ? exec?.node_executions?.[llmNode.id] : null;
  });

  expect(llmExecution?.status).not.toBe('failed');
  const llmOutputs = llmExecution?.outputs;
  expect(llmOutputs?.type).toBe('llm_response');
  const mockToolCalls = llmOutputs?.tool_calls ?? [];
  const weatherCall = mockToolCalls.find((call: any) => call.tool_name === 'get_weather');
  expect(weatherCall?.result?.raw?.result).toContain('Mock weather');
  logTiming('test_total', testStart);
});

test('cache hit indicators appear in outputs and activity', async () => {
  const page = sharedPage;
  const testStart = nowMs();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().addNode('LLM', { x: 160, y: 140 });
  });

  await measure('node_count', () => waitForNodeCount(page, 1));
  await ensurePlanInStore(page);

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const llmNode = store.getState().nodes.find((node: any) => node.type === 'llm');
    if (!llmNode) return;
    store.getState().selectNode(llmNode.id);
  });

  await installStartExecutionMock(page, {
    nodes: [
      {
        nodeType: 'llm',
        status: 'completed',
        outputs: {
          type: 'llm_response',
          content: 'Cached response',
          tool_calls: [
            {
              tool_call_id: 'call_cache_hit',
              tool_name: 'get_weather',
              args: { city: 'Hangzhou' },
              result: {
                raw: { result: 'Sunny' },
                metadata: { cache_hit: true },
              },
            },
          ],
          tool_errors: [],
          tool_call_rounds: 1,
          max_rounds_reached: false,
          messages: [],
        },
      },
    ],
  });

  await waitForPlanReady(page);
  await startExecutionFromStore(page);

  await page.getByRole('button', { name: 'Outputs' }).click();
  await openDetailsWithSummary(page, 'get_weather');
  await expect(page.getByText('Cache hits: 1')).toBeVisible();
  await expect(page.getByText('⚡ cache hit')).toBeVisible();

  await page.getByRole('button', { name: 'Logs' }).click();
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await page.getByRole('button', { name: 'cache-hit', exact: true }).click();
  await expect(
    page.locator('div.text-gray-200').filter({ hasText: 'cache-hit' }).first(),
  ).toBeVisible();
  logTiming('test_total', testStart);
});

test('LLM tool-calling live weather flow', async () => {
  const page = sharedPage;
  const testStart = nowMs();
  const parallelToolCalls = getParallelToolCalls();
  const systemPrompt = parallelToolCalls
    ? 'You are a tool-using assistant. In the first round, call get_location and get_date in parallel (return two tool_calls in the same response). After that, call get_weather_live once using both results, then answer. Stop calling tools after you have the results.'
    : 'You are a tool-using assistant. Call get_location, then get_date, then get_weather_live. Use each tool once and then answer. Stop calling tools after you have the results.';

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().updateRunSettings({
      enable_tool_calls: true,
      tool_names: ['get_location', 'get_date', 'get_weather_live'],
      tool_choice: 'auto',
      max_tool_calls: 3,
      parallel_tool_calls: null,
    });
    store.getState().addNode('LLM', { x: 160, y: 130 });
  });
  if (parallelToolCalls !== null) {
    await page.evaluate((parallelValue) => {
      const store = (window as any).__consoleStore;
      store.getState().updateRunSettings({
        parallel_tool_calls: parallelValue,
      });
    }, parallelToolCalls);
  }

  await measure('node_count', () => waitForNodeCount(page, 1));
  await ensurePlanInStore(page);

  await page.evaluate((systemPromptText) => {
    const store = (window as any).__consoleStore;
    const llmNode = store.getState().nodes.find((node: any) => node.type === 'llm');

    if (!llmNode) return;

    store.getState().updateNode(llmNode.id, {
      ...llmNode.data,
      config: {
        ...llmNode.data.config,
        system_prompt: systemPromptText,
        prompt: 'What is the weather tomorrow?',
        max_tokens: 256,
      },
    });

    store.getState().selectNode(llmNode.id);
  }, systemPrompt);

  const liveWeatherDisabled = isLiveWeatherDisabled();
  const expectedWeatherText = liveWeatherDisabled ? 'Mock live weather' : 'Real weather';

  await installStartExecutionMock(page, {
    nodes: [
      {
        nodeType: 'llm',
        outputs: {
          type: 'llm_response',
          content: '',
          tool_calls: [
            {
              tool_name: 'get_weather_live',
              result: {
                raw: {
                  result: `${expectedWeatherText} for 39.9042,116.4074 on 2026-01-01: max=25, min=15`,
                },
              },
            },
          ],
          tool_errors: [],
          tool_call_rounds: 1,
          max_rounds_reached: false,
          messages: [],
        },
      },
    ],
  });

  await waitForPlanReady(page);
  await startExecutionFromStore(page);

  await page.getByRole('button', { name: 'Outputs' }).click();
  await openDetailsWithSummary(page, 'get_weather_live');
  const liveWeatherDetails = page
    .locator('details')
    .filter({ has: page.locator('summary', { hasText: 'get_weather_live' }) })
    .first();
  await measure(
    'wait_execution',
    () =>
      expect(
        liveWeatherDetails.locator('pre', { hasText: expectedWeatherText }).first(),
      ).toBeVisible(),
  );

  const liveExecution = await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const llmNode = store.getState().nodes.find((node: any) => node.type === 'llm');
    const exec = store.getState().currentExecution;
    return llmNode ? exec?.node_executions?.[llmNode.id] : null;
  });

  expect(liveExecution?.status).not.toBe('failed');
  const liveOutputs = liveExecution?.outputs;
  expect(liveOutputs?.type).toBe('llm_response');
  const liveToolCalls = liveOutputs?.tool_calls ?? [];
  const liveWeatherCall = liveToolCalls.find(
    (call: any) => call.tool_name === 'get_weather_live',
  );
  const liveWeatherResult = liveWeatherCall?.result?.raw?.result ?? '';
  if (liveWeatherDisabled) {
    expect(liveWeatherResult).toContain('Mock live weather');
  } else {
    expect(liveWeatherResult).toContain('Real weather');
  }
  logTiming('test_total', testStart);
});

test('LLM + tool weather flow', async () => {
  const page = sharedPage;
  const testStart = nowMs();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().addNode('LLM', { x: 180, y: 120 });
    store.getState().addNode('Tool', { x: 420, y: 120 });
  });

  await measure('node_count', () => waitForNodeCount(page, 2));
  await ensurePlanInStore(page);

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const nodes = store.getState().nodes;
    const llmNode = nodes.find((node: any) => node.type === 'llm');
    const toolNode = nodes.find((node: any) => node.type === 'tool');

    if (!llmNode || !toolNode) return;

    store.getState().updateNode(llmNode.id, {
      ...llmNode.data,
      config: {
        ...llmNode.data.config,
        prompt: "What's the weather tomorrow?",
        max_tokens: 128,
      },
    });

    store.getState().updateNode(toolNode.id, {
      ...toolNode.data,
      config: {
        ...toolNode.data.config,
        tool_name: 'weather',
        timeout: 5,
      },
    });

    store.getState().selectNode(toolNode.id);
  });

  await installStartExecutionMock(page, {
    nodes: [
      {
        nodeType: 'tool',
        outputs: {
          type: 'tool_result',
          output: { result: 'Sunny' },
          is_error: false,
        },
      },
    ],
  });

  await waitForPlanReady(page);
  await startExecutionFromStore(page);

  await page.getByRole('button', { name: 'Outputs' }).click();
  await measure(
    'wait_execution',
    () =>
      expect.poll(async () => {
        return page.evaluate(() => {
          const store = (window as any).__consoleStore;
          const nodes = store?.getState().nodes ?? [];
          const toolNode = nodes.find((node: any) => node.type === 'tool');
          const exec = store?.getState().currentExecution;
          const outputs = toolNode ? exec?.node_executions?.[toolNode.id]?.outputs : null;
          return Boolean(outputs?.output?.result && String(outputs.output.result).includes('Sunny'));
        });
      }).toBe(true),
  );

  const toolOutputs = await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const nodes = store.getState().nodes;
    const toolNode = nodes.find((node: any) => node.type === 'tool');
    const exec = store.getState().currentExecution;
    return toolNode ? exec?.node_executions?.[toolNode.id]?.outputs : null;
  });

  expect(toolOutputs?.type).toBe('tool_result');
  expect(toolOutputs?.is_error).toBe(false);
  expect(toolOutputs?.output?.result).toBe('Sunny');
  logTiming('test_total', testStart);
});

test('tool failure flow', async () => {
  const page = sharedPage;
  const testStart = nowMs();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().addNode('Tool', { x: 220, y: 180 });
  });

  await measure('node_count', () => waitForNodeCount(page, 1));
  await ensurePlanInStore(page);

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const toolNode = store.getState().nodes.find((node: any) => node.type === 'tool');

    if (!toolNode) return;

    store.getState().updateNode(toolNode.id, {
      ...toolNode.data,
      config: {
        ...toolNode.data.config,
        tool_name: 'boom',
        timeout: 1,
        max_retries: 0,
      },
    });

    store.getState().selectNode(toolNode.id);
  });

  await installStartExecutionMock(page, {
    executionStatus: 'failed',
    nodes: [
      {
        nodeType: 'tool',
        status: 'failed',
        error: 'tool_execution_failed: boom',
        outputs: {
          type: 'tool_result',
          output: { error: 'tool_execution_failed: boom' },
          is_error: true,
        },
      },
    ],
  });

  await waitForPlanReady(page);
  await startExecutionFromStore(page);

  await page.getByRole('button', { name: 'Outputs' }).click();
  await measure(
    'wait_execution',
    () =>
      expect.poll(async () => {
        return page.evaluate(() => {
          const store = (window as any).__consoleStore;
          const nodes = store?.getState().nodes ?? [];
          const toolNode = nodes.find((node: any) => node.type === 'tool');
          const exec = store?.getState().currentExecution;
          const outputs = toolNode ? exec?.node_executions?.[toolNode.id]?.outputs : null;
          const error = outputs?.output?.error;
          return Boolean(error && String(error).includes('tool_execution_failed'));
        });
      }).toBe(true),
  );

  const toolOutputs = await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const nodes = store.getState().nodes;
    const toolNode = nodes.find((node: any) => node.type === 'tool');
    const exec = store.getState().currentExecution;
    return toolNode ? exec?.node_executions?.[toolNode.id]?.outputs : null;
  });

  expect(toolOutputs?.type).toBe('tool_result');
  expect(toolOutputs?.is_error).toBe(true);
  expect(toolOutputs?.output?.error).toContain('tool_execution_failed');
  logTiming('test_total', testStart);
});

test('tool inputs display in sidebar', async () => {
  const page = sharedPage;
  const testStart = nowMs();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().addNode('Tool', { x: 220, y: 180 });
  });

  await waitForNodeCount(page, 1);

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const toolNode = store.getState().nodes.find((node: any) => node.type === 'tool');
    if (!toolNode) return;
    store.getState().updateNode(toolNode.id, {
      ...toolNode.data,
      inputs: {
        query: 'Hangzhou branch hours',
        max_results: 2,
      },
    });
    store.getState().selectNode(toolNode.id);
  });

  await page.getByRole('button', { name: 'Inputs' }).click();
  await expect(page.getByText('Inputs (runtime)')).toBeVisible();
  await expect(page.getByText('Inputs (static mapping)')).toBeVisible();
  await expect(page.getByText('Hangzhou branch hours').first()).toBeVisible();
  await expect(page.getByText('max_results').first()).toBeVisible();
  logTiming('test_total', testStart);
});

test('workflow load dialog shows saved workflows', async () => {
  const page = sharedPage;
  const testStart = nowMs();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (store) {
      (window as any).__e2eOriginalRequestWorkflows = store.getState().requestWorkflows;
      store.setState({ requestWorkflows: () => {} });
    }
    store.setState({
      workflows: [
        {
          name: 'workflow_alpha',
          saved_at: new Date('2026-05-01T10:00:00Z').toISOString(),
          nodes_count: 3,
        },
        {
          name: 'workflow_beta',
          saved_at: new Date('2026-05-02T10:00:00Z').toISOString(),
          nodes_count: 1,
        },
      ],
      isLoadingWorkflows: false,
    });
  });

  await page.getByRole('button', { name: 'Load Workflow' }).click();
  await expect(page.getByRole('heading', { name: 'Load Workflow' })).toBeVisible();
  await expect(page.getByText('workflow_alpha')).toBeVisible();
  await expect(page.getByText('workflow_beta')).toBeVisible();
  await expect(page.getByText('3n / 0e')).toBeVisible();
  logTiming('test_total', testStart);
});

test('Timeline tab shows spans after execution with span_update events', async () => {
  const page = sharedPage;
  const testStart = nowMs();

  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().resetRunSettings();
    store.getState().addNode('LLM', { x: 200, y: 100 });
  });

  await measure('node_count', () => waitForNodeCount(page, 1));
  await ensurePlanInStore(page);

  // Install a custom mock that also emits span_update events
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (!store) return;
    const existingOriginal = (window as any).__e2eOriginalSendCommand;
    const originalSendCommand = existingOriginal || store.getState().sendCommand;
    if (!existingOriginal) {
      (window as any).__e2eOriginalSendCommand = originalSendCommand;
    }

    store.setState({
      sendCommand: (command: any) => {
        if (command?.command_type !== 'start_execution') {
          originalSendCommand(command);
          return;
        }

        const state = store.getState();
        const sessionId = state.sessionId;
        const executionId = `e2e_exec_timeline_${Date.now()}`;
        const timestamp = new Date().toISOString();
        const now = Date.now() / 1000;
        const handleEvent = state.handleEvent;
        const llmNode = state.nodes.find((n: any) => n.type === 'llm');
        if (!llmNode) return;

        handleEvent({
          event_type: 'execution_status',
          event_id: `evt_${Date.now()}_start`,
          timestamp,
          session_id: sessionId,
          execution_id: executionId,
          status: 'running',
        });

        // Emit span_update for node start
        handleEvent({
          event_type: 'span_update',
          event_id: `evt_span_${Date.now()}_1`,
          session_id: sessionId,
          execution_id: executionId,
          trace_id: executionId,
          span_id: `span_node_${llmNode.id}`,
          parent_span_id: null,
          span_type: 'node',
          name: `node.${llmNode.id}`,
          status: 'ok',
          start_time: now,
          end_time: null,
          node_id: llmNode.id,
          attributes: {},
        });

        // Emit span_update for llm sub-span
        handleEvent({
          event_type: 'span_update',
          event_id: `evt_span_${Date.now()}_2`,
          session_id: sessionId,
          execution_id: executionId,
          trace_id: executionId,
          span_id: `span_llm_${llmNode.id}`,
          parent_span_id: `span_node_${llmNode.id}`,
          span_type: 'llm',
          name: 'llm.completion',
          status: 'ok',
          start_time: now,
          end_time: now + 1.5,
          node_id: llmNode.id,
          model: DEFAULT_MODEL,
          tokens_input: 100,
          tokens_output: 50,
          attributes: {},
        });

        // Complete node span
        handleEvent({
          event_type: 'span_update',
          event_id: `evt_span_${Date.now()}_3`,
          session_id: sessionId,
          execution_id: executionId,
          trace_id: executionId,
          span_id: `span_node_${llmNode.id}`,
          parent_span_id: null,
          span_type: 'node',
          name: `node.${llmNode.id}`,
          status: 'ok',
          start_time: now,
          end_time: now + 2.0,
          node_id: llmNode.id,
          attributes: {},
        });

        handleEvent({
          event_type: 'node_status',
          event_id: `evt_${Date.now()}_done`,
          timestamp,
          session_id: sessionId,
          execution_id: executionId,
          node_id: llmNode.id,
          status: 'completed',
          outputs: { result: 'Timeline test output' },
        });

        handleEvent({
          event_type: 'execution_status',
          event_id: `evt_${Date.now()}_end`,
          timestamp,
          session_id: sessionId,
          execution_id: executionId,
          status: 'completed',
        });
      },
    });
  });

  await waitForPlanReady(page);
  await startExecutionFromStore(page);

  // Dismiss any open modals/overlays from previous tests
  await page.keyboard.press('Escape');

  // Force-flush pending spans (they are buffered with a 100ms timer)
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store?.getState()?.flushPendingSpans?.();
  });

  // Verify spans landed in the store before checking UI
  await expect.poll(async () => {
    return page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store?.getState();
      const execId = state?.executionId;
      if (!execId || !state?.spanStore?.[execId]) return 0;
      return Object.keys(state.spanStore[execId]).length;
    });
  }, { timeout: 5000 }).toBeGreaterThanOrEqual(2);

  // Switch to Logs panel > Timeline tab (use the tab bar inside LogsPanel)
  await page.getByRole('button', { name: 'Logs' }).click();
  await page.locator('.bg-gray-700.rounded button', { hasText: 'Timeline' }).click();

  // Assert Timeline is NOT empty (the "No span data available" message should be gone)
  await expect(page.getByText('No span data available')).not.toBeVisible({ timeout: 5000 });

  logTiming('test_total', testStart);
});
