import { test, expect } from '@playwright/test';

const waitForNodeCount = async (page: any, count: number): Promise<void> => {
  await expect(page.locator('.react-flow__node')).toHaveCount(count);
};

const installStartExecutionMock = async (page: any): Promise<void> => {
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (!store) return;

    const buildNodeOutputs = (node: any, index: number) => {
      const nodeType = String(node?.type ?? '').toLowerCase();
      const nodeId = String(node?.id ?? `node_${index}`);

      if (nodeType === 'llm') {
        const toolCallId = `call_${nodeId}_${index}`;
        return {
          type: 'llm_response',
          content: 'Smoke test: execution strategy summary (mocked).',
          tool_calls: [
            {
              tool_name: 'web_search',
              tool_call_id: toolCallId,
              args: {
                query: 'DeepSeek R1 vs DeepSeek V3 differences',
                max_results: 3,
                provider: 'ddg',
              },
              result: {
                raw: {
                  results: [
                    {
                      title: 'DeepSeek: model overview',
                      url: 'https://example.com/deepseek',
                      snippet: 'Mock search result for smoke test.',
                    },
                  ],
                },
                content: 'Mock search result for smoke test.',
                is_error: false,
                metadata: {},
              },
              tool_override: null,
            },
          ],
          finish_reason: 'stop',
          tool_finish_reason: 'stop',
          tool_call_rounds: 1,
          max_rounds_reached: false,
          tool_errors: [],
          messages: [],
        };
      }

      if (nodeType === 'tool') {
        const toolName = String(node?.data?.config?.tool_name ?? 'web_search');
        return {
          type: 'tool_result',
          output: {
            result: {
              tool_name: toolName,
              results: [
                {
                  title: `Mock ${toolName} result`,
                  url: 'https://example.com/result',
                  content: 'Mock content for smoke test.',
                },
              ],
            },
          },
          is_error: false,
          metadata: {},
        };
      }

      if (nodeType === 'verify') {
        return {
          type: 'verify_result',
          passed: true,
          rules_checked: 2,
          failures: [],
          verified: true,
          summary: 'Verification passed (mocked).',
        };
      }

      if (nodeType === 'logic') {
        return {
          type: 'logic_result',
          result: 'final_report',
          preview: 'Smoke test summary payload (mocked).',
        };
      }

      return { type: 'smoke_ok', content: 'ok' };
    };

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

        handleEvent({
          event_type: 'execution_status',
          event_id: `evt_${Date.now()}_start`,
          timestamp,
          session_id: sessionId,
          execution_id: executionId,
          status: 'running',
        });

        const nodes = state.nodes ?? [];
        nodes.forEach((node: any, index: number) => {
          const outputs = buildNodeOutputs(node, index);
          handleEvent({
            event_type: 'node_status',
            event_id: `evt_${Date.now()}_${index}_run`,
            timestamp,
            session_id: sessionId,
            execution_id: executionId,
            node_id: node.id,
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
            node_id: node.id,
            status: 'completed',
            outputs,
          });
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
};

test('console UI loads core panels', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
  await expect(page.getByLabel('Timeline')).toBeVisible();
  await page.locator('button.px-4:has-text("Checkpoints")').first().click();
  await expect(page.getByText('No checkpoints created yet')).toBeVisible();
});

test('smoke: load position_test workflow and run end-to-end', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();

  // Open workflow list (left sidebar) to fetch workflows from backend.
  await page.getByTitle('Show workflow list').click();
  await expect(page.getByText('📄 position_test')).toBeVisible();

  // Load the workflow.
  await page.getByText('📄 position_test').click();
  await waitForNodeCount(page, 5);

  // Deterministic E2E: mock start_execution so smoke doesn't depend on external services.
  await installStartExecutionMock(page);

  await page.getByRole('button', { name: 'Start Execution' }).click();

  await expect
    .poll(async () => {
      return page.evaluate(() => {
        const store = (window as any).__consoleStore;
        const status = store?.getState?.().currentExecution?.status;
        return status || null;
      });
    })
    .toBe('completed');
});
