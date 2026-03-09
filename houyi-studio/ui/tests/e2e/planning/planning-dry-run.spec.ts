import { expect, test, type Page } from '@playwright/test';

const waitForWsConnected = async (page: Page) => {
  await expect
    .poll(
      async () =>
        page.evaluate(() => {
          const store = (window as any).__consoleStore;
          return store?.getState?.().connectionStatus ?? 'disconnected';
        }),
      { message: 'waiting for WS to connect', timeout: 15_000 },
    )
    .toBe('connected');
};

const sendCommandAndWait = async (
  page: Page,
  command: Record<string, unknown>,
  eventType: string,
  timeoutMs = 10_000,
) => {
  return page.evaluate(
    ({ cmd, evtType, timeout }) =>
      new Promise<any>((resolve) => {
        const store = (window as any).__consoleStore;
        if (!store) return resolve(null);
        const state = store.getState();
        const unsub = state.registerSkillEventHandler(evtType, (event: any) => {
          unsub();
          resolve(event);
        });
        state.sendCommand(cmd);
        setTimeout(() => resolve(null), timeout);
      }),
    { cmd: command, evtType: eventType, timeout: timeoutMs },
  );
};

test.describe('Planning skill dry-run', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);
  });

  test('returns concise schema error for negative subtask_index', async ({ page }) => {
    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: `cmd_${Date.now()}_negative_index`,
        session_id: 'planning_integration',
        skill_name: 'planning-with-files',
        tool_name: 'planning-with-files',
        input: { action: 'update', subtask_index: -1 },
      },
      'dry_run_result',
    );

    expect(result).not.toBeNull();
    expect(result.result.valid).toBe(false);
    expect(result.result.schema_errors).toContain('subtask_index: must be >= 0');
    expect((result.result.schema_errors as string[]).join('\n')).not.toContain('For further information visit');
    expect((result.result.schema_errors as string[]).join('\n')).not.toContain('input_value=');
  });

  test('keeps planning create validation as readable domain error', async ({ page }) => {
    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: `cmd_${Date.now()}_create_missing_task`,
        session_id: 'planning_integration',
        skill_name: 'planning-with-files',
        tool_name: 'planning-with-files',
        input: { action: 'create' },
      },
      'dry_run_result',
    );

    expect(result).not.toBeNull();
    expect(result.result.valid).toBe(true);
    expect(result.result.schema_errors).toEqual([]);
  });
});
