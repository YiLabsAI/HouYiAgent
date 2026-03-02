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
  timeoutMs = 12_000,
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
        setTimeout(() => {
          unsub();
          resolve(null);
        }, timeout);
      }),
    { cmd: command, evtType: eventType, timeout: timeoutMs },
  );
};

test.describe('Skill dry-run workflows', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);
  });

  test('notebooklm detail exposes auditable workflow candidates', async ({ page }) => {
    const detailEvent = await sendCommandAndWait(
      page,
      {
        command_type: 'get_skill_detail',
        command_id: `cmd_${Date.now()}_detail_notebooklm`,
        session_id: 'e2e_skill_workflows',
        skill_name: 'notebooklm',
      },
      'skill_detail',
    );

    expect(detailEvent).not.toBeNull();
    expect(detailEvent.skill.name).toBe('notebooklm');
    expect(Array.isArray(detailEvent.skill.available_workflows)).toBe(true);
    expect(detailEvent.skill.available_workflows.length).toBeGreaterThan(0);

    const first = detailEvent.skill.available_workflows[0];
    expect(typeof first.id).toBe('string');
    expect(typeof first.command).toBe('string');
    expect(['frontmatter', 'instructions']).toContain(first.source);
    expect(typeof first.evidence).toBe('string');
    expect(['low', 'medium', 'high']).toContain(first.confidence);
    expect(typeof first.confidence_score).toBe('number');
    expect(['pass', 'warn', 'fail']).toContain(first.validation?.status);
    expect(Array.isArray(first.validation?.missing_dependencies)).toBe(true);
  });

  test('dry-run returns workflow candidates and accepts explicit workflow_id for notebooklm', async ({ page }) => {
    const dryRunEvent = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: `cmd_${Date.now()}_dryrun_notebooklm`,
        session_id: 'e2e_skill_workflows',
        skill_name: 'notebooklm',
        tool_name: 'notebooklm',
        input: { workflow_id: 'template_1' },
      },
      'dry_run_result',
    );

    expect(dryRunEvent).not.toBeNull();
    expect(dryRunEvent.skill_name).toBe('notebooklm');
    expect(Array.isArray(dryRunEvent.result.available_workflows)).toBe(true);
    expect(dryRunEvent.result.available_workflows.length).toBeGreaterThan(0);

    const first = dryRunEvent.result.available_workflows[0];
    expect(first).toHaveProperty('validation');
    expect(first.validation).toHaveProperty('status');
  });

  test('planning-with-files dry-run validation remains non-regressive', async ({ page }) => {
    const dryRunEvent = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: `cmd_${Date.now()}_dryrun_planning`,
        session_id: 'e2e_skill_workflows',
        skill_name: 'planning-with-files',
        tool_name: 'planning-with-files',
        input: { action: 'create' },
      },
      'dry_run_result',
    );

    expect(dryRunEvent).not.toBeNull();
    expect(dryRunEvent.result.valid).toBe(true);
    expect(dryRunEvent.result.schema_errors).toEqual([]);
  });
});
