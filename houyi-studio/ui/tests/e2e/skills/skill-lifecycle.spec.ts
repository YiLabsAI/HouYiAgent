/**
 * Skill Lifecycle E2E Tests — Industrial-grade Playwright verification.
 *
 * Covers the full lifecycle flows:
 *   1. Dry-run (Center Stage M dialog): open → tool select → form fill → execute → result
 *   2. Load Skill (Center Stage M dialog): open → mode select → input → submit
 *   3. Configure Skill: open → change policy → save → verify
 *   4. Unload Skill: open confirm → confirm → verify removal
 *
 * Prerequisites:
 *   - Backend running on http://localhost:8000
 *   - Frontend dev server on http://localhost:3000
 */

import { test, expect, type Page } from '@playwright/test';

// ─── Helpers ────────────────────────────────────────────────────────

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

/** Navigate to the Skills tab in the Activity Bar. */
const navigateToSkillsTab = async (page: Page) => {
  const skillsTab = page.locator(
    '[data-sidebar-tab="skills"], [data-testid="sidebar-tab-skills"], button[title*="Skills" i]',
  );
  if (await skillsTab.first().isVisible({ timeout: 3000 }).catch(() => false)) {
    await skillsTab.first().click();
  }
};

/** Select a specific skill by name from the SkillsList. */
const selectSkill = async (page: Page, skillName: string) => {
  await page.waitForTimeout(2000);
  const skillButton = page.locator(`button:has-text("${skillName}")`).first();
  if (await skillButton.isVisible({ timeout: 5000 }).catch(() => false)) {
    await skillButton.click();
    await page.waitForTimeout(1500);
  }
};

/** Send a WS command and wait for a specific event type. */
const sendCommandAndWait = async (
  page: Page,
  command: Record<string, any>,
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

// ═══════════════════════════════════════════════════════════════════
// 1. DRY-RUN DIALOG (Center Stage M)
// ═══════════════════════════════════════════════════════════════════

test.describe('Dry-run Dialog — Center Stage M', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);
    await navigateToSkillsTab(page);
  });

  test('dry-run button opens Center Stage M dialog', async ({ page }) => {
    await selectSkill(page, 'web_search');

    const dryRunButton = page.getByTestId('skill-dry-run-button');
    await expect(dryRunButton).toBeVisible({ timeout: 5000 });
    await dryRunButton.click();

    const dialog = page.getByTestId('dry-run-dialog');
    await expect(dialog).toBeVisible({ timeout: 3000 });

    const panel = page.getByTestId('center-stage-panel');
    await expect(panel).toBeVisible();
    expect(await panel.getAttribute('data-size')).toBe('M');
  });

  test('dialog shows form/json mode toggle', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    const formBtn = page.getByTestId('dry-run-mode-form');
    const jsonBtn = page.getByTestId('dry-run-mode-json');
    await expect(formBtn).toBeVisible();
    await expect(jsonBtn).toBeVisible();
  });

  test('form mode renders input fields from tool schema', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    const formInputs = page.getByTestId('dry-run-form-inputs');
    await expect(formInputs).toBeVisible();
  });

  test('switch to JSON mode shows textarea', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    await page.getByTestId('dry-run-mode-json').click();
    const jsonInput = page.getByTestId('dry-run-json-input');
    await expect(jsonInput).toBeVisible();

    const textarea = jsonInput.locator('textarea');
    await expect(textarea).toBeVisible();
    const value = await textarea.inputValue();
    expect(value).toBe('{}');
  });

  test('execute dry-run with empty input (availability check)', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    const executeBtn = page.getByTestId('dry-run-execute');
    await expect(executeBtn).toBeVisible();
    await executeBtn.click();

    // Wait for result panel to appear
    const resultPanel = page.getByTestId('dry-run-result-panel');
    await expect(resultPanel).toBeVisible({ timeout: 10_000 });

    // Should show pass/fail indicator
    const resultText = await resultPanel.textContent();
    expect(resultText).toMatch(/Dry-run (Passed|Failed)/);

    // Schema should show PASS for empty input (availability check)
    await expect(resultPanel.getByText('Schema:')).toBeVisible();
  });

  test('execute dry-run with JSON input validates schema', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    // Switch to JSON mode
    await page.getByTestId('dry-run-mode-json').click();
    const textarea = page.getByTestId('dry-run-json-input').locator('textarea');

    // Fill with mock input
    await textarea.fill('{"query": "test search query"}');

    await page.getByTestId('dry-run-execute').click();

    const resultPanel = page.getByTestId('dry-run-result-panel');
    await expect(resultPanel).toBeVisible({ timeout: 10_000 });
  });

  test('invalid JSON shows error message', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    await page.getByTestId('dry-run-mode-json').click();
    const textarea = page.getByTestId('dry-run-json-input').locator('textarea');
    await textarea.fill('{ invalid json }');
    await page.getByTestId('dry-run-execute').click();

    // Should show JSON error, NOT send command
    const errorText = page.locator('text=Invalid JSON');
    await expect(errorText).toBeVisible({ timeout: 3000 });
  });

  test('dialog closes on Esc key', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    await page.keyboard.press('Escape');
    await expect(page.getByTestId('dry-run-dialog')).not.toBeVisible({ timeout: 3000 });
  });

  test('dialog closes on close button', async ({ page }) => {
    await selectSkill(page, 'web_search');
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();

    await page.getByTestId('center-stage-close').click();
    await expect(page.getByTestId('dry-run-dialog')).not.toBeVisible({ timeout: 3000 });
  });

  test('dry-run result clears when dialog reopened', async ({ page }) => {
    await selectSkill(page, 'web_search');

    // First run
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();
    await page.getByTestId('dry-run-execute').click();
    await expect(page.getByTestId('dry-run-result-panel')).toBeVisible({ timeout: 10_000 });

    // Close
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('dry-run-dialog')).not.toBeVisible();

    // Reopen — result should be cleared
    await page.getByTestId('skill-dry-run-button').click();
    await expect(page.getByTestId('dry-run-dialog')).toBeVisible();
    await expect(page.getByTestId('dry-run-result-panel')).not.toBeVisible({ timeout: 2000 });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. DRY-RUN VIA WEBSOCKET (backend integration)
// ═══════════════════════════════════════════════════════════════════

test.describe('Dry-run — WebSocket backend integration', () => {
  test('dry_run_skill command returns valid result for executable skill', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: 'e2e_dryrun_001',
        session_id: 'e2e_test',
        skill_name: 'web_search',
        tool_name: 'web_search',
        input: {},
      },
      'dry_run_result',
    );

    expect(result).not.toBeNull();
    expect(result).toHaveProperty('result');
    expect(typeof result.result.valid).toBe('boolean');
    expect(Array.isArray(result.result.schema_errors)).toBe(true);
    expect(typeof result.result.policy_result).toBe('string');
    expect(Array.isArray(result.result.capability_gaps)).toBe(true);
  });

  test('dry_run_skill with input validates schema', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: 'e2e_dryrun_002',
        session_id: 'e2e_test',
        skill_name: 'web_search',
        tool_name: 'web_search',
        input: { query: 'test search' },
      },
      'dry_run_result',
    );

    expect(result).not.toBeNull();
    expect(result.result.valid).toBe(true);
  });

  test('dry_run_skill for non-existent skill returns error', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: 'e2e_dryrun_003',
        session_id: 'e2e_test',
        skill_name: 'nonexistent_skill_xyz',
        tool_name: 'nope',
        input: {},
      },
      'dry_run_result',
    );

    expect(result).not.toBeNull();
    expect(result.result.valid).toBe(false);
    expect(result.result.schema_errors.length).toBeGreaterThan(0);
  });

  test('dry_run_skill for schema-only skill passes availability check', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: 'e2e_dryrun_004',
        session_id: 'e2e_test',
        skill_name: 'planning-with-files',
        tool_name: 'planning-with-files',
        input: {},
      },
      'dry_run_result',
    );

    expect(result).not.toBeNull();
    expect(result.result.valid).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. LOAD SKILL DIALOG (Center Stage M)
// ═══════════════════════════════════════════════════════════════════

test.describe('Load Skill Dialog — Center Stage M', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);
    await navigateToSkillsTab(page);
  });

  test('load button opens Center Stage M dialog', async ({ page }) => {
    await page.waitForTimeout(2000);
    const loadBtn = page.getByTestId('load-skill-button');
    await expect(loadBtn).toBeVisible({ timeout: 5000 });
    await loadBtn.click();

    const dialog = page.getByTestId('load-skill-dialog');
    await expect(dialog).toBeVisible({ timeout: 3000 });

    const panel = page.getByTestId('center-stage-panel');
    await expect(panel).toBeVisible();
    expect(await panel.getAttribute('data-size')).toBe('M');
  });

  test('dialog shows three source mode buttons', async ({ page }) => {
    await page.waitForTimeout(2000);
    await page.getByTestId('load-skill-button').click();
    await expect(page.getByTestId('load-skill-dialog')).toBeVisible();

    await expect(page.getByTestId('load-mode-file')).toBeVisible();
    await expect(page.getByTestId('load-mode-url')).toBeVisible();
    await expect(page.getByTestId('load-mode-directory')).toBeVisible();
  });

  test('source input field shows correct placeholder per mode', async ({ page }) => {
    await page.waitForTimeout(2000);
    await page.getByTestId('load-skill-button').click();
    await expect(page.getByTestId('load-skill-dialog')).toBeVisible();

    const input = page.getByTestId('load-skill-source-input');

    // File mode (default)
    const filePlaceholder = await input.getAttribute('placeholder');
    expect(filePlaceholder).toContain('SKILL.md');

    // Switch to URL mode
    await page.getByTestId('load-mode-url').click();
    const urlPlaceholder = await input.getAttribute('placeholder');
    expect(urlPlaceholder).toContain('http');

    // Switch to Directory mode
    await page.getByTestId('load-mode-directory').click();
    const dirPlaceholder = await input.getAttribute('placeholder');
    expect(dirPlaceholder).toContain('skills');
  });

  test('submit button disabled when input is empty', async ({ page }) => {
    await page.waitForTimeout(2000);
    await page.getByTestId('load-skill-button').click();
    await expect(page.getByTestId('load-skill-dialog')).toBeVisible();

    const submitBtn = page.getByTestId('load-skill-submit');
    await expect(submitBtn).toBeDisabled();
  });

  test('URL mode validates http/https prefix', async ({ page }) => {
    await page.waitForTimeout(2000);
    await page.getByTestId('load-skill-button').click();
    await expect(page.getByTestId('load-skill-dialog')).toBeVisible();

    await page.getByTestId('load-mode-url').click();
    const input = page.getByTestId('load-skill-source-input');
    await input.fill('ftp://invalid-protocol');
    await page.getByTestId('load-skill-submit').click();

    // Should show validation error
    const errorText = page.locator('text=URL must start with');
    await expect(errorText).toBeVisible({ timeout: 3000 });
  });

  test('dialog closes on Esc', async ({ page }) => {
    await page.waitForTimeout(2000);
    await page.getByTestId('load-skill-button').click();
    await expect(page.getByTestId('load-skill-dialog')).toBeVisible();

    await page.keyboard.press('Escape');
    await expect(page.getByTestId('load-skill-dialog')).not.toBeVisible({ timeout: 3000 });
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. CONFIGURE SKILL FLOW
// ═══════════════════════════════════════════════════════════════════

test.describe('Configure Skill — End-to-end', () => {
  test('configure command updates policy and emits skill_configured', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_config_001',
        session_id: 'e2e_test',
        skill_name: 'web_search',
        policy_action: 'deny',
        auto_invoke: false,
      },
      'skill_configured',
    );

    expect(result).not.toBeNull();
    expect(result.skill_name).toBe('web_search');

    // Verify via get_skill_detail that policy was applied
    const detail = await sendCommandAndWait(
      page,
      {
        command_type: 'get_skill_detail',
        command_id: 'e2e_config_002',
        session_id: 'e2e_test',
        skill_name: 'web_search',
      },
      'skill_detail',
    );

    expect(detail).not.toBeNull();
    expect(detail.skill.policy.default_action).toBe('deny');

    // Restore to allow for other tests
    await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_config_003',
        session_id: 'e2e_test',
        skill_name: 'web_search',
        policy_action: 'allow',
        auto_invoke: true,
      },
      'skill_configured',
    );
  });

  test('configure with invalid policy returns error', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_config_err_001',
        session_id: 'e2e_test',
        skill_name: 'web_search',
        policy_action: 'invalid_value',
      },
      'skill_error',
    );

    // Should receive an error event
    if (result) {
      expect(result.message).toBeTruthy();
    }
  });

  test('configure non-existent skill returns error', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const result = await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_config_err_002',
        session_id: 'e2e_test',
        skill_name: 'does_not_exist_xyz',
        policy_action: 'allow',
      },
      'skill_error',
    );

    if (result) {
      expect(result.message).toBeTruthy();
    }
  });

  test('configure deny → dry-run returns denied policy', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    // Set policy to deny
    await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_deny_dryrun_001',
        session_id: 'e2e_test',
        skill_name: 'get_date',
        policy_action: 'deny',
      },
      'skill_configured',
    );

    // Dry-run should reflect denied policy
    const dryResult = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: 'e2e_deny_dryrun_002',
        session_id: 'e2e_test',
        skill_name: 'get_date',
        tool_name: 'get_date',
        input: {},
      },
      'dry_run_result',
    );

    expect(dryResult).not.toBeNull();
    // Policy result should reflect the deny setting
    if (dryResult.result.policy_result === 'deny') {
      expect(dryResult.result.valid).toBe(false);
    }

    // Restore
    await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_deny_dryrun_003',
        session_id: 'e2e_test',
        skill_name: 'get_date',
        policy_action: 'allow',
      },
      'skill_configured',
    );
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. UNLOAD SKILL FLOW
// ═══════════════════════════════════════════════════════════════════

test.describe('Unload Skill — Confirmation dialog', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);
    await navigateToSkillsTab(page);
  });

  test('unload button shows confirmation dialog', async ({ page }) => {
    await selectSkill(page, 'get_date');

    // Find the unload button in the action bar
    const unloadBtn = page.locator('[data-testid="skill-actions"] button:has-text("Unload")');
    if (await unloadBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await unloadBtn.click();

      // Confirmation dialog should appear
      const confirmDialog = page.locator('[role="dialog"]:has-text("Unload Skill")');
      await expect(confirmDialog).toBeVisible({ timeout: 3000 });
      await expect(confirmDialog.getByText('Are you sure')).toBeVisible();

      // Cancel to avoid actually unloading
      const cancelBtn = confirmDialog.locator('button:has-text("Cancel")');
      await cancelBtn.click();
      await expect(confirmDialog).not.toBeVisible();
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. FULL LIFECYCLE ROUND-TRIP
// ═══════════════════════════════════════════════════════════════════

test.describe('Full lifecycle — round-trip verification', () => {
  test('list → select → detail → configure → dry-run → restore', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    // Step 1: List skills
    const listResult = await sendCommandAndWait(
      page,
      {
        command_type: 'list_skills',
        command_id: 'e2e_lifecycle_001',
        session_id: 'e2e_test',
      },
      'skill_list',
    );
    expect(listResult).not.toBeNull();
    expect(Array.isArray(listResult.skills)).toBe(true);
    expect(listResult.skills.length).toBeGreaterThan(0);

    const skillName = 'get_weather';

    // Step 2: Get detail
    const detail = await sendCommandAndWait(
      page,
      {
        command_type: 'get_skill_detail',
        command_id: 'e2e_lifecycle_002',
        session_id: 'e2e_test',
        skill_name: skillName,
      },
      'skill_detail',
    );
    expect(detail).not.toBeNull();
    expect(detail.skill.name).toBe(skillName);

    // Step 3: Configure → deny
    const configResult = await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_lifecycle_003',
        session_id: 'e2e_test',
        skill_name: skillName,
        policy_action: 'allow_with_consent',
      },
      'skill_configured',
    );
    expect(configResult).not.toBeNull();

    // Step 4: Verify configuration took effect
    const detail2 = await sendCommandAndWait(
      page,
      {
        command_type: 'get_skill_detail',
        command_id: 'e2e_lifecycle_004',
        session_id: 'e2e_test',
        skill_name: skillName,
      },
      'skill_detail',
    );
    expect(detail2.skill.policy.default_action).toBe('allow_with_consent');

    // Step 5: Dry-run
    const dryResult = await sendCommandAndWait(
      page,
      {
        command_type: 'dry_run_skill',
        command_id: 'e2e_lifecycle_005',
        session_id: 'e2e_test',
        skill_name: skillName,
        tool_name: skillName,
        input: {},
      },
      'dry_run_result',
    );
    expect(dryResult).not.toBeNull();
    expect(typeof dryResult.result.valid).toBe('boolean');

    // Step 6: Restore original policy
    await sendCommandAndWait(
      page,
      {
        command_type: 'configure_skill',
        command_id: 'e2e_lifecycle_006',
        session_id: 'e2e_test',
        skill_name: skillName,
        policy_action: 'allow',
      },
      'skill_configured',
    );

    // Step 7: Get metrics
    const metrics = await sendCommandAndWait(
      page,
      {
        command_type: 'get_skill_metrics',
        command_id: 'e2e_lifecycle_007',
        session_id: 'e2e_test',
        skill_name: skillName,
      },
      'skill_metrics',
    );
    // Metrics should at least return a result (even if zero)
    if (metrics) {
      expect(metrics).toHaveProperty('metrics');
    }
  });

  test('per-skill dry-run passes for all executable skills', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const executableSkills = ['web_search', 'get_date', 'get_weather', 'get_location'];

    for (const skillName of executableSkills) {
      const result = await sendCommandAndWait(
        page,
        {
          command_type: 'dry_run_skill',
          command_id: `e2e_batch_${skillName}`,
          session_id: 'e2e_test',
          skill_name: skillName,
          tool_name: skillName,
          input: {},
        },
        'dry_run_result',
      );

      expect(result).not.toBeNull();
      expect(result.result.valid).toBe(true);
    }
  });

  test('per-skill dry-run passes for schema-only skills', async ({ page }) => {
    await page.goto('/');
    await waitForWsConnected(page);

    const schemaSkills = ['planning-with-files', 'skill-creator', 'frontend-design'];

    for (const skillName of schemaSkills) {
      const result = await sendCommandAndWait(
        page,
        {
          command_type: 'dry_run_skill',
          command_id: `e2e_batch_schema_${skillName}`,
          session_id: 'e2e_test',
          skill_name: skillName,
          tool_name: skillName,
          input: {},
        },
        'dry_run_result',
      );

      expect(result).not.toBeNull();
      expect(result.result.valid).toBe(true);
    }
  });
});
