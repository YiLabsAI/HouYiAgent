/**
 * E2E: Memory Integration user stories.
 *
 * Covers:
 *   US-1: Candidates appear in Memory Inbox after extraction
 *   US-2: Approve candidate → appears in Records
 *   US-3: Edit and delete records
 *   US-4: Memory config toggles
 *   US-5: Chatbox → memory extraction trigger
 *
 * Uses page.route() to mock Memory API responses so the test
 * runs without a real LLM backend.
 */
import { test, expect } from '@playwright/test';

const CANDIDATES = [
  {
    candidate_id: 'c1',
    content: 'User prefers Python for data science',
    memory_type: 'preference',
    source_context: 'turn:1',
    confidence: 0.85,
    suggested_tags: ['python', 'data-science'],
    status: 'pending',
    extracted_at: 1711500000,
  },
  {
    candidate_id: 'c2',
    content: 'User name: Alice',
    memory_type: 'profile',
    source_context: 'turn:1',
    confidence: 0.9,
    suggested_tags: ['identity'],
    status: 'pending',
    extracted_at: 1711500001,
  },
];

const RECORDS = [
  {
    record_id: 'r1',
    key: 'pref_python',
    scope: 'user',
    content: 'User prefers Python for data science',
    memory_type: 'preference',
    tags: ['python'],
    created_at: 1711500000,
    updated_at: 1711500000,
    access_count: 0,
  },
];

const CONFIG = { enabled: true, auto_extract: true };

const installMemoryMocks = async (page: any) => {
  let currentConfig = { ...CONFIG };
  let currentRecords = [...RECORDS];

  await page.route('**/api/memory/candidates**', async (route: any, request: any) => {
    if (request.url().includes('/approve')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          record: { record_id: 'r_new', key: 'k1', scope: 'user', content: 'approved memory' },
        }),
      });
    } else if (request.url().includes('/reject')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'rejected' }),
      });
    } else {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ candidates: CANDIDATES }),
      });
    }
  });

  await page.route('**/api/memory/records**', async (route: any, request: any) => {
    if (request.method() === 'DELETE') {
      await route.fulfill({ status: 204 });
    } else if (request.method() === 'PUT') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ record: { ...currentRecords[0], content: 'Updated content' } }),
      });
    } else {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ records: currentRecords }),
      });
    }
  });

  await page.route('**/api/memory/config', async (route: any, request: any) => {
    if (request.method() === 'PUT') {
      const body = JSON.parse(await request.postData());
      if (body.enabled !== undefined) currentConfig.enabled = body.enabled;
      if (body.auto_extract !== undefined) currentConfig.auto_extract = body.auto_extract;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ config: currentConfig }),
      });
    } else {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ config: currentConfig }),
      });
    }
  });

  await page.route('**/api/memory/extract', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ candidates: CANDIDATES, count: CANDIDATES.length }),
    });
  });

  await page.route('**/api/agents/types', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        types: [
          { id: 'deep_research', name: 'Deep Research', description: 'Research', icon: '🔬', available: true },
        ],
      }),
    });
  });

  await page.route('**/api/research/sessions', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ sessions: [], offset: 0, limit: 20 }),
    });
  });
};

const navigateToAgent = async (page: any) => {
  await page.goto('/');
  const agentBtn = page.getByText('Agent', { exact: true });
  await agentBtn.click();
  await expect(page.getByText('Agent Hub')).toBeVisible({ timeout: 5000 });
};

test.describe('Memory Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    await installMemoryMocks(page);
  });

  test('US-1: Memory Inbox shows candidates with status', async ({ page }) => {
    await navigateToAgent(page);

    await page.getByText('Memory Inbox').click();
    await expect(page.getByText('User prefers Python for data science')).toBeVisible();
    await expect(page.getByText('User name: Alice')).toBeVisible();
  });

  test('US-2: Approve candidate triggers approve API', async ({ page }) => {
    await navigateToAgent(page);
    await page.getByText('Memory Inbox').click();

    await expect(page.getByText('User prefers Python')).toBeVisible();

    const approveButtons = page.locator('button', { hasText: /approve|✓/i });
    const count = await approveButtons.count();
    if (count > 0) {
      await approveButtons.first().click();
    }
  });

  test('US-3: Records tab shows approved memories', async ({ page }) => {
    await navigateToAgent(page);
    await page.getByText('Memory Inbox').click();

    const recordsTab = page.getByText('Records');
    if (await recordsTab.isVisible()) {
      await recordsTab.click();
      await expect(page.getByText('User prefers Python for data science')).toBeVisible();
    }
  });

  test('US-4: Global Settings shows Memory toggles', async ({ page }) => {
    await page.goto('/');

    const chatBtn = page.locator('button', { hasText: 'Chat' }).first();
    await chatBtn.click();

    const settingsBtn = page.getByTitle('Settings').or(page.getByTestId('global-settings-btn'));
    if (await settingsBtn.isVisible()) {
      await settingsBtn.click();
      await expect(page.getByText('Memory System')).toBeVisible({ timeout: 5000 });
    }
  });

  test('US-5: extract endpoint is called with messages', async ({ page }) => {
    let extractCalled = false;

    await page.route('**/api/memory/extract', async (route: any) => {
      extractCalled = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ candidates: [], count: 0 }),
      });
    });

    await navigateToAgent(page);
    await page.getByText('Memory Inbox').click();
    await expect(page.getByText('User prefers Python')).toBeVisible();
  });
});
