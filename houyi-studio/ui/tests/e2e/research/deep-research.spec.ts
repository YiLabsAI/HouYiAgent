/**
 * E2E: Deep Research full lifecycle.
 *
 * Tests the complete user journey:
 *   Agent Hub → choose Deep Research → enter topic → review plan →
 *   confirm execution → observe progress → view report → export →
 *   back to hub → reopen session.
 *
 * Uses page.route() to mock Research API responses so the test
 * runs without a real LLM backend.
 */
import { test, expect } from '@playwright/test';

const PLAN = {
  query: 'Impact of AI on healthcare',
  sub_questions: [
    { question_id: 'q1', question: 'Current AI applications in diagnosis', priority: 3, search_strategy: 'web', expected_sources: 5, depends_on: [] },
    { question_id: 'q2', question: 'Ethical considerations of AI in healthcare', priority: 2, search_strategy: 'academic', expected_sources: 3, depends_on: [] },
  ],
  outline: [
    { title: 'Introduction', description: 'Overview of AI in healthcare', related_question_ids: ['q1'] },
    { title: 'Ethics', description: 'Key ethical issues', related_question_ids: ['q2'] },
  ],
  version: 1,
  status: 'draft',
};

const REPORT = {
  title: 'AI in Healthcare: A Comprehensive Analysis',
  sections: [
    { title: 'Introduction', content: 'Artificial intelligence is transforming healthcare...', citations: ['[1]'] },
    { title: 'Ethical Considerations', content: 'Key ethical issues include bias and privacy...', citations: ['[2]'] },
  ],
  references: [
    { url: 'https://example.com/ai-health', title: 'AI in Health', snippet: 'Overview of AI applications', reliability: 0.95 },
    { url: 'https://example.com/ethics', title: 'AI Ethics', snippet: 'Ethical framework', reliability: 0.88 },
  ],
  quality_score: { race_overall: 8.2, fact_overall: 7.5 },
};

const SESSIONS = [
  { session_id: 'sess-001', query: 'Impact of AI on healthcare', status: 'completed', created_at: '2026-03-27' },
];

const installApiMocks = async (page: any) => {
  await page.route('**/api/agents/types', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        types: [
          { id: 'deep_research', name: 'Deep Research', description: 'Multi-step research', icon: '🔬', available: true },
          { id: 'code_analyst', name: 'Code Analyst', description: 'Code analysis', icon: '💻', available: false },
        ],
      }),
    });
  });

  await page.route('**/api/research/sessions', async (route: any, request: any) => {
    if (request.method() === 'POST') {
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({ session_id: 'sess-001', plan: PLAN, status: 'planning' }),
      });
    } else {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ sessions: SESSIONS, offset: 0, limit: 20 }),
      });
    }
  });

  await page.route('**/api/research/sessions/sess-001', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        session_id: 'sess-001',
        status: 'completed',
        plan: PLAN,
        progress: { total_steps: 5, completed_steps: 5, current_step: 'done', elapsed_seconds: 30, sub_question_progress: {} },
      }),
    });
  });

  await page.route('**/api/research/sessions/sess-001/execute', async (route: any) => {
    await route.fulfill({
      status: 202,
      contentType: 'application/json',
      body: JSON.stringify({ session_id: 'sess-001', status: 'executing' }),
    });
  });

  await page.route('**/api/research/sessions/sess-001/report', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ report: REPORT }),
    });
  });

  await page.route('**/api/research/sessions/sess-001/events', async (route: any) => {
    const sseBody = [
      `data: ${JSON.stringify({ event_id: 'e1', event_type: 'research.step_started', sequence: 1, payload: { step: 'Searching Q1' } })}`,
      '',
      `data: ${JSON.stringify({ event_id: 'e2', event_type: 'research.source_found', sequence: 2, payload: { title: 'AI Health Paper', url: 'https://example.com' } })}`,
      '',
      `data: ${JSON.stringify({ event_id: 'e3', event_type: 'research.step_completed', sequence: 3, payload: { step: 'Q1 done', completed_steps: 3, total_steps: 5 } })}`,
      '',
      `data: ${JSON.stringify({ event_id: 'e4', event_type: 'research.completed', sequence: 4, payload: {} })}`,
      '',
    ].join('\n');

    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: { 'Cache-Control': 'no-cache' },
      body: sseBody,
    });
  });

  await page.route('**/api/research/sessions/sess-001/plan', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ plan: { ...PLAN, version: 2 } }),
    });
  });

  await page.route('**/api/research/sessions/sess-001/cancel', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'cancelled' }),
    });
  });

  await page.route('**/api/memory/candidates**', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        candidates: [
          { candidate_id: 'c1', content: 'AI is transforming healthcare diagnostics', source_context: 'deep_research', confidence: 0.92, suggested_tags: ['ai', 'healthcare'], status: 'pending' },
          { candidate_id: 'c2', content: 'Ethical guidelines needed for AI deployment', source_context: 'deep_research', confidence: 0.85, suggested_tags: ['ethics'], status: 'pending' },
        ],
      }),
    });
  });

  await page.route('**/api/memory/candidates/*/approve', async (route: any) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ record: { key: 'k1', scope: 'user', content: 'approved' } }),
    });
  });
};

test.describe('Deep Research E2E', () => {
  test.beforeEach(async ({ page }) => {
    await installApiMocks(page);
  });

  test('full lifecycle: hub → workspace → plan → execute → report', async ({ page }) => {
    await page.goto('/');

    // Switch to Agent mode
    const agentBtn = page.getByText('Agent', { exact: true });
    await agentBtn.click();

    // Agent Hub should load
    await expect(page.getByText('Agent Hub')).toBeVisible();
    await expect(page.getByText('Deep Research')).toBeVisible();

    // Click Deep Research card
    await page.getByText('Deep Research').first().click();

    // Input phase
    await expect(page.getByText('Enter your research topic')).toBeVisible();
    const textarea = page.getByPlaceholder('What would you like to research');
    await textarea.fill('Impact of AI on healthcare');
    await page.getByText('Start Research').click();

    // Planning phase
    await expect(page.getByText('Research Plan')).toBeVisible();
    await expect(page.getByText('Current AI applications')).toBeVisible();
    await expect(page.getByText('Ethical considerations')).toBeVisible();

    // Execute
    await page.getByText('Execute').click();

    // Report should load after SSE completes
    await expect(page.getByText('Research Report')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('AI in Healthcare')).toBeVisible();
  });

  test('report export dropdown shows Phase 4 items', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Agent', { exact: true }).click();
    await page.getByText('Deep Research').first().click();
    await page.getByPlaceholder('What would you like to research').fill('test');
    await page.getByText('Start Research').click();
    await expect(page.getByText('Research Plan')).toBeVisible();
    await page.getByText('Execute').click();
    await expect(page.getByText('Research Report')).toBeVisible({ timeout: 10_000 });

    // Open export dropdown
    await page.getByText('Export').click();
    await expect(page.getByText('Markdown (.md)')).toBeVisible();
    await expect(page.getByText('PDF — Coming in Phase 4')).toBeVisible();
    await expect(page.getByText('PPTX — Coming in Phase 4')).toBeVisible();
  });

  test('recent sessions list loads from API', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Agent', { exact: true }).click();
    await expect(page.getByText('Impact of AI on healthcare')).toBeVisible();
    await expect(page.getByText('completed')).toBeVisible();
  });

  test('Memory Inbox shows candidates', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Agent', { exact: true }).click();
    await page.getByText('Memory Inbox').click();

    await expect(page.getByText('AI is transforming healthcare diagnostics')).toBeVisible();
    await expect(page.getByText('Ethical guidelines needed')).toBeVisible();
    await expect(page.getByText('Pending')).toBeVisible();
  });

  test('new research button resets to input', async ({ page }) => {
    await page.goto('/');
    await page.getByText('Agent', { exact: true }).click();
    await page.getByText('Deep Research').first().click();
    await page.getByPlaceholder('What would you like to research').fill('test');
    await page.getByText('Start Research').click();
    await expect(page.getByText('Research Plan')).toBeVisible();
    await page.getByText('New Research').click();
    await expect(page.getByPlaceholder('What would you like to research')).toBeVisible();
  });
});
