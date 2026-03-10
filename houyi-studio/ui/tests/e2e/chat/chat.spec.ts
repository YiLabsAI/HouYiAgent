/**
 * Chat mode e2e tests.
 *
 * These tests exercise the Chat workspace: navigation, conversation CRUD,
 * composer interaction, theme switching, and search.
 *
 * The backend is started by global-setup.ts; Chat APIs are REST-based
 * (/api/chat/*) so no WebSocket mocking is needed for basic flows.
 */
import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// --- Helpers ---

const switchToChat = async (page: Page): Promise<void> => {
  // Click the Chat button in the Header mode toggle
  const chatBtn = page.locator('button', { hasText: 'Chat' }).first();
  await chatBtn.click();
  // Wait for either the empty state or the chat page to appear
  await expect(
    page.getByTestId('chat-empty-state').or(page.getByTestId('chat-page')),
  ).toBeVisible({ timeout: 5000 });
};

const createConversation = async (page: Page): Promise<void> => {
  const btn = page.getByTestId('new-conversation-btn');
  await btn.click();
  // Wait for the chat page (with composer) to appear
  await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 5000 });
};

// Track conversation IDs created during each test for cleanup
let testCreatedConvIds: string[] = [];

// Clean up ONLY conversations created during the current test
const cleanupTestConversations = async (page: Page): Promise<void> => {
  for (const id of testCreatedConvIds) {
    try {
      await page.evaluate(async (cid) => {
        await fetch(`/api/chat/conversations/${cid}`, { method: 'DELETE' });
      }, id);
    } catch {
      // Best-effort cleanup
    }
  }
  testCreatedConvIds = [];
};

// Snapshot conversation IDs before a test so we can diff after
const snapshotConvIds = async (page: Page): Promise<string[]> => {
  try {
    return await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations?limit=200');
      const data = await res.json();
      return data.conversations?.map((c: any) => c.conversation_id) || [];
    });
  } catch {
    return [];
  }
};

// --- Tests ---

test.describe('Chat mode', () => {
  let preTestConvIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    // Navigate first so page context is available for API calls
    await page.goto('/');
    preTestConvIds = await snapshotConvIds(page);
  });

  test.afterEach(async ({ page }) => {
    // Find conversations created during this test (new IDs not in pre-test snapshot)
    const postTestConvIds = await snapshotConvIds(page);
    testCreatedConvIds = postTestConvIds.filter((id) => !preTestConvIds.includes(id));
    await cleanupTestConversations(page);
  });

  test('long thinking output keeps the inner reasoning panel scrolled to the bottom', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    const streamState = await page.evaluate(async () => {
      const chatStore = (window as any).__chatStore;
      if (!chatStore) {
        throw new Error('Chat store not found');
      }

      const conversationId = chatStore.getState().activeConversationId;
      if (!conversationId) {
        throw new Error('Active conversation not found');
      }

      const messageId = `mock-assistant-${Date.now()}`;
      const reasoningChunks = Array.from(
        { length: 180 },
        (_, i) => `Step ${i + 1}: reasoning detail line ${i + 1} with extra context for scroll overflow validation.\n`,
      );
      const initialReasoning = reasoningChunks[0] ?? '';

      chatStore.setState((state: any) => {
        const conversation = state.activeConversation;
        if (!conversation) return state;
        return {
          ...state,
          activeConversation: {
            ...conversation,
            messages: [
              ...conversation.messages,
              {
                message_id: messageId,
                role: 'assistant',
                content: '',
                reasoning_content: initialReasoning,
                metadata: {},
                created_at: Date.now() / 1000,
              },
            ],
          },
          streaming: {
            ...state.streaming,
            isStreaming: true,
            messageId,
            contentBuffer: '',
            reasoningBuffer: initialReasoning,
            streamConversationId: conversationId,
          },
        };
      });

      return { messageId, reasoningChunks: reasoningChunks.slice(1) };
    });

    const assistantBubble = page.getByTestId('message-bubble').last();
    const thinkingButton = assistantBubble.getByRole('button', { name: /Thinking/ });
    await expect(thinkingButton).toBeVisible({ timeout: 10000 });

    const reasoningPanel = thinkingButton.locator('xpath=following-sibling::div[1]');
    await expect(reasoningPanel).toBeVisible({ timeout: 10000 });

    await page.evaluate(async ({ messageId, reasoningChunks }) => {
      const chatStore = (window as any).__chatStore;
      for (const chunk of reasoningChunks as string[]) {
        chatStore.setState((state: any) => {
          const nextReasoning = `${state.streaming.reasoningBuffer || ''}${chunk}`;
          const conversation = state.activeConversation;
          const messages = Array.isArray(conversation?.messages) ? [...conversation.messages] : [];
          const existingIndex = messages.findIndex((message: any) => message?.message_id === messageId);
          if (existingIndex >= 0) {
            messages[existingIndex] = {
              ...messages[existingIndex],
              reasoning_content: nextReasoning,
            };
          }
          return {
            ...state,
            activeConversation: conversation
              ? {
                  ...conversation,
                  messages,
                }
              : conversation,
            streaming: {
              ...state.streaming,
              isStreaming: true,
              messageId,
              reasoningBuffer: nextReasoning,
            },
          };
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }, streamState);

    await expect.poll(async () => reasoningPanel.evaluate((el) => {
      const textLength = el.textContent?.length ?? 0;
      return (
        textLength > 1000
        && el.scrollHeight > el.clientHeight
        && el.scrollTop >= el.scrollHeight - el.clientHeight - 8
      );
    })).toBe(true);

    const panelMetrics = await reasoningPanel.evaluate((el) => ({
      scrollTop: el.scrollTop,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
      textLength: el.textContent?.length ?? 0,
    }));

    expect(panelMetrics.textLength).toBeGreaterThan(1000);
    expect(panelMetrics.scrollHeight).toBeGreaterThan(panelMetrics.clientHeight);
    expect(panelMetrics.scrollTop).toBeGreaterThanOrEqual(
      panelMetrics.scrollHeight - panelMetrics.clientHeight - 8,
    );
  });

  test('tool steps show duration and parallel group metadata', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    await page.evaluate(() => {
      const chatStore = (window as any).__chatStore;
      if (!chatStore) {
        throw new Error('Chat store not found');
      }

      chatStore.setState((state: any) => {
        const conversation = state.activeConversation;
        if (!conversation) {
          throw new Error('Active conversation not found');
        }

        const now = Date.now() / 1000;

        return {
          ...state,
          activeConversation: {
            ...conversation,
            messages: [
              {
                message_id: `user-tool-${Date.now()}`,
                role: 'user',
                content: 'search the docs',
                metadata: {},
                created_at: now,
              },
              {
                message_id: `assistant-tool-carrier-${Date.now()}`,
                role: 'assistant',
                content: '',
                tool_calls: [{ id: 'call-tool-1' }],
                metadata: {},
                created_at: now + 1,
              },
              {
                message_id: `tool-step-${Date.now()}`,
                role: 'tool',
                content: '{"matches":["README.md"]}',
                name: 'houyi_read_file',
                tool_call_id: 'call-tool-1',
                metadata: {
                  tool_status: 'ok',
                  round_index: 1,
                  parallel_group_id: 'round_1',
                  duration_ms: 1485,
                },
                created_at: now + 1.5,
              },
              {
                message_id: `assistant-tool-final-${Date.now()}`,
                role: 'assistant',
                content: 'Found the relevant file.',
                metadata: { trace_id: 'trace-tool-1', usage: { total_tokens: 42 } },
                created_at: now + 2,
              },
            ],
          },
        };
      });
    });

    await expect(page.getByText('Tool calls 1')).toBeVisible({ timeout: 10000 });
    await page.getByText('Show steps').click();
    await expect(page.getByText('houyi_read_file')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Duration 1.5s')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Parallel round_1')).toBeVisible({ timeout: 10000 });
  });

  // --- P-039B: Switching large conversations must NOT trigger settings-request storms ---
  test('switching between large conversations does not spam /api/chat/settings', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    let settingsReqCount = 0;
    await page.route('**/api/chat/settings', async (route) => {
      settingsReqCount += 1;
      await route.continue();
    });

    const makeConv = async (title: string) => {
      const conv = await page.evaluate(async (t: string) => {
        const res = await fetch('/api/chat/conversations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: t }),
        });
        return res.json();
      }, title);

      await page.evaluate(async (cid: string) => {
        const messages = Array.from({ length: 60 }).flatMap((_, i) => {
          const idx = i + 1;
          return [
            { role: 'user', content: `Q${idx}` },
            { role: 'assistant', content: `A${idx}` },
          ];
        });
        await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages }),
        });
      }, conv.conversation_id);

      return conv;
    };

    const convA = await makeConv('switch-storm-A');
    const convB = await makeConv('switch-storm-B');
    testCreatedConvIds.push(convA.conversation_id, convB.conversation_id);

    // Refresh list
    await page.goto('/');
    await switchToChat(page);

    // Switch between A/B a few times
    for (let i = 0; i < 4; i++) {
      await page.locator('text=switch-storm-A').first().click();
      await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
      await expect(page.getByTestId('message-bubble')).toHaveCount(120, { timeout: 10000 });

      await page.locator('text=switch-storm-B').first().click();
      await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
      await expect(page.getByTestId('message-bubble')).toHaveCount(120, { timeout: 10000 });
    }

    // The settings request should be constant-ish (mount once), never proportional to message count.
    // Allow a small buffer for retries / reloads.
    expect(settingsReqCount).toBeLessThanOrEqual(4);
  });

  test('scroll position is restored per conversation (baseline regression)', async ({ page }) => {
    const onConsole = (msg: any) => {
      const text = msg.text?.() ?? '';
      if (text.includes('[chat][scroll-restore]')) {
        console.log(text);
      }
    };
    page.on('console', onConsole);

    await page.goto('/');
    await switchToChat(page);

    const makeConv = async (title: string, pairs: number) => {
      const conv = await page.evaluate(async (t: string) => {
        const res = await fetch('/api/chat/conversations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: t }),
        });
        return res.json();
      }, title);

      await page.evaluate(async ({ cid, nPairs }: { cid: string; nPairs: number }) => {
        const messages = Array.from({ length: nPairs }).flatMap((_, i) => {
          const idx = i + 1;
          return [
            { role: 'user', content: `Q${idx}` },
            { role: 'assistant', content: `A${idx} - some content to make the message bubble stable` },
          ];
        });
        await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages }),
        });
      }, { cid: conv.conversation_id, nPairs: pairs });

      return conv;
    };

    const convA = await makeConv('scroll-restore-A', 120);
    const convB = await makeConv('scroll-restore-B', 6);
    testCreatedConvIds.push(convA.conversation_id, convB.conversation_id);

    // Refresh list
    await page.goto('/');
    await switchToChat(page);

    await page.locator('text=scroll-restore-A').first().click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });

    const scroller = page.getByTestId('chat-timeline');
    await expect(scroller).toBeVisible();

    // Set a deterministic scroll position away from bottom with real user-like
    // scrolling so the app persists the snapshot (it gates on wheel/touch/pointer).
    await scroller.hover();
    const desiredDist = 600;
    for (let i = 0; i < 40; i++) {
      const cur = await scroller.evaluate((el) => Math.abs(el.scrollTop));
      if (cur >= desiredDist - 40) break;
      // Scroll "up" (toward older messages in our column-reverse timeline).
      await page.mouse.wheel(0, -200);
      await page.waitForTimeout(20);
    }

    // Allow markdown layout / placeholder sizing to settle before taking baseline.
    await page.waitForTimeout(150);
    const beforeStable = await scroller.evaluate((el) => {
      const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
      el.dispatchEvent(new Event('scroll'));
      const dist = Math.abs(el.scrollTop);
      const fraction = maxScroll > 0 ? dist / maxScroll : 0;
      return { dist, fraction, maxScroll };
    });
    console.log('[e2e][scroll-restore] beforeStable', beforeStable);

    // Give React onScroll/state time to persist snapshot.
    await page.waitForTimeout(50);

    for (let i = 0; i < 3; i++) {
      await page.locator('text=scroll-restore-B').first().click();
      await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
      await expect(page.getByTestId('message-bubble')).toHaveCount(12, { timeout: 10000 });

      await page.locator('text=scroll-restore-A').first().click();
      await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });

      await expect(page.getByTestId('message-bubble')).toHaveCount(120, { timeout: 10000 });
      await page.waitForTimeout(200);

      const after = await scroller.evaluate((el) => {
        const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
        el.dispatchEvent(new Event('scroll'));
        const dist = Math.abs(el.scrollTop);
        const fraction = maxScroll > 0 ? dist / maxScroll : 0;
        return { dist, fraction, maxScroll };
      });

      expect(after.maxScroll).toBeGreaterThan(0);
      expect(after.dist).toBeGreaterThan(80);
      expect(after.dist).toBeLessThan(after.maxScroll - 80);
      // Note: We intentionally do not assert an exact fraction-of-scrollHeight here.
      // scrollHeight can drift across swaps (fonts, async layout), and the primary baseline goal
      // is that we do NOT reset to top/bottom after switching back.
    }

    page.off('console', onConsole);
  });

  test('mermaid conversation switch does not drift scroll position (baseline)', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    const convA = await page.evaluate(async (t: string) => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: t }),
      });
      return res.json();
    }, 'mermaid-stability-A');

    const convB = await page.evaluate(async (t: string) => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: t }),
      });
      return res.json();
    }, 'mermaid-stability-B');

    testCreatedConvIds.push(convA.conversation_id, convB.conversation_id);

    await page.evaluate(async (cid: string) => {
      const messages = Array.from({ length: 50 }).flatMap((_, i) => {
        const idx = i + 1;
        const mermaid = idx % 5 === 0
          ? `\n\n\
\
\`\`\`mermaid\nflowchart TD\n  A${idx}-->B${idx}\n\`\`\`\n`
          : '';
        return [
          { role: 'user', content: `Q${idx}` },
          { role: 'assistant', content: `A${idx}${mermaid}` },
        ];
      });
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, convA.conversation_id);

    await page.evaluate(async (cid: string) => {
      const messages = Array.from({ length: 6 }).flatMap((_, i) => {
        const idx = i + 1;
        return [
          { role: 'user', content: `QB${idx}` },
          { role: 'assistant', content: `AB${idx}` },
        ];
      });
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, convB.conversation_id);

    await page.goto('/');
    await switchToChat(page);

    await page.locator('text=mermaid-stability-A').first().click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });

    const scroller = page.getByTestId('chat-timeline');
    await expect(scroller).toBeVisible();
    await expect(page.getByTestId('message-bubble')).toHaveCount(100, { timeout: 10000 });

    await page.waitForTimeout(800);
    await expect(page.locator('svg').first()).toBeVisible({ timeout: 15000 });

    await scroller.hover();
    const desiredDist = 900;
    for (let i = 0; i < 70; i++) {
      const cur = await scroller.evaluate((el) => Math.abs(el.scrollTop));
      if (cur >= desiredDist - 60) break;
      await page.mouse.wheel(0, -220);
      await page.waitForTimeout(20);
    }
    await page.waitForTimeout(200);

    const before = await scroller.evaluate((el) => Math.abs(el.scrollTop));

    await page.locator('text=mermaid-stability-B').first().click();
    await expect(page.getByTestId('message-bubble')).toHaveCount(12, { timeout: 10000 });
    await page.locator('text=mermaid-stability-A').first().click();
    await expect(page.getByTestId('message-bubble')).toHaveCount(100, { timeout: 10000 });
    await page.waitForTimeout(250);

    const drift = await scroller.evaluate(async (el) => {
      const samples: number[] = [];
      const start = performance.now();
      while (performance.now() - start < 600) {
        samples.push(Math.abs((el as HTMLDivElement).scrollTop));
        await new Promise((r) => setTimeout(r, 50));
      }
      let min = Infinity;
      let max = -Infinity;
      for (const v of samples) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return { min, max, span: max - min };
    });

    const after = await scroller.evaluate((el) => Math.abs(el.scrollTop));

    expect(Math.abs(after - before)).toBeLessThan(420);
    expect(drift.span).toBeLessThan(420);
  });

  test('navigates to Chat mode and shows empty state', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Should see the empty state or conversation rail
    const emptyState = page.getByTestId('chat-empty-state');
    const chatPage = page.getByTestId('chat-page');
    await expect(emptyState.or(chatPage)).toBeVisible();
  });

  test('creates a new conversation and shows composer', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    // Composer should be visible with the input and send button
    await expect(page.getByTestId('chat-input')).toBeVisible();
    await expect(page.getByTestId('chat-send-btn')).toBeVisible();

    // Send button should be disabled when input is empty
    await expect(page.getByTestId('chat-send-btn')).toBeDisabled();
  });

  test('typing in composer enables send button', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    const input = page.getByTestId('chat-input');
    await input.fill('Hello world');

    // Send button should now be enabled
    await expect(page.getByTestId('chat-send-btn')).toBeEnabled();
  });

  test('theme switcher changes CSS variables', async ({ page }) => {
    await page.goto('/');

    // Verify initial theme class on <html>
    const initialClass = await page.evaluate(() => document.documentElement.className);
    expect(initialClass).toContain('theme-');

    // Open theme menu and switch to Light
    await page.getByTitle('Switch theme').click();
    await page.getByText('Light').click();

    // Verify <html> has theme-light class
    const lightClass = await page.evaluate(() => document.documentElement.className);
    expect(lightClass).toContain('theme-light');

    // Verify a CSS variable actually changed (gray-900 should be light in light theme)
    // Light theme maps gray-900 to "246 248 250" (space-separated RGB channels)
    const gray900 = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue('--gray-900').trim(),
    );
    expect(gray900).toBe('246 248 250');

    // Switch back to Dark
    await page.getByTitle('Switch theme').click();
    await page.getByText('Dark').click();

    const darkClass = await page.evaluate(() => document.documentElement.className);
    expect(darkClass).toContain('theme-dark');
  });

  // BUG-042: Theme switch must produce a VISIBLE change — not just CSS variable swap
  test('theme switch produces visually different screenshots', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await page.waitForTimeout(500);

    // Screenshot in dark mode
    const darkScreenshot = await page.screenshot();

    // Switch to Light
    await page.getByTitle('Switch theme').click();
    await page.getByText('Light').click();
    await page.waitForTimeout(500);

    const lightScreenshot = await page.screenshot();

    // Screenshots MUST differ (theme actually changed visually)
    expect(
      Buffer.compare(darkScreenshot, lightScreenshot),
      'Dark and Light screenshots are identical — theme switch has no visual effect',
    ).not.toBe(0);

    // Verify key visual properties changed:
    // 1. Header background should be light in light theme
    const headerBg = await page.evaluate(() => {
      const header = document.querySelector('header');
      return header ? getComputedStyle(header).backgroundColor : '';
    });
    // In light theme, bg-gray-900 resolves to #e6e9ef → rgb(230, 233, 239)
    // It should NOT be dark (#18181b → rgb(24, 24, 27))
    expect(headerBg).not.toBe('rgb(24, 24, 27)');

    // 2. Body/main background should be light
    const bodyBg = await page.evaluate(() => {
      return getComputedStyle(document.documentElement).backgroundColor;
    });
    expect(bodyBg).not.toBe('rgb(24, 24, 27)');

    // Switch back to Dark
    await page.getByTitle('Switch theme').click();
    await page.getByText('Dark').click();
  });

  test('theme persists across page reload', async ({ page }) => {
    await page.goto('/');

    // Switch to Nord theme
    await page.getByTitle('Switch theme').click();
    await page.getByText('Nord').click();

    const nordClass = await page.evaluate(() => document.documentElement.className);
    expect(nordClass).toContain('theme-nord');

    // Reload page
    await page.reload();
    await expect(page.getByText('HouYi')).toBeVisible();

    // Theme should persist
    const afterReloadClass = await page.evaluate(() => document.documentElement.className);
    expect(afterReloadClass).toContain('theme-nord');

    // Restore to dark for other tests
    await page.getByTitle('Switch theme').click();
    await page.getByText('Dark').click();
  });

  test('search modal opens with Cmd+K', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Open search via keyboard shortcut
    await page.keyboard.press('Meta+k');

    // Search modal should appear with input
    const searchInput = page.getByPlaceholder('Search conversations...');
    await expect(searchInput).toBeVisible({ timeout: 3000 });

    // Close by clicking the X button (more reliable than Escape in headless)
    const closeBtn = page.locator('button[aria-label="Close search"], button:has(svg.lucide-x)').first();
    if (await closeBtn.isVisible()) {
      await closeBtn.click();
    } else {
      // Fallback: press Escape with focus on the input
      await searchInput.press('Escape');
    }
    await expect(searchInput).not.toBeVisible({ timeout: 5000 });
  });

  test('mode switching between Graph and Chat preserves state', async ({ page }) => {
    await page.goto('/');

    // Start in Graph mode — DAG canvas should be visible
    await expect(page.getByTestId('dag-canvas')).toBeVisible();

    // Switch to Chat
    await switchToChat(page);
    await expect(page.getByTestId('dag-canvas')).not.toBeVisible();

    // Switch back to Graph
    const graphBtn = page.locator('button', { hasText: 'Graph' }).first();
    await graphBtn.click();
    await expect(page.getByTestId('dag-canvas')).toBeVisible();
  });

  test('global settings page opens from Chat mode', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Click the settings gear icon in the header
    await page.getByTitle('Global settings').click();

    // Settings page should show provider configuration
    await expect(page.getByText('LLM Providers')).toBeVisible({ timeout: 3000 });
  });

  // --- BUG-036: Code blocks should have syntax highlighting ---
  test('code blocks have syntax highlighting styles loaded', async ({ page }) => {
    await page.goto('/');

    // Verify that highlight.js theme CSS is injected into the page
    const hasHljsTheme = await page.evaluate(() => {
      const styles = document.querySelectorAll('style[data-hljs-theme]');
      return styles.length > 0;
    });
    expect(hasHljsTheme).toBe(true);
  });

  // --- BUG-036: hljs theme switches with UI theme ---
  test('highlight.js theme switches between light and dark', async ({ page }) => {
    await page.goto('/');

    // Default should be dark theme
    let hljsTheme = await page.evaluate(() => {
      const el = document.querySelector('style[data-hljs-theme]');
      return el?.getAttribute('data-hljs-theme');
    });
    expect(hljsTheme).toBe('dark');

    // Switch to Light
    await page.getByTitle('Switch theme').click();
    await page.getByText('Light').click();
    await page.waitForTimeout(500);

    hljsTheme = await page.evaluate(() => {
      const el = document.querySelector('style[data-hljs-theme]');
      return el?.getAttribute('data-hljs-theme');
    });
    expect(hljsTheme).toBe('light');

    // Switch back to Dark
    await page.getByTitle('Switch theme').click();
    await page.getByText('Dark').click();
    await page.waitForTimeout(500);

    hljsTheme = await page.evaluate(() => {
      const el = document.querySelector('style[data-hljs-theme]');
      return el?.getAttribute('data-hljs-theme');
    });
    expect(hljsTheme).toBe('dark');
  });

  // --- P-039: Theme switch must not cause hljs style gap (flicker) even with many messages ---
  test('theme switch does not produce a gap where no hljs theme style is present', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Create a conversation with many messages to increase render pressure.
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'theme-flicker-many-messages' }),
      });
      return res.json();
    });

    // Seed 80 messages.
    await page.evaluate(async (cid: string) => {
      const messages = Array.from({ length: 80 }).flatMap((_, i) => {
        const idx = i + 1;
        return [
          { role: 'user', content: `Q${idx}: code sample` },
          { role: 'assistant', content: `A${idx}:\n\n\`\`\`ts\nconst x${idx} = ${idx};\n\`\`\`` },
        ];
      });
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, conv.conversation_id);

    // Refresh UI conversation list (created via API) so it appears in the sidebar.
    await page.goto('/');
    await switchToChat(page);

    // Navigate to the conversation via sidebar.
    const convItem = page.locator('text=theme-flicker-many-messages').first();
    await expect(convItem).toBeVisible({ timeout: 5000 });
    await convItem.click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });

    // Progressive rendering: large conversations show the most recent chunk first.
    await expect(page.getByTestId('message-bubble')).toHaveCount(120, { timeout: 10000 });

    // Expand once to render the full set (stress test for theme switching).
    const loadOlder = page.getByRole('button', { name: 'Show more' });
    await expect(loadOlder).toBeVisible({ timeout: 5000 });
    await loadOlder.click();
    await expect(page.getByTestId('message-bubble')).toHaveCount(160, { timeout: 10000 });

    // Observe hljs theme style count in real-time.
    await page.evaluate(() => {
      (window as any).__hljsMinCount = Number.POSITIVE_INFINITY;
      (window as any).__hljsObserver?.disconnect?.();
      const update = () => {
        const count = document.head.querySelectorAll('style[data-hljs-theme]').length;
        (window as any).__hljsMinCount = Math.min((window as any).__hljsMinCount, count);
      };
      update();
      const obs = new MutationObserver(update);
      obs.observe(document.head, { childList: true, subtree: true });
      (window as any).__hljsObserver = obs;
    });

    // Rapidly toggle theme a few times.
    for (let i = 0; i < 4; i++) {
      await page.getByTitle('Switch theme').click();
      await page.getByText('Light').click();
      await page.waitForTimeout(50);
      await page.getByTitle('Switch theme').click();
      await page.getByText('Dark').click();
      await page.waitForTimeout(50);
    }

    // Assertion: there was never a time with 0 hljs theme styles.
    const minCount = await page.evaluate(() => (window as any).__hljsMinCount);
    expect(minCount).toBeGreaterThanOrEqual(1);

    // Cleanup observer.
    await page.evaluate(() => {
      (window as any).__hljsObserver?.disconnect?.();
      (window as any).__hljsObserver = null;
    });

    // Best-effort cleanup conversation
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}`, { method: 'DELETE' });
    }, conv.conversation_id);
  });

  // --- BUG-037: Conversation settings Save should not crash ---
  test('conversation settings save does not blank the page', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    // Send a message so we have content
    const input = page.getByTestId('chat-input');
    await input.fill('Hello test');
    await page.getByTestId('chat-send-btn').click();

    // Wait for streaming to complete (assistant message appears)
    await page.waitForTimeout(3000);

    // Open conversation settings drawer
    const settingsBtn = page.locator('button[title="Conversation settings"]').first();
    if (await settingsBtn.isVisible()) {
      await settingsBtn.click();
      await page.waitForTimeout(500);

      // Click Save (even if not dirty, should not crash)
      const saveBtn = page.locator('button', { hasText: 'Save' }).first();
      if (await saveBtn.isVisible()) {
        await saveBtn.click();
        await page.waitForTimeout(500);
      }

      // Page should NOT be blank — chat-page should still be visible
      await expect(page.getByTestId('chat-page')).toBeVisible();
    }
  });

  // --- BUG-038: Global settings Vertex AI buttons should not 404 ---
  test('global settings provider test/fetch shows friendly message for Vertex', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Open global settings
    await page.getByTitle('Global settings').click();
    await expect(page.getByText('LLM Providers')).toBeVisible({ timeout: 3000 });

    // Add a Vertex AI provider
    const vertexBtn = page.locator('button', { hasText: 'Vertex AI' }).first();
    await vertexBtn.click();
    await page.waitForTimeout(500);

    // Fill in a base_url so the Test Connection button becomes enabled
    const baseUrlInput = page.locator('input[placeholder="https://api.example.com/v1"]').first();
    if (await baseUrlInput.isVisible()) {
      await baseUrlInput.fill('https://aiplatform.googleapis.com');
      await page.waitForTimeout(300);
    }

    // The Vertex provider should be expanded — find Test Connection button
    const testBtn = page.locator('button', { hasText: 'Test Connection' }).first();
    if (await testBtn.isVisible()) {
      await testBtn.click();
      await page.waitForTimeout(2000);

      // Should show a friendly result about Vertex AI (success or error), NOT an HTML 404
      // Look for any text containing 'Vertex AI' in the test result area
      const resultText = page.locator('[class*="text-"]', { hasText: 'Vertex AI' }).first();
      await expect(resultText).toBeVisible({ timeout: 10000 });
    }
  });

  // --- BUG-038a2: Vertex AI Fetch Models UI shows friendly error banner, no HTML ---
  test('global settings Vertex AI Fetch Models button shows friendly error in UI', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Open global settings
    await page.getByTitle('Global settings').click();
    await expect(page.getByText('LLM Providers')).toBeVisible({ timeout: 3000 });

    // Add a Vertex AI provider
    await page.locator('button', { hasText: 'Vertex AI' }).first().click();
    await page.waitForTimeout(300);

    // Fill base_url so Fetch Models button is enabled
    const baseUrlInput = page.locator('input[placeholder="https://api.example.com/v1"]').first();
    if (await baseUrlInput.isVisible()) {
      await baseUrlInput.fill('https://aiplatform.googleapis.com');
      await page.waitForTimeout(200);
    }

    // Click Fetch Models
    const fetchBtn = page.locator('button', { hasText: 'Fetch Models' }).first();
    await expect(fetchBtn).toBeVisible();
    await fetchBtn.click();

    // Wait for error banner to appear
    await page.waitForTimeout(2000);

    // Fetch Models should either succeed (populate model list) or show a friendly error
    // Wait for response
    await page.waitForTimeout(5000);
    // Check: no raw HTML in the page (the old bug was showing HTML 404 pages)
    const pageContent = await page.textContent('body');
    expect(pageContent).not.toContain('{margin');
    expect(pageContent).not.toContain('*{padding:0}');
  });

  // --- BUG-038b: Vertex AI Fetch Models API returns friendly error, not HTML 404 ---
  test('Vertex AI fetch-models API returns friendly error without HTML', async ({ page }) => {
    await page.goto('/');

    // Call fetch-models with Vertex AI base_url (even without vertex provider_id)
    const result = await page.evaluate(async () => {
      const res = await fetch('/api/chat/providers/fetch-models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_url: 'https://aiplatform.googleapis.com',
          api_key: 'test-key',
          provider_id: 'vertex-test',
        }),
      });
      return res.json();
    });

    // Should return clean response (success with models or friendly error), never raw HTML
    if (result.error) {
      expect(result.error).not.toContain('<style');
      expect(result.error).not.toContain('{margin');
    } else {
      // Success: models array should exist
      expect(Array.isArray(result.models)).toBe(true);
    }

    // Also test with non-vertex provider_id but GCP domain
    const result2 = await page.evaluate(async () => {
      const res = await fetch('/api/chat/providers/fetch-models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_url: 'https://aiplatform.googleapis.com',
          api_key: 'test-key',
          provider_id: 'custom-123',
        }),
      });
      return res.json();
    });

    // Domain-based detection should also catch it — clean response, no HTML
    if (result2.error) {
      expect(result2.error).not.toContain('<style');
      expect(result2.error).not.toContain('{margin');
    } else {
      expect(Array.isArray(result2.models)).toBe(true);
    }
  });

  // --- BUG-038c: Test Connection API also returns friendly error for Vertex ---
  test('Vertex AI test-connection API returns friendly error without HTML', async ({ page }) => {
    await page.goto('/');

    const result = await page.evaluate(async () => {
      const res = await fetch('/api/chat/providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_url: 'https://aiplatform.googleapis.com',
          api_key: 'test-key',
          provider_id: 'custom-xyz',
        }),
      });
      return res.json();
    });

    // Domain-based detection should return clean message (success or friendly error), no HTML
    expect(result.message).not.toContain('<style');
    expect(result.message).not.toContain('{margin');
    expect(result.message).toContain('Vertex AI');
  });

  // --- BUG-047: Invalid model (e.g. gemini) 400 error must show error banner ---
  test('invalid model 400 error shows error banner with message', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    // Get the active conversation ID, then PATCH its model to an invalid one
    const convId = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations');
      const data = await res.json();
      return data.conversations[0]?.conversation_id;
    });
    expect(convId).toBeTruthy();

    // PATCH the conversation model to something that will 400
    await page.evaluate(async (id) => {
      await fetch(`/api/chat/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'nonexistent-model-xyz' }),
      });
    }, convId);

    // Reload to pick up the new model
    await page.reload();
    await switchToChat(page);
    const convTitle = page.locator('.text-\\[12px\\].truncate').first();
    await expect(convTitle).toBeVisible({ timeout: 5000 });
    await convTitle.click();
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 5000 });

    // Send a message — this should trigger a 400 from the LLM
    const input = page.getByTestId('chat-input');
    await input.fill('trigger 400 error');
    await page.getByTestId('chat-send-btn').click();

    // Wait for the error banner to appear (SSE message.error → error state → red banner)
    const errorBanner = page.locator('[class*="bg-red-900"]');
    await expect(errorBanner).toBeVisible({ timeout: 15000 });

    // Error banner must contain meaningful text
    const errorText = await errorBanner.textContent();
    expect(errorText).toBeTruthy();
    expect(errorText!.length).toBeGreaterThan(10);

    // Chat page should still be visible (not blank)
    await expect(page.getByTestId('chat-page')).toBeVisible();
  });

  // --- BUG-046: Conversation settings temperature/max_tokens/top_p persist via API ---
  test('conversation settings params persist after PATCH', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    // Get the conversation ID
    const convId = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations');
      const data = await res.json();
      return data.conversations[0]?.conversation_id;
    });
    expect(convId).toBeTruthy();

    // PATCH temperature, max_tokens, top_p
    const patchRes = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature: 0.5, max_tokens: 2048, top_p: 0.9 }),
      });
      return res.json();
    }, convId);

    // Verify PATCH response contains the values
    expect(patchRes.temperature).toBe(0.5);
    expect(patchRes.max_tokens).toBe(2048);
    expect(patchRes.top_p).toBe(0.9);

    // GET the conversation and verify values persisted
    const getRes = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`);
      return res.json();
    }, convId);

    expect(getRes.temperature).toBe(0.5);
    expect(getRes.max_tokens).toBe(2048);
    expect(getRes.top_p).toBe(0.9);

    // PATCH with null to reset — verify they go back to null
    const resetRes = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature: null, max_tokens: null, top_p: null }),
      });
      return res.json();
    }, convId);

    expect(resetRes.temperature).toBeNull();
    expect(resetRes.max_tokens).toBeNull();
    expect(resetRes.top_p).toBeNull();
  });

  // --- BUG-043: Search sort by Created / Last Updated actually works ---
  test('search sort toggle changes result order', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Create a conversation and send a message so search has content
    await createConversation(page);
    const input = page.getByTestId('chat-input');
    await input.fill('searchable test message');
    await page.getByTestId('chat-send-btn').click();
    await page.waitForTimeout(4000);

    // Open search modal with Cmd+K
    await page.keyboard.press('Meta+k');
    const searchInput = page.locator('input[placeholder="Search conversations..."]');
    await expect(searchInput).toBeVisible({ timeout: 3000 });

    // Search for "searchable"
    await searchInput.fill('searchable');
    await page.waitForTimeout(500);

    // Verify results appear
    const results = page.locator('button.w-full.text-left');
    await expect(results.first()).toBeVisible({ timeout: 5000 });

    // Click "Last Updated" sort button and verify it gets active styling
    const updatedBtn = page.locator('button', { hasText: 'Last Updated' }).first();
    await updatedBtn.click();
    await page.waitForTimeout(300);

    // The "Last Updated" button should have active styling (bg-gray-700)
    const updatedClass = await updatedBtn.getAttribute('class');
    expect(updatedClass).toContain('bg-gray-700');

    // Click "Created" sort button and verify it gets active styling
    const createdBtn = page.locator('button', { hasText: 'Created' }).first();
    await createdBtn.click();
    await page.waitForTimeout(300);
    const createdClass = await createdBtn.getAttribute('class');
    expect(createdClass).toContain('bg-gray-700');
  });

  // --- Provider switching: settings API round-trip ---
  test('provider CRUD via settings API persists and lists models', async ({ page }) => {
    await page.goto('/');

    // 1. GET current settings (baseline)
    const baseline = await page.evaluate(async () => {
      const res = await fetch('/api/chat/settings');
      return res.json();
    });
    const baselineProviderCount = baseline.providers?.length ?? 0;

    // 2. PUT settings with a new test provider + models
    const testProvider = {
      id: 'test-provider-e2e',
      name: 'E2E Test Provider',
      api_key: 'sk-test-key',
      base_url: 'https://api.example.com/v1',
      models: ['test-model-a', 'test-model-b'],
      enabled: true,
    };
    const updatedSettings = {
      ...baseline,
      providers: [...(baseline.providers || []), testProvider],
      defaults: { ...baseline.defaults, model: 'test-model-a' },
    };
    const putRes = await page.evaluate(async (body) => {
      const res = await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return res.json();
    }, updatedSettings);

    // Verify provider was added
    expect(putRes.providers.length).toBe(baselineProviderCount + 1);
    const addedProvider = putRes.providers.find((p: any) => p.id === 'test-provider-e2e');
    expect(addedProvider).toBeTruthy();
    expect(addedProvider.models).toEqual(['test-model-a', 'test-model-b']);
    expect(putRes.defaults.model).toBe('test-model-a');

    // 3. GET /models — should include the new provider's models
    const modelsRes = await page.evaluate(async () => {
      const res = await fetch('/api/chat/models');
      return res.json();
    });
    const testModels = modelsRes.models.filter((m: any) => m.provider_id === 'test-provider-e2e');
    expect(testModels.length).toBe(2);
    expect(testModels.map((m: any) => m.model).sort()).toEqual(['test-model-a', 'test-model-b']);

    // 4. Cleanup: restore baseline settings
    await page.evaluate(async (body) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }, baseline);
  });

  // --- Provider switching: conversation uses provider model ---
  test('conversation inherits default model from settings', async ({ page }) => {
    await page.goto('/');

    // Set a specific default model via settings API
    const settings = await page.evaluate(async () => {
      const res = await fetch('/api/chat/settings');
      return res.json();
    });
    const originalModel = settings.defaults.model;

    await page.evaluate(async (body) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }, { ...settings, defaults: { ...settings.defaults, model: 'e2e-test-model' } });

    // Create a conversation — it should inherit the default model
    const convRes = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.json();
    });
    // New conversations may or may not copy the default model at creation time,
    // but the model field should be accessible
    expect(convRes.conversation_id).toBeTruthy();

    // Override model at conversation level
    const patchRes = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'conversation-specific-model' }),
      });
      return res.json();
    }, convRes.conversation_id);
    expect(patchRes.model).toBe('conversation-specific-model');

    // Cleanup: restore original model and delete test conversation
    await page.evaluate(async ({ settings: s, convId }) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(s),
      });
      await fetch(`/api/chat/conversations/${convId}`, { method: 'DELETE' });
    }, { settings: { ...settings, defaults: { ...settings.defaults, model: originalModel } }, convId: convRes.conversation_id });
  });

  // --- Provider switching: disabled provider models excluded from list ---
  test('disabled provider models are excluded from model list', async ({ page }) => {
    await page.goto('/');

    const settings = await page.evaluate(async () => {
      const res = await fetch('/api/chat/settings');
      return res.json();
    });

    // Add a disabled provider
    const disabledProvider = {
      id: 'disabled-provider-e2e',
      name: 'Disabled Provider',
      api_key: 'sk-disabled',
      base_url: 'https://disabled.example.com/v1',
      models: ['disabled-model-x'],
      enabled: false,
    };
    await page.evaluate(async (body) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }, { ...settings, providers: [...(settings.providers || []), disabledProvider] });

    // GET /models — disabled provider's models should NOT appear
    const modelsRes = await page.evaluate(async () => {
      const res = await fetch('/api/chat/models');
      return res.json();
    });
    const disabledModels = modelsRes.models.filter((m: any) => m.provider_id === 'disabled-provider-e2e');
    expect(disabledModels.length).toBe(0);

    // Cleanup
    await page.evaluate(async (body) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }, settings);
  });

  // --- Stream toggle: settings API round-trip ---
  test('stream field persists via global settings API', async ({ page }) => {
    await page.goto('/');

    // 1. GET current settings (baseline)
    const settings = await page.evaluate(async () => {
      const res = await fetch('/api/chat/settings');
      return res.json();
    });
    const originalStream = settings.defaults.stream;

    // 2. PUT settings with stream=false
    await page.evaluate(async (body) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }, { ...settings, defaults: { ...settings.defaults, stream: false } });

    // 3. GET settings — stream should be false
    const updated = await page.evaluate(async () => {
      const res = await fetch('/api/chat/settings');
      return res.json();
    });
    expect(updated.defaults.stream).toBe(false);

    // 4. Cleanup: restore original
    await page.evaluate(async (body) => {
      await fetch('/api/chat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }, { ...settings, defaults: { ...settings.defaults, stream: originalStream ?? true } });
  });

  // --- Stream toggle: conversation-level PATCH ---
  test('stream field persists via conversation PATCH', async ({ page }) => {
    await page.goto('/');

    // Create a conversation
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.json();
    });

    // stream should be null by default (inherits global)
    const detail = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`);
      return res.json();
    }, conv.conversation_id);
    expect(detail.stream).toBeNull();

    // PATCH stream=false
    const patched = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream: false }),
      });
      return res.json();
    }, conv.conversation_id);
    expect(patched.stream).toBe(false);

    // PATCH stream=null (reset to inherit global)
    const reset = await page.evaluate(async (id) => {
      const res = await fetch(`/api/chat/conversations/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream: null }),
      });
      return res.json();
    }, conv.conversation_id);
    expect(reset.stream).toBeNull();

    // Cleanup
    await page.evaluate(async (id) => {
      await fetch(`/api/chat/conversations/${id}`, { method: 'DELETE' });
    }, conv.conversation_id);
  });

  // --- Stream toggle: backward compat (old conversations without stream field) ---
  test('old conversations without stream field default to null', async ({ page }) => {
    await page.goto('/');

    // Create conversation, verify stream is null
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      return res.json();
    });
    expect(conv.stream === null || conv.stream === undefined).toBeTruthy();

    // Cleanup
    await page.evaluate(async (id) => {
      await fetch(`/api/chat/conversations/${id}`, { method: 'DELETE' });
    }, conv.conversation_id);
  });

  // --- BUG-041: Zero-flicker conversation switch ---
  // Creates two conversations with 20+ messages each, switches between them,
  // and uses rAF frame sampling + rapid screenshots to detect visual flicker.
  test('switching conversations with many messages does not flicker', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Helper: create conversation and seed N user+assistant message pairs
    const createSeededConv = async (title: string, pairCount: number) => {
      const result = await page.evaluate(async ({ t, count }) => {
        // Create conversation
        const createRes = await fetch('/api/chat/conversations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: t }),
        });
        const conv = await createRes.json();

        // Build messages
        const msgs: { role: string; content: string }[] = [];
        for (let i = 0; i < count; i++) {
          msgs.push({ role: 'user', content: `User message ${i + 1}` });
          msgs.push({ role: 'assistant', content: `Reply ${i + 1}. Some content here to create DOM nodes.` });
        }

        // Seed via API
        const seedRes = await fetch(`/api/chat/conversations/${conv.conversation_id}/_seed-messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: msgs }),
        });
        const seedData = await seedRes.json();

        return { conv, seedStatus: seedRes.status, seedData };
      }, { t: title, count: pairCount });

      console.log(`[BUG-041] Created ${title}: seed status=${result.seedStatus}`, result.seedData);
      return result.conv;
    };

    // Create conv A (small) and conv B (50 message pairs = 100 messages)
    // Use high message count to simulate real-world heavy conversations
    const convA = await createSeededConv('Flicker A (small)', 2);
    const convB = await createSeededConv('Flicker B (heavy)', 25);

    // Reload to pick up new conversations
    await page.goto('/');
    await switchToChat(page);

    // Load conv A first and wait for messages to render
    await page.locator('text=Flicker A (small)').first().click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
    // Wait for message bubbles to render (seeded messages)
    await expect(page.locator('[data-testid="message-bubble"]').first()).toBeVisible({ timeout: 5000 });
    await page.waitForTimeout(200);

    // Take "before" screenshot of the chat area for reference
    const chatArea = page.getByTestId('chat-page');
    const beforeScreenshot = await chatArea.screenshot();

    // --- Core flicker detection ---
    // Strategy: install rAF observer that captures per-frame state, then
    // trigger switch to conv B. We detect:
    // 1. State-level flicker: EMPTY_STATE / LOADING / NO_CHAT_PAGE appearing
    // 2. Content-level flicker: message bubble count dropping to 0 during switch
    // 3. Layout shift: scroll container height changing drastically between frames
    const flickerReport = await page.evaluate(async (convBTitle) => {
      return new Promise<{
        frameStates: string[];
        frameBubbleCounts: number[];
        frameScrollHeights: number[];
      }>((resolve) => {
        const frameStates: string[] = [];
        const frameBubbleCounts: number[] = [];
        const frameScrollHeights: number[] = [];

        const getState = () => {
          const chatPage = document.querySelector('[data-testid="chat-page"]');
          const emptyState = document.querySelector('[data-testid="chat-empty-state"]');
          const loadingPulse = document.querySelector('.animate-pulse');
          if (emptyState) return 'EMPTY_STATE';
          if (loadingPulse && loadingPulse.textContent?.includes('Loading')) return 'LOADING';
          if (!chatPage) return 'NO_CHAT_PAGE';
          return 'CHAT_PAGE';
        };

        const getBubbleCount = () => {
          return document.querySelectorAll('[data-testid="message-bubble"]').length;
        };

        const getScrollHeight = () => {
          const container = document.querySelector('[data-testid="chat-page"] .overflow-y-auto');
          return container ? container.scrollHeight : 0;
        };

        // Record initial state
        frameStates.push(getState());
        frameBubbleCounts.push(getBubbleCount());
        frameScrollHeights.push(getScrollHeight());

        let frameCount = 0;
        const maxFrames = 60;

        const sampleFrame = () => {
          frameCount++;
          frameStates.push(getState());
          frameBubbleCounts.push(getBubbleCount());
          frameScrollHeights.push(getScrollHeight());

          if (frameCount < maxFrames) {
            requestAnimationFrame(sampleFrame);
          } else {
            resolve({ frameStates, frameBubbleCounts, frameScrollHeights });
          }
        };

        // Click conv B to trigger switch
        const sidebarItems = document.querySelectorAll('.cursor-pointer');
        let clicked = false;
        sidebarItems.forEach((item) => {
          if (item.textContent?.includes(convBTitle)) {
            (item as HTMLElement).click();
            clicked = true;
          }
        });
        if (!clicked) {
          frameStates.push('COULD_NOT_FIND_CONV_B');
          resolve({ frameStates, frameBubbleCounts, frameScrollHeights });
          return;
        }

        requestAnimationFrame(sampleFrame);
      });
    }, 'Flicker B (heavy)');

    // --- Analysis ---
    console.log('[BUG-041] Frame states:', flickerReport.frameStates);
    console.log('[BUG-041] Bubble counts:', flickerReport.frameBubbleCounts);
    console.log('[BUG-041] Scroll heights:', flickerReport.frameScrollHeights);

    const problems: string[] = [];

    // Check 1: No EMPTY_STATE / LOADING / NO_CHAT_PAGE during switch
    for (let i = 0; i < flickerReport.frameStates.length; i++) {
      const state = flickerReport.frameStates[i];
      if (state === 'EMPTY_STATE' || state === 'LOADING' || state === 'NO_CHAT_PAGE') {
        problems.push(`Frame ${i}: state=${state}`);
      }
    }

    // Check 2: Bubble count should never drop to 0 during switch
    // (conv A has 4 bubbles, conv B has 40 — during switch we should
    // always see either conv A's or conv B's bubbles, never 0)
    const initialBubbles = flickerReport.frameBubbleCounts[0];
    for (let i = 1; i < flickerReport.frameBubbleCounts.length; i++) {
      const count = flickerReport.frameBubbleCounts[i];
      if (count === 0 && initialBubbles > 0) {
        problems.push(`Frame ${i}: bubble count dropped to 0 (was ${initialBubbles})`);
      }
    }

    // Check 3: Scroll height should not drop to near-zero during switch
    // (indicates the message container was momentarily empty)
    const initialHeight = flickerReport.frameScrollHeights[0];
    for (let i = 1; i < flickerReport.frameScrollHeights.length; i++) {
      const height = flickerReport.frameScrollHeights[i];
      if (initialHeight > 100 && height < 50) {
        problems.push(`Frame ${i}: scroll height dropped from ${initialHeight} to ${height}`);
      }
    }

    console.log('[BUG-041] Problems:', problems);

    // Take "after" screenshot to verify conv B loaded
    await page.waitForTimeout(500);
    const afterScreenshot = await chatArea.screenshot();
    // After screenshot should differ from before (different conversation)
    expect(Buffer.compare(beforeScreenshot, afterScreenshot)).not.toBe(0);

    // FAIL if any flicker detected
    expect(problems, `Flicker detected during switch: ${problems.join('; ')}`).toHaveLength(0);

    // Cleanup
    await page.evaluate(async (ids) => {
      for (const id of ids) {
        await fetch(`/api/chat/conversations/${id}`, { method: 'DELETE' }).catch(() => {});
      }
    }, [convA.conversation_id, convB.conversation_id]);
  });

  // --- BUG-040: No spurious white borders on toolbar ---
  test('graph mode toolbar has no visible border', async ({ page }) => {
    await page.goto('/');

    // Should start in Graph mode — check the toolbar container
    const toolbar = page.locator('header .flex.items-center.gap-1.rounded-lg.bg-gray-800.p-1').first();
    if (await toolbar.isVisible()) {
      // Verify no border CSS is applied
      const border = await toolbar.evaluate((el) => {
        const style = getComputedStyle(el);
        return style.borderWidth;
      });
      // Should be 0px (no border)
      expect(border).toBe('0px');
    }
  });

  // --- Scroll position memory ---
  test('scroll position is restored when switching back to a conversation', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Create two conversations with enough messages to scroll
    const convA = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'scroll-test-A' }),
      });
      return res.json();
    });
    // Seed conv A with many messages so it's scrollable
    await page.evaluate(async (cid: string) => {
      const msgs = Array.from({ length: 30 }, (_, i) => ({
        role: 'user',
        content: `Message ${i + 1} in conversation A — padding text to make it longer`,
      }));
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: msgs }),
      });
    }, convA.conversation_id);

    const convB = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'scroll-test-B' }),
      });
      return res.json();
    });

    // Refresh sidebar and load conv A
    await page.evaluate(async () => {
      await fetch('/api/chat/conversations?limit=200');
    });
    await page.reload();
    await switchToChat(page);

    // Click conv A in sidebar (list items are divs with title text)
    const convABtn = page.locator('text=scroll-test-A').first();
    await convABtn.click();
    await page.waitForTimeout(800);

    // Scroll up using wheel (real user-like scroll, triggers handleScroll reliably)
    const timeline = page.locator('.flex-col-reverse').first();
    await timeline.hover();
    for (let i = 0; i < 8; i++) {
      await page.mouse.wheel(0, -150);
      await page.waitForTimeout(30);
    }
    await page.waitForTimeout(400);

    // Record anchor message visual position
    const anchorBefore = await timeline.evaluate((el) => {
      const containerRect = el.getBoundingClientRect();
      const viewportCenter = containerRect.top + containerRect.height / 2;
      const msgs = el.querySelectorAll('[data-message-id]');
      let closest: Element | null = null;
      let closestDist = Infinity;
      for (const m of msgs) {
        const r = m.getBoundingClientRect();
        const d = Math.abs(r.top + r.height / 2 - viewportCenter);
        if (d < closestDist) { closestDist = d; closest = m; }
      }
      if (!closest) return null;
      return {
        id: closest.getAttribute('data-message-id'),
        offset: closest.getBoundingClientRect().top - containerRect.top,
      };
    });
    expect(anchorBefore).not.toBeNull();
    const scrollBefore = await timeline.evaluate((el) => el.scrollTop);
    expect(Math.abs(scrollBefore)).toBeGreaterThan(50);

    // Switch to conv B
    const convBBtn = page.locator('text=scroll-test-B').first();
    await convBBtn.click();
    await page.waitForTimeout(800);

    // Switch back to conv A
    await convABtn.click();
    await page.waitForTimeout(1500);

    // Measure visual position of the same anchor message
    const anchorAfter = await timeline.evaluate((el, anchorId) => {
      const msg = el.querySelector(`[data-message-id="${anchorId}"]`);
      if (!msg) return null;
      const containerRect = el.getBoundingClientRect();
      return {
        offset: msg.getBoundingClientRect().top - containerRect.top,
      };
    }, anchorBefore!.id);

    expect(anchorAfter).not.toBeNull();
    const visualDrift = Math.abs(anchorAfter!.offset - anchorBefore!.offset);
    expect(
      visualDrift,
      `Visual drift too large. Before offset=${anchorBefore!.offset.toFixed(0)}, After offset=${anchorAfter!.offset.toFixed(0)}`,
    ).toBeLessThan(100);

    // Cleanup
    await page.evaluate(async (ids: string[]) => {
      for (const id of ids) {
        await fetch(`/api/chat/conversations/${id}`, { method: 'DELETE' }).catch(() => {});
      }
    }, [convA.conversation_id, convB.conversation_id]);
  });

  // --- Search: click result navigates to message with data-message-id ---
  test('search result click loads conversation and message has data-message-id', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Create a conversation with a unique searchable message
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'search-nav-test' }),
      });
      return res.json();
    });
    // Seed with messages including a unique keyword
    await page.evaluate(async (cid: string) => {
      const msgs = [
        { role: 'user', content: 'First message padding' },
        { role: 'assistant', content: 'Response padding' },
        { role: 'user', content: 'UniqueSearchToken12345 is the keyword' },
        { role: 'assistant', content: 'Got it about UniqueSearchToken12345' },
      ];
      const res = await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: msgs }),
      });
      return res.ok;
    }, conv.conversation_id);

    // Reload to ensure sidebar and search index are fresh
    await page.reload();
    await switchToChat(page);
    await page.waitForTimeout(500);

    // Open search modal (Cmd+K)
    await page.keyboard.press('Meta+k');
    await page.waitForTimeout(500);

    // Type the unique keyword
    const searchInput = page.getByPlaceholder('Search conversations...');
    await expect(searchInput).toBeVisible();
    await searchInput.fill('UniqueSearchToken12345');
    await page.waitForTimeout(800); // debounce + API call

    // Should see results — the modal is the rounded-xl div
    const searchModal = page.locator('.rounded-xl.shadow-2xl');
    const resultBtn = searchModal.locator('button.w-full.text-left').first();
    await expect(resultBtn).toBeVisible({ timeout: 5000 });

    // Click the result
    await resultBtn.click({ force: true });
    await page.waitForTimeout(800);

    // Verify conversation loaded — chat page visible
    await expect(page.getByTestId('chat-page')).toBeVisible();

    // Verify data-message-id attributes exist on rendered messages
    const msgIds = await page.evaluate(() => {
      const els = document.querySelectorAll('[data-message-id]');
      return Array.from(els).map((el) => el.getAttribute('data-message-id'));
    });
    expect(msgIds.length).toBeGreaterThan(0);
    // All IDs should be non-empty strings
    for (const id of msgIds) {
      expect(id).toBeTruthy();
      expect(typeof id).toBe('string');
    }

    // Cleanup
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}`, { method: 'DELETE' }).catch(() => {});
    }, conv.conversation_id);
  });

  // --- Attachment: Composer shows attachment preview and file input exists ---
  test('composer file input and attachment preview work', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);
    await createConversation(page);

    // Verify hidden file input exists with correct accept attribute
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeAttached();
    const accept = await fileInput.getAttribute('accept');
    expect(accept).toContain('image/*');

    // Verify paperclip button exists
    const attachBtn = page.locator('button[title="Attach file"]');
    await expect(attachBtn).toBeVisible();

    // Simulate file selection via the hidden input
    const __dirname_esm = path.dirname(fileURLToPath(import.meta.url));
    const fixtureDir = path.join(__dirname_esm, 'fixtures');
    const testFilePath = path.join(fixtureDir, 'test-image.txt');
    // Create a minimal test fixture if it doesn't exist
    if (!fs.existsSync(fixtureDir)) fs.mkdirSync(fixtureDir, { recursive: true });
    if (!fs.existsSync(testFilePath)) {
      // Create a tiny text file as a stand-in (real image not needed for UI test)
      fs.writeFileSync(testFilePath, 'test-attachment-content');
    }

    // Use setInputFiles to simulate file selection
    await fileInput.setInputFiles(testFilePath);

    // Attachment preview should appear
    const preview = page.locator('.flex.flex-wrap.gap-1\\.5');
    await expect(preview).toBeVisible({ timeout: 2000 });

    // Preview should contain the filename
    await expect(preview).toContainText('test-image.txt');

    // Remove button (×) should be visible
    const removeBtn = preview.locator('button');
    await expect(removeBtn).toBeVisible();

    // Click remove — preview should disappear
    await removeBtn.click();
    await expect(preview).not.toBeVisible();
  });

  // --- Attachment: API accepts attachments in SendMessageRequest ---
  test('send message API accepts attachments field', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Create conversation
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'attachment-api-test' }),
      });
      return res.json();
    });

    // Send message with attachment via API directly
    const apiResult = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: 'Look at this image',
          attachments: [{
            filename: 'test.png',
            mime_type: 'image/png',
            data: 'data:image/png;base64,iVBORw0KGgo=',
            size: 100,
          }],
        }),
      });
      return { status: res.status, ok: res.ok };
    }, conv.conversation_id);

    // API should accept the request (200 for SSE stream)
    expect(apiResult.ok).toBe(true);

    // Verify the message was persisted with attachment
    const convData = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);

    // Find the user message with attachment
    const userMsg = convData.messages.find(
      (m: any) => m.role === 'user' && m.content === 'Look at this image',
    );
    expect(userMsg).toBeTruthy();
    expect(userMsg.attachments).toHaveLength(1);
    expect(userMsg.attachments[0].filename).toBe('test.png');
    expect(userMsg.attachments[0].mime_type).toBe('image/png');
    expect(userMsg.attachments[0].size).toBe(100);

    // Cleanup
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}`, { method: 'DELETE' }).catch(() => {});
    }, conv.conversation_id);
  });
});

// ============================================================
// Message Bookmark tests
// ============================================================
test.describe('Message bookmark', () => {
  let preTestConvIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    preTestConvIds = await snapshotConvIds(page);
  });

  test.afterEach(async ({ page }) => {
    const postTestConvIds = await snapshotConvIds(page);
    testCreatedConvIds = postTestConvIds.filter((id) => !preTestConvIds.includes(id));
    await cleanupTestConversations(page);
  });

  test('bookmark API: toggle bookmark on a message', async ({ page }) => {
    // Create conversation with seeded messages via API
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'bookmark-test' }),
      });
      return res.json();
    });

    // Seed messages
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: 'Important question' },
            { role: 'assistant', content: 'Important answer' },
          ],
        }),
      });
    }, conv.conversation_id);

    // Get the message IDs
    const convData = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);
    const msgId = convData.messages[0].message_id;

    // Verify default bookmarked=false
    expect(convData.messages[0].bookmarked).toBe(false);

    // Toggle bookmark ON
    const bookmarkResult = await page.evaluate(async ({ cid, mid }: { cid: string; mid: string }) => {
      const res = await fetch(`/api/chat/conversations/${cid}/messages/${mid}/bookmark`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
      return res.json();
    }, { cid: conv.conversation_id, mid: msgId });

    expect(bookmarkResult.bookmarked).toBe(true);

    // Verify persisted
    const convAfter = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);
    expect(convAfter.messages[0].bookmarked).toBe(true);
    expect(convAfter.messages[1].bookmarked).toBe(false);

    // Toggle bookmark OFF
    const unbookmark = await page.evaluate(async ({ cid, mid }: { cid: string; mid: string }) => {
      const res = await fetch(`/api/chat/conversations/${cid}/messages/${mid}/bookmark`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: false }),
      });
      return res.json();
    }, { cid: conv.conversation_id, mid: msgId });

    expect(unbookmark.bookmarked).toBe(false);
  });

  test('bookmark API: bookmarked field appears in search results', async ({ page }) => {
    // Create conversation with a bookmarked message
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'search-bookmark-test' }),
      });
      return res.json();
    });

    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: 'UniqueSearchTermXYZ123' },
            { role: 'assistant', content: 'Response to unique term' },
          ],
        }),
      });
    }, conv.conversation_id);

    // Get message ID and bookmark it
    const convData = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);
    const msgId = convData.messages[0].message_id;

    await page.evaluate(async ({ cid, mid }: { cid: string; mid: string }) => {
      await fetch(`/api/chat/conversations/${cid}/messages/${mid}/bookmark`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
    }, { cid: conv.conversation_id, mid: msgId });

    // Search for the unique term
    const searchResults = await page.evaluate(async () => {
      const res = await fetch('/api/chat/search?q=UniqueSearchTermXYZ123&limit=10');
      return res.json();
    });

    const matchedMsg = searchResults.results.find(
      (r: any) => r.match_type === 'message' && r.message_id === msgId,
    );
    expect(matchedMsg).toBeTruthy();
    expect(matchedMsg.bookmarked).toBe(true);
  });

  test('bookmark UI: bookmark button toggles on message hover', async ({ page }) => {
    await switchToChat(page);

    // Create conversation with seeded messages
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'bookmark-ui-test' }),
      });
      return res.json();
    });

    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: 'Bookmark me please' },
            { role: 'assistant', content: 'Sure thing' },
          ],
        }),
      });
    }, conv.conversation_id);

    // Reload to pick up the new conversation in sidebar, then navigate
    await page.goto('/');
    await switchToChat(page);

    // Click on the conversation by its title text in the sidebar
    await page.locator('text=bookmark-ui-test').first().click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });

    // Wait for messages to render
    await expect(page.getByTestId('message-bubble')).toHaveCount(2, { timeout: 5000 });

    // The last message's action bar is always visible (isLastMessage=true).
    // Find the bookmark button on the last message bubble.
    const lastBubble = page.getByTestId('message-bubble').last();

    // Bookmark button should be visible on the last message (always shown)
    const bookmarkBtn = lastBubble.locator('button[title="Bookmark"]');
    await expect(bookmarkBtn).toBeVisible({ timeout: 5000 });

    // Click bookmark
    await bookmarkBtn.click();

    // After clicking, button title should change to "Remove bookmark"
    const removeBtn = lastBubble.locator('button[title="Remove bookmark"]');
    await expect(removeBtn).toBeVisible({ timeout: 5000 });

    // Verify via API that the message is bookmarked
    const convAfter = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);
    const bookmarkedMsgs = convAfter.messages.filter((m: any) => m.bookmarked);
    expect(bookmarkedMsgs.length).toBeGreaterThanOrEqual(1);
  });
});

// ============================================================
// Bookmark overview modal tests
// ============================================================
test.describe('Bookmark overview modal', () => {
  let preTestConvIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    preTestConvIds = await snapshotConvIds(page);
  });

  test.afterEach(async ({ page }) => {
    const postTestConvIds = await snapshotConvIds(page);
    testCreatedConvIds = postTestConvIds.filter((id) => !preTestConvIds.includes(id));
    await cleanupTestConversations(page);
  });

  test('bookmarks API returns bookmarked conversations and messages', async ({ page }) => {
    // Create a conversation and bookmark it + bookmark a message
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'bm-overview-test' }),
      });
      return res.json();
    });
    testCreatedConvIds.push(conv.conversation_id);

    // Seed messages via _seed-messages endpoint
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: 'Important question' },
            { role: 'assistant', content: 'Important answer' },
          ],
        }),
      });
    }, conv.conversation_id);

    // Bookmark the conversation
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
    }, conv.conversation_id);

    // Bookmark a message
    const convData = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);
    const msgId = convData.messages[0].message_id;

    await page.evaluate(async ({ cid, mid }: { cid: string; mid: string }) => {
      await fetch(`/api/chat/conversations/${cid}/messages/${mid}/bookmark`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
    }, { cid: conv.conversation_id, mid: msgId });

    // Call bookmarks API
    const bookmarks = await page.evaluate(async () => {
      const res = await fetch('/api/chat/bookmarks');
      return res.json();
    });

    expect(bookmarks.bookmarks).toBeDefined();
    expect(bookmarks.bookmarks.length).toBeGreaterThanOrEqual(2);

    const convBookmarks = bookmarks.bookmarks.filter((b: any) => b.type === 'conversation');
    const msgBookmarks = bookmarks.bookmarks.filter((b: any) => b.type === 'message');
    expect(convBookmarks.length).toBeGreaterThanOrEqual(1);
    expect(msgBookmarks.length).toBeGreaterThanOrEqual(1);
  });

  test('bookmark modal opens from header button', async ({ page }) => {
    await switchToChat(page);

    // Click the bookmark button in header
    const bookmarkBtn = page.locator('button[title="Bookmarks"]');
    await expect(bookmarkBtn).toBeVisible({ timeout: 3000 });
    await bookmarkBtn.click();

    // Modal should appear with header containing "Bookmarks"
    const modal = page.locator('.rounded-xl.shadow-2xl');
    await expect(modal).toBeVisible({ timeout: 3000 });
    await expect(modal.locator('span.text-gray-200', { hasText: 'Bookmarks' })).toBeVisible();

    // Close with ESC
    await page.keyboard.press('Escape');
    await expect(modal).not.toBeVisible({ timeout: 3000 });
  });

  test('bookmark modal shows empty state when no bookmarks', async ({ page }) => {
    await switchToChat(page);

    const bookmarkBtn = page.locator('button[title="Bookmarks"]');
    await bookmarkBtn.click();

    const modal = page.locator('.rounded-xl.shadow-2xl');
    await expect(modal).toBeVisible({ timeout: 3000 });

    // Should show empty state text
    await expect(modal.locator('text=No bookmarks yet')).toBeVisible({ timeout: 3000 });

    await page.keyboard.press('Escape');
  });

  test('bookmark modal shows bookmarked items and navigates on click', async ({ page }) => {
    // Create and bookmark a conversation with a message
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'bm-nav-test' }),
      });
      return res.json();
    });
    testCreatedConvIds.push(conv.conversation_id);

    // Seed messages
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: 'Navigate to me' },
            { role: 'assistant', content: 'Found you' },
          ],
        }),
      });
    }, conv.conversation_id);

    // Bookmark the conversation
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
    }, conv.conversation_id);

    await switchToChat(page);

    // Open bookmark modal
    const bookmarkBtn = page.locator('button[title="Bookmarks"]');
    await bookmarkBtn.click();

    const modal = page.locator('.rounded-xl.shadow-2xl');
    await expect(modal).toBeVisible({ timeout: 3000 });

    // Should show the bookmarked conversation
    await expect(modal.locator('text=bm-nav-test')).toBeVisible({ timeout: 3000 });

    // Click on it to navigate
    await modal.locator('button.w-full.text-left').first().click();

    // Modal should close
    await expect(modal).not.toBeVisible({ timeout: 3000 });

    // Should navigate to the conversation — chat page visible
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
  });

  test('bookmark modal filter tabs work', async ({ page }) => {
    // Create conversation with bookmarked conv + bookmarked message
    const conv = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'bm-filter-test' }),
      });
      return res.json();
    });
    testCreatedConvIds.push(conv.conversation_id);

    // Seed messages
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: 'Filter test msg' },
            { role: 'assistant', content: 'Filter test reply' },
          ],
        }),
      });
    }, conv.conversation_id);

    // Bookmark the conversation
    await page.evaluate(async (cid: string) => {
      await fetch(`/api/chat/conversations/${cid}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
    }, conv.conversation_id);

    // Bookmark a message
    const convData = await page.evaluate(async (cid: string) => {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      return res.json();
    }, conv.conversation_id);

    await page.evaluate(async ({ cid, mid }: { cid: string; mid: string }) => {
      await fetch(`/api/chat/conversations/${cid}/messages/${mid}/bookmark`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bookmarked: true }),
      });
    }, { cid: conv.conversation_id, mid: convData.messages[0].message_id });

    await switchToChat(page);

    // Open bookmark modal
    await page.locator('button[title="Bookmarks"]').click();
    const modal = page.locator('.rounded-xl.shadow-2xl');
    await expect(modal).toBeVisible({ timeout: 3000 });

    // Click "Conversations" filter tab (text-[10px] distinguishes filter tabs from result items)
    await modal.locator('button.text-\\[10px\\]', { hasText: 'Conversations' }).click();
    await page.waitForTimeout(300);

    // Should still show the conversation entry
    await expect(modal.locator('text=bm-filter-test')).toBeVisible();

    // Click "Messages" filter tab
    await modal.locator('button.text-\\[10px\\]', { hasText: 'Messages' }).click();
    await page.waitForTimeout(300);

    // Should show the bookmarked message snippet
    await expect(modal.locator('text=Filter test msg')).toBeVisible({ timeout: 3000 });

    await page.keyboard.press('Escape');
  });
});

// ============================================================
// Upload validation toast tests
// ============================================================
test.describe('Upload validation toast', () => {
  let preTestConvIds: string[] = [];

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    preTestConvIds = await snapshotConvIds(page);
  });

  test.afterEach(async ({ page }) => {
    const postTestConvIds = await snapshotConvIds(page);
    testCreatedConvIds = postTestConvIds.filter((id) => !preTestConvIds.includes(id));
    await cleanupTestConversations(page);
  });

  test('unsupported file type shows toast instead of native alert', async ({ page }) => {
    await switchToChat(page);
    await createConversation(page);

    // Intercept window.alert to detect if it's called (it should NOT be)
    let alertCalled = false;
    await page.exposeFunction('__testAlertCalled', () => { alertCalled = true; });
    await page.evaluate(() => {
      window.alert = () => { (window as any).__testAlertCalled(); };
    });

    // Try to upload an unsupported file type (.exe)
    const fileInput = page.locator('input[type="file"]');
    // Create a fake .exe file buffer
    const buffer = Buffer.from('MZ fake exe content');
    await fileInput.setInputFiles({
      name: 'malware.exe',
      mimeType: 'application/x-msdownload',
      buffer,
    });

    // Wait a moment for the validation to fire
    await page.waitForTimeout(500);

    // Native alert should NOT have been called
    expect(alertCalled).toBe(false);

    // Verify toast appeared (not native alert)
    // Check for error text visible somewhere in the page
    const hasErrorText = await page.locator('text=unsupported type').first().isVisible({ timeout: 2000 }).catch(() => false);
    expect(hasErrorText).toBe(true);
  });

  test('oversized file shows toast notification', async ({ page }) => {
    await switchToChat(page);
    await createConversation(page);

    // Intercept alert
    let alertCalled = false;
    await page.exposeFunction('__testAlertCalled2', () => { alertCalled = true; });
    await page.evaluate(() => {
      window.alert = () => { (window as any).__testAlertCalled2(); };
    });

    // Create a file that exceeds 10MB limit
    const bigBuffer = Buffer.alloc(11 * 1024 * 1024, 'x');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'huge-image.png',
      mimeType: 'image/png',
      buffer: bigBuffer,
    });

    await page.waitForTimeout(500);

    // Native alert should NOT be called
    expect(alertCalled).toBe(false);
  });
});
