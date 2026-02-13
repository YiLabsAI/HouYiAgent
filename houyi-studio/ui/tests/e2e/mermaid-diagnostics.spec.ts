/**
 * Mermaid rendering regression tests (baseline).
 *
 * Coverage:
 * - Node-level assertions: every expected label must appear in the SVG
 * - Font-size: diagram text ≤ 15px (matches body text, no bloat)
 * - SVG normalization: viewBox present, no fixed width/height, no upscaling
 * - Sanitization: diagrams with parentheses / special edge labels render
 * - Subgraphs: complex multi-node diagrams render all nodes
 * - Scroll position preserved across conversation switches
 * - No large jitter/flicker on Mermaid conversation switch
 */
import { test, expect } from '@playwright/test';

const switchToChat = async (page: import('@playwright/test').Page) => {
  const chatBtn = page.locator('button:has-text("Chat"), [data-testid="mode-chat"]').first();
  if (await chatBtn.isVisible()) {
    await chatBtn.click();
  }
  await expect(page.getByTestId('chat-page').or(page.getByTestId('chat-empty-state'))).toBeVisible({ timeout: 5000 });
};

const testCreatedConvIds: string[] = [];

// Helper: create a conversation, seed with mermaid code, navigate & wait for render.
// Returns info about the rendered SVG for assertions.
async function seedMermaidAndRender(
  page: import('@playwright/test').Page,
  title: string,
  mermaidCode: string,
) {
  // Must be on a page with the app origin so relative fetch URLs work
  await page.goto('/');
  await switchToChat(page);

  const conv = await page.evaluate(async (t: string) => {
    const res = await fetch('/api/chat/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: t }),
    });
    return res.json();
  }, title);
  testCreatedConvIds.push(conv.conversation_id);

  await page.evaluate(async ({ cid, code }: { cid: string; code: string }) => {
    const messages = [
      { role: 'user', content: 'Draw a diagram' },
      { role: 'assistant', content: `Here you go:\n\n\`\`\`mermaid\n${code}\n\`\`\`` },
    ];
    await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages }),
    });
  }, { cid: conv.conversation_id, code: mermaidCode });

  await page.goto('/');
  await switchToChat(page);
  await page.locator(`text=${title}`).first().click();
  await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });

  // Wait for Mermaid render (500ms debounce + render)
  const svgLocator = page.locator('svg[id^="mermaid"]').first();
  await expect(svgLocator).toBeVisible({ timeout: 15000 });

  return { conv, svgLocator };
}

// Helper: extract all text labels and measurements from the rendered Mermaid SVG.
async function measureMermaidSvg(page: import('@playwright/test').Page) {
  return page.evaluate(() => {
    const svg = document.querySelector('svg[id^="mermaid"]') as SVGElement;
    if (!svg) return null;

    const container = svg.closest('.bg-gray-950') as HTMLElement;
    const containerRect = container
      ? container.getBoundingClientRect()
      : svg.parentElement!.getBoundingClientRect();
    const svgRect = svg.getBoundingClientRect();
    const viewBox = svg.getAttribute('viewBox');
    const svgStyle = svg.getAttribute('style') || '';

    // Collect all visible text from foreignObject spans AND SVG <text> elements
    const spans = svg.querySelectorAll('foreignObject span');
    const textEls = svg.querySelectorAll('text');
    const labels: string[] = [];
    const fontSizes: number[] = [];

    spans.forEach((s) => {
      const txt = (s as HTMLElement).textContent?.trim();
      if (txt) labels.push(txt);
      const fs = parseFloat(window.getComputedStyle(s).fontSize);
      if (fs > 0) fontSizes.push(fs);
    });
    textEls.forEach((t) => {
      const txt = t.textContent?.trim();
      if (txt) labels.push(txt);
      const fs = parseFloat(window.getComputedStyle(t).fontSize);
      if (fs > 0) fontSizes.push(fs);
    });

    // Check SVG normalization
    const hasViewBox = !!viewBox;
    const hasFixedWidth = svg.hasAttribute('width');
    const hasFixedHeight = svg.hasAttribute('height');

    return {
      labels,
      fontSizes,
      maxFontSize: fontSizes.length > 0 ? Math.max(...fontSizes) : 0,
      avgFontSize: fontSizes.length > 0 ? fontSizes.reduce((a, b) => a + b, 0) / fontSizes.length : 0,
      containerWidth: containerRect.width,
      svgRenderedWidth: svgRect.width,
      svgRenderedHeight: svgRect.height,
      viewBox,
      svgStyle,
      hasViewBox,
      hasFixedWidth,
      hasFixedHeight,
      spanCount: spans.length,
      textElCount: textEls.length,
    };
  });
}

test.afterEach(async ({ page }) => {
  for (const cid of testCreatedConvIds) {
    try {
      await page.evaluate(async (id: string) => {
        await fetch(`/api/chat/conversations/${id}`, { method: 'DELETE' });
      }, cid);
    } catch {}
  }
  testCreatedConvIds.length = 0;
});

test.describe('Mermaid Diagnostics', () => {
  // ── 1. Simple flowchart: node labels + font sizes + SVG normalization ──
  test('simple flowchart renders all nodes with correct font size', async ({ page }) => {
    const code = `flowchart TD
  A[Start] --> B{Condition}
  B -->|Yes| C[Action A]
  B -->|No| D[Action B]
  C --> E[End]
  D --> E`;

    await seedMermaidAndRender(page, 'mermaid-simple', code);
    const m = await measureMermaidSvg(page);
    console.log('[simple-flowchart]', JSON.stringify(m, null, 2));

    expect(m).not.toBeNull();
    // ── Element assertions: every expected label must be present ──
    for (const label of ['Start', 'Condition', 'Action A', 'Action B', 'End']) {
      expect(m!.labels, `Missing label: ${label}`).toContain(label);
    }
    // Edge labels
    expect(m!.labels.some((l) => l.includes('Yes'))).toBe(true);
    expect(m!.labels.some((l) => l.includes('No'))).toBe(true);

    // ── Font-size: must find text, max ≤ 15px (body text is ~14px) ──
    expect(m!.spanCount + m!.textElCount).toBeGreaterThan(0);
    expect(m!.maxFontSize).toBeLessThanOrEqual(15);

    // ── SVG normalization: viewBox present, no fixed width/height, no upscaling ──
    expect(m!.hasViewBox).toBe(true);
    expect(m!.hasFixedWidth).toBe(false);
    expect(m!.hasFixedHeight).toBe(false);
    expect(m!.svgRenderedWidth).toBeLessThanOrEqual(m!.containerWidth + 2);
  });

  // ── 2. Subgraph with mixed text ──
  test('subgraph diagram renders all nodes including subgraph title', async ({ page }) => {
    const code = `flowchart TD
    subgraph sandbox["Secure Sandbox"]
        exec["Execute LLM Call"]
        tool["Invoke Tools/Functions"]
    end
    dispatch["Re-dispatch"] --> sandbox
    sandbox --> validate{Validate Result}
    validate -->|Pass| output["Output"]
    validate -->|Fail| dispatch`;

    await seedMermaidAndRender(page, 'mermaid-subgraph', code);
    const m = await measureMermaidSvg(page);
    console.log('[subgraph]', JSON.stringify(m, null, 2));

    expect(m).not.toBeNull();
    // ── Element assertions ──
    for (const label of ['Secure Sandbox', 'Execute LLM Call', 'Invoke Tools/Functions', 'Re-dispatch', 'Validate Result', 'Output']) {
      expect(m!.labels, `Missing label: ${label}`).toContain(label);
    }
    expect(m!.labels.some((l) => l.includes('Pass'))).toBe(true);
    expect(m!.labels.some((l) => l.includes('Fail'))).toBe(true);

    // ── Font & sizing ──
    expect(m!.maxFontSize).toBeLessThanOrEqual(15);
    expect(m!.svgRenderedWidth).toBeLessThanOrEqual(m!.containerWidth + 2);
  });

  // ── 3. Sanitization: parentheses in node labels ──
  test('diagram with parentheses in labels renders correctly', async ({ page }) => {
    const code = `flowchart LR
    A[LLM Service (OpenAI, Claude etc)] --> B[Message Queue (Kafka)]
    B --> C[Analytics Engine]`;

    await seedMermaidAndRender(page, 'mermaid-parens', code);
    const m = await measureMermaidSvg(page);
    console.log('[parens]', JSON.stringify(m, null, 2));

    expect(m).not.toBeNull();
    // Should render (not show "Diagram rendering failed")
    expect(m!.labels.length).toBeGreaterThan(0);
    // Check for key text (sanitization may quote labels, so check substrings)
    expect(m!.labels.some((l) => l.includes('LLM') || l.includes('OpenAI'))).toBe(true);
    expect(m!.labels.some((l) => l.includes('Kafka') || l.includes('Message Queue'))).toBe(true);
    expect(m!.labels.some((l) => l.includes('Analytics Engine'))).toBe(true);
  });

  // ── 4. No-upscale: narrow diagram must NOT stretch to fill container ──
  test('narrow diagram is not upscaled beyond its viewBox width', async ({ page }) => {
    // A simple 2-node LR diagram produces a narrow SVG (~250-400px viewBox width).
    // It must NOT be stretched to the full container width (~700px).
    const code = `flowchart LR
    A[Input] --> B[Output]`;

    await seedMermaidAndRender(page, 'mermaid-narrow', code);
    const m = await measureMermaidSvg(page);
    console.log('[narrow-no-upscale]', JSON.stringify(m, null, 2));

    expect(m).not.toBeNull();
    expect(m!.labels).toContain('Input');
    expect(m!.labels).toContain('Output');

    // Parse viewBox width from the SVG
    const vbParts = m!.viewBox?.split(/[\s,]+/).map(Number);
    const viewBoxW = vbParts && vbParts.length >= 3 ? vbParts[2] : 0;
    expect(viewBoxW).toBeGreaterThan(0);

    // Core assertion: rendered width must NOT exceed viewBox width (no upscaling).
    // Allow small tolerance for sub-pixel rounding.
    expect(m!.svgRenderedWidth).toBeLessThanOrEqual(viewBoxW + 2);

    // If container is wider than viewBox, SVG should be significantly smaller
    if (m!.containerWidth > viewBoxW + 50) {
      expect(m!.svgRenderedWidth).toBeLessThan(m!.containerWidth * 0.85);
    }

    // SVG style should include max-width cap
    expect(m!.svgStyle).toContain('max-width');
  });

  // ── 5. Complex architecture diagram with subgraphs + feedback loops ──
  // Mirrors a complex conversation's diagram structure: many nodes,
  // subgraph, feedback edges. ALL nodes must render (no content loss).
  test('complex architecture diagram renders all nodes completely', async ({ page }) => {
    const code = `flowchart TD
    User["User Request"] --> Router["Request Router"]
    Router --> LLM["LLM Inference Engine"]
    LLM --> ToolCall{Tool Call?}
    ToolCall -->|Yes| subgraph1
    ToolCall -->|No| Response["Generate Response"]

    subgraph subgraph1["Secure Sandbox"]
        Exec["Execute Code"]
        Fetch["Network Request"]
        FileOp["File Operation"]
    end

    subgraph1 --> Validate{Validate Result}
    Validate -->|Pass| Response
    Validate -->|Fail| Retry["Retry Strategy"]
    Retry --> LLM
    Response --> User`;

    await seedMermaidAndRender(page, 'mermaid-complex-arch', code);
    const m = await measureMermaidSvg(page);
    console.log('[complex-arch]', JSON.stringify(m, null, 2));

    expect(m).not.toBeNull();

    // ALL expected node labels must be present — no content loss from normalization
    const expectedLabels = [
      'User Request', 'Request Router', 'LLM Inference Engine', 'Tool Call?',
      'Secure Sandbox', 'Execute Code', 'Network Request', 'File Operation',
      'Validate Result', 'Generate Response', 'Retry Strategy',
    ];
    for (const label of expectedLabels) {
      expect(m!.labels.some((l) => l.includes(label)), `Missing node: "${label}"`).toBe(true);
    }
    // Edge labels
    expect(m!.labels.some((l) => l.includes('Pass'))).toBe(true);
    expect(m!.labels.some((l) => l.includes('Fail'))).toBe(true);

    // Total label count must be high — detects wholesale content loss
    expect(m!.labels.length).toBeGreaterThanOrEqual(12);

    // Font & normalization
    expect(m!.maxFontSize).toBeLessThanOrEqual(15);
    expect(m!.hasViewBox).toBe(true);
    expect(m!.hasFixedWidth).toBe(false);
  });

  // ── 6. Edge labels with dots and special chars (common failure) ──
  test('diagram with special edge labels renders', async ({ page }) => {
    const code = `flowchart TD
    AppServer["App Server"] -->|"2. Call Track API"| Client["Client SDK"]
    Client -->|"3. Async Batch Send"| MQ["Message Queue"]
    MQ --> Processor["Processing Engine"]`;

    await seedMermaidAndRender(page, 'mermaid-edge-labels', code);
    const m = await measureMermaidSvg(page);
    console.log('[edge-labels]', JSON.stringify(m, null, 2));

    expect(m).not.toBeNull();
    expect(m!.labels.length).toBeGreaterThan(0);
    expect(m!.labels.some((l) => l.includes('App Server'))).toBe(true);
    expect(m!.labels.some((l) => l.includes('Processing Engine'))).toBe(true);
  });

  test('scroll position preserved across 10 conversation switches', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    // Create two conversations
    const convA = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'scroll-stress-A' }),
      });
      return res.json();
    });

    const convB = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'scroll-stress-B' }),
      });
      return res.json();
    });

    testCreatedConvIds.push(convA.conversation_id, convB.conversation_id);

    // Seed A with many messages including Mermaid
    await page.evaluate(async (cid: string) => {
      const messages = Array.from({ length: 40 }).flatMap((_, i) => {
        const idx = i + 1;
        const mermaid = idx % 8 === 0
          ? `\n\n\`\`\`mermaid\nflowchart LR\n  X${idx}-->Y${idx}\n\`\`\`\n`
          : '';
        return [
          { role: 'user', content: `Question ${idx}` },
          { role: 'assistant', content: `Answer ${idx}${mermaid}` },
        ];
      });
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, convA.conversation_id);

    // Seed B with few messages
    await page.evaluate(async (cid: string) => {
      const messages = [
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello!' },
      ];
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, convB.conversation_id);

    await page.goto('/');
    await switchToChat(page);

    // Open conversation A
    await page.locator('text=scroll-stress-A').first().click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
    await expect(page.getByTestId('message-bubble')).toHaveCount(80, { timeout: 10000 });

    // Wait for Mermaid diagrams to render (they have 500ms debounce + render time)
    await page.waitForTimeout(3000);

    const scroller = page.getByTestId('chat-timeline');
    await expect(scroller).toBeVisible();

    // Scroll to a specific position using wheel
    await scroller.hover();
    for (let i = 0; i < 30; i++) {
      await page.mouse.wheel(0, -150);
      await page.waitForTimeout(30);
    }
    // Wait for scroll to settle and snapshot to be saved
    await page.waitForTimeout(500);

    // Find a message near the center of viewport and measure its visual position
    const anchorInfo = await scroller.evaluate((el) => {
      const bubbles = el.querySelectorAll('[data-testid="message-bubble"]');
      const containerRect = el.getBoundingClientRect();
      const viewportCenter = containerRect.top + containerRect.height / 2;
      let closest: Element | null = null;
      let closestDist = Infinity;
      for (const b of bubbles) {
        const r = b.getBoundingClientRect();
        const dist = Math.abs(r.top + r.height / 2 - viewportCenter);
        if (dist < closestDist) {
          closestDist = dist;
          closest = b;
        }
      }
      if (!closest) return null;
      const idx = Array.from(bubbles).indexOf(closest);
      const rect = closest.getBoundingClientRect();
      return {
        messageIndex: idx,
        offsetFromContainerTop: rect.top - containerRect.top,
        scrollTop: Math.abs(el.scrollTop),
        scrollHeight: el.scrollHeight,
      };
    });

    console.log('[scroll-stress] initial anchor:', JSON.stringify(anchorInfo));
    expect(anchorInfo).not.toBeNull();
    const anchorIdx = anchorInfo!.messageIndex;

    // Switch back and forth 10 times
    const visualDrifts: number[] = [];
    for (let i = 0; i < 10; i++) {
      await page.locator('text=scroll-stress-B').first().click();
      await expect(page.getByTestId('message-bubble')).toHaveCount(2, { timeout: 5000 });
      await page.waitForTimeout(100);

      await page.locator('text=scroll-stress-A').first().click();
      await expect(page.getByTestId('message-bubble')).toHaveCount(80, { timeout: 5000 });
      // Wait for Mermaid to re-render and scroll to stabilize
      await page.waitForTimeout(800);

      const info = await scroller.evaluate((el, idx) => {
        const bubbles = el.querySelectorAll('[data-testid="message-bubble"]');
        const target = bubbles[idx];
        if (!target) return null;
        const containerRect = el.getBoundingClientRect();
        const rect = target.getBoundingClientRect();
        return {
          offsetFromContainerTop: rect.top - containerRect.top,
          scrollTop: Math.abs(el.scrollTop),
          scrollHeight: el.scrollHeight,
        };
      }, anchorIdx);

      const visualDrift = info ? Math.abs(info.offsetFromContainerTop - anchorInfo!.offsetFromContainerTop) : 9999;
      visualDrifts.push(visualDrift);
      console.log(`[scroll-stress] switch ${i + 1}: scrollTop=${info?.scrollTop} scrollHeight=${info?.scrollHeight} visualDrift=${visualDrift.toFixed(0)}px`);
    }

    const maxVisualDrift = Math.max(...visualDrifts);
    console.log('[scroll-stress] max visual drift:', maxVisualDrift.toFixed(0), 'px');

    // Visual position of the anchor message should stay within 300px
    expect(maxVisualDrift).toBeLessThan(300);
  });

  test('mermaid conversation switch should not cause large scroll jumps', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    const convA = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'jitter-test-A' }),
      });
      return res.json();
    });

    const convB = await page.evaluate(async () => {
      const res = await fetch('/api/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'jitter-test-B' }),
      });
      return res.json();
    });

    testCreatedConvIds.push(convA.conversation_id, convB.conversation_id);

    // Seed A with multiple Mermaid diagrams
    await page.evaluate(async (cid: string) => {
      const messages = Array.from({ length: 20 }).flatMap((_, i) => {
        const idx = i + 1;
        const mermaid = `\n\n\`\`\`mermaid\nflowchart TD\n  Start${idx}-->Process${idx}-->End${idx}\n\`\`\`\n`;
        return [
          { role: 'user', content: `Draw diagram ${idx}` },
          { role: 'assistant', content: `Here is diagram ${idx}:${mermaid}` },
        ];
      });
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, convA.conversation_id);

    await page.evaluate(async (cid: string) => {
      const messages = [
        { role: 'user', content: 'Test' },
        { role: 'assistant', content: 'OK' },
      ];
      await fetch(`/api/chat/conversations/${cid}/_seed-messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });
    }, convB.conversation_id);

    await page.goto('/');
    await switchToChat(page);

    await page.locator('text=jitter-test-A').first().click();
    await expect(page.getByTestId('chat-page')).toBeVisible({ timeout: 5000 });
    await expect(page.getByTestId('message-bubble')).toHaveCount(40, { timeout: 10000 });

    // Wait for all Mermaid diagrams to render
    await page.waitForTimeout(3000);

    const scroller = page.getByTestId('chat-timeline');

    // Scroll to middle
    await scroller.hover();
    for (let i = 0; i < 20; i++) {
      await page.mouse.wheel(0, -100);
      await page.waitForTimeout(20);
    }
    await page.waitForTimeout(300);

    const beforeSwitch = await scroller.evaluate((el) => Math.abs(el.scrollTop));

    // Switch to B and back
    await page.locator('text=jitter-test-B').first().click();
    await expect(page.getByTestId('message-bubble')).toHaveCount(2, { timeout: 5000 });

    await page.locator('text=jitter-test-A').first().click();
    await expect(page.getByTestId('message-bubble')).toHaveCount(40, { timeout: 5000 });

    // Sample scroll position over 1 second to detect jitter
    const jitterData = await scroller.evaluate(async (el) => {
      const samples: number[] = [];
      const start = performance.now();
      while (performance.now() - start < 1000) {
        samples.push(Math.abs((el as HTMLDivElement).scrollTop));
        await new Promise((r) => setTimeout(r, 30));
      }
      const min = Math.min(...samples);
      const max = Math.max(...samples);
      return { samples, min, max, span: max - min, final: samples[samples.length - 1] };
    });

    console.log('[jitter-test] before:', beforeSwitch);
    console.log('[jitter-test] jitter span:', jitterData.span);
    console.log('[jitter-test] final:', jitterData.final);
    console.log('[jitter-test] drift from before:', Math.abs(jitterData.final - beforeSwitch));

    // Jitter span should be small (no large jumps during render)
    expect(jitterData.span).toBeLessThan(300);
    // Final position should be close to before
    expect(Math.abs(jitterData.final - beforeSwitch)).toBeLessThan(400);
  });
});
