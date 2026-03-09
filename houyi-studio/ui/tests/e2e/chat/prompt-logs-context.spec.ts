import { test, expect } from '@playwright/test';

test('prompt/logs/context panels', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Context' }).click();
  await expect(page.getByText('No execution context available')).toBeVisible();

  await page.getByRole('button', { name: 'Logs' }).click();
  await expect(page.getByRole('button', { name: 'Execution', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Activity', exact: true })).toBeVisible();
  await expect(page.getByPlaceholder('Search logs')).toBeVisible();

  await expect(page.getByTestId('dag-canvas')).toBeVisible();
  await expect(page.getByText('Live')).toBeVisible();
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    store.getState().addNode('LLM', { x: 200, y: 200 });
  });

  await expect(page.locator('.react-flow__node')).toHaveCount(1);
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    const firstNodeId = store.getState().nodes?.[0]?.id;
    if (firstNodeId) {
      store.getState().selectNode(firstNodeId);
    }
  });

  await page.getByRole('button', { name: 'Prompt' }).click();
  await expect(page.getByText('System Prompt')).toBeVisible();
  await expect(page.getByText('User Message')).toBeVisible();
  await expect(page.getByText('Template Variables')).toBeVisible();
  await expect(page.getByText('Evaluate Prompt')).toBeVisible();
});
