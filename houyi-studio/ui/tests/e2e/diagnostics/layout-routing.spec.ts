import { test, expect, type Page } from '@playwright/test';

const switchToChat = async (page: Page): Promise<void> => {
  await page.locator('button', { hasText: 'Chat' }).first().click();
  await expect(
    page.getByTestId('chat-empty-state').or(page.getByTestId('chat-page')),
  ).toBeVisible({ timeout: 5000 });
};

const switchToGraph = async (page: Page): Promise<void> => {
  await page.locator('button', { hasText: 'Graph' }).first().click();
  await expect(page.getByTestId('dag-canvas')).toBeVisible({ timeout: 5000 });
};

test.describe('Layout routing baseline', () => {
  test('graph mode routes activity tabs and keeps bottom panel visible', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByTestId('dag-canvas')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Knowledge', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Knowledge', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Skills', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Skills', exact: true })).toBeVisible();

    await expect(page.getByRole('button', { name: 'Observability', exact: true })).toBeVisible();
  });

  test('chat mode routes activity tabs and hides graph bottom panel', async ({ page }) => {
    await page.goto('/');
    await switchToChat(page);

    await expect(page.getByTestId('dag-canvas')).not.toBeVisible();
    await expect(page.getByRole('button', { name: 'Conversations', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Workflow', exact: true })).toHaveCount(0);

    await page.getByRole('button', { name: 'Knowledge', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Knowledge', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Skills', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Skills', exact: true })).toBeVisible();

    await expect(page.getByRole('button', { name: 'Observability' })).not.toBeVisible();

    await switchToGraph(page);
    await expect(page.getByRole('button', { name: 'Workflow', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Observability', exact: true })).toBeVisible();
  });
});
