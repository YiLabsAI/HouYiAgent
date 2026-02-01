import { test, expect } from '@playwright/test';

test('console UI loads core panels', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
  await expect(page.getByLabel('Timeline')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Compare' })).toBeVisible();

  await page.getByRole('button', { name: 'Compare' }).click();
  await expect(page.getByText('Need at least 2 checkpoints to compare')).toBeVisible();
});
