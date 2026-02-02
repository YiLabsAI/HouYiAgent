import { defineConfig } from '@playwright/test';

const useSystemChrome = (globalThis as any).process?.env?.HOUYI_USE_SYSTEM_CHROME === '1';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  globalSetup: './tests/e2e/global-setup.ts',
  globalTeardown: './tests/e2e/global-teardown.ts',
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    video: useSystemChrome ? 'off' : 'retain-on-failure',
  },
  webServer: {
    command: 'pnpm run dev:strict-3000',
    url: 'http://localhost:3000',
    reuseExistingServer: !(globalThis as any).process?.env?.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: 'chromium',
      use: useSystemChrome ? { browserName: 'chromium', channel: 'chrome' } : { browserName: 'chromium' },
    },
  ],
});
