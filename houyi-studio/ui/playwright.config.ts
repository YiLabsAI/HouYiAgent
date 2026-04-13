import { defineConfig } from '@playwright/test';

const useSystemChrome = (globalThis as any).process?.env?.HOUYI_USE_SYSTEM_CHROME === '1';
const e2eBackendPort = Number((globalThis as any).process?.env?.HOUYI_E2E_BACKEND_PORT || '19000');
const e2eUiPort = Number((globalThis as any).process?.env?.HOUYI_E2E_UI_PORT || '13100');
const e2eBaseUrl = `http://127.0.0.1:${e2eUiPort}`;

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  globalSetup: './tests/e2e/global-setup.ts',
  globalTeardown: './tests/e2e/global-teardown.ts',
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL: e2eBaseUrl,
    trace: 'retain-on-failure',
    video: useSystemChrome ? 'off' : 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: useSystemChrome ? { browserName: 'chromium', channel: 'chrome' } : { browserName: 'chromium' },
    },
  ],
});
