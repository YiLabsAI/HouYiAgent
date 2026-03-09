/// <reference types="node" />

import { spawn } from 'child_process';
import fs from 'fs';
import http from 'http';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const E2E_BACKEND_PORT = Number(process.env.HOUYI_E2E_BACKEND_PORT || '9000');
const SERVER_URL = `http://127.0.0.1:${E2E_BACKEND_PORT}/`;
const SERVER_PORT = E2E_BACKEND_PORT;
const PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e.pid');
const UV_PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e-uv.pid');
const SKIP_KILL_PORT = process.env.SKIP_KILL_E2E_PORT === '1';
const CONFIG_PATH = process.env.HOUYI_E2E_CONFIG;

type E2EConfig = {
  quiet?: boolean;
  logLevel?: string;
  disableLiveWeather?: boolean;
  toolcallTimeout?: number;
  toolcallRetries?: number;
  toolcallTiming?: boolean;
  toolcallAdapter?: string;
  toolcallModel?: string | null;
  toolcallMaxTokens?: number | null;
  disablePlanPersistence?: boolean;
  parallelToolCalls?: boolean | string | null;
  toolcallToolLatencyMs?: number | null;
  toolcallFastPath?: boolean;
};

const loadE2EConfig = (): E2EConfig => {
  if (!CONFIG_PATH) {
    return {};
  }
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf-8');
    return JSON.parse(raw) as E2EConfig;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`[E2E] Failed to read config ${CONFIG_PATH}: ${message}`);
  }
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const isServerUp = async (): Promise<boolean> => {
  return new Promise((resolve) => {
    const request = http.get(SERVER_URL, (response: http.IncomingMessage) => {
      response.resume();
      resolve(Boolean(response.statusCode && response.statusCode < 500));
    });

    request.on('error', () => resolve(false));
    request.setTimeout(1000, () => {
      request.destroy();
      resolve(false);
    });
  });
};

const waitForServerDown = async (retries = 10, delayMs = 300): Promise<boolean> => {
  for (let attempt = 0; attempt < retries; attempt += 1) {
    if (!(await isServerUp())) {
      return true;
    }
    await sleep(delayMs);
  }
  return !(await isServerUp());
};

const tryKillPid = (pid: number, signal: NodeJS.Signals): boolean => {
  try {
    process.kill(pid, signal);
    return true;
  } catch (error) {
    console.warn(`[E2E] Failed to send ${signal} to process ${pid}:`, error);
    return false;
  }
};

export default async function globalSetup(): Promise<void> {
  if (await isServerUp()) {
    if (SKIP_KILL_PORT) {
      console.log(`[E2E] Reusing existing server on 127.0.0.1:${SERVER_PORT} (SKIP_KILL_E2E_PORT=1)`);
      return;
    }

    if (fs.existsSync(PID_FILE)) {
      const pidRaw = fs.readFileSync(PID_FILE, 'utf-8');
      const pid = Number(pidRaw);
      if (Number.isFinite(pid)) {
        tryKillPid(pid, 'SIGTERM');
      }
      fs.unlinkSync(PID_FILE);
      await waitForServerDown();
    }

    if (await isServerUp()) {
      throw new Error(
        `E2E backend port 127.0.0.1:${SERVER_PORT} is already in use by a non-e2e process. `
        + 'For safety, Playwright setup will not kill arbitrary listeners. '
        + `Stop that process, choose another HOUYI_E2E_BACKEND_PORT, or set SKIP_KILL_E2E_PORT=1 to reuse it intentionally.`,
      );
    }
  }

  if (fs.existsSync(PID_FILE)) {
    fs.unlinkSync(PID_FILE);
  }
  if (fs.existsSync(UV_PID_FILE)) {
    fs.unlinkSync(UV_PID_FILE);
  }

  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const repoRoot = path.resolve(__dirname, '../../../../');
  const config = loadE2EConfig();
  const quietLogs = config.quiet ?? false;
  const logLevel = (config.logLevel ?? (quietLogs ? 'WARNING' : 'INFO')).toUpperCase();
  const disableLiveWeather = config.disableLiveWeather ?? true;
  const toolcallTimeout = config.toolcallTimeout ?? 20;
  const toolcallRetries = config.toolcallRetries ?? 0;
  const toolcallTiming = config.toolcallTiming ?? false;
  const toolcallAdapter = (config.toolcallAdapter ?? 'real').toLowerCase();
  const toolcallModel = config.toolcallModel ?? null;
  const toolcallMaxTokens = config.toolcallMaxTokens ?? null;
  const disablePlanPersistence = config.disablePlanPersistence ?? true;
  const parallelToolCalls = config.parallelToolCalls ?? null;
  const toolcallToolLatencyMs = config.toolcallToolLatencyMs ?? null;
  const toolcallFastPath = config.toolcallFastPath ?? false;
  const serverRoot = path.join(repoRoot, 'houyi-studio', 'server');
  const workflowsDir = path.join(repoRoot, 'tests', 'integration', 'fixtures', 'workflows');
  const pythonPath = process.env.PYTHONPATH
    ? `${serverRoot}${path.delimiter}${repoRoot}${path.delimiter}${process.env.PYTHONPATH}`
    : `${serverRoot}${path.delimiter}${repoRoot}`;

  // CRITICAL: Use isolated data directory for e2e tests to prevent
  // deleting user data. See acceptance doc §14 for incident report.
  const e2eDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'houyi-e2e-chat-'));
  const e2eSettingsPath = path.join(e2eDataDir, 'settings.json');
  console.log(`[E2E] Using isolated chat data dir: ${e2eDataDir}`);
  console.log(`[E2E] Starting isolated backend on 127.0.0.1:${SERVER_PORT}`);

  const child = spawn('uv', ['run', 'python', 'tests/integration/fixtures/console_tools.py'], {
    cwd: repoRoot,
    stdio: quietLogs ? 'ignore' : 'inherit',
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONPATH: pythonPath,
      HOUYI_CHAT_DATA_DIR: e2eDataDir,
      HOUYI_CHAT_SETTINGS_PATH: e2eSettingsPath,
      HOUYI_E2E_QUIET: quietLogs ? '1' : '0',
      HOUYI_E2E_BACKEND_PORT: String(SERVER_PORT),
      HOUYI_LOG_LEVEL: logLevel,
      HOUYI_PORT: String(SERVER_PORT),
      HOUYI_TOOLCALL_MAX_RETRIES: String(toolcallRetries),
      HOUYI_TOOLCALL_TIMEOUT: String(toolcallTimeout),
      HOUYI_TOOLCALL_TIMING: toolcallTiming ? '1' : '0',
      HOUYI_DISABLE_LIVE_WEATHER: disableLiveWeather ? '1' : '0',
      HOUYI_DISABLE_PLAN_PERSISTENCE: disablePlanPersistence ? '1' : '0',
      HOUYI_TOOLCALL_ADAPTER: toolcallAdapter,
      HOUYI_WORKFLOWS_DIR: workflowsDir,
      ...(toolcallModel ? { HOUYI_TOOLCALL_MODEL: toolcallModel } : {}),
      ...(toolcallMaxTokens != null ? { HOUYI_TOOLCALL_MAX_TOKENS: String(toolcallMaxTokens) } : {}),
      ...(parallelToolCalls !== null
        ? { HOUYI_PARALLEL_TOOL_CALLS: String(parallelToolCalls) }
        : {}),
      ...(toolcallToolLatencyMs != null
        ? { HOUYI_TOOLCALL_TOOL_LATENCY_MS: String(toolcallToolLatencyMs) }
        : {}),
      HOUYI_TOOLCALL_FAST_PATH: toolcallFastPath ? '1' : '0',
    },
  });

  child.unref();

  if (child.pid) {
    fs.writeFileSync(UV_PID_FILE, String(child.pid));
  }

  if (!child.pid) {
    throw new Error('Failed to start console server process for E2E tests.');
  }

  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (await isServerUp()) {
      return;
    }

    if (child.exitCode !== null) {
      throw new Error('Console server exited before becoming ready.');
    }

    await sleep(500);
  }

  child.kill('SIGTERM');
  throw new Error('Console server did not become ready within timeout.');
}
