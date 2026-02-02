#!/usr/bin/env node
import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';

let args = process.argv.slice(2);
if (args[0] === '--') {
  args = args.slice(1);
}
const separatorIndex = args.indexOf('--');
const optionArgs = separatorIndex >= 0 ? args.slice(0, separatorIndex) : args;
const passthroughArgs = separatorIndex >= 0 ? args.slice(separatorIndex + 1) : [];

const env = { ...process.env };
const CONFIG_PATH = path.join(os.tmpdir(), 'houyi-console-e2e-config.json');
let quiet;
let logLevel;
let disableLiveWeather;
let toolcallTimeout;
let toolcallRetries;
let toolcallTiming;
let toolcallAdapter;
let toolcallModel;
let toolcallMaxTokens;
let disablePlanPersistence;
let parallelToolCalls;
let toolcallToolLatencyMs;
let toolcallFastPath;

const takeValue = (index, arg, flagName) => {
  const inlineValue = arg.includes('=') ? arg.split('=')[1] : undefined;
  if (inlineValue !== undefined && inlineValue !== '') {
    return { value: inlineValue, nextIndex: index };
  }
  const nextValue = optionArgs[index + 1];
  if (!nextValue || nextValue.startsWith('-')) {
    throw new Error(`Missing value for ${flagName}`);
  }
  return { value: nextValue, nextIndex: index + 1 };
};

const usage = () => {
  console.log(`\nUsage: pnpm run test:e2e:cli -- [options] -- [playwright args]\n\nOptions:\n  --quiet | --no-quiet                     Silence console server logs\n  --log-level <level>                      Override log level (INFO, WARNING, ERROR, DEBUG)\n  --live-weather                           Use real live weather data\n  --mock-weather                           Use mock live weather data\n  --toolcall-timeout <seconds>             Override tool call timeout\n  --toolcall-retries <count>               Override tool call max retries\n  --toolcall-timing | --no-toolcall-timing Enable tool call timing logs\n  --toolcall-adapter <real|fake>           Use real or fake tool-calling adapter\n  --fake-toolcall-adapter                  Shortcut for --toolcall-adapter fake\n  --toolcall-model <model>                 Override tool-calling model\n  --toolcall-max-tokens <count>            Override tool-calling max tokens\n  --toolcall-tool-latency <ms>             Simulate per-tool latency (E2E only)\n  --toolcall-fast-path                     Enable single-round tool-call planning\n  --no-toolcall-fast-path                  Disable single-round tool-call planning\n  --disable-plan-persistence               Skip plan file writes for faster E2E\n  --enable-plan-persistence                Force plan file writes during E2E\n  --parallel-tool-calls                    Enable parallel tool calls in E2E\n  --no-parallel-tool-calls                 Disable parallel tool calls in E2E\n\nExamples:\n  pnpm run test:e2e:cli -- --quiet --mock-weather -- tests/e2e/tool-execution.spec.ts\n  pnpm run test:e2e:cli -- --live-weather --toolcall-timeout 10 -- tool-execution.spec.ts\n`);
};

try {
  for (let i = 0; i < optionArgs.length; i += 1) {
    const arg = optionArgs[i];
    if (!arg) continue;

    if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    }

    if (arg === '--quiet') {
      quiet = true;
      continue;
    }
    if (arg === '--no-quiet') {
      quiet = false;
      continue;
    }
    if (arg.startsWith('--log-level')) {
      const { value, nextIndex } = takeValue(i, arg, '--log-level');
      logLevel = value.toUpperCase();
      i = nextIndex;
      continue;
    }
    if (arg === '--live-weather') {
      disableLiveWeather = false;
      continue;
    }
    if (arg === '--mock-weather' || arg === '--disable-live-weather') {
      disableLiveWeather = true;
      continue;
    }
    if (arg.startsWith('--toolcall-timeout')) {
      const { value, nextIndex } = takeValue(i, arg, '--toolcall-timeout');
      toolcallTimeout = value;
      i = nextIndex;
      continue;
    }
    if (arg.startsWith('--toolcall-retries')) {
      const { value, nextIndex } = takeValue(i, arg, '--toolcall-retries');
      toolcallRetries = value;
      i = nextIndex;
      continue;
    }
    if (arg.startsWith('--toolcall-model')) {
      const { value, nextIndex } = takeValue(i, arg, '--toolcall-model');
      toolcallModel = value;
      i = nextIndex;
      continue;
    }
    if (arg.startsWith('--toolcall-max-tokens')) {
      const { value, nextIndex } = takeValue(i, arg, '--toolcall-max-tokens');
      toolcallMaxTokens = value;
      i = nextIndex;
      continue;
    }
    if (arg.startsWith('--toolcall-tool-latency')) {
      const { value, nextIndex } = takeValue(i, arg, '--toolcall-tool-latency');
      toolcallToolLatencyMs = value;
      i = nextIndex;
      continue;
    }
    if (arg === '--toolcall-fast-path') {
      toolcallFastPath = true;
      continue;
    }
    if (arg === '--no-toolcall-fast-path') {
      toolcallFastPath = false;
      continue;
    }
    if (arg === '--toolcall-timing') {
      toolcallTiming = true;
      continue;
    }
    if (arg === '--no-toolcall-timing') {
      toolcallTiming = false;
      continue;
    }
    if (arg === '--disable-plan-persistence') {
      disablePlanPersistence = true;
      continue;
    }
    if (arg === '--enable-plan-persistence') {
      disablePlanPersistence = false;
      continue;
    }
    if (arg === '--parallel-tool-calls') {
      parallelToolCalls = true;
      continue;
    }
    if (arg === '--no-parallel-tool-calls') {
      parallelToolCalls = false;
      continue;
    }
    if (arg === '--fake-toolcall-adapter') {
      toolcallAdapter = 'fake';
      continue;
    }
    if (arg.startsWith('--toolcall-adapter')) {
      const { value, nextIndex } = takeValue(i, arg, '--toolcall-adapter');
      const normalized = value.toLowerCase();
      if (!['real', 'fake'].includes(normalized)) {
        throw new Error(`Invalid --toolcall-adapter value: ${value}`);
      }
      toolcallAdapter = normalized;
      i = nextIndex;
      continue;
    }

    console.warn(`[run-e2e] Unknown option ignored: ${arg}`);
  }
} catch (error) {
  console.error(`[run-e2e] ${error instanceof Error ? error.message : error}`);
  usage();
  process.exit(1);
}

const resolveNumber = (value, flagName) => {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid value for ${flagName}: ${value}`);
  }
  return parsed;
};

const resolvedQuiet = quiet ?? false;
const resolvedLogLevel = (logLevel ?? (resolvedQuiet ? 'WARNING' : 'INFO')).toUpperCase();
const resolvedDisableLiveWeather = disableLiveWeather ?? true;
const resolvedToolcallTimeout = resolveNumber(toolcallTimeout, '--toolcall-timeout') ?? 10;
const resolvedToolcallRetries = resolveNumber(toolcallRetries, '--toolcall-retries') ?? 0;
const resolvedToolcallTiming = toolcallTiming ?? false;
const resolvedToolcallAdapter = toolcallAdapter ?? 'real';
const resolvedToolcallMaxTokens = resolveNumber(toolcallMaxTokens, '--toolcall-max-tokens');
const resolvedDisablePlanPersistence = disablePlanPersistence ?? true;
const resolvedParallelToolCalls = parallelToolCalls ?? null;
const resolvedToolcallToolLatencyMs = resolveNumber(
  toolcallToolLatencyMs,
  '--toolcall-tool-latency',
);
const resolvedToolcallFastPath = toolcallFastPath ?? false;

const configPayload = {
  quiet: resolvedQuiet,
  logLevel: resolvedLogLevel,
  disableLiveWeather: resolvedDisableLiveWeather,
  toolcallTimeout: resolvedToolcallTimeout,
  toolcallRetries: resolvedToolcallRetries,
  toolcallTiming: resolvedToolcallTiming,
  toolcallAdapter: resolvedToolcallAdapter,
  toolcallModel: toolcallModel ?? null,
  toolcallMaxTokens: resolvedToolcallMaxTokens,
  disablePlanPersistence: resolvedDisablePlanPersistence,
  parallelToolCalls: resolvedParallelToolCalls,
  toolcallToolLatencyMs: resolvedToolcallToolLatencyMs,
  toolcallFastPath: resolvedToolcallFastPath,
};

fs.writeFileSync(CONFIG_PATH, `${JSON.stringify(configPayload, null, 2)}\n`, 'utf-8');
env.HOUYI_E2E_CONFIG = CONFIG_PATH;

const command = process.platform === 'win32' ? 'npx.cmd' : 'npx';
const child = spawn(command, ['playwright', 'test', ...passthroughArgs], {
  stdio: 'inherit',
  env,
});

const cleanupConfig = () => {
  try {
    fs.unlinkSync(CONFIG_PATH);
  } catch (error) {
    if (error && error.code !== 'ENOENT') {
      console.warn(`[run-e2e] Failed to remove config file: ${error.message || error}`);
    }
  }
};

child.on('exit', (code) => {
  cleanupConfig();
  process.exit(code ?? 1);
});

child.on('error', (error) => {
  cleanupConfig();
  console.error(`[run-e2e] Failed to start Playwright: ${error}`);
  process.exit(1);
});
