/// <reference types="node" />

import fs from 'fs';
import os from 'os';
import path from 'path';

const PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e.pid');
const UV_PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e-uv.pid');
const UI_PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e-ui.pid');

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const isPidAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

const waitForExit = async (pid: number, retries = 20, delayMs = 250): Promise<void> => {
  for (let attempt = 0; attempt < retries; attempt += 1) {
    if (!isPidAlive(pid)) {
      return;
    }
    await sleep(delayMs);
  }
};

export default async function globalTeardown(): Promise<void> {
  let backendPid: number | null = null;
  let uvPid: number | null = null;
  let uiPid: number | null = null;

  if (fs.existsSync(UI_PID_FILE)) {
    const uiPidRaw = fs.readFileSync(UI_PID_FILE, 'utf-8');
    const parsedUiPid = Number(uiPidRaw);
    if (Number.isFinite(parsedUiPid)) {
      uiPid = parsedUiPid;
      try {
        process.kill(parsedUiPid, 'SIGTERM');
      } catch (error) {
        console.warn('[E2E] Failed to stop UI server:', error);
      }
    }
    fs.unlinkSync(UI_PID_FILE);
  }

  if (fs.existsSync(PID_FILE)) {
    const pidRaw = fs.readFileSync(PID_FILE, 'utf-8');
    const pid = Number(pidRaw);
    if (Number.isFinite(pid)) {
      backendPid = pid;
      try {
        process.kill(pid, 'SIGTERM');
      } catch {
        // ignore
      }
    }
    fs.unlinkSync(PID_FILE);
  }

  if (fs.existsSync(UV_PID_FILE)) {
    const uvPidRaw = fs.readFileSync(UV_PID_FILE, 'utf-8');
    const parsedUvPid = Number(uvPidRaw);
    if (Number.isFinite(parsedUvPid)) {
      uvPid = parsedUvPid;
      try {
        process.kill(parsedUvPid, 'SIGTERM');
      } catch (error) {
        console.warn('[E2E] Failed to stop uv wrapper:', error);
      }
    }
    fs.unlinkSync(UV_PID_FILE);
  }

  if (backendPid !== null) {
    await waitForExit(backendPid);
  }
  if (uvPid !== null) {
    await waitForExit(uvPid);
  }
  if (uiPid !== null) {
    await waitForExit(uiPid);
  }
}
