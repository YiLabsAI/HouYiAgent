/// <reference types="node" />

import fs from 'fs';
import os from 'os';
import path from 'path';

const PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e.pid');
const UV_PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e-uv.pid');

export default async function globalTeardown(): Promise<void> {
  if (fs.existsSync(PID_FILE)) {
    const pidRaw = fs.readFileSync(PID_FILE, 'utf-8');
    const pid = Number(pidRaw);
    if (Number.isFinite(pid)) {
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
    const uvPid = Number(uvPidRaw);
    if (Number.isFinite(uvPid)) {
      try {
        process.kill(uvPid, 'SIGTERM');
      } catch (error) {
        console.warn('[E2E] Failed to stop uv wrapper:', error);
      }
    }
    fs.unlinkSync(UV_PID_FILE);
  }
}
