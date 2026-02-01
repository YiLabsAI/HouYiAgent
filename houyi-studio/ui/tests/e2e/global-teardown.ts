/// <reference types="node" />

import fs from 'fs';
import os from 'os';
import path from 'path';

const PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e.pid');
const CONDA_PID_FILE = path.join(os.tmpdir(), 'houyi-console-e2e-conda.pid');

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

  if (fs.existsSync(CONDA_PID_FILE)) {
    const condaPidRaw = fs.readFileSync(CONDA_PID_FILE, 'utf-8');
    const condaPid = Number(condaPidRaw);
    if (Number.isFinite(condaPid)) {
      try {
        process.kill(condaPid, 'SIGTERM');
      } catch (error) {
        console.warn('[E2E] Failed to stop conda wrapper:', error);
      }
    }
    fs.unlinkSync(CONDA_PID_FILE);
  }
}
