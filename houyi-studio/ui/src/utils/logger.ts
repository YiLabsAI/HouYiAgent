/**
 * Frontend Logger - Bridges browser console logs to the backend terminal.
 *
 * Usage:
 *   import { logger } from '@/utils/logger';
 *   logger.info('Knowledge', 'Creating library:', name);
 *   logger.error('WebSocket', 'Connection failed:', error);
 *
 * Logs are printed to browser console AND sent to the backend terminal
 * when a WebSocket connection is available.
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

type SendFn = ((command: any) => void) | null;

class FrontendLogger {
  private _sendCommand: SendFn = null;
  private _sessionId: string | null = null;
  private _enabled = true;

  /**
   * Connect the logger to the WebSocket command sender.
   * Called once when the store initializes.
   */
  connect(sendCommand: SendFn, sessionId: string): void {
    this._sendCommand = sendCommand;
    this._sessionId = sessionId;
  }

  disconnect(): void {
    this._sendCommand = null;
    this._sessionId = null;
  }

  /** Enable or disable backend bridging */
  setEnabled(enabled: boolean): void {
    this._enabled = enabled;
  }

  debug(category: string, ...args: any[]): void {
    this._log('debug', category, args);
  }

  info(category: string, ...args: any[]): void {
    this._log('info', category, args);
  }

  warn(category: string, ...args: any[]): void {
    this._log('warn', category, args);
  }

  error(category: string, ...args: any[]): void {
    this._log('error', category, args);
  }

  private _log(level: LogLevel, category: string, args: any[]): void {
    const prefix = `[${category}]`;
    const message = args
      .map((a) => (typeof a === 'string' ? a : JSON.stringify(a)))
      .join(' ');

    // 1. Always log to browser console
    switch (level) {
      case 'debug':
        console.debug(prefix, ...args);
        break;
      case 'info':
        console.log(prefix, ...args);
        break;
      case 'warn':
        console.warn(prefix, ...args);
        break;
      case 'error':
        console.error(prefix, ...args);
        break;
    }

    // 2. Send to backend if connected and enabled
    if (this._enabled && this._sendCommand && this._sessionId) {
      try {
        this._sendCommand({
          command_type: 'frontend_log',
          command_id: `log_${Date.now()}`,
          session_id: this._sessionId,
          level,
          category,
          message,
          timestamp: new Date().toISOString(),
        });
      } catch {
        // Silently fail - don't recurse into error logging
      }
    }
  }
}

/** Singleton logger instance */
export const logger = new FrontendLogger();
