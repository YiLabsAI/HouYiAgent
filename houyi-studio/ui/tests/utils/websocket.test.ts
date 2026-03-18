/**
 * Tests for ConsoleWebSocket heartbeat and session behavior.
 *
 * These tests verify:
 * 1. Client replies with pong when server sends ping
 * 2. server_info events are silently consumed (no handler dispatch)
 * 3. Normal events are dispatched to handlers
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// We test the message handling logic by simulating the ReconnectingWebSocket
// behavior. Since ConsoleWebSocket wraps ReconnectingWebSocket, we mock it.

let capturedListeners: Record<string, Function[]> = {};
let sentMessages: string[] = [];
let wsReadyState = 1; // WebSocket.OPEN

vi.mock('reconnecting-websocket', () => {
  return {
    default: class MockRWS {
      readyState = wsReadyState;

      constructor() {
        capturedListeners = {};
        sentMessages = [];
      }

      addEventListener(event: string, handler: Function) {
        if (!capturedListeners[event]) capturedListeners[event] = [];
        capturedListeners[event].push(handler);
      }

      send(data: string) {
        sentMessages.push(data);
      }

      close() {}
    },
  };
});

// Must import after mock setup
import { ConsoleWebSocket } from '@/utils/websocket';

function triggerMessage(data: any) {
  const handlers = capturedListeners['message'] || [];
  handlers.forEach((h) => h({ data: JSON.stringify(data) }));
}

function triggerOpen() {
  const handlers = capturedListeners['open'] || [];
  handlers.forEach((h) => h());
}

describe('ConsoleWebSocket heartbeat', () => {
  let ws: ConsoleWebSocket;

  beforeEach(() => {
    wsReadyState = 1;
    ws = new ConsoleWebSocket('test-session');
    ws.connect();
    triggerOpen();
  });

  it('replies with pong to ping', () => {
    triggerMessage({ event_type: 'ping' });

    expect(sentMessages).toHaveLength(1);
    const pong = JSON.parse(sentMessages[0]);
    expect(pong.command_type).toBe('pong');
  });

  it('skips ping handlers', () => {
    const handler = vi.fn();
    ws.onEvent(handler);

    triggerMessage({ event_type: 'ping' });

    expect(handler).not.toHaveBeenCalled();
  });

  it('skips server info handlers', () => {
    const handler = vi.fn();
    ws.onEvent(handler);

    triggerMessage({ event_type: 'server_info', server_boot_id: 'abc123' });

    expect(handler).not.toHaveBeenCalled();
  });

  it('dispatches normal events', () => {
    const handler = vi.fn();
    ws.onEvent(handler);

    triggerMessage({ event_type: 'execution_status', execution_id: 'e1', status: 'running' });

    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith(
      expect.objectContaining({ event_type: 'execution_status' }),
    );
  });

  it('ignores malformed messages', () => {
    const handlers = capturedListeners['message'] || [];
    // Send non-JSON
    expect(() => {
      handlers.forEach((h) => h({ data: 'not-json' }));
    }).not.toThrow();
  });

  it('sends pong when visible', () => {
    // The client sends a proactive pong when the tab becomes visible,
    // resetting the server's liveness timer after background throttling.
    sentMessages.length = 0;

    // Simulate tab becoming visible
    Object.defineProperty(document, 'visibilityState', { value: 'visible', configurable: true });
    document.dispatchEvent(new Event('visibilitychange'));

    const pongs = sentMessages.filter((m) => JSON.parse(m).command_type === 'pong');
    expect(pongs.length).toBeGreaterThanOrEqual(1);
  });

  it('cleans up on disconnect', () => {
    const removeSpy = vi.spyOn(document, 'removeEventListener');
    ws.disconnect();
    expect(removeSpy).toHaveBeenCalledWith('visibilitychange', expect.any(Function));
    removeSpy.mockRestore();
  });
});
