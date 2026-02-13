/**
 * WebSocket client for console server
 */

import ReconnectingWebSocket from 'reconnecting-websocket';
import type { AnyServerEvent, AnyClientCommand } from '@/types/websocket';

export type EventHandler = (event: AnyServerEvent) => void;
export type StatusHandler = (status: 'connected' | 'disconnected' | 'error') => void;

export class ConsoleWebSocket {
  private ws: ReconnectingWebSocket | null = null;
  private eventHandlers: Set<EventHandler> = new Set();
  private statusHandlers: Set<StatusHandler> = new Set();
  private sessionId: string;
  private pendingCommands: AnyClientCommand[] = [];
  private visibilityHandler: (() => void) | null = null;
  private lastServerBootId: string | null = null;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
  }

  connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    // Use environment variable or default to localhost:8000
    const wsHost = (import.meta as any).env?.VITE_WS_HOST || 'localhost:8000';
    const wsUrl = `ws://${wsHost}/ws/session/${this.sessionId}`;

    console.log('[WebSocket] Attempting to connect to:', wsUrl);

    this.ws = new ReconnectingWebSocket(wsUrl, [], {
      maxRetries: 20,
      reconnectionDelayGrowFactor: 2,
      maxReconnectionDelay: 30000,
      minReconnectionDelay: 2000,
    });

    this.ws.addEventListener('open', () => {
      console.log('[WebSocket] ✅ Connected successfully');
      this.statusHandlers.forEach((handler) => handler('connected'));
      if (this.pendingCommands.length > 0) {
        console.log('[WebSocket] 📤 Sending queued commands:', this.pendingCommands.length);
        const queued = [...this.pendingCommands];
        this.pendingCommands = [];
        queued.forEach((command) => this.sendCommand(command));
      }
    });

    this.ws.addEventListener('message', (event) => {
      try {
        const raw = JSON.parse(event.data);
        const eventType = raw?.event_type;

        // Reply to server heartbeat pings with pong (bidirectional keepalive)
        if (eventType === 'ping') {
          this.ws?.send(JSON.stringify({ command_type: 'pong' }));
          return;
        }

        // Detect server restart via boot_id change
        if (eventType === 'server_info') {
          const bootId = raw?.server_boot_id;
          if (bootId && this.lastServerBootId && bootId !== this.lastServerBootId) {
            console.log('[WebSocket] Server restarted (boot_id changed), reloading for fresh session');
            // Reload page — module-level _currentSessionId in App.tsx resets on reload
            window.location.reload();
            return;
          }
          this.lastServerBootId = bootId || null;
          return;
        }

        const data = raw as AnyServerEvent;

        // Special logging for streaming_output
        if (data.event_type === 'streaming_output') {
          console.log('🔥 [STREAMING]', (data as any).chunk);
        } else {
          console.log('[WebSocket] Received event:', data.event_type);
        }

        this.handleEvent(data);
      } catch (error) {
        console.error('[WebSocket] Failed to parse message:', error);
      }
    });

    this.ws.addEventListener('close', (event) => {
      console.log('[WebSocket] ❌ Disconnected, code:', event.code, 'reason:', event.reason);
      this.statusHandlers.forEach((handler) => handler('disconnected'));
      // ReconnectingWebSocket handles reconnection automatically
    });

    this.ws.addEventListener('error', (error) => {
      console.error('[WebSocket] ❌ Connection error:', error);
      this.statusHandlers.forEach((handler) => handler('error'));
    });

    // When tab becomes visible after being backgrounded, immediately
    // send pong to reset the server's liveness timer. Browsers throttle
    // background-tab timers to ~60s, so server pings may not be replied
    // to promptly. This proactive pong prevents false timeout detection.
    this.visibilityHandler = () => {
      if (document.visibilityState === 'visible' && this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ command_type: 'pong' }));
      }
    };
    document.addEventListener('visibilitychange', this.visibilityHandler);
  }

  disconnect(): void {
    if (this.visibilityHandler) {
      document.removeEventListener('visibilitychange', this.visibilityHandler);
      this.visibilityHandler = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendCommand(command: AnyClientCommand): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] ⚠️ Not connected, queueing command:', command.command_type);
      this.pendingCommands.push(command);
      return;
    }

    console.log('[WebSocket] 📤 Sending command:', command.command_type);
    this.ws.send(JSON.stringify(command));
    console.log('[WebSocket] ✅ Command sent successfully');
  }

  onEvent(handler: EventHandler): () => void {
    this.eventHandlers.add(handler);

    // Return unsubscribe function
    return () => {
      this.eventHandlers.delete(handler);
    };
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);

    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  private handleEvent(event: AnyServerEvent): void {
    console.log('[WebSocket] Event:', event.event_type, event);

    this.eventHandlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error('[WebSocket] Event handler error:', error);
      }
    });
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  isConnecting(): boolean {
    return this.ws?.readyState === WebSocket.CONNECTING;
  }
}
