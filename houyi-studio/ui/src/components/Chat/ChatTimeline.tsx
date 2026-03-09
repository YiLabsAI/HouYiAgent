/**
 * ChatTimeline: scrollable message list for the active conversation.
 *
 * BUG-041 zero-flicker design:
 * Uses CSS `flex-direction: column-reverse` on the scroll container so the
 * browser natively anchors content to the BOTTOM.  This means:
 *   - On conversation switch, the last message is visible immediately
 *     without any JS scroll — zero flicker, zero intermediate frames.
 *   - Scrolling up works naturally (user scrolls toward older messages).
 *   - New messages appear at the bottom automatically.
 *
 * The inner wrapper uses `flex-direction: column` to restore visual order
 * (oldest at top, newest at bottom).
 *
 * Auto-scroll during streaming uses scrollTop=0 (which is the bottom in
 * a column-reverse container).
 */
import React from 'react';
import { MessageCircle } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';
import { useChatStore } from '@/stores/useChatStore';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';
import { Bot } from 'lucide-react';

const INITIAL_RENDER_LIMIT = 120;
const RENDER_LIMIT_INCREMENT = 120;

// Module-level scroll position cache — survives re-renders, no state needed.
// Uses message-based anchoring: stores which message was at the viewport center
// and its pixel offset from the container top. This is robust against scrollHeight
// changes caused by Mermaid diagram re-rendering.
type ScrollSnapshot = {
  // Primary: message-based anchor
  anchorMessageId: string;
  anchorOffsetFromContainerTop: number;
  // Fallback: pixel-based (used when anchor message not found in DOM)
  distFromBottom: number;
  scrollTopSign: 1 | -1;
};

const scrollPositionCache = new Map<string, ScrollSnapshot>();

function getScrollSnapshot(el: HTMLDivElement): ScrollSnapshot {
  const distFromBottom = Math.abs(el.scrollTop);
  const scrollTopSign: 1 | -1 = el.scrollTop < 0 ? -1 : 1;

  // Find the message closest to the viewport center
  const containerRect = el.getBoundingClientRect();
  const viewportCenter = containerRect.top + containerRect.height / 2;
  const msgEls = el.querySelectorAll('[data-message-id]');
  let closestEl: Element | null = null;
  let closestDist = Infinity;
  for (const m of msgEls) {
    const r = m.getBoundingClientRect();
    const d = Math.abs(r.top + r.height / 2 - viewportCenter);
    if (d < closestDist) {
      closestDist = d;
      closestEl = m;
    }
  }

  const anchorMessageId = closestEl?.getAttribute('data-message-id') ?? '';
  const anchorOffsetFromContainerTop = closestEl
    ? closestEl.getBoundingClientRect().top - containerRect.top
    : 0;

  return { anchorMessageId, anchorOffsetFromContainerTop, distFromBottom, scrollTopSign };
}

function restoreScrollFromSnapshot(el: HTMLDivElement, snap: ScrollSnapshot) {
  const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
  if (maxScroll === 0) {
    el.scrollTop = 0;
    return;
  }

  // Restore via distFromBottom. During useLayoutEffect, content-visibility:auto
  // means off-screen messages have estimated heights (contain-intrinsic-size),
  // so getBoundingClientRect on the anchor message would be inaccurate.
  // The ResizeObserver will refine the position using the message anchor
  // once content-visibility expands the real heights.
  const dist = Math.max(0, Math.min(maxScroll, snap.distFromBottom));
  el.scrollTop = snap.scrollTopSign * dist;
  if (dist > 1 && Math.abs(el.scrollTop) < dist * 0.5) {
    el.scrollTop = -snap.scrollTopSign * dist;
  }
}

// Refine scroll position using message anchor after content has fully expanded.
// Called by ResizeObserver when content-visibility reveals real element heights.
function refineScrollWithAnchor(el: HTMLDivElement, snap: ScrollSnapshot) {
  if (!snap.anchorMessageId) return;
  const anchor = el.querySelector(`[data-message-id="${snap.anchorMessageId}"]`);
  if (!anchor) return;

  const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
  if (maxScroll === 0) return;

  const containerRect = el.getBoundingClientRect();
  const anchorRect = anchor.getBoundingClientRect();
  const currentOffset = anchorRect.top - containerRect.top;
  const moveDown = snap.anchorOffsetFromContainerTop - currentOffset;

  if (Math.abs(moveDown) > 2) {
    const currentDist = Math.abs(el.scrollTop);
    const newDist = Math.max(0, Math.min(maxScroll, currentDist + moveDown));
    const sign = el.scrollTop < 0 ? -1 : 1;
    el.scrollTop = sign * newDist;
    if (newDist > 1 && Math.abs(el.scrollTop) < newDist * 0.5) {
      el.scrollTop = -sign * newDist;
    }
  }
}

interface ChatTimelineProps {
  messages: ChatMessage[];
  streamingMessageId: string | null;
  conversationId: string | null;
  isLastMessage?: (msg: ChatMessage) => boolean;
  isWaitingForResponse?: boolean;
  onOpenTrace?: (traceId: string) => void;
}

type TimelineItem = {
  message: ChatMessage;
  toolSteps: ChatMessage[];
};

const formatDateDivider = (timestamp: number): string => {
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    month: '2-digit',
    day: '2-digit',
    weekday: 'short',
  });
};

const getDateDividerKey = (timestamp: number): string => {
  return new Date(timestamp * 1000).toISOString().slice(0, 10);
};

export const ChatTimeline: React.FC<ChatTimelineProps> = ({
  messages,
  streamingMessageId,
  conversationId,
  isWaitingForResponse = false,
  onOpenTrace,
}) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const contentRef = React.useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = React.useState(true);
  const [highlightedMsgId, setHighlightedMsgId] = React.useState<string | null>(null);
  const [renderLimit, setRenderLimit] = React.useState(INITIAL_RENDER_LIMIT);
  const restoreRafRef = React.useRef<number | null>(null);
  const resizeObserverRef = React.useRef<ResizeObserver | null>(null);
  const resizeRafRef = React.useRef<number | null>(null);
  const programmaticScrollRef = React.useRef<boolean>(false);
  const lastUserScrollAtRef = React.useRef<number>(0);
  const isPointerDownInTimelineRef = React.useRef<boolean>(false);
  // Tracks the last wheel event timestamp.  Used to distinguish user-initiated
  // wheel scrolls from layout-shift-induced scroll events during the post-restore
  // protection window.  Without this, wheel scrolling doesn't save snapshots.
  const lastWheelAtRef = React.useRef<number>(0);
  const lastRestoreAtRef = React.useRef<number>(0);
  const prevMessageCountRef = React.useRef(messages.length);
  const prevConversationIdRef = React.useRef<string | null>(conversationId);
  // The conversation ID we need to restore scroll for.  null = no pending restore.
  // We store the target ID (not just a boolean) because BUG-041's store strategy
  // updates conversationId first while keeping old messages until the fetch
  // completes.  We must wait until messages actually belong to this conversation.
  const pendingRestoreForRef = React.useRef<string | null>(null);
  // Fingerprint of the current messages array — used to detect when the store
  // has swapped in a genuinely different messages list (new conversation data).
  const prevMessagesFingerprintRef = React.useRef<string>('');
  const scrollToMessageId = useChatStore((s) => s.scrollToMessageId);
  const clearScrollTarget = useChatStore((s) => s.clearScrollTarget);
  const streamingReasoningLength = useChatStore((s) => (
    streamingMessageId && s.streaming.messageId === streamingMessageId
      ? s.streaming.reasoningBuffer.length
      : 0
  ));

  // Cheap fingerprint: first message ID + length.  Changes when the store
  // replaces activeConversation with data from a different conversation.
  const messagesFingerprint = messages.length > 0
    ? `${messages[0].message_id}:${messages.length}`
    : '';

  // Progressive rendering: for large conversations, render the most recent N messages first.
  const visibleMessages = React.useMemo(() => {
    if (messages.length <= renderLimit) return messages;
    return messages.slice(-renderLimit);
  }, [messages, renderLimit]);

  const timelineItems = React.useMemo<TimelineItem[]>(() => {
    const items: TimelineItem[] = [];
    let pendingToolSteps: ChatMessage[] = [];
    let latestAssistantCarrier: ChatMessage | null = null;
    const appendPendingToolStep = (step: ChatMessage) => {
      const incomingCallId = typeof step.tool_call_id === 'string' ? step.tool_call_id : null;
      const existingIndex = pendingToolSteps.findIndex((pending) => {
        const pendingCallId = typeof pending.tool_call_id === 'string' ? pending.tool_call_id : null;
        if (incomingCallId && pendingCallId) return pendingCallId === incomingCallId;
        return pending.message_id === step.message_id;
      });
      if (existingIndex >= 0) {
        pendingToolSteps[existingIndex] = step;
        return;
      }
      pendingToolSteps.push(step);
    };

    for (const msg of visibleMessages) {
      const isAssistantToolCallCarrier = msg.role === 'assistant'
        && Array.isArray(msg.tool_calls)
        && msg.tool_calls.length > 0;
      if (isAssistantToolCallCarrier) {
        latestAssistantCarrier = msg;
        msg.tool_calls?.forEach((toolCall, index) => {
          const callPayload =
            toolCall && typeof toolCall === 'object'
              ? (toolCall as Record<string, any>)
              : {};
          const fnPayload =
            callPayload.function && typeof callPayload.function === 'object'
              ? (callPayload.function as Record<string, any>)
              : {};
          const rawArgs = fnPayload.arguments;
          const argsText = typeof rawArgs === 'string'
            ? rawArgs
            : rawArgs != null
              ? JSON.stringify(rawArgs)
              : '';
          appendPendingToolStep({
            message_id: `${msg.message_id}-tool-call-${index}`,
            role: 'tool',
            content: argsText,
            name: typeof fnPayload.name === 'string' ? fnPayload.name : 'tool',
            tool_call_id: typeof callPayload.id === 'string' ? callPayload.id : null,
            metadata: {
              tool_status: 'ok',
              round_index: Number(msg.metadata?.round_index || 0) || undefined,
            },
            created_at: msg.created_at,
          });
        });
        continue;
      }

      if (msg.role === 'tool') {
        appendPendingToolStep(msg);
        continue;
      }

      if (msg.role === 'assistant') {
        latestAssistantCarrier = msg;
        items.push({ message: msg, toolSteps: pendingToolSteps });
        pendingToolSteps = [];
        continue;
      }

      if (pendingToolSteps.length > 0) {
        for (const step of pendingToolSteps) {
          items.push({ message: step, toolSteps: [] });
        }
        pendingToolSteps = [];
      }

      items.push({ message: msg, toolSteps: [] });
    }

    if (pendingToolSteps.length > 0) {
      if (latestAssistantCarrier) {
        const lastIndex = items.length - 1;
        if (lastIndex >= 0 && items[lastIndex].message.message_id === latestAssistantCarrier.message_id) {
          items[lastIndex] = {
            message: items[lastIndex].message,
            toolSteps: [...items[lastIndex].toolSteps, ...pendingToolSteps],
          };
        } else {
          items.push({ message: latestAssistantCarrier, toolSteps: pendingToolSteps });
        }
      } else {
        for (const step of pendingToolSteps) {
          items.push({ message: step, toolSteps: [] });
        }
      }
    }

    return items;
  }, [visibleMessages]);

  const priorityIds = React.useMemo(() => {
    const ids = new Set<string>();
    if (visibleMessages.length === 0) return ids;

    // Always prioritize a small tail of messages for instant paint.
    const tail = visibleMessages.slice(-3);
    for (const m of tail) {
      ids.add(m.message_id);
    }

    // Keep the actively streaming message fully visible. Long reasoning output
    // grows inside a nested scroll panel; if content-visibility remains auto
    // here, the outer timeline can continue scrolling while the live text falls
    // outside the visible paint region.
    if (streamingMessageId) {
      ids.add(streamingMessageId);
    }

    // Also prioritize any message containing Mermaid blocks. Mermaid rendering
    // can be expensive and interacts poorly with content-visibility during
    // scroll (especially scrollbar dragging), causing visible jitter.
    for (const m of visibleMessages) {
      if (m.content && m.content.includes('```mermaid')) {
        ids.add(m.message_id);
      }
    }

    return ids;
  }, [visibleMessages, streamingMessageId]);

  // Track which conversation's messages are currently rendered in the DOM.
  // Updated by effect 2 after a successful restore (i.e. after the store
  // has swapped in the new conversation's data).
  const renderedConvIdRef = React.useRef<string | null>(conversationId);

  // Keep scroll position stable when content height changes after restore
  // (e.g. Mermaid diagrams finish rendering, images decode).
  // Uses the message-based anchor from the saved snapshot to re-anchor.
  React.useEffect(() => {
    const contentEl = contentRef.current;
    if (!contentEl) return;

    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect();
    }

    resizeObserverRef.current = new ResizeObserver(() => {
      if (!containerRef.current) return;
      if (autoScroll) return;
      if (streamingMessageId || isWaitingForResponse) return;

      // If the user is actively scrolling / dragging the scrollbar, do not
      // fight their scroll position.
      if (Date.now() - lastUserScrollAtRef.current < 800) return;

      // Only re-anchor if we recently restored (within 3s window for Mermaid).
      if (Date.now() - lastRestoreAtRef.current > 3000) return;

      if (resizeRafRef.current) {
        cancelAnimationFrame(resizeRafRef.current);
      }

      resizeRafRef.current = requestAnimationFrame(() => {
        resizeRafRef.current = null;
        const el = containerRef.current;
        if (!el) return;

        const cid = renderedConvIdRef.current;
        if (!cid) return;
        const snap = scrollPositionCache.get(cid);
        if (!snap) return;

        // Refine position using message anchor now that content-visibility
        // has expanded the real element heights.
        programmaticScrollRef.current = true;
        refineScrollWithAnchor(el, snap);
        requestAnimationFrame(() => {
          programmaticScrollRef.current = false;
        });
      });
    });

    resizeObserverRef.current.observe(contentEl);

    return () => {
      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;
      if (resizeRafRef.current) {
        cancelAnimationFrame(resizeRafRef.current);
        resizeRafRef.current = null;
      }
    };
  }, [autoScroll, streamingMessageId, isWaitingForResponse]);

  // Treat scrollbar dragging / pointer interactions in the timeline as user intent.
  // This prevents ResizeObserver-based restores from fighting the user's scroll,
  // and makes scroll snapshot persistence more reliable when the user drags the scrollbar.
  React.useEffect(() => {
    const onPointerDownCapture = (e: PointerEvent) => {
      const el = containerRef.current;
      if (!el) return;
      const target = e.target as Node | null;
      if (target && el.contains(target)) {
        isPointerDownInTimelineRef.current = true;
        lastUserScrollAtRef.current = Date.now();
      }
    };

    const onPointerMoveCapture = (e: PointerEvent) => {
      if (!isPointerDownInTimelineRef.current) return;
      // Keep refreshing while dragging; some browsers emit ResizeObserver callbacks
      // during scroll/drag and we want to avoid fighting user scroll.
      if (e.buttons === 1) {
        lastUserScrollAtRef.current = Date.now();
      }
    };

    const onPointerUpCapture = () => {
      isPointerDownInTimelineRef.current = false;
      lastUserScrollAtRef.current = Date.now();
    };

    window.addEventListener('pointerdown', onPointerDownCapture, true);
    window.addEventListener('pointermove', onPointerMoveCapture, true);
    window.addEventListener('pointerup', onPointerUpCapture, true);
    window.addEventListener('pointercancel', onPointerUpCapture, true);

    return () => {
      window.removeEventListener('pointerdown', onPointerDownCapture, true);
      window.removeEventListener('pointermove', onPointerMoveCapture, true);
      window.removeEventListener('pointerup', onPointerUpCapture, true);
      window.removeEventListener('pointercancel', onPointerUpCapture, true);
    };
  }, []);

  // --- 1. Mark pending restore when conversationId changes ---
  React.useLayoutEffect(() => {
    // Do NOT recompute/overwrite the snapshot here.
    // During a conversation switch, the DOM may transiently reset scrollTop
    // (especially with flex-col-reverse), and overwriting the cached snapshot
    // would destroy the user's last-known position.
    // We rely on handleScroll's continuous updates instead.
    pendingRestoreForRef.current = conversationId;
    prevConversationIdRef.current = conversationId;
  }, [conversationId]);

  // Reset progressive window only when switching conversations.
  // Keep current window during in-conversation mutations (delete/edit/new stream)
  // so the viewport does not jump back to the progressive-loading header.
  React.useLayoutEffect(() => {
    setRenderLimit(INITIAL_RENDER_LIMIT);
  }, [conversationId]);

  // --- 2. Restore scroll position after new conversation's messages render ---
  // Fires on every messages change.  Only acts when:
  //   a) We have a pending restore target, AND
  //   b) The messages fingerprint has actually changed (meaning the store
  //      swapped in the new conversation's data).
  // This avoids restoring on the old conversation's DOM.
  React.useLayoutEffect(() => {
    if (!pendingRestoreForRef.current) return;
    if (messagesFingerprint === prevMessagesFingerprintRef.current) {
      // Messages haven't changed yet — store still has old conversation data.
      return;
    }
    prevMessagesFingerprintRef.current = messagesFingerprint;

    const targetId = pendingRestoreForRef.current;
    pendingRestoreForRef.current = null;
    prevMessageCountRef.current = messages.length;
    renderedConvIdRef.current = targetId;

    const saved = targetId ? scrollPositionCache.get(targetId) : undefined;
    if (saved && containerRef.current) {
      if (restoreRafRef.current) {
        cancelAnimationFrame(restoreRafRef.current);
        restoreRafRef.current = null;
      }
      // Restore synchronously in useLayoutEffect — before paint.
      const el = containerRef.current;
      programmaticScrollRef.current = true;
      restoreScrollFromSnapshot(el, saved);
      setAutoScroll(saved.distFromBottom < 60);
      lastRestoreAtRef.current = Date.now();
      // After paint, content-visibility will expand real element heights,
      // changing scrollHeight. Refine position using message anchor.
      restoreRafRef.current = requestAnimationFrame(() => {
        programmaticScrollRef.current = true;
        refineScrollWithAnchor(el, saved);
        // Second pass: content-visibility may still be expanding
        requestAnimationFrame(() => {
          refineScrollWithAnchor(el, saved);
          programmaticScrollRef.current = false;
          restoreRafRef.current = null;
        });
      });
    } else {
      // No saved position — scroll to bottom (default for new conversations)
      if (containerRef.current) containerRef.current.scrollTop = 0;
      setAutoScroll(true);
    }
  }, [messages, conversationId, messagesFingerprint]);

  // --- 3. Scroll to a specific message (from search navigation) ---
  React.useEffect(() => {
    if (!scrollToMessageId || !containerRef.current) return;
    requestAnimationFrame(() => {
      const el = containerRef.current?.querySelector(`[data-message-id="${scrollToMessageId}"]`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setHighlightedMsgId(scrollToMessageId);
        setTimeout(() => setHighlightedMsgId(null), 2000);
      }
      clearScrollTarget();
    });
  }, [scrollToMessageId, clearScrollTarget]);

  // Track streaming content length for scroll trigger
  const streamingContentLength = streamingMessageId
    ? timelineItems.find((item) => item.message.message_id === streamingMessageId)?.message.content?.length ?? 0
    : 0;

  // --- 4. Auto-scroll during streaming or when a new message is added ---
  // In a column-reverse container, scrollTop=0 is the BOTTOM.
  React.useEffect(() => {
    // If we just restored scroll for a conversation switch, don't let the
    // messageCountChanged heuristic override it and snap back to bottom.
    if (Date.now() - lastRestoreAtRef.current < 500) {
      prevMessageCountRef.current = messages.length;
      return;
    }

    // During conversation switches we restore scroll in effect (2).
    // Do not let this effect force scrollTop=0 based on a transient
    // messageCountChanged (e.g. switching from a short chat to a long chat).
    if (pendingRestoreForRef.current) {
      prevMessageCountRef.current = messages.length;
      return;
    }

    // There is a very small window where conversationId has changed but
    // pendingRestoreForRef hasn't been set yet (before the layout effect runs).
    // In that window, do not mutate scroll/autoScroll state.
    if (conversationId !== renderedConvIdRef.current) {
      prevMessageCountRef.current = messages.length;
      return;
    }

    const isStreaming = !!streamingMessageId;
    const messageCountChanged = messages.length > prevMessageCountRef.current;

    if (messageCountChanged) {
      // New message added during normal chat flow — scroll to bottom
      setAutoScroll(true);
      if (containerRef.current) containerRef.current.scrollTop = 0;
      prevMessageCountRef.current = messages.length;
      return;
    }

    if (!autoScroll || !containerRef.current) {
      prevMessageCountRef.current = messages.length;
      return;
    }

    if (isStreaming || isWaitingForResponse) {
      containerRef.current.scrollTop = 0;
    }

    prevMessageCountRef.current = messages.length;
  }, [messages.length, autoScroll, streamingContentLength, streamingReasoningLength, streamingMessageId, isWaitingForResponse]);

  // Mark wheel events as user-initiated scrolling.  Wheel events don't
  // trigger pointerdown, so without this the post-restore protection guard
  // in handleScroll would suppress snapshot saving for wheel scrolls.
  const handleWheel = React.useCallback(() => {
    lastWheelAtRef.current = Date.now();
  }, []);

  // Detect manual scroll to disable auto-scroll.
  // In column-reverse, scrollTop is 0 at bottom and negative (or large) when scrolled up.
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    // During conversation switches, the DOM may emit scroll events while
    // renderedConvIdRef still points at the previous conversation (because
    // the store swaps conversationId first, then message list). If we record
    // snapshots here we can accidentally overwrite the saved position for the
    // previous conversation with a transient "bottom" snapshot.
    if (pendingRestoreForRef.current) return;
    if (conversationId !== renderedConvIdRef.current) return;
    if (programmaticScrollRef.current) return;

    // After a restore, content may still be loading (e.g. Mermaid diagrams).
    // During this window, scroll events are caused by layout shifts and React
    // re-renders, not user interaction. Don't update anything — the restore
    // already set autoScroll and the snapshot correctly.
    // EXCEPT: if the user is actively scrolling via wheel (lastWheelAtRef)
    // or pointer (isPointerDownInTimelineRef), honour their intent.
    const recentlyRestored = Date.now() - lastRestoreAtRef.current < 2000;
    const userWheelActive = Date.now() - lastWheelAtRef.current < 150;
    if (recentlyRestored && !isPointerDownInTimelineRef.current && !userWheelActive) {
      return;
    }

    lastUserScrollAtRef.current = Date.now();
    // In column-reverse, scrollTop=0 means at bottom.
    // When scrolled up, scrollTop becomes negative (some browsers) or
    // the distance from bottom increases.  We check if near bottom.
    const snap = getScrollSnapshot(el);
    const maxScroll = Math.max(0, el.scrollHeight - el.clientHeight);
    const distanceFromBottom = Math.min(maxScroll, snap.distFromBottom);
    const atBottom = distanceFromBottom < 60;
    setAutoScroll(atBottom);

    // Continuously save scroll position for the currently rendered conversation.
    // This is more reliable than saving only on conversation switch, because
    // BUG-041's store strategy may keep old messages visible while the new
    // conversation's data is being fetched.
    const cid = renderedConvIdRef.current;
    if (cid) {
      scrollPositionCache.set(cid, snap);
    }
  };

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
        <MessageCircle size={32} className="mb-3 opacity-40" />
        <p className="text-sm">Start a conversation</p>
        <p className="text-xs mt-1 opacity-60">Type a message below to begin</p>
      </div>
    );
  }

  const lastMsg = timelineItems[timelineItems.length - 1]?.message;

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      onWheel={handleWheel}
      data-testid="chat-timeline"
      className="flex-1 overflow-y-auto min-h-0 flex flex-col-reverse"
      style={{ scrollbarGutter: 'stable' }}
    >
      {/* Inner wrapper restores visual order: oldest at top, newest at bottom */}
      <div ref={contentRef} className="py-2">
        {messages.length > visibleMessages.length && (
          <div className="px-4 py-2">
            <button
              type="button"
              data-testid="chat-timeline-show-more"
              className="text-[11px] text-gray-500 hover:text-gray-300 underline"
              onClick={() => setRenderLimit((n) => Math.min(messages.length, n + RENDER_LIMIT_INCREMENT))}
            >
              Show more
            </button>
            <span className="ml-2 text-[10px] text-gray-600 tabular-nums">
              Showing {visibleMessages.length} / {messages.length}
            </span>
          </div>
        )}

        {timelineItems.map(({ message: msg, toolSteps }, index) => {
          const dividerKey = getDateDividerKey(msg.created_at);
          const prevMsg = timelineItems[index - 1]?.message;
          const shouldShowDateDivider = !prevMsg || getDateDividerKey(prevMsg.created_at) !== dividerKey;

          return (
            <React.Fragment key={msg.message_id}>
              {shouldShowDateDivider && (
                <div className="px-4 py-2" data-testid="chat-date-divider">
                  <div className="flex items-center gap-3 text-[10px] text-gray-500">
                    <div className="h-px flex-1 bg-gray-800" />
                    <span className="shrink-0 rounded-full border border-gray-700 bg-gray-900 px-2 py-0.5 tabular-nums">
                      {formatDateDivider(msg.created_at)}
                    </span>
                    <div className="h-px flex-1 bg-gray-800" />
                  </div>
                </div>
              )}
              <div
                data-message-id={msg.message_id}
                className={`houyi-message-item ${priorityIds.has(msg.message_id) ? 'houyi-message-item--priority' : ''} ${highlightedMsgId === msg.message_id ? 'bg-yellow-500/10 rounded-lg transition-colors duration-300' : ''}`}
              >
                <MessageBubble
                  message={msg}
                  toolSteps={toolSteps}
                  isStreaming={msg.message_id === streamingMessageId}
                  isLastMessage={Boolean(lastMsg && msg.message_id === lastMsg.message_id && !isWaitingForResponse)}
                  onOpenTrace={onOpenTrace}
                />
              </div>
            </React.Fragment>
          );
        })}
        {/* Ghost assistant bubble: shown after user sends a message but before
            the first SSE event arrives (no assistant message in the list yet) */}
        {isWaitingForResponse && (
          <div className="flex gap-3 px-4 py-3" data-testid="typing-indicator">
            <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-gray-600">
              <Bot size={14} />
            </div>
            <div className="flex flex-col items-start">
              <span className="text-[10px] text-gray-500 mb-1">Assistant</span>
              <div className="px-3 py-2 rounded-lg bg-gray-700">
                <TypingIndicator />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
