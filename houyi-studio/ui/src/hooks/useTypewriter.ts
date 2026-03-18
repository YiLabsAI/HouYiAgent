/**
 * useTypewriter: smoothly animates large text chunks during streaming.
 *
 * Problem: Gemini API sends large chunks (entire sentences/paragraphs at once),
 * causing text to "jump" instead of flowing smoothly like a typewriter.
 *
 * Solution: When streaming, the hook uses setInterval (not rAF) to progressively
 * reveal `targetText`.  The interval self-clears when caught up and restarts when
 * new content arrives.  When not streaming, returns full text immediately.
 *
 * Uses setInterval instead of requestAnimationFrame because:
 * - rAF never stops firing while scheduled, blocking Playwright's idle detection
 * - setInterval with self-clear is deterministic and test-friendly
 */
import { useState, useEffect, useRef } from 'react';

const CHARS_PER_TICK = 1;
const TICK_INTERVAL_MS = 16; // ~60fps
const CATCH_UP_MULTIPLIER = 3;

export function useTypewriter(targetText: string, isStreaming: boolean): string {
  // When mounting mid-stream (e.g. switching back to a conversation that is
  // still streaming), skip to near the end so we don't re-animate everything
  // from scratch.  Show at most the last ~40 chars as typewriter reveal.
  const [displayedLen, setDisplayedLen] = useState(() => {
    if (isStreaming && targetText.length > 40) {
      return targetText.length - 40;
    }
    return isStreaming ? 0 : targetText.length;
  });
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const targetLenRef = useRef(targetText.length);

  targetLenRef.current = targetText.length;

  // When streaming stops, show full content and clear interval.
  useEffect(() => {
    if (!isStreaming) {
      setDisplayedLen(targetText.length);
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
  }, [isStreaming, targetText.length]);

  // Start/restart interval when new content arrives during streaming.
  useEffect(() => {
    if (!isStreaming) return;
    // Already running — the existing interval will pick up the new target
    if (intervalRef.current !== null) return;

    intervalRef.current = setInterval(() => {
      setDisplayedLen((prev) => {
        const target = targetLenRef.current;
        if (prev >= target) {
          // Caught up — stop interval.  Will restart when targetText grows.
          if (intervalRef.current !== null) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          return prev;
        }
        const gap = target - prev;
        const step = gap > 80 ? CHARS_PER_TICK * CATCH_UP_MULTIPLIER : CHARS_PER_TICK;
        return Math.min(prev + step, target);
      });
    }, TICK_INTERVAL_MS);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isStreaming, targetText.length]);

  if (isStreaming) {
    return targetText.slice(0, displayedLen);
  }
  return targetText;
}
