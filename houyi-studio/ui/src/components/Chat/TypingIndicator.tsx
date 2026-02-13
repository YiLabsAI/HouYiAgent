/**
 * TypingIndicator: animated three-dot "thinking" indicator.
 *
 * Shown while waiting for an assistant response (both streaming and
 * non-streaming modes).  Each dot bounces with a staggered delay to
 * create a smooth wave effect.
 */
import React from 'react';

export const TypingIndicator: React.FC = () => (
  <span className="inline-flex items-center gap-[3px] py-1" aria-label="Assistant is thinking">
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        className="w-[6px] h-[6px] rounded-full bg-gray-400"
        style={{
          animation: 'typing-bounce 1.2s ease-in-out infinite',
          animationDelay: `${i * 0.2}s`,
        }}
      />
    ))}
  </span>
);
