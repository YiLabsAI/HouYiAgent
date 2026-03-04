import React from 'react';
import { ChevronDown, ChevronRight, Wrench } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';

interface ToolCallBubbleProps {
  message: ChatMessage;
}

export const ToolCallBubble: React.FC<ToolCallBubbleProps> = ({ message }) => {
  const [expanded, setExpanded] = React.useState(false);
  const status = String(message.metadata?.tool_status || 'done');
  const statusClass =
    status === 'error'
      ? 'text-red-300 bg-red-500/10 border-red-500/30'
      : status === 'running'
        ? 'text-amber-300 bg-amber-500/10 border-amber-500/30'
        : 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30';

  const title = message.name || message.tool_call_id || 'tool';
  const preview = React.useMemo(() => {
    if (!message.content) return '(empty)';
    try {
      const parsed = JSON.parse(message.content);
      if (parsed && typeof parsed === 'object') {
        const success = (parsed as Record<string, unknown>).success;
        const payload = (parsed as Record<string, unknown>).data;
        const info = (parsed as Record<string, unknown>).message;
        if (typeof success === 'boolean' && success === false && typeof info === 'string' && info) {
          return info;
        }
        if (payload && typeof payload === 'object') {
          const p = payload as Record<string, unknown>;
          if (Array.isArray(p.matches)) return `matches: ${p.matches.length}`;
          if (Array.isArray(p.entries)) return `entries: ${p.entries.length}`;
          if (typeof p.path === 'string') return p.path;
        }
        return 'structured payload';
      }
    } catch {
      // Ignore parse failures and fallback to plain text preview.
    }
    const text = String(message.content).replace(/\s+/g, ' ').trim();
    return text.length > 120 ? `${text.slice(0, 120)}…` : text;
  }, [message.content]);

  return (
    <div className="w-full rounded-md border border-gray-700/80 bg-gray-900/60 px-2.5 py-2">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="min-w-0 flex items-center gap-2 text-[12px] text-gray-200">
          <Wrench size={13} className="shrink-0 text-gray-400" />
          <span className="truncate font-medium">{title}</span>
          <span className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${statusClass}`}>{status}</span>
        </span>
        {expanded ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
      </button>
      <div className="mt-1 text-[11px] text-gray-400">{preview}</div>
      {expanded && (
        <pre className="mt-2 max-h-56 overflow-auto rounded border border-gray-800 bg-gray-950/80 p-2 text-[11px] leading-relaxed text-gray-300 whitespace-pre-wrap break-words">
          {message.content || '(empty)'}
        </pre>
      )}
    </div>
  );
};
