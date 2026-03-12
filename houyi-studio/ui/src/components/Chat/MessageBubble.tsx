/**
 * MessageBubble: renders a single chat message.
 *
 * Supports user and assistant roles with distinct styling.
 * Assistant messages show streaming cursor when actively streaming.
 * Reasoning content is rendered in a collapsible section.
 */
import React from 'react';
import { User, Bot, ChevronDown, ChevronRight, Check, X, Send } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';
import { useChatStore } from '@/stores/useChatStore';
import { useSettingsStore } from '@/stores/useSettingsStore';
import { MarkdownRenderer } from './MarkdownRenderer';
import { MessageActionBar } from './MessageActionBar';
import { TypingIndicator } from './TypingIndicator';
import { ImageLightbox } from './ImageLightbox';
import { ToolCallBubble } from './ToolCallBubble';
import { useTypewriter } from '@/hooks/useTypewriter';

const formatDurationMs = (durationMs: number | null): string | null => {
  if (!Number.isFinite(durationMs) || !durationMs || durationMs <= 0) return null;
  return durationMs >= 1000
    ? `${(durationMs / 1000).toFixed(durationMs >= 10_000 ? 0 : 1)}s`
    : `${Math.round(durationMs)} ms`;
};

const sanitizeAssistantToolMarkers = (raw: string): string => {
  const hasToolMarker = /<\|tool_[^|]+\|>|<tool_call\b|<\/tool_call>|<arg_[^>]+>|<\/?think>/i.test(raw);
  if (!hasToolMarker) return raw;

  const stripped = raw
    .replace(/<tool_call[^>]*>[\s\S]*?<\/tool_call>/gi, ' ')
    .replace(/<tool_call[^>]*>/gi, ' ')
    .replace(/<\/tool_call>/gi, ' ')
    .replace(/<arg_[^>]+>[\s\S]*?<\/arg_[^>]+>/gi, ' ')
    .replace(/<\/?think>/gi, ' ')
    .replace(/<\|tool_calls_section_begin\|>/gi, ' ')
    .replace(/<\|tool_calls_section_end\|>/gi, ' ')
    .replace(/<\|tool_call_begin\|>/gi, ' ')
    .replace(/<\|tool_call_end\|>/gi, ' ')
    .replace(/<\|tool_call_argument_begin\|>/gi, ' ')
    .replace(/<\|tool_call_argument_end\|>/gi, ' ')
    .replace(/<\|tool_[^|]+\|>/gi, ' ')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  if (stripped) return stripped;

  return raw
    .replace(/<tool_call[^>]*>/gi, ' ')
    .replace(/<\/tool_call>/gi, ' ')
    .replace(/<arg_[^>]+>/gi, ' ')
    .replace(/<\/arg_[^>]+>/gi, ' ')
    .replace(/<\/?think>/gi, ' ')
    .replace(/<\|tool_calls_section_begin\|>/gi, ' ')
    .replace(/<\|tool_calls_section_end\|>/gi, ' ')
    .replace(/<\|tool_call_begin\|>/gi, ' ')
    .replace(/<\|tool_call_end\|>/gi, ' ')
    .replace(/<\|tool_call_argument_begin\|>/gi, ' ')
    .replace(/<\|tool_call_argument_end\|>/gi, ' ')
    .replace(/<\|tool_[^|]+\|>/gi, ' ')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

const looksLikePlainTextToolCommandReplay = (
  raw: string,
  toolSteps: ChatMessage[],
  hasAssistantToolCalls: boolean,
): boolean => {
  const text = raw.trim();
  if (!text) return false;
  if (!hasAssistantToolCalls && toolSteps.length === 0) return false;

  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0 || lines.length > 3) return false;

  const toolNames = new Set<string>();
  toolSteps.forEach((step) => {
    if (typeof step.name === 'string' && step.name.trim()) {
      toolNames.add(step.name.trim());
    }
  });

  const isCommandLine = (line: string): boolean => {
    const firstToken = line.split(/\s+/)[0] || '';
    const normalizedToolName = firstToken.replace(/^functions\./, '');
    const matchesKnownTool = toolNames.has(firstToken) || toolNames.has(normalizedToolName);
    const matchesHouyiTool = /^houyi_[a-z0-9_]+$/i.test(normalizedToolName);
    if (!matchesKnownTool && !matchesHouyiTool) return false;
    if (/[.!?。；;]\s*$/.test(line)) return false;
    if (/(^|\s)(let me|i will|next|then|because|looks like|让我|我来|看起来)/i.test(line)) {
      return false;
    }
    return line.split(/\s+/).length >= 2;
  };

  return lines.every(isCommandLine);
};

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
  isLastMessage?: boolean;
  toolSteps?: ChatMessage[];
  onOpenTrace?: (traceId: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isStreaming = false,
  isLastMessage = false,
  toolSteps = [],
  onOpenTrace,
}) => {
  const [showReasoning, setShowReasoning] = React.useState(false);
  const [isHovered, setIsHovered] = React.useState(false);
  const [showMetricsTooltip, setShowMetricsTooltip] = React.useState(false);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editText, setEditText] = React.useState('');
  const editRef = React.useRef<HTMLTextAreaElement>(null);
  const reasoningContentRef = React.useRef<HTMLDivElement>(null);
  const metricsTooltipTimerRef = React.useRef<number | null>(null);
  const editMessage = useChatStore((s) => s.editMessage);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const streamingReasoning = useChatStore((s) =>
    isStreaming ? s.streaming.reasoningBuffer : '',
  );
  const display = useSettingsStore((s) => s.display);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const isTool = message.role === 'tool';
  const [showToolSteps, setShowToolSteps] = React.useState(false);
  const rawContent = typeof message.content === 'string' ? message.content : String(message.content ?? '');
  const rawReasoning = typeof message.reasoning_content === 'string'
    ? message.reasoning_content
    : String(message.reasoning_content ?? '');
  const hasAssistantToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
  const markerSanitizedAssistantContent = isAssistant
    ? sanitizeAssistantToolMarkers(rawContent)
    : rawContent;
  const normalizedAssistantContent = isAssistant
    && looksLikePlainTextToolCommandReplay(
      markerSanitizedAssistantContent,
      toolSteps,
      hasAssistantToolCalls,
    )
    ? ''
    : markerSanitizedAssistantContent;
  const normalizedAssistantReasoning = isAssistant
    ? sanitizeAssistantToolMarkers(rawReasoning)
    : rawReasoning;
  const normalizedStreamingReasoning = isAssistant
    ? sanitizeAssistantToolMarkers(streamingReasoning)
    : streamingReasoning;
  const shouldRenderAssistantContent = !isAssistant
    || Boolean(normalizedAssistantContent.trim())
    || (isStreaming && !normalizedStreamingReasoning);
  const shouldAutoExpandReasoning = isStreaming && !rawContent;
  const traceId = typeof message.metadata?.trace_id === 'string' ? message.metadata.trace_id : null;
  const usagePromptTokens = Number(message.metadata?.usage?.prompt_tokens || 0);
  const usageCompletionTokens = Number(message.metadata?.usage?.completion_tokens || 0);
  const usageTotalTokens = Number(message.metadata?.usage?.total_tokens || 0);
  const usageInputTokens = Number(message.metadata?.usage?.input_tokens || 0);
  const budget = message.metadata?.budget as Record<string, any> | undefined;
  const budgetGuardrailApplied = Boolean(budget?.max_tokens_guardrail_applied);
  const firstTokenLatencyMs = Number(message.metadata?.first_token_latency_ms || 0);
  const tokensPerSecond = Number(message.metadata?.tokens_per_second || 0);
  const decodeTokensPerSecond = Number(message.metadata?.decode_tokens_per_second || 0);
  const endToEndTokensPerSecond = Number(message.metadata?.end_to_end_tokens_per_second || 0);
  const hasToolSummary = toolSteps.length > 0;
  const canRevealToolSteps = hasToolSummary;
  const assistantMetricCount = [
    usageTotalTokens > 0,
    usagePromptTokens > 0,
    usageCompletionTokens > 0,
    firstTokenLatencyMs > 0,
    tokensPerSecond > 0,
  ].filter(Boolean).length;
  const shouldShowMetaPanel = isAssistant
    && (traceId || assistantMetricCount > 0 || canRevealToolSteps);
  const shouldShowUserMetaPanel = isUser && usageInputTokens > 0;
  const roundCount = toolSteps.length > 0
    ? new Set(toolSteps.map((step) => Number(step.metadata?.round_index || 0)).filter((v) => Number.isFinite(v) && v > 0)).size
    : 0;
  const hoverMetricCount = [
    firstTokenLatencyMs > 0,
    tokensPerSecond > 0,
    decodeTokensPerSecond > 0,
    endToEndTokensPerSecond > 0,
  ].filter(Boolean).length;
  const toolStatusCounts = toolSteps.reduce<Record<string, number>>((acc, step) => {
    const rawStatus = String(step.metadata?.tool_status || '').trim().toLowerCase();
    const status = rawStatus || 'ok';
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {});
  const hasErroredTool = Object.keys(toolStatusCounts).some((status) => status === 'error' || status === 'failed');
  const hasRunningTool = Object.keys(toolStatusCounts).some((status) => status === 'running' || status === 'pending');
  const toolActivityStatus = hasErroredTool ? 'error' : hasRunningTool ? 'running' : 'done';
  const toolActivityStatusClass = toolActivityStatus === 'error'
    ? 'text-red-300 border-red-500/30 bg-red-500/10'
    : toolActivityStatus === 'running'
      ? 'text-amber-300 border-amber-500/30 bg-amber-500/10'
      : 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10';
  const totalToolDurationMs = toolSteps.reduce((sum, step) => {
    const durationMs = Number(step.metadata?.duration_ms);
    return Number.isFinite(durationMs) && durationMs > 0 ? sum + durationMs : sum;
  }, 0);
  const formattedToolDuration = formatDurationMs(totalToolDurationMs || null);
  const toolActivityLabel = `${toolSteps.length} tool${toolSteps.length > 1 ? 's' : ''}`;
  const isEmptyAssistantPlaceholder =
    isAssistant
    && !isStreaming
    && !normalizedAssistantContent.trim()
    && !(normalizedAssistantReasoning.trim())
    && (!message.attachments || message.attachments.length === 0)
    && toolSteps.length === 0
    && !hasAssistantToolCalls;

  // Keep hook order stable across placeholder/non-placeholder transitions.
  const displayContent = useTypewriter(normalizedAssistantContent, isStreaming && isAssistant);

  React.useLayoutEffect(() => {
    if (!isStreaming || !(showReasoning || shouldAutoExpandReasoning)) return;
    const panel = reasoningContentRef.current;
    if (!panel) return;
    panel.scrollTop = panel.scrollHeight;
  }, [isStreaming, showReasoning, shouldAutoExpandReasoning, normalizedStreamingReasoning, normalizedAssistantReasoning]);

  React.useEffect(() => {
    return () => {
      if (metricsTooltipTimerRef.current !== null) {
        window.clearTimeout(metricsTooltipTimerRef.current);
      }
    };
  }, []);

  if (isEmptyAssistantPlaceholder) {
    return null;
  }

  const handleStartEdit = () => {
    setEditText(rawContent);
    setIsEditing(true);
    setTimeout(() => editRef.current?.focus(), 0);
  };

  const handleSaveEdit = async () => {
    const trimmed = editText.trim();
    if (trimmed && trimmed !== message.content) {
      await editMessage(message.message_id, trimmed);
    }
    setIsEditing(false);
  };

  const handleSaveAndResend = async () => {
    const trimmed = editText.trim();
    if (!trimmed) return;
    if (trimmed !== message.content) {
      await editMessage(message.message_id, trimmed);
    }
    setIsEditing(false);
    sendMessage(trimmed);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
  };

  const handleMetricsEnter = () => {
    if (metricsTooltipTimerRef.current !== null) {
      window.clearTimeout(metricsTooltipTimerRef.current);
    }
    metricsTooltipTimerRef.current = window.setTimeout(() => {
      setShowMetricsTooltip(true);
      metricsTooltipTimerRef.current = null;
    }, 90);
  };

  const handleMetricsLeave = () => {
    if (metricsTooltipTimerRef.current !== null) {
      window.clearTimeout(metricsTooltipTimerRef.current);
    }
    metricsTooltipTimerRef.current = window.setTimeout(() => {
      setShowMetricsTooltip(false);
      metricsTooltipTimerRef.current = null;
    }, 120);
  };

  return (
    <div
      className={`relative flex gap-3 px-4 py-3 ${isUser ? 'flex-row-reverse' : ''}`}
      data-testid="message-bubble"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Avatar */}
      <div
        className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-600' : 'bg-gray-600'
        }`}
      >
        {isUser
          ? (display.user_avatar ? <span className="text-sm">{display.user_avatar}</span> : <User size={14} />)
          : (display.assistant_avatar ? <span className="text-sm">{display.assistant_avatar}</span> : <Bot size={14} />)
        }
      </div>

      {/* Content */}
      <div className={`flex flex-col min-w-0 flex-1 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Role label */}
        <span className="text-[10px] text-gray-500 mb-1">
          {isUser ? display.user_name : display.assistant_name}
        </span>

        {/* Reasoning (collapsible, auto-expand during streaming) */}
        {isAssistant && (normalizedAssistantReasoning || (isStreaming && normalizedStreamingReasoning)) && (
          <div className="mb-1 w-full">
            <button
              onClick={() => setShowReasoning((v) => !v)}
              className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
              type="button"
            >
              {(showReasoning || shouldAutoExpandReasoning) ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              <span className="flex items-center gap-1">
                Thinking
                {isStreaming && normalizedStreamingReasoning && !normalizedAssistantContent && (
                  <span className="inline-block w-1 h-1 bg-blue-400 rounded-full animate-pulse" />
                )}
              </span>
            </button>
            {(showReasoning || shouldAutoExpandReasoning) && (
              <div
                ref={reasoningContentRef}
                className="mt-1 pl-3 py-1.5 border-l-2 border-gray-500 text-[11px] text-gray-400 whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto"
              >
                {normalizedAssistantReasoning || normalizedStreamingReasoning}
                {isStreaming && !normalizedAssistantContent && (
                  <span className="inline-block w-1 h-3 ml-0.5 bg-gray-400 animate-pulse rounded-sm" />
                )}
              </div>
            )}
          </div>
        )}

        {/* Image attachments with lightbox */}
        {message.attachments && message.attachments.length > 0 && (
          <AttachmentGallery attachments={message.attachments} isUser={isUser} />
        )}

        {/* Message content */}
        {isTool ? (
          <ToolCallBubble message={message} />
        ) : isEditing ? (
          <div className="w-full">
            <textarea
              ref={editRef}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-[13px] text-gray-100 resize-none focus:outline-none focus:border-blue-500"
              rows={Math.max(3, Math.min(12, editText.split('\n').length + 1))}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSaveEdit();
                }
                if (e.key === 'Escape') handleCancelEdit();
              }}
            />
            <div className="flex gap-1 mt-1">
              <button
                onClick={handleSaveEdit}
                className="flex items-center gap-1 px-2 py-0.5 bg-blue-600 hover:bg-blue-700 rounded text-[11px] text-white transition-colors"
                type="button"
              >
                <Check size={11} /> Save
              </button>
              <button
                onClick={handleSaveAndResend}
                className="flex items-center gap-1 px-2 py-0.5 bg-green-600 hover:bg-green-700 rounded text-[11px] text-white transition-colors"
                type="button"
                title="Save changes and resend this message"
              >
                <Send size={11} /> Save & Resend
              </button>
              <button
                onClick={handleCancelEdit}
                className="flex items-center gap-1 px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-[11px] text-gray-300 transition-colors"
                type="button"
              >
                <X size={11} /> Cancel
              </button>
            </div>
          </div>
        ) : shouldRenderAssistantContent ? (
          <div
            className={`px-3 py-2 rounded-lg text-[13px] leading-relaxed break-words ${
              isUser
                ? 'bg-blue-600 text-white whitespace-pre-wrap'
                : 'bg-gray-700 text-gray-100 markdown-body'
            }`}
          >
            {isAssistant ? (
              normalizedAssistantContent ? (
                <>
                  <MarkdownRenderer content={displayContent} />
                  {isStreaming && (
                    <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-300 animate-pulse rounded-sm" />
                  )}
                </>
              ) : isStreaming && !normalizedStreamingReasoning ? (
                <TypingIndicator />
              ) : null
            ) : (
              rawContent
            )}
          </div>
        ) : null}

        {isAssistant && hasToolSummary && (
          <div className="mt-2 w-full rounded-lg border border-cyan-500/20 bg-gray-900/60 px-2.5 py-2">
            <button
              type="button"
              className="flex w-full items-center justify-between gap-2 text-left"
              onClick={() => setShowToolSteps((v) => !v)}
              aria-expanded={showToolSteps}
              aria-label={`Tool activity ${toolSteps.length}`}
            >
              <span className="min-w-0 flex items-center gap-2">
                <span className="text-[12px] font-medium text-gray-200">Tool activity</span>
                <span className="text-[11px] text-gray-400">{toolActivityLabel}</span>
                <span className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${toolActivityStatusClass}`}>
                  {toolActivityStatus}
                </span>
                {hasRunningTool && (
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
                )}
              </span>
              {showToolSteps ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
            </button>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
              <span>{toolActivityLabel}</span>
              {formattedToolDuration && <span>{`Duration ${formattedToolDuration}`}</span>}
              {roundCount > 0 && <span>{`Rounds ${roundCount}`}</span>}
            </div>
            {showToolSteps && canRevealToolSteps && (
              <div className="mt-2 space-y-1.5 rounded-md border border-gray-700/80 bg-gray-900/50 p-1.5">
                {toolSteps.map((step) => (
                  <ToolCallBubble key={step.message_id} message={step} />
                ))}
              </div>
            )}
          </div>
        )}

        {shouldShowUserMetaPanel && (
          <div className="mt-1.5 flex items-center gap-1.5 text-[10px] text-gray-300">
            <span className="text-[11px] text-gray-500 tabular-nums">
              {`Tokens: ${usageInputTokens}`}
            </span>
          </div>
        )}

        {/* Timestamp + edited indicator */}
        {shouldShowMetaPanel && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] text-gray-300">
            {usageTotalTokens > 0 && (
              <div
                className="relative"
                onMouseEnter={handleMetricsEnter}
                onMouseLeave={handleMetricsLeave}
              >
                <span className="text-[11px] text-gray-500 tabular-nums">
                  {`Tokens: ${usageTotalTokens}`}
                  {usagePromptTokens > 0 && ` ↑${usagePromptTokens}`}
                  {usageCompletionTokens > 0 && ` ↓${usageCompletionTokens}`}
                </span>
                {hoverMetricCount > 0 && showMetricsTooltip && (
                  <div className={`pointer-events-none absolute ${isUser ? 'right-0' : 'left-0'} ${isLastMessage ? 'top-full mt-1' : 'bottom-full mb-1'} z-10 whitespace-nowrap rounded-2xl bg-white px-4 py-3 text-[12px] text-gray-800 shadow-lg ring-1 ring-black/5`}>
                    <div className="flex items-center gap-1">
                      {firstTokenLatencyMs > 0 && <span>{`First token ${Math.round(firstTokenLatencyMs)} ms`}</span>}
                      {firstTokenLatencyMs > 0 && (tokensPerSecond > 0 || decodeTokensPerSecond > 0 || endToEndTokensPerSecond > 0) && <span>|</span>}
                      {decodeTokensPerSecond > 0 && <span>{`Decode ${Math.round(decodeTokensPerSecond)} tokens/s`}</span>}
                      {decodeTokensPerSecond > 0 && endToEndTokensPerSecond > 0 && <span>|</span>}
                      {endToEndTokensPerSecond > 0 && <span>{`E2E ${Math.round(endToEndTokensPerSecond)} tokens/s`}</span>}
                      {endToEndTokensPerSecond === 0 && tokensPerSecond > 0 && <span>{`${Math.round(tokensPerSecond)} tokens/s`}</span>}
                    </div>
                    <div className={`absolute ${isLastMessage ? '-top-1.5' : '-bottom-1.5'} h-3 w-3 rotate-45 bg-white ring-1 ring-black/5 ${isUser ? 'right-5' : 'left-5'}`} />
                  </div>
                )}
              </div>
            )}
            {roundCount > 0 && (
              <span className="rounded-md border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-violet-200">
                Rounds {roundCount}
              </span>
            )}
            {budgetGuardrailApplied && (
              <span className="rounded-md border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 text-rose-200">
                Guardrail
              </span>
            )}
            {traceId && onOpenTrace && (
              <button
                type="button"
                className="rounded-md border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-violet-200 hover:bg-violet-500/20"
                onClick={() => onOpenTrace(traceId)}
              >
                View trace
              </button>
            )}
          </div>
        )}

        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-600">
            {new Date(message.created_at * 1000).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {message.metadata?.edited && (
            <span className="text-[9px] text-gray-600 italic">(edited)</span>
          )}
        </div>

        {/* Action bar: always rendered to preserve layout; visibility toggled to avoid content shift */}
        {!isTool && !isStreaming && !isEditing && (
          <div className={`mt-0.5 ${isUser ? 'self-end' : 'self-start'} transition-opacity duration-100 ${
            isLastMessage || isHovered ? 'opacity-100' : 'opacity-0 pointer-events-none'
          }`}>
            <MessageActionBar
              message={message}
              onStartEdit={isUser ? handleStartEdit : undefined}
            />
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * AttachmentGallery: renders image thumbnails with lightbox and non-image file chips.
 */
const AttachmentGallery: React.FC<{ attachments: import('@/types/chat').Attachment[]; isUser: boolean }> = ({ attachments, isUser }) => {
  const [lightboxSrc, setLightboxSrc] = React.useState<{ src: string; alt: string } | null>(null);
  const images = attachments.filter(a => a.mime_type.startsWith('image/'));
  const files = attachments.filter(a => !a.mime_type.startsWith('image/'));

  return (
    <>
      <div className={`flex flex-wrap gap-2 mb-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {images.map((att, i) => (
          <img
            key={`${att.filename}-${i}`}
            src={att.data}
            alt={att.filename}
            className="max-w-[240px] max-h-[180px] rounded-lg object-cover border border-gray-600 cursor-zoom-in hover:opacity-90 transition-opacity"
            title="Click to enlarge"
            onClick={() => setLightboxSrc({ src: att.data, alt: att.filename })}
          />
        ))}
        {files.map((att, i) => (
          <div
            key={`file-${att.filename}-${i}`}
            className="flex items-center gap-1.5 px-2 py-1 bg-gray-700 rounded text-[11px] text-gray-300"
          >
            <span className="opacity-60">📎</span>
            <span className="max-w-[150px] truncate">{att.filename}</span>
            <span className="text-gray-500">({Math.round(att.size / 1024)}KB)</span>
          </div>
        ))}
      </div>
      {lightboxSrc && (
        <ImageLightbox
          src={lightboxSrc.src}
          alt={lightboxSrc.alt}
          onClose={() => setLightboxSrc(null)}
        />
      )}
    </>
  );
};
