/**
 * MessageBubble: renders a single chat message.
 *
 * Supports user and assistant roles with distinct styling.
 * Assistant messages show streaming cursor when actively streaming.
 * Reasoning content is rendered in a collapsible section.
 */
import React from 'react';
import { createPortal } from 'react-dom';
import { User, Bot, ChevronDown, ChevronRight, Check, X, Send } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';
import { useChatStore } from '@/stores/useChatStore';
import { MarkdownRenderer } from './MarkdownRenderer';
import { MessageActionBar } from './MessageActionBar';
import { TypingIndicator } from './TypingIndicator';
import { ImageLightbox } from './ImageLightbox';
import { ToolCallBubble } from './ToolCallBubble';
import { useTypewriter } from '@/hooks/useTypewriter';
import { useSettingsStore } from '@/stores/useSettingsStore';

const formatDurationMs = (durationMs: number | null): string | null => {
  if (!Number.isFinite(durationMs) || !durationMs || durationMs <= 0) return null;
  return durationMs >= 1000
    ? `${(durationMs / 1000).toFixed(durationMs >= 10_000 ? 0 : 1)}s`
    : `${Math.round(durationMs)}ms`;
};

const resolveTimingMetric = (...candidates: unknown[]): number => {
  for (const candidate of candidates) {
    if (candidate === undefined || candidate === null || candidate === '') continue;
    const parsed = Number(candidate);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return 0;
};

const sanitizeAssistantToolMarkers = (raw: string): string => {
  const hasToolMarker = /\[tool_call\]|\[tool call\]|<\|tool_[^|]+\|>|<tool_call\b|<\/tool_call>|<arg_[^>]+>|<\/?think>|(?:^|\n)\s*tool\s*:\s*[a-zA-Z_][\w.-]*\s*&args\s*:/i.test(raw);
  if (!hasToolMarker) return raw;

  const stripped = raw
    .replace(/\[tool_call\]/gi, ' ')
    .replace(/\[tool call\]/gi, ' ')
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
    .replace(/(?:^|\n)\s*tool\s*:\s*[a-zA-Z_][\w.-]*\s*&args\s*:\s*[^\n]*/gi, ' ')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  if (stripped) return stripped;

  return raw
    .replace(/\[tool_call\]/gi, ' ')
    .replace(/\[tool call\]/gi, ' ')
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
    .replace(/(?:^|\n)\s*tool\s*:\s*[a-zA-Z_][\w.-]*\s*&args\s*:\s*[^\n]*/gi, ' ')
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

const extractBracketToolReplay = (
  raw: string,
  message: ChatMessage,
): { content: string; syntheticToolSteps: ChatMessage[] } => {
  const text = String(raw || '');
  const match = text.match(/(^|\n)\s*\[tool:([^\]]+)\]\s*([\s\S]*)$/i);
  if (!match) {
    return { content: text, syntheticToolSteps: [] };
  }

  const toolName = String(match[2] || '').trim();
  const payload = String(match[3] || '').trim();
  if (!toolName || !payload) {
    return { content: text, syntheticToolSteps: [] };
  }

  const prefix = text.slice(0, match.index).trimEnd();
  return {
    content: prefix,
    syntheticToolSteps: [
      {
        message_id: `${message.message_id}-synthetic-tool-replay`,
        role: 'tool',
        content: payload,
        name: toolName,
        metadata: {
          tool_status: 'ok',
        },
        created_at: message.created_at,
      },
    ],
  };
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
  toolSteps = [],
  onOpenTrace,
}) => {
  const [showReasoning, setShowReasoning] = React.useState(false);
  const [showMetricsTooltip, setShowMetricsTooltip] = React.useState(false);
  const [metricsTooltipRect, setMetricsTooltipRect] = React.useState<{ left: number; top: number; arrowLeft: number } | null>(null);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editText, setEditText] = React.useState('');
  const editRef = React.useRef<HTMLTextAreaElement>(null);
  const reasoningContentRef = React.useRef<HTMLDivElement>(null);
  const metricsAnchorRef = React.useRef<HTMLDivElement>(null);
  const metricsTooltipTimerRef = React.useRef<number | null>(null);
  const editMessage = useChatStore((s) => s.editMessage);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const streamingReasoning = useChatStore((s) =>
    isStreaming ? s.streaming.reasoningBuffer : '',
  );
  const displaySettings = useSettingsStore((s) => s.display);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const isTool = message.role === 'tool';
  const [showToolSteps, setShowToolSteps] = React.useState(false);
  const rawContent = typeof message.content === 'string' ? message.content : String(message.content ?? '');
  const rawReasoning = typeof message.reasoning_content === 'string'
    ? message.reasoning_content
    : String(message.reasoning_content ?? '');
  const bracketToolReplay = isAssistant ? extractBracketToolReplay(rawContent, message) : { content: rawContent, syntheticToolSteps: [] };
  const effectiveToolSteps = toolSteps.length > 0 ? toolSteps : bracketToolReplay.syntheticToolSteps;
  const hasAssistantToolCalls = Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
  const markerSanitizedAssistantContent = isAssistant
    ? sanitizeAssistantToolMarkers(bracketToolReplay.content)
    : rawContent;
  const normalizedAssistantContent = isAssistant
    && looksLikePlainTextToolCommandReplay(
      markerSanitizedAssistantContent,
      effectiveToolSteps,
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
  const userLabel = displaySettings.user_name?.trim() || 'You';
  const userAvatar = displaySettings.user_avatar?.trim() || null;
  const assistantLabel = displaySettings.assistant_name?.trim() || 'Assistant';
  const assistantAvatar = displaySettings.assistant_avatar?.trim() || null;
  const shouldRenderAssistantContent = !isAssistant
    || Boolean(normalizedAssistantContent.trim())
    || (isStreaming && !normalizedStreamingReasoning);
  const shouldAutoExpandReasoning = isStreaming && !rawContent;
  const traceId = typeof message.metadata?.trace_id === 'string' ? message.metadata.trace_id : null;
  const messageMetadata = (message.metadata as Record<string, any> | undefined) ?? undefined;
  const usageMetrics = (messageMetadata?.usage as Record<string, any> | undefined)
    ?? undefined;
  const usagePromptTokens = Number(message.metadata?.usage?.prompt_tokens || usageMetrics?.prompt_tokens || 0);
  const usageCompletionTokens = Number(message.metadata?.usage?.completion_tokens || usageMetrics?.completion_tokens || 0);
  const usageTotalTokens = Number(message.metadata?.usage?.total_tokens || usageMetrics?.total_tokens || 0);
  const usageInputTokens = Number(message.metadata?.usage?.input_tokens || usageMetrics?.input_tokens || 0);
  const budget = message.metadata?.budget as Record<string, any> | undefined;
  const budgetGuardrailApplied = Boolean(budget?.max_tokens_guardrail_applied);
  const firstTokenLatencyMs = resolveTimingMetric(
    messageMetadata?.first_token_latency_ms,
    messageMetadata?.first_token_ms,
    usageMetrics?.first_token_latency_ms,
    usageMetrics?.first_token_ms,
  );
  const decodeTokensPerSecond = resolveTimingMetric(
    messageMetadata?.decode_tokens_per_second,
    usageMetrics?.decode_tokens_per_second,
  );
  const tokensPerSecond = resolveTimingMetric(
    messageMetadata?.tokens_per_second,
    usageMetrics?.tokens_per_second,
  );
  const endToEndTokensPerSecond = resolveTimingMetric(
    messageMetadata?.end_to_end_tokens_per_second,
    usageMetrics?.end_to_end_tokens_per_second,
  );
  const throughputTokensPerSecond = endToEndTokensPerSecond || decodeTokensPerSecond || tokensPerSecond;
  const hasToolSummary = effectiveToolSteps.length > 0;
  const canRevealToolSteps = hasToolSummary;
  const assistantMetricCount = [
    usageTotalTokens > 0,
    usagePromptTokens > 0,
    usageCompletionTokens > 0,
    firstTokenLatencyMs > 0,
    throughputTokensPerSecond > 0,
  ].filter(Boolean).length;
  const shouldShowMetaPanel = isAssistant
    && (traceId || assistantMetricCount > 0 || canRevealToolSteps || budgetGuardrailApplied);
  const shouldShowUserMetaPanel = isUser && usageInputTokens > 0;
  const roundCount = effectiveToolSteps.length > 0
    ? new Set(effectiveToolSteps.map((step) => Number(step.metadata?.round_index || 0)).filter((v) => Number.isFinite(v) && v > 0)).size
    : 0;
  const hoverMetricCount = [
    firstTokenLatencyMs > 0,
    throughputTokensPerSecond > 0,
  ].filter(Boolean).length;
  const showLegacyTokenTooltip = hoverMetricCount === 0 && usageTotalTokens > 0;
  const metricsAnchorLabel = usageTotalTokens > 0
    ? `${`Tokens: ${usageTotalTokens}`}${usagePromptTokens > 0 ? ` ↑${usagePromptTokens}` : ''}${usageCompletionTokens > 0 ? ` ↓${usageCompletionTokens}` : ''}`
    : 'Metrics';
  const toolStatusCounts = effectiveToolSteps.reduce<Record<string, number>>((acc, step) => {
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
  const toolActivityErrorSummary = React.useMemo(() => {
    if (!hasErroredTool) return null;
    for (const step of effectiveToolSteps) {
      const rawStatus = String(step.metadata?.tool_status || '').trim().toLowerCase();
      if (rawStatus !== 'error' && rawStatus !== 'failed') continue;
      try {
        const parsed = JSON.parse(String(step.content || '')) as Record<string, unknown>;
        const data = parsed.data && typeof parsed.data === 'object'
          ? parsed.data as Record<string, unknown>
          : null;
        const candidates = [parsed.message, parsed.error, data?.message, data?.error, data?.stderr];
        for (const candidate of candidates) {
          if (typeof candidate === 'string' && candidate.trim()) {
            const compact = candidate.replace(/\s+/g, ' ').trim();
            return compact.length > 72 ? `${compact.slice(0, 72)}…` : compact;
          }
        }
      } catch {
        const compact = String(step.content || '').replace(/\s+/g, ' ').trim();
        if (compact) return compact.length > 72 ? `${compact.slice(0, 72)}…` : compact;
      }
    }
    return null;
  }, [effectiveToolSteps, hasErroredTool]);
  const totalToolDurationMs = effectiveToolSteps.reduce((sum, step) => {
    const durationMs = Number(step.metadata?.duration_ms);
    return Number.isFinite(durationMs) && durationMs > 0 ? sum + durationMs : sum;
  }, 0);
  const formattedToolDuration = formatDurationMs(totalToolDurationMs || null);
  const toolActivityLabel = `${effectiveToolSteps.length} tool${effectiveToolSteps.length > 1 ? 's' : ''}`;
  const isEmptyAssistantPlaceholder =
    isAssistant
    && !isStreaming
    && !normalizedAssistantContent.trim()
    && !(normalizedAssistantReasoning.trim())
    && (!message.attachments || message.attachments.length === 0)
    && effectiveToolSteps.length === 0
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

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditText(rawContent);
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

  const updateMetricsTooltipRect = React.useCallback(() => {
    const anchor = metricsAnchorRef.current;
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    setMetricsTooltipRect({
      left: rect.left,
      top: rect.top,
      arrowLeft: Math.min(20, Math.max(12, rect.width / 2)),
    });
  }, []);

  const handleMetricsEnter = () => {
    if (metricsTooltipTimerRef.current != null) {
      window.clearTimeout(metricsTooltipTimerRef.current);
    }
    metricsTooltipTimerRef.current = window.setTimeout(() => {
      updateMetricsTooltipRect();
      setShowMetricsTooltip(true);
      metricsTooltipTimerRef.current = null;
    }, 90);
  };

  const handleMetricsLeave = () => {
    if (metricsTooltipTimerRef.current != null) {
      window.clearTimeout(metricsTooltipTimerRef.current);
    }
    metricsTooltipTimerRef.current = window.setTimeout(() => {
      setShowMetricsTooltip(false);
      setMetricsTooltipRect(null);
      metricsTooltipTimerRef.current = null;
    }, 120);
  };

  React.useEffect(() => {
    if (!showMetricsTooltip) return undefined;
    updateMetricsTooltipRect();
    window.addEventListener('scroll', updateMetricsTooltipRect, true);
    window.addEventListener('resize', updateMetricsTooltipRect);
    return () => {
      window.removeEventListener('scroll', updateMetricsTooltipRect, true);
      window.removeEventListener('resize', updateMetricsTooltipRect);
    };
  }, [showMetricsTooltip, updateMetricsTooltipRect]);

  const metricsTooltip = showMetricsTooltip
    && metricsTooltipRect
    && (hoverMetricCount > 0 || showLegacyTokenTooltip)
    && typeof document !== 'undefined'
    ? createPortal(
      <div
        className="pointer-events-none fixed z-[100] whitespace-nowrap rounded-2xl bg-white px-4 py-3 text-[12px] text-gray-800 shadow-lg ring-1 ring-black/5"
        style={{ left: metricsTooltipRect.left, top: metricsTooltipRect.top - 12, transform: 'translateY(-100%)' }}
      >
        {hoverMetricCount > 0 ? (
          <div className="flex items-center gap-1">
            {firstTokenLatencyMs > 0 && <span>{`First token ${Math.round(firstTokenLatencyMs)} ms`}</span>}
            {firstTokenLatencyMs > 0 && throughputTokensPerSecond > 0 && <span>|</span>}
            {throughputTokensPerSecond > 0 && <span>{`Throughput ${Math.round(throughputTokensPerSecond)} tokens/s`}</span>}
          </div>
        ) : (
          <div className="flex items-center gap-1 font-medium text-gray-900">
            <span>{`Tokens ${usageTotalTokens}`}</span>
            {usagePromptTokens > 0 && <span>{`↑${usagePromptTokens}`}</span>}
            {usageCompletionTokens > 0 && <span>{`↓${usageCompletionTokens}`}</span>}
          </div>
        )}
        <div
          className="absolute -bottom-1.5 h-3 w-3 rotate-45 bg-white ring-1 ring-black/5"
          style={{ left: metricsTooltipRect.arrowLeft }}
        />
      </div>,
      document.body,
    )
    : null;

  return (
    <>
      <div
        data-testid="message-bubble"
        className={`group flex w-full min-w-0 gap-3 overflow-x-hidden px-4 py-3 ${isUser ? 'justify-end' : 'justify-start'}`}
      >
        {!isUser && (
          <div className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center ${isTool ? 'bg-cyan-700' : 'bg-gray-600'}`}>
            {isTool ? <ChevronRight size={14} /> : assistantAvatar ? <span className="text-[14px] leading-none">{assistantAvatar}</span> : <Bot size={14} />}
          </div>
        )}

        <div className={`flex min-w-0 max-w-[min(100%,72rem)] flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <span className="text-[10px] text-gray-500 mb-1">{isUser ? userLabel : isTool ? 'Tool' : assistantLabel}</span>

          {(message.attachments?.length ?? 0) > 0 && !isTool && (
            <AttachmentGallery attachments={message.attachments || []} isUser={isUser} />
          )}

          {(normalizedAssistantReasoning || normalizedStreamingReasoning) && !isTool && (
            <div className="mb-1 w-full max-w-full">
              <button
                type="button"
                onClick={() => setShowReasoning((value) => !value)}
                className="flex items-center gap-1 text-[11px] text-gray-400 hover:text-gray-200"
              >
                {showReasoning || shouldAutoExpandReasoning ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                <span>Thinking</span>
              </button>
              {(showReasoning || shouldAutoExpandReasoning) && (
                <div
                  ref={reasoningContentRef}
                  className="mt-1 max-h-56 overflow-y-auto rounded-lg border border-gray-700/70 bg-gray-900/60 px-3 py-2 text-[12px] leading-relaxed text-gray-300"
                >
                  <MarkdownRenderer content={isStreaming ? normalizedStreamingReasoning || normalizedAssistantReasoning : normalizedAssistantReasoning} />
                </div>
              )}
            </div>
          )}

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
                    void handleSaveEdit();
                  }
                  if (e.key === 'Escape') handleCancelEdit();
                }}
              />
              <div className="flex gap-1 mt-1">
                <button onClick={() => void handleSaveEdit()} className="flex items-center gap-1 px-2 py-0.5 bg-blue-600 hover:bg-blue-700 rounded text-[11px] text-white transition-colors" type="button">
                  <Check size={11} /> Save
                </button>
                <button onClick={() => void handleSaveAndResend()} className="flex items-center gap-1 px-2 py-0.5 bg-green-600 hover:bg-green-700 rounded text-[11px] text-white transition-colors" type="button" title="Save changes and resend this message">
                  <Send size={11} /> Save & Resend
                </button>
                <button onClick={handleCancelEdit} className="flex items-center gap-1 px-2 py-0.5 bg-gray-700 hover:bg-gray-600 rounded text-[11px] text-gray-300 transition-colors" type="button">
                  <X size={11} /> Cancel
                </button>
              </div>
            </div>
          ) : shouldRenderAssistantContent ? (
            <div className={`px-3 py-2 rounded-lg text-[13px] leading-relaxed break-words ${isUser ? 'bg-blue-600 text-white whitespace-pre-wrap' : 'bg-gray-700 text-gray-100 markdown-body'}`}>
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
            <div className="mt-2 w-full min-w-0 max-w-full overflow-x-hidden rounded-lg border border-cyan-500/20 bg-gray-900/60 px-2.5 py-2">
              <button
                type="button"
                className="flex w-full min-w-0 items-center justify-between gap-2 text-left"
                onClick={() => setShowToolSteps((v) => !v)}
                aria-expanded={showToolSteps}
                aria-label={`Tool activity ${effectiveToolSteps.length}`}
              >
                <span className="flex min-w-0 flex-wrap items-center gap-2">
                  <span className="shrink-0 text-[12px] font-medium text-gray-200">Tool activity</span>
                  <span className="min-w-0 break-words text-[11px] text-gray-400">{toolActivityLabel}</span>
                  <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${toolActivityStatusClass}`}>
                    {toolActivityStatus}
                  </span>
                  {toolActivityErrorSummary && (
                    <span className="min-w-0 break-words text-[11px] text-red-300/90">{toolActivityErrorSummary}</span>
                  )}
                  {hasRunningTool && <span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-amber-400 animate-pulse" />}
                </span>
                {showToolSteps ? <ChevronDown size={14} className="shrink-0 text-gray-500" /> : <ChevronRight size={14} className="shrink-0 text-gray-500" />}
              </button>
              <div className="mt-1 flex min-w-0 flex-wrap items-center gap-2 text-[10px] text-gray-500">
                <span>{toolActivityLabel}</span>
                {formattedToolDuration && <span>{`Duration ${formattedToolDuration}`}</span>}
                {roundCount > 0 && <span>{`Rounds ${roundCount}`}</span>}
              </div>
              {showToolSteps && canRevealToolSteps && (
                <div className="mt-2 min-w-0 max-w-full space-y-1.5 overflow-x-hidden rounded-md border border-gray-700/80 bg-gray-900/50 p-1.5">
                  {effectiveToolSteps.map((step) => (
                    <ToolCallBubble key={step.message_id} message={step} />
                  ))}
                </div>
              )}
            </div>
          )}

          {shouldShowUserMetaPanel && (
            <div className="mt-1.5 flex min-w-0 max-w-full flex-wrap items-center gap-1.5 overflow-x-hidden text-[10px] text-gray-300">
              <span className="min-w-0 break-words text-[11px] text-gray-500 tabular-nums">{`Tokens: ${usageInputTokens}`}</span>
            </div>
          )}

          {shouldShowMetaPanel && (
            <div className="mt-1.5 flex min-w-0 max-w-full flex-wrap items-center gap-1.5 overflow-x-hidden text-[10px] text-gray-300">
              {(usageTotalTokens > 0 || hoverMetricCount > 0) && (
                <div
                  ref={metricsAnchorRef}
                  className="relative min-w-0 max-w-full"
                  onMouseEnter={handleMetricsEnter}
                  onMouseLeave={handleMetricsLeave}
                >
                  <span className="block min-w-0 max-w-full break-all text-[11px] text-gray-500 tabular-nums">
                    {metricsAnchorLabel}
                  </span>
                </div>
              )}
              {budgetGuardrailApplied && <span className="shrink-0 rounded-md border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 text-rose-200">Guardrail</span>}
              {traceId && onOpenTrace && (
                <button
                  type="button"
                  className="shrink-0 rounded-md border border-violet-500/30 bg-violet-500/10 px-1.5 py-0.5 text-violet-200 hover:bg-violet-500/20"
                  onClick={() => onOpenTrace(traceId)}
                >
                  View trace
                </button>
              )}
            </div>
          )}

          <MessageActionBar
            message={message}
            onStartEdit={isUser ? handleStartEdit : undefined}
          />
        </div>

        {isUser && (
          <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-blue-700">
            {userAvatar ? <span className="text-[14px] leading-none">{userAvatar}</span> : <User size={14} />}
          </div>
        )}
      </div>

      {metricsTooltip}
    </>
  );
};

const AttachmentGallery: React.FC<{ attachments: import('@/types/chat').Attachment[]; isUser: boolean }> = ({ attachments, isUser }) => {
  const [lightboxSrc, setLightboxSrc] = React.useState<{ src: string; alt: string } | null>(null);
  const images = attachments.filter((a) => a.mime_type.startsWith('image/'));
  const files = attachments.filter((a) => !a.mime_type.startsWith('image/'));

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
