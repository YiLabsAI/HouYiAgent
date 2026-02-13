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
import { useTypewriter } from '@/hooks/useTypewriter';

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
  isLastMessage?: boolean;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isStreaming = false, isLastMessage = false }) => {
  const [showReasoning, setShowReasoning] = React.useState(false);
  const [isHovered, setIsHovered] = React.useState(false);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editText, setEditText] = React.useState('');
  const editRef = React.useRef<HTMLTextAreaElement>(null);
  const editMessage = useChatStore((s) => s.editMessage);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const streamingReasoning = useChatStore((s) =>
    isStreaming ? s.streaming.reasoningBuffer : '',
  );
  const display = useSettingsStore((s) => s.display);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  // Smooth typewriter animation for streaming content (fixes Gemini chunky output)
  const displayContent = useTypewriter(message.content || '', isStreaming && isAssistant);

  const handleStartEdit = () => {
    setEditText(message.content);
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
        {isAssistant && (message.reasoning_content || (isStreaming && streamingReasoning)) && (
          <div className="mb-1 w-full">
            <button
              onClick={() => setShowReasoning((v) => !v)}
              className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
              type="button"
            >
              {(showReasoning || isStreaming) ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              <span className="flex items-center gap-1">
                Thinking
                {isStreaming && streamingReasoning && !message.content && (
                  <span className="inline-block w-1 h-1 bg-blue-400 rounded-full animate-pulse" />
                )}
              </span>
            </button>
            {(showReasoning || isStreaming) && (
              <div className="mt-1 pl-3 py-1.5 border-l-2 border-gray-500 text-[11px] text-gray-400 whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto">
                {message.reasoning_content || streamingReasoning}
                {isStreaming && !message.content && (
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
        {isEditing ? (
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
        ) : (
          <div
            className={`px-3 py-2 rounded-lg text-[13px] leading-relaxed break-words ${
              isUser
                ? 'bg-blue-600 text-white whitespace-pre-wrap'
                : 'bg-gray-700 text-gray-100 markdown-body'
            }`}
          >
            {isAssistant ? (
              message.content ? (
                <>
                  <MarkdownRenderer content={displayContent} />
                  {isStreaming && (
                    <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-300 animate-pulse rounded-sm" />
                  )}
                </>
              ) : isStreaming && !streamingReasoning ? (
                <TypingIndicator />
              ) : null
            ) : (
              message.content
            )}
          </div>
        )}

        {/* Timestamp + edited indicator */}
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-600">
            {new Date(message.created_at * 1000).toLocaleTimeString()}
          </span>
          {message.metadata?.edited && (
            <span className="text-[9px] text-gray-600 italic">(edited)</span>
          )}
        </div>

        {/* Action bar: always rendered to preserve layout; visibility toggled to avoid content shift */}
        {!isStreaming && !isEditing && (
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
