/**
 * Composer: message input area with send/stop controls.
 *
 * Features:
 * - Auto-expanding textarea
 * - Send on Enter (Shift+Enter for newline)
 * - Stop button during streaming
 * - Context usage indicator
 */
import React from 'react';
import { Send, Square, Paperclip, BrainCircuit, Globe, Search } from 'lucide-react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { useChatStore } from '@/stores/useChatStore';

interface ComposerProps {
  conversationId?: string | null;
  onSend: (content: string, options?: {
    enableReasoning?: boolean;
    enableToolCalls?: boolean;
    toolCallStrategy?: 'conservative' | 'balanced' | 'aggressive';
    enableWebSearch?: boolean;
    enableDeepResearch?: boolean;
    maxTokens?: number;
    attachments?: File[];
  }) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

// Upload constraints
const MAX_FILES = 5;
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;
const MIN_COMPOSER_HEIGHT = 88;
const DEFAULT_COMPOSER_HEIGHT = 88;

// Allowed MIME type prefixes
const ALLOWED_MIME_PREFIXES = ['image/', 'text/'];
// Allowed exact MIME types
const ALLOWED_MIME_EXACT = new Set([
  'application/pdf',
  'application/json',
  'application/xml',
  'application/javascript',
  'application/typescript',
  'application/x-python',
  'application/x-yaml',
  'application/x-sh',
  'application/toml',
  'application/sql',
  // Office documents (binary — sent as filename description)
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',   // .docx
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',         // .xlsx
  'application/vnd.openxmlformats-officedocument.presentationml.presentation', // .pptx
  'application/msword',        // .doc
  'application/vnd.ms-excel',  // .xls
]);
// Allowed extensions (fallback when MIME is empty or generic).
const ALLOWED_EXTENSIONS = new Set([
  '.txt', '.md', '.markdown', '.csv', '.tsv', '.json', '.jsonl',
  '.xml', '.html', '.htm', '.css', '.js', '.ts', '.jsx', '.tsx',
  '.py', '.rb', '.go', '.rs', '.java', '.kt', '.c', '.cpp', '.h',
  '.hpp', '.cs', '.swift', '.sh', '.bash', '.zsh', '.fish',
  '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
  '.sql', '.graphql', '.proto', '.r', '.lua', '.pl', '.pm',
  '.tex', '.bib', '.rst', '.adoc', '.org',
  '.log', '.diff', '.patch',
  '.pdf', '.doc', '.docx', '.xlsx', '.pptx',
]);

function getFileExtension(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
}

function isFileAllowed(file: File): boolean {
  // Check MIME prefix
  if (ALLOWED_MIME_PREFIXES.some((p) => file.type.startsWith(p))) return true;
  // Check exact MIME
  if (ALLOWED_MIME_EXACT.has(file.type)) return true;
  // Fallback: check extension (handles empty/generic MIME)
  if (ALLOWED_EXTENSIONS.has(getFileExtension(file.name))) return true;
  return false;
}

function validateFiles(files: File[], existing: File[]): { accepted: File[]; errors: string[] } {
  const errors: string[] = [];
  const accepted: File[] = [];
  const remaining = MAX_FILES - existing.length;

  for (const file of files) {
    if (accepted.length >= remaining) {
      errors.push(`Max ${MAX_FILES} files allowed — extra files skipped`);
      break;
    }
    if (file.size > MAX_FILE_SIZE) {
      errors.push(`${file.name}: exceeds ${MAX_FILE_SIZE_MB}MB limit`);
      continue;
    }
    if (!isFileAllowed(file)) {
      errors.push(`${file.name}: unsupported type (${file.type || getFileExtension(file.name) || 'unknown'})`);
      continue;
    }
    accepted.push(file);
  }
  return { accepted, errors };
}

export const Composer: React.FC<ComposerProps> = ({
  conversationId,
  onSend,
  onStop,
  isStreaming,
  disabled = false,
}) => {
  const composerUiState = useChatStore((s) =>
    conversationId ? s.composerUiByConversation[conversationId] : undefined,
  );
  const setComposerUiState = useChatStore((s) => s.setComposerUiState);
  const [text, setText] = React.useState('');
  const showToast = useConsoleStore((s) => s.showToast);
  const toolCallsEnabled = useConsoleStore((s) => s.runSettings.enable_tool_calls);
  const toolCallStrategy = useConsoleStore((s) => s.runSettings.tool_call_strategy);
  const [attachments, setAttachments] = React.useState<File[]>([]);
  const [composerHeight, setComposerHeight] = React.useState(DEFAULT_COMPOSER_HEIGHT);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const isComposingRef = React.useRef(false);
  const resizeDragRef = React.useRef<{ startY: number; startHeight: number } | null>(null);

  const clampComposerHeight = React.useCallback((height: number) => {
    if (typeof window === 'undefined') return Math.max(MIN_COMPOSER_HEIGHT, Math.round(height));
    const maxHeight = Math.max(MIN_COMPOSER_HEIGHT, Math.floor(window.innerHeight * 0.72));
    return Math.max(MIN_COMPOSER_HEIGHT, Math.min(maxHeight, Math.round(height)));
  }, []);

  const stopResizeDrag = React.useCallback(() => {
    resizeDragRef.current = null;
    document.body.style.userSelect = '';
    document.body.style.cursor = '';
  }, []);

  const handleResizeMove = React.useCallback((event: PointerEvent) => {
    const drag = resizeDragRef.current;
    if (!drag) return;
    const deltaY = event.clientY - drag.startY;
    setComposerHeight(clampComposerHeight(drag.startHeight + deltaY));
  }, [clampComposerHeight]);

  React.useEffect(() => {
    const handlePointerUp = () => stopResizeDrag();
    window.addEventListener('pointermove', handleResizeMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handleResizeMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerUp);
      stopResizeDrag();
    };
  }, [handleResizeMove, stopResizeDrag]);

  React.useEffect(() => {
    setAttachments([]);
  }, [conversationId]);

  const enableReasoning = composerUiState?.enableReasoning ?? false;
  const enableWebSearch = composerUiState?.enableWebSearch ?? false;
  const enableDeepResearch = composerUiState?.enableDeepResearch ?? false;
  const showAdvanced = composerUiState?.showAdvanced ?? false;
  const maxTokensDraft = composerUiState?.maxTokensDraft ?? '';

  const startResizeDrag = (e: React.PointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    resizeDragRef.current = { startY: e.clientY, startHeight: composerHeight };
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'ns-resize';
  };

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled || isStreaming) return;
    const parsedMaxTokens = maxTokensDraft.trim() ? parseInt(maxTokensDraft.trim(), 10) : NaN;
    const maxTokens = Number.isFinite(parsedMaxTokens) && parsedMaxTokens > 0 ? parsedMaxTokens : undefined;
    onSend(trimmed, {
      enableReasoning: enableReasoning || undefined,
      enableToolCalls: toolCallsEnabled,
      toolCallStrategy,
      enableWebSearch: enableWebSearch || undefined,
      enableDeepResearch: enableDeepResearch || undefined,
      maxTokens,
      attachments: attachments.length > 0 ? attachments : undefined,
    });
    setText('');
    setAttachments([]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const { accepted, errors } = validateFiles(Array.from(files), attachments);
      if (errors.length > 0) errors.forEach((msg) => showToast(msg, 'error'));
      if (accepted.length > 0) setAttachments((prev) => [...prev, ...accepted]);
    }
    // Reset input so same file can be selected again
    e.target.value = '';
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposingRef.current) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
  };

  // Clipboard paste: support pasting images from clipboard
  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const imageFiles: File[] = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          const named = new File([file], `clipboard-${Date.now()}.${file.type.split('/')[1] || 'png'}`, { type: file.type });
          imageFiles.push(named);
        }
      }
    }
    if (imageFiles.length > 0) {
      e.preventDefault();
      const { accepted, errors } = validateFiles(imageFiles, attachments);
      if (errors.length > 0) errors.forEach((msg) => showToast(msg, 'error'));
      if (accepted.length > 0) setAttachments((prev) => [...prev, ...accepted]);
    }
  };

  return (
    <div className="border-t border-gray-700 bg-gray-800 px-4 py-3">
      {/* Attachment preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {attachments.map((file, i) => (
            <div
              key={`${file.name}-${i}`}
              className="flex items-center gap-1 px-2 py-0.5 bg-gray-700 rounded text-[11px] text-gray-300"
            >
              <Paperclip size={10} />
              <span className="max-w-[120px] truncate">{file.name}</span>
              <button
                onClick={() => removeAttachment(i)}
                className="ml-0.5 text-gray-500 hover:text-gray-300"
                type="button"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
        <div
          className="relative flex-1 overflow-hidden flex flex-col bg-gray-900 border border-gray-700 rounded-lg focus-within:border-gray-500"
          style={{ height: `${composerHeight}px` }}
        >
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            onCompositionStart={() => { isComposingRef.current = true; }}
            onCompositionEnd={() => { setTimeout(() => { isComposingRef.current = false; }, 0); }}
            placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
            disabled={disabled}
            rows={1}
            data-testid="chat-input"
            className="min-h-0 flex-1 resize-none overflow-y-auto bg-transparent px-3 py-2 text-[13px] text-gray-100 placeholder:text-gray-500 focus:outline-none disabled:opacity-50"
          />

          {/* Toolbar */}
          <div className="flex items-center gap-0.5 px-2 py-1 border-t border-gray-800">
            {/* File upload */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileSelect}
              className="hidden"
              accept="image/*,.pdf,.txt,.md,.csv,.json,.xml,.html,.py,.js,.ts,.tsx,.jsx"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-1 rounded hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors"
              title="Attach file"
              type="button"
            >
              <Paperclip size={14} />
            </button>

            {/* Thinking mode */}
            <button
              onClick={() => {
                if (!conversationId) return;
                setComposerUiState(conversationId, { enableReasoning: !enableReasoning });
              }}
              className={`p-1 rounded transition-colors ${
                enableReasoning
                  ? 'bg-purple-600/20 text-purple-400 hover:bg-purple-600/30'
                  : 'hover:bg-gray-800 text-gray-500 hover:text-gray-300'
              }`}
              title={enableReasoning ? 'Thinking mode ON' : 'Thinking mode OFF'}
              type="button"
            >
              <BrainCircuit size={14} />
            </button>

            {/* Web search */}
            <button
              onClick={() => {
                if (!conversationId) return;
                setComposerUiState(conversationId, { enableWebSearch: !enableWebSearch });
              }}
              className={`p-1 rounded transition-colors ${
                enableWebSearch
                  ? 'bg-blue-600/20 text-blue-400 hover:bg-blue-600/30'
                  : 'hover:bg-gray-800 text-gray-500 hover:text-gray-300'
              }`}
              title={enableWebSearch ? 'Web search ON' : 'Web search OFF'}
              type="button"
            >
              <Globe size={14} />
            </button>

            {/* Deep Research mode toggle */}
            <button
              onClick={() => {
                if (!conversationId) return;
                setComposerUiState(conversationId, { enableDeepResearch: !enableDeepResearch });
              }}
              className={`p-1 rounded transition-colors ${
                enableDeepResearch
                  ? 'bg-green-600/20 text-green-400 hover:bg-green-600/30'
                  : 'hover:bg-gray-800 text-gray-500 hover:text-gray-300'
              }`}
              title={enableDeepResearch ? 'Deep Research ON — multi-step search & synthesis' : 'Deep Research OFF'}
              type="button"
              data-testid="deep-research-toggle"
            >
              <Search size={14} />
            </button>

            <button
              onClick={() => {
                if (!conversationId) return;
                setComposerUiState(conversationId, { showAdvanced: !showAdvanced });
              }}
              className={`ml-1 px-2 py-0.5 rounded text-[11px] transition-colors ${
                showAdvanced
                  ? 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                  : 'bg-gray-900 text-gray-500 hover:bg-gray-800 hover:text-gray-300'
              }`}
              title={showAdvanced ? 'Options open' : 'Message options'}
              type="button"
              data-testid="composer-advanced-toggle"
            >
              Options
            </button>
          </div>

          {showAdvanced && (
            <div className="flex items-center gap-2 px-3 py-2 border-t border-gray-800 text-[11px] text-gray-400">
              <label className="flex items-center gap-2">
                <span className="text-gray-500">Max tokens</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={maxTokensDraft}
                  onChange={(e) => {
                    if (!conversationId) return;
                    setComposerUiState(conversationId, { maxTokensDraft: e.target.value });
                  }}
                  placeholder="(default)"
                  className="w-[110px] bg-gray-800 border border-gray-700 rounded px-2 py-1 text-[11px] text-gray-200 focus:outline-none focus:border-blue-500"
                />
              </label>
              <button
                type="button"
                className="text-gray-500 hover:text-gray-300 underline"
                onClick={() => {
                  if (!conversationId) return;
                  setComposerUiState(conversationId, { maxTokensDraft: '' });
                }}
              >
                reset
              </button>
            </div>
          )}

          <button
            type="button"
            aria-label="Resize composer"
            data-testid="composer-resize-handle"
            className="absolute bottom-1 right-1 h-4 w-4 cursor-nwse-resize rounded-sm text-gray-500 hover:text-gray-300"
            onPointerDown={startResizeDrag}
          >
            <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
              <path d="M4 12h8" />
              <path d="M7 9h5" />
              <path d="M10 6h2" />
            </svg>
          </button>
        </div>

        {isStreaming ? (
          <button
            onClick={onStop}
            className="shrink-0 p-2 bg-red-600 hover:bg-red-700 rounded-lg text-white transition-colors"
            title="Stop generating"
            type="button"
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!text.trim() || disabled}
            className="shrink-0 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-white transition-colors"
            title="Send message"
            type="button"
            data-testid="chat-send-btn"
          >
            <Send size={16} />
          </button>
        )}
      </div>
    </div>
  );
};
