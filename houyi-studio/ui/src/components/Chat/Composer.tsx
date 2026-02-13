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
import { Send, Square, Paperclip, Brain, Globe } from 'lucide-react';
import { useConsoleStore } from '@/stores/useConsoleStore';

interface ComposerProps {
  onSend: (content: string, options?: { enableReasoning?: boolean; enableWebSearch?: boolean; attachments?: File[] }) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}

// Upload constraints
const MAX_FILES = 5;
const MAX_FILE_SIZE_MB = 10;
const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;

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
// Referenced from CherryStudio's textExts + documentExts.
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
  onSend,
  onStop,
  isStreaming,
  disabled = false,
}) => {
  const [text, setText] = React.useState('');
  const [enableReasoning, setEnableReasoning] = React.useState(false);
  const [enableWebSearch, setEnableWebSearch] = React.useState(false);
  const showToast = useConsoleStore((s) => s.showToast);
  const [attachments, setAttachments] = React.useState<File[]>([]);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const isComposingRef = React.useRef(false);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled || isStreaming) return;
    onSend(trimmed, {
      enableReasoning: enableReasoning || undefined,
      enableWebSearch: enableWebSearch || undefined,
      attachments: attachments.length > 0 ? attachments : undefined,
    });
    setText('');
    setAttachments([]);
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
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
    // Auto-expand
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
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
        <div className="flex-1 flex flex-col bg-gray-900 border border-gray-700 rounded-lg focus-within:border-gray-500">
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
            className="flex-1 resize-none bg-transparent px-3 py-2 text-[13px] text-gray-100 placeholder:text-gray-500 focus:outline-none disabled:opacity-50 max-h-40"
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
              onClick={() => setEnableReasoning((v) => !v)}
              className={`p-1 rounded transition-colors ${
                enableReasoning
                  ? 'bg-purple-600/20 text-purple-400 hover:bg-purple-600/30'
                  : 'hover:bg-gray-800 text-gray-500 hover:text-gray-300'
              }`}
              title={enableReasoning ? 'Thinking mode ON' : 'Thinking mode OFF'}
              type="button"
            >
              <Brain size={14} />
            </button>

            {/* Web search */}
            <button
              onClick={() => setEnableWebSearch((v) => !v)}
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
          </div>
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
