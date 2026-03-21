import React from 'react';
import { ChevronDown, ChevronRight, Wrench } from 'lucide-react';
import type { ChatMessage } from '@/types/chat';

interface ToolCallBubbleProps {
  message: ChatMessage;
}

type JsonRecord = Record<string, unknown>;

type ToolSection = {
  label: string;
  value: string;
  code?: boolean;
};

type StructuredSections = {
  input: ToolSection[];
  output: ToolSection[];
  rawPayload: string;
  preview: string;
  showRawPayload: boolean;
  outputLooksRedundant: boolean;
};

const formatDuration = (durationMs: number | null): string | null => {
  if (!Number.isFinite(durationMs) || !durationMs || durationMs <= 0) return null;
  return durationMs >= 1000
    ? `${(durationMs / 1000).toFixed(durationMs >= 10_000 ? 0 : 1)}s`
    : `${Math.round(durationMs)} ms`;
};

const safeJsonParse = (value: string): unknown => {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const stringifyPretty = (value: unknown): string => {
  if (value == null) return '(empty)';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const textPreview = (value: unknown, max = 120): string => {
  const text = stringifyPretty(value).replace(/\s+/g, ' ').trim();
  if (!text) return '(empty)';
  return text.length > max ? `${text.slice(0, max)}…` : text;
};

const pushIfPresent = (
  sections: ToolSection[],
  label: string,
  value: unknown,
  code = false,
): void => {
  if (value == null) return;
  if (typeof value === 'string' && !value.trim()) return;
  sections.push({ label, value: stringifyPretty(value), code });
};

const formatInlineValue = (value: unknown): string => {
  if (value == null) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const buildArgsSummary = (toolName: string, toolArgs: unknown): string | null => {
  if (!toolArgs || typeof toolArgs !== 'object') return null;
  const args = toolArgs as JsonRecord;
  if (toolName === 'houyi_shell_exec') {
    if (typeof args.command === 'string' && args.command.trim()) return textPreview(args.command, 160);
  }
  if (toolName === 'houyi_find_files') {
    const root = typeof args.root === 'string' ? args.root : typeof args.path === 'string' ? args.path : null;
    const pattern = typeof args.pattern === 'string' ? args.pattern : null;
    if (pattern && root) return `${pattern} @ ${root}`;
    if (pattern) return pattern;
  }
  if (toolName === 'houyi_grep') {
    const query = typeof args.query === 'string' ? args.query : typeof args.pattern === 'string' ? args.pattern : null;
    const path = typeof args.path === 'string' ? args.path : typeof args.root === 'string' ? args.root : null;
    if (query && path) return `${query} @ ${path}`;
    if (query) return query;
  }
  if (toolName === 'houyi_read_file') {
    if (typeof args.file_path === 'string' && args.file_path.trim()) return args.file_path;
    if (typeof args.path === 'string' && args.path.trim()) return args.path;
  }
  const pairs = Object.entries(args)
    .filter(([, value]) => value != null && !(typeof value === 'string' && !value.trim()))
    .slice(0, 3)
    .map(([key, value]) => `${key}=${formatInlineValue(value)}`);
  return pairs.length > 0 ? pairs.join(', ') : null;
};

const buildStructuredSections = (title: string, content: string, toolArgs: unknown): StructuredSections => {
  const parsed = safeJsonParse(content);
  const input: ToolSection[] = [];
  const output: ToolSection[] = [];
  const rawPayload = content || '(empty)';
  let preview = buildArgsSummary(title, toolArgs) || '(empty)';
  let showRawPayload = true;
  let outputLooksRedundant = false;

  if (parsed && typeof parsed === 'object') {
    const root = parsed as JsonRecord;
    const data = root.data && typeof root.data === 'object' ? root.data as JsonRecord : null;
    const message = typeof root.message === 'string' ? root.message : null;
    pushIfPresent(input, 'Arguments', toolArgs);

    if (title === 'houyi_shell_exec' && data) {
      input.length = 0;
      pushIfPresent(input, 'Command', (toolArgs as JsonRecord | null)?.command ?? data.command, true);
      pushIfPresent(input, 'Working directory', (toolArgs as JsonRecord | null)?.cwd ?? data.cwd);
      pushIfPresent(
        input,
        'Timeout',
        typeof ((toolArgs as JsonRecord | null)?.timeout_seconds ?? data.timeout_seconds) === 'number'
          ? `${(toolArgs as JsonRecord | null)?.timeout_seconds ?? data.timeout_seconds}s`
          : null,
      );
      pushIfPresent(input, 'Retry count', data.retry_count);

      pushIfPresent(output, 'Message', data.message);
      pushIfPresent(output, 'Stdout', data.stdout, true);
      pushIfPresent(output, 'Stderr', data.stderr, true);
      pushIfPresent(output, 'Timed out', typeof data.timed_out === 'boolean' ? String(data.timed_out) : null);
      pushIfPresent(output, 'Return code', data.returncode);
      preview = typeof data.command === 'string' && data.command.trim()
        ? textPreview(data.command)
        : buildArgsSummary(title, toolArgs) || message || 'shell command';
      outputLooksRedundant = false;
      showRawPayload = true;
      return { input, output, rawPayload, preview, showRawPayload, outputLooksRedundant };
    }

    if (data) {
      if (Array.isArray(data.matches)) {
        preview = buildArgsSummary(title, toolArgs) || `matches: ${data.matches.length}`;
        pushIfPresent(output, 'Matches', data.matches);
      } else if (Array.isArray(data.entries)) {
        preview = buildArgsSummary(title, toolArgs) || `entries: ${data.entries.length}`;
        pushIfPresent(output, 'Entries', data.entries);
      } else if (typeof data.path === 'string') {
        preview = buildArgsSummary(title, toolArgs) || data.path;
        pushIfPresent(output, 'Path', data.path);
      } else if (buildArgsSummary(title, toolArgs)) {
        preview = buildArgsSummary(title, toolArgs) || preview;
      } else if (typeof message === 'string' && message) {
        preview = message;
      } else {
        preview = textPreview(data);
      }

      const knownOutputKeys = new Set([
        'matches',
        'entries',
        'path',
        'stdout',
        'stderr',
        'command',
        'cwd',
        'duration_ms',
        'retry_count',
        'returncode',
        'timed_out',
        'timeout_seconds',
        'message',
        'success',
      ]);
      const argRecord = toolArgs && typeof toolArgs === 'object' ? toolArgs as JsonRecord : null;
      const inputCandidate = Object.fromEntries(
        Object.entries(argRecord ?? data).filter(([key]) => !knownOutputKeys.has(key)),
      );
      if (Object.keys(inputCandidate).length > 0 && input.length === 0) pushIfPresent(input, 'Parameters', inputCandidate);
      if (output.length === 0) {
        pushIfPresent(output, 'Result', data);
      }
      if (message && !output.some((section) => section.label === 'Message')) {
        pushIfPresent(output, 'Message', message);
      }
      if (root._truncated === true) {
        pushIfPresent(output, 'Truncated', 'true');
      }
      if (typeof root._truncated_message === 'string') {
        pushIfPresent(output, 'Truncation note', root._truncated_message);
      }
      outputLooksRedundant = output.length === 1 && output[0].label === 'Result';
      showRawPayload = !outputLooksRedundant;
      return { input, output, rawPayload, preview, showRawPayload, outputLooksRedundant };
    }

    preview = buildArgsSummary(title, toolArgs) || message || textPreview(root);
    if (message) pushIfPresent(output, 'Message', message);
    pushIfPresent(output, 'Result', root);
    outputLooksRedundant = output.length === 1 && output[0].label === 'Result';
    showRawPayload = !outputLooksRedundant;
    return { input, output, rawPayload, preview, showRawPayload, outputLooksRedundant };
  }

  pushIfPresent(input, 'Arguments', toolArgs);
  preview = buildArgsSummary(title, toolArgs) || textPreview(content);
  pushIfPresent(output, 'Output', content, true);
  return { input, output, rawPayload, preview, showRawPayload: true, outputLooksRedundant: false };
};

const extractErrorSummary = (content: string): string | null => {
  const parsed = safeJsonParse(content);
  if (!parsed || typeof parsed !== 'object') return null;
  const root = parsed as JsonRecord;
  const data = root.data && typeof root.data === 'object' ? root.data as JsonRecord : null;
  const candidates = [
    root.message,
    root.error,
    data?.message,
    data?.error,
    data?.stderr,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return textPreview(candidate, 160);
    }
  }
  return null;
};

const Section: React.FC<{ title: string; entries: ToolSection[] }> = ({ title, entries }) => {
  if (entries.length === 0) return null;
  return (
    <div className="mt-2 min-w-0 max-w-full rounded border border-gray-800 bg-gray-950/60 p-2">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-gray-500">{title}</div>
      <div className="min-w-0 space-y-1.5">
        {entries.map((entry) => (
          <div key={`${title}-${entry.label}`} className="min-w-0">
            <div className="text-[10px] text-gray-500">{entry.label}</div>
            {entry.code ? (
              <pre className="mt-0.5 max-w-full overflow-auto rounded border border-gray-800 bg-gray-950/90 p-2 text-[11px] leading-relaxed text-gray-300 whitespace-pre-wrap break-words">
                {entry.value}
              </pre>
            ) : (
              <div className="mt-0.5 min-w-0 max-w-full text-[11px] text-gray-300 whitespace-pre-wrap break-words">{entry.value}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};


const MetaChips: React.FC<{ entries: ToolSection[] }> = ({ entries }) => {
  if (entries.length === 0) return null;
  return (
    <div className="mt-2 min-w-0 max-w-full rounded border border-gray-800 bg-gray-950/60 p-2">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-gray-500">Meta</div>
      <div className="flex min-w-0 flex-wrap gap-1.5">
        {entries.map((entry) => (
          <div
            key={`meta-${entry.label}`}
            className="inline-flex min-w-0 max-w-full items-center gap-1 rounded border border-gray-800 bg-gray-950/90 px-2 py-1 text-[11px] text-gray-300"
          >
            <span className="shrink-0 text-gray-500">{entry.label}</span>
            <span className="min-w-0 break-all">{entry.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export const ToolCallBubble: React.FC<ToolCallBubbleProps> = ({ message }) => {
  const [expanded, setExpanded] = React.useState(false);
  const status = String(message.metadata?.tool_status || 'done');
  const parallelGroupId = typeof message.metadata?.parallel_group_id === 'string'
    ? message.metadata.parallel_group_id
    : null;
  const roundIndex = Number(message.metadata?.round_index);
  const durationMs = Number(message.metadata?.duration_ms);
  const formattedDuration = formatDuration(Number.isFinite(durationMs) ? durationMs : null);
  const statusClass =
    status === 'error'
      ? 'text-red-300 bg-red-500/10 border-red-500/30'
      : status === 'running'
        ? 'text-amber-300 bg-amber-500/10 border-amber-500/30'
        : 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30';

  const title = message.name || message.tool_call_id || 'tool';
  const toolArgs = message.metadata?.tool_args;
  const errorSummary = React.useMemo(
    () => (status === 'error'
      ? extractErrorSummary(typeof message.content === 'string' ? message.content : String(message.content ?? ''))
      : null),
    [message.content, status],
  );
  const sections = React.useMemo(
    () => buildStructuredSections(
      title,
      typeof message.content === 'string' ? message.content : String(message.content ?? ''),
      toolArgs,
    ),
    [message.content, title, toolArgs],
  );

  const metaEntries: ToolSection[] = [
    { label: 'Tool', value: title },
    { label: 'Status', value: status },
  ];
  if (formattedDuration) metaEntries.push({ label: 'Duration', value: formattedDuration });
  if (Number.isFinite(roundIndex) && roundIndex > 0) metaEntries.push({ label: 'Round', value: String(roundIndex) });
  if (parallelGroupId) metaEntries.push({ label: 'Parallel group', value: parallelGroupId });
  if (errorSummary) metaEntries.push({ label: 'Error', value: errorSummary });

  return (
    <div className="min-w-0 w-full max-w-full rounded-md border border-gray-700/80 bg-gray-900/60 px-2.5 py-2">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="flex min-w-0 items-center gap-2 text-[12px] text-gray-200">
          <Wrench size={13} className="shrink-0 text-gray-400" />
          <span className="truncate font-medium">{title}</span>
          <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${statusClass}`}>{status}</span>
        </span>
        {expanded ? <ChevronDown size={14} className="shrink-0 text-gray-500" /> : <ChevronRight size={14} className="shrink-0 text-gray-500" />}
      </button>
      {(formattedDuration || parallelGroupId) && (
        <div className="mt-1 flex min-w-0 flex-wrap items-center gap-2 text-[10px] text-gray-500">
          {formattedDuration && <span>Duration {formattedDuration}</span>}
          {parallelGroupId && <span>Parallel {parallelGroupId}</span>}
        </div>
      )}
      <div className="mt-1 min-w-0 max-w-full break-words text-[11px] text-gray-400">
        {errorSummary ? `Error: ${errorSummary}` : sections.preview}
      </div>
      {expanded && (
        <div className="mt-2 min-w-0 max-w-full">
          <MetaChips entries={metaEntries} />
          <Section title="Input" entries={sections.input} />
          {!sections.outputLooksRedundant && <Section title="Output" entries={sections.output} />}
          {sections.showRawPayload && (
            <details className="mt-2 min-w-0 max-w-full rounded border border-gray-800 bg-gray-950/60 p-2">
              <summary className="cursor-pointer text-[10px] font-medium uppercase tracking-wide text-gray-500">Raw payload</summary>
              <pre className="mt-2 max-h-56 max-w-full overflow-auto rounded border border-gray-800 bg-gray-950/90 p-2 text-[11px] leading-relaxed text-gray-300 whitespace-pre-wrap break-words">
                {stringifyPretty(safeJsonParse(sections.rawPayload) ?? sections.rawPayload)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
};
