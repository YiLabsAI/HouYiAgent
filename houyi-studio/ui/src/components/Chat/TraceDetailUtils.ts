export const formatDuration = (value: unknown): string => {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms < 0) return '—';
  if (ms > 0 && ms < 1) return '<1ms';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

export const formatInt = (value: unknown): string => {
  const num = Number(value);
  if (!Number.isFinite(num)) return '—';
  return num.toLocaleString();
};

export const hasValue = (value: unknown): boolean => value !== null && value !== undefined && value !== '';
