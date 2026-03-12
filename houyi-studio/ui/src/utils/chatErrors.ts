export const CHAT_ERROR_SIGNALS = {
  rateLimit: ['RESOURCE_EXHAUSTED', '429', 'RATE LIMIT', 'RATE_LIMIT'],
  auth: ['UNAUTHENTICATED', '401'],
  permission: ['PERMISSION_DENIED', 'FORBIDDEN', '403'],
  timeout: ['TIMEOUT', 'TIMED OUT', 'DEADLINE_EXCEEDED'],
  network: ['UNEXPECTED EOF', 'ECONNRESET', 'NETWORK ERROR', 'CONNECTION RESET'],
} as const;

export const CHAT_ERROR_MESSAGES = {
  rateLimit:
    'The model is temporarily rate limited. Please retry in a moment, reduce request frequency, or switch to another model if the issue persists.',
  auth:
    'The request failed due to authentication issues. Check the configured credentials before retrying.',
  permission:
    'The request failed due to missing permissions. Check the configured credentials and project access before retrying.',
  timeout:
    'The request timed out before the model finished responding. Please retry or reduce the request size.',
  network:
    'The connection to the model was interrupted. Please retry in a moment.',
  geminiRateLimit:
    'Gemini is temporarily rate limited. Please retry in a moment, reduce request frequency, or switch to another model if the issue persists.',
  geminiAuth:
    'Gemini request failed due to authentication or permission issues. Check the configured Vertex AI credentials and project access.',
  unknown:
    'The model request failed. Please retry in a moment.',
} as const;

export type ChatErrorCategory = 'rate_limit' | 'auth' | 'permission' | 'timeout' | 'network' | 'unknown';
export type ChatErrorProvider = 'generic' | 'gemini';

function includesAnySignal(messageUpper: string, signals: readonly string[]): boolean {
  return signals.some((signal) => messageUpper.includes(signal));
}

export function classifyChatError(raw: string): ChatErrorCategory {
  const message = raw.trim();
  const upper = message.toUpperCase();

  if (includesAnySignal(upper, CHAT_ERROR_SIGNALS.rateLimit)) {
    return 'rate_limit';
  }
  if (includesAnySignal(upper, CHAT_ERROR_SIGNALS.auth)) {
    return 'auth';
  }
  if (includesAnySignal(upper, CHAT_ERROR_SIGNALS.permission)) {
    return 'permission';
  }
  if (includesAnySignal(upper, CHAT_ERROR_SIGNALS.timeout)) {
    return 'timeout';
  }
  if (includesAnySignal(upper, CHAT_ERROR_SIGNALS.network)) {
    return 'network';
  }
  return 'unknown';
}

export function formatChatErrorMessage(
  raw: string,
  provider: ChatErrorProvider = 'generic',
): string {
  const message = raw.trim();
  switch (classifyChatError(message)) {
    case 'rate_limit':
      return provider === 'gemini' ? CHAT_ERROR_MESSAGES.geminiRateLimit : CHAT_ERROR_MESSAGES.rateLimit;
    case 'auth':
      return provider === 'gemini' ? CHAT_ERROR_MESSAGES.geminiAuth : CHAT_ERROR_MESSAGES.auth;
    case 'permission':
      return provider === 'gemini' ? CHAT_ERROR_MESSAGES.geminiAuth : CHAT_ERROR_MESSAGES.permission;
    case 'timeout':
      return CHAT_ERROR_MESSAGES.timeout;
    case 'network':
      return CHAT_ERROR_MESSAGES.network;
    default:
      return CHAT_ERROR_MESSAGES.unknown;
  }
}

export function buildVisibleChatError(
  raw: string,
  provider: ChatErrorProvider = 'generic',
): string {
  return formatChatErrorMessage(raw, provider);
}
