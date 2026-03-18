export const CHAT_ERROR_MESSAGES = {
  rateLimit: 'The model is temporarily rate limited. Please retry in a moment.',
  auth: 'The request failed due to authentication issues. Check the configured credentials before retrying.',
  permission:
    'The request failed due to missing permissions. Check the configured credentials and project access before retrying.',
  timeout: 'The request timed out before the model finished responding. Please retry or reduce the request size.',
  network: 'The connection to the model was interrupted. Please retry in a moment.',
  unknown: 'The model request failed. Please retry in a moment.',
} as const;

export type ChatErrorCode =
  | 'provider_rate_limited'
  | 'provider_auth_failed'
  | 'provider_permission_denied'
  | 'provider_timeout'
  | 'provider_network_error'
  | 'provider_request_failed';
export type ChatErrorProvider = 'generic' | 'gemini';

export interface ChatErrorPayload {
  error?: string;
  error_code?: string;
  public_message?: string;
  retryable?: boolean;
  status_code?: number | null;
  provider_code?: string | null;
}

export function classifyChatError(code?: string | null): ChatErrorCode | 'unknown' {
  switch (code) {
    case 'provider_rate_limited':
    case 'provider_auth_failed':
    case 'provider_permission_denied':
    case 'provider_timeout':
    case 'provider_network_error':
    case 'provider_request_failed':
      return code;
    default:
      return 'unknown';
  }
}

export function formatChatErrorMessage(
  payload: string | ChatErrorPayload,
  _provider: ChatErrorProvider = 'generic',
): string {
  if (typeof payload !== 'string' && typeof payload.public_message === 'string' && payload.public_message.trim()) {
    return payload.public_message.trim();
  }

  const code = typeof payload === 'string' ? undefined : payload.error_code;
  switch (classifyChatError(code)) {
    case 'provider_rate_limited':
      return CHAT_ERROR_MESSAGES.rateLimit;
    case 'provider_auth_failed':
      return CHAT_ERROR_MESSAGES.auth;
    case 'provider_permission_denied':
      return CHAT_ERROR_MESSAGES.permission;
    case 'provider_timeout':
      return CHAT_ERROR_MESSAGES.timeout;
    case 'provider_network_error':
      return CHAT_ERROR_MESSAGES.network;
    default:
      return CHAT_ERROR_MESSAGES.unknown;
  }
}

export function buildVisibleChatError(
  payload: string | ChatErrorPayload,
  provider: ChatErrorProvider = 'generic',
): string {
  return formatChatErrorMessage(payload, provider);
}
