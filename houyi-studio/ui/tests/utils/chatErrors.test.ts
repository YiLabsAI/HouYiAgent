import { describe, expect, it } from 'vitest';
import {
  buildVisibleChatError,
  classifyChatError,
  formatChatErrorMessage,
} from '@/utils/chatErrors';

describe('chatErrors', () => {
  it('classifies normalized rate limit codes', () => {
    expect(classifyChatError('provider_rate_limited')).toBe('provider_rate_limited');
  });

  it('classifies permission errors separately from auth', () => {
    expect(classifyChatError('provider_permission_denied')).toBe('provider_permission_denied');
  });

  it('returns unknown for unrecognized codes', () => {
    expect(classifyChatError('unavailable: unexpected EOF')).toBe('unknown');
  });

  it('formats timeout errors', () => {
    expect(formatChatErrorMessage({ error_code: 'provider_timeout' })).toBe(
      'The request timed out before the model finished responding. Please retry or reduce the request size.',
    );
  });

  it('prefers backend public messages', () => {
    expect(
      formatChatErrorMessage({
        error_code: 'provider_rate_limited',
        public_message: 'Backend says retry later.',
      }, 'gemini'),
    ).toBe(
      'Backend says retry later.',
    );
  });

  it('formats permission errors', () => {
    expect(formatChatErrorMessage({ error_code: 'provider_permission_denied' }, 'gemini')).toBe(
      'The request failed due to missing permissions. Check the configured credentials and project access before retrying.',
    );
  });

  it('builds fallback errors', () => {
    expect(buildVisibleChatError({ error: 'something custom failed' })).toBe(
      'The model request failed. Please retry in a moment.',
    );
  });

  it('hides raw details', () => {
    expect(buildVisibleChatError({ error_code: 'provider_rate_limited', error: '429 RESOURCE_EXHAUSTED' })).toBe(
      'The model is temporarily rate limited. Please retry in a moment.',
    );
  });
});
