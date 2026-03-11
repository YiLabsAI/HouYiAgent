import { describe, expect, it } from 'vitest';
import {
  buildVisibleChatError,
  classifyChatError,
  formatChatErrorMessage,
} from '@/utils/chatErrors';

describe('chatErrors', () => {
  it('classifies rate limit errors generically', () => {
    expect(classifyChatError('429 RESOURCE_EXHAUSTED')).toBe('rate_limit');
  });

  it('classifies permission errors separately from auth', () => {
    expect(classifyChatError('403 PERMISSION_DENIED')).toBe('permission');
  });

  it('classifies network interruption signals', () => {
    expect(classifyChatError('unavailable: unexpected EOF')).toBe('network');
  });

  it('formats generic timeout errors with provider-agnostic copy', () => {
    expect(formatChatErrorMessage('request timed out')).toBe(
      'The request timed out before the model finished responding. Please retry or reduce the request size.',
    );
  });

  it('formats gemini rate limit errors with provider-specific copy', () => {
    expect(formatChatErrorMessage('429 RESOURCE_EXHAUSTED', 'gemini')).toBe(
      'Gemini is temporarily rate limited. Please retry in a moment, reduce request frequency, or switch to another model if the issue persists.',
    );
  });

  it('formats gemini auth and permission errors with provider-specific copy', () => {
    expect(formatChatErrorMessage('403 PERMISSION_DENIED', 'gemini')).toBe(
      'Gemini request failed due to authentication or permission issues. Check the configured Vertex AI credentials and project access.',
    );
  });

  it('builds visible fallback errors for unknown messages', () => {
    expect(buildVisibleChatError('something custom failed')).toBe(
      'LLM Error: something custom failed',
    );
  });
});
