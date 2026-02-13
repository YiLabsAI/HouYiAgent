import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { MessageActionBar } from '@/components/Chat/MessageActionBar';
import type { ChatMessage } from '@/types/chat';

// Mock useChatStore
const mockDeleteMessage = vi.fn();
const mockRegenerateMessage = vi.fn();
const mockSendMessage = vi.fn();
let mockIsStreaming = false;

vi.mock('@/stores/useChatStore', () => ({
  useChatStore: (selector: any) => {
    const state = {
      deleteMessage: mockDeleteMessage,
      regenerateMessage: mockRegenerateMessage,
      sendMessage: mockSendMessage,
      streaming: { isStreaming: mockIsStreaming },
    };
    return selector(state);
  },
}));

// Mock clipboard
Object.assign(navigator, {
  clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
});

const userMsg: ChatMessage = {
  message_id: 'u1',
  role: 'user',
  content: 'Hello world',
  created_at: Date.now() / 1000,
  metadata: {},
};

const assistantMsg: ChatMessage = {
  message_id: 'a1',
  role: 'assistant',
  content: 'Hi there',
  created_at: Date.now() / 1000,
  metadata: {},
};

describe('MessageActionBar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsStreaming = false;
  });

  it('shows resend, edit, copy, delete for user messages', () => {
    const onStartEdit = vi.fn();
    render(<MessageActionBar message={userMsg} onStartEdit={onStartEdit} />);

    expect(screen.getByTitle('Resend')).toBeInTheDocument();
    expect(screen.getByTitle('Edit')).toBeInTheDocument();
    expect(screen.getByTitle('Copy')).toBeInTheDocument();
    expect(screen.getByTitle('Delete')).toBeInTheDocument();
    expect(screen.queryByTitle('Regenerate')).not.toBeInTheDocument();
  });

  it('shows regenerate, copy, delete for assistant messages', () => {
    render(<MessageActionBar message={assistantMsg} />);

    expect(screen.getByTitle('Regenerate')).toBeInTheDocument();
    expect(screen.getByTitle('Copy')).toBeInTheDocument();
    expect(screen.getByTitle('Delete')).toBeInTheDocument();
    expect(screen.queryByTitle('Resend')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Edit')).not.toBeInTheDocument();
  });

  it('calls sendMessage on resend click', () => {
    render(<MessageActionBar message={userMsg} />);
    fireEvent.click(screen.getByTitle('Resend'));
    expect(mockSendMessage).toHaveBeenCalledWith('Hello world');
  });

  it('calls deleteMessage on double-click delete (confirmation)', () => {
    render(<MessageActionBar message={userMsg} />);
    // First click: enters confirmation state
    fireEvent.click(screen.getByTitle('Delete'));
    expect(mockDeleteMessage).not.toHaveBeenCalled();
    // Second click: confirms deletion
    fireEvent.click(screen.getByTitle('Click again to confirm delete'));
    expect(mockDeleteMessage).toHaveBeenCalledWith('u1');
  });

  it('calls regenerateMessage on regenerate click', () => {
    render(<MessageActionBar message={assistantMsg} />);
    fireEvent.click(screen.getByTitle('Regenerate'));
    expect(mockRegenerateMessage).toHaveBeenCalledWith('a1');
  });

  it('calls onStartEdit on edit click', () => {
    const onStartEdit = vi.fn();
    render(<MessageActionBar message={userMsg} onStartEdit={onStartEdit} />);
    fireEvent.click(screen.getByTitle('Edit'));
    expect(onStartEdit).toHaveBeenCalledTimes(1);
  });

  it('copies message content to clipboard', async () => {
    render(<MessageActionBar message={userMsg} />);
    await act(async () => {
      fireEvent.click(screen.getByTitle('Copy'));
    });
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Hello world');
  });

  it('does not resend or regenerate while streaming', () => {
    mockIsStreaming = true;
    render(<MessageActionBar message={userMsg} />);
    fireEvent.click(screen.getByTitle('Resend'));
    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});
