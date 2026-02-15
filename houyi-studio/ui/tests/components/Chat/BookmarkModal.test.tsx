import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { BookmarkModal } from '@/components/Chat/BookmarkModal';

const mockLoadConversation = vi.fn();
vi.mock('@/stores/useChatStore', () => ({
  useChatStore: (selector: (state: { loadConversation: typeof mockLoadConversation }) => unknown) =>
    selector({ loadConversation: mockLoadConversation }),
}));

const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('BookmarkModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = mockFetch;
  });

  it('returns null when closed', () => {
    const { container } = render(<BookmarkModal isOpen={false} onClose={vi.fn()} />);
    expect(container.innerHTML).toBe('');
  });

  it('loads bookmarks when opened', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        bookmarks: [
          {
            type: 'conversation',
            conversation_id: 'conv_1',
            title: 'Architecture Discussion',
            created_at: 1700000000,
            updated_at: 1700000100,
          },
        ],
      }),
    });

    render(<BookmarkModal isOpen={true} onClose={vi.fn()} />);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/chat/bookmarks');
    });

    expect(screen.getByText('Architecture Discussion')).toBeInTheDocument();
    expect(screen.getByText('Bookmarks')).toBeInTheDocument();
  });

  it('navigates to conversation when an item is clicked', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        bookmarks: [
          {
            type: 'message',
            conversation_id: 'conv_2',
            message_id: 'msg_9',
            title: 'Bug Triage',
            role: 'assistant',
            snippet: 'Need to reproduce this issue first',
            created_at: 1700000200,
            updated_at: 1700000201,
          },
        ],
      }),
    });

    const onClose = vi.fn();
    render(<BookmarkModal isOpen={true} onClose={onClose} />);

    const row = await screen.findByRole('button', { name: /Bug Triage/i });
    fireEvent.click(row);

    expect(mockLoadConversation).toHaveBeenCalledWith('conv_2', 'msg_9');
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
