import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { SearchModal } from '@/components/Chat/SearchModal';

// Mock useChatStore
const mockLoadConversation = vi.fn();
vi.mock('@/stores/useChatStore', () => ({
  useChatStore: (selector: any) => {
    const state = { loadConversation: mockLoadConversation };
    return selector(state);
  },
}));

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('SearchModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns null when not open', () => {
    const { container } = render(<SearchModal isOpen={false} onClose={vi.fn()} />);
    expect(container.innerHTML).toBe('');
  });

  it('renders search input when open', () => {
    render(<SearchModal isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByPlaceholderText('Search conversations...')).toBeInTheDocument();
  });

  it('shows empty state text when no query', () => {
    render(<SearchModal isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByText('Type to search conversation history')).toBeInTheDocument();
  });

  it('calls onClose when backdrop clicked', () => {
    const onClose = vi.fn();
    render(<SearchModal isOpen={true} onClose={onClose} />);
    // The backdrop is the first fixed div
    const backdrop = document.querySelector('.fixed.inset-0');
    fireEvent.click(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose on Escape key', () => {
    const onClose = vi.fn();
    render(<SearchModal isOpen={true} onClose={onClose} />);
    const modal = document.querySelector('.fixed.top-\\[10\\%\\]');
    fireEvent.keyDown(modal!, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('fetches search results after debounce', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            conversation_id: 'c1',
            title: 'Test Conv',
            match_type: 'message',
            message_id: 'm1',
            role: 'user',
            snippet: 'hello world',
            created_at: 1000,
          },
        ],
      }),
    });

    render(<SearchModal isOpen={true} onClose={vi.fn()} />);
    const input = screen.getByPlaceholderText('Search conversations...');

    fireEvent.change(input, { target: { value: 'hello' } });

    // Advance past debounce
    await vi.advanceTimersByTimeAsync(300);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/chat/search?q=hello&limit=50')
      );
    });

    await waitFor(() => {
      // highlightText splits text into <mark> + plain nodes, so use container query
      const container = document.querySelector('.overflow-y-auto');
      expect(container?.textContent).toContain('Test Conv');
      expect(container?.textContent).toContain('hello world');
    });
  });

  it('navigates to conversation on result click', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        results: [
          {
            conversation_id: 'c1',
            title: 'Test Conv',
            match_type: 'title',
            message_id: null,
            role: null,
            snippet: 'Test Conv',
            created_at: 1000,
          },
        ],
      }),
    });

    const onClose = vi.fn();
    render(<SearchModal isOpen={true} onClose={onClose} />);
    const input = screen.getByPlaceholderText('Search conversations...');

    fireEvent.change(input, { target: { value: 'Test' } });
    await vi.advanceTimersByTimeAsync(300);

    await waitFor(() => {
      const results = document.querySelector('.overflow-y-auto');
      expect(results?.textContent).toContain('Test Conv');
    });

    // Click the result button
    const resultBtn = document.querySelector('.overflow-y-auto button');
    fireEvent.click(resultBtn!);
    expect(mockLoadConversation).toHaveBeenCalledWith('c1');
    expect(onClose).toHaveBeenCalled();
  });

  it('shows no results message for empty results', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ results: [] }),
    });

    render(<SearchModal isOpen={true} onClose={vi.fn()} />);
    const input = screen.getByPlaceholderText('Search conversations...');

    fireEvent.change(input, { target: { value: 'nonexistent' } });
    await vi.advanceTimersByTimeAsync(300);

    await waitFor(() => {
      expect(screen.getByText(/No results for/)).toBeInTheDocument();
    });
  });
});
