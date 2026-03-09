import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { Composer } from '@/components/Chat/Composer';
import { useConsoleStore } from '@/stores/useConsoleStore';

type MockContextUsage = {
  used_tokens: number;
  reserved_output_tokens: number;
  available_tokens: number;
  max_context_tokens: number;
} | null;

const { mockUseChatStore, mockUseConsoleStore } = vi.hoisted(() => ({
  mockUseChatStore: vi.fn((selector?: (state: { contextUsage: MockContextUsage }) => unknown) => (
    selector ? selector({ contextUsage: null }) : { contextUsage: null }
  )),
  mockUseConsoleStore: vi.fn(),
}));

vi.mock('@/stores/useChatStore', () => ({
  useChatStore: mockUseChatStore,
}));

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: mockUseConsoleStore,
}));

describe('Composer', () => {
  let onSend: ReturnType<typeof vi.fn>;
  let onStop: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    const state = {
      showToast: vi.fn(),
      runSettings: {
        enable_tool_calls: true,
        tool_call_strategy: 'balanced' as const,
      },
    };
    mockUseChatStore.mockReset();
    mockUseChatStore.mockImplementation((selector?: (store: { contextUsage: MockContextUsage }) => unknown) => {
      const store = { contextUsage: null };
      return selector ? selector(store) : store;
    });
    mockUseConsoleStore.mockReset();
    vi.mocked(useConsoleStore).mockImplementation((((selector?: (store: typeof state) => unknown) => (
      selector ? selector(state) : state
    )) as unknown as typeof useConsoleStore));
    onSend = vi.fn();
    onStop = vi.fn();
  });

  it('renders textarea and send button', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/) as HTMLTextAreaElement;
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveClass('resize-none');
    expect(textarea).toHaveClass('overflow-y-auto');
    expect(screen.getByTestId('composer-resize-handle')).toBeInTheDocument();
    expect(screen.getByTitle('Send message')).toBeInTheDocument();
  });

  it('renders context token stats when context usage is available', async () => {
    mockUseChatStore.mockImplementation((selector?: (store: { contextUsage: MockContextUsage }) => unknown) => {
      const store = {
        contextUsage: {
          used_tokens: 1200,
          reserved_output_tokens: 800,
          available_tokens: 6000,
          max_context_tokens: 8000,
        },
      };
      return selector ? selector(store) : store;
    });

    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);

    expect(screen.queryByText('Context')).not.toBeInTheDocument();
    expect(screen.queryByText('Used 1,200')).not.toBeInTheDocument();
    expect(screen.queryByText('Reserve 800')).not.toBeInTheDocument();
    expect(screen.queryByText('Available 6,000')).not.toBeInTheDocument();
    expect(screen.queryByText('Max 8,000')).not.toBeInTheDocument();
    expect(screen.queryByText('15% used')).not.toBeInTheDocument();
  });

  it('sends message on Enter key', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/);

    fireEvent.change(textarea, { target: { value: 'hello' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSend).toHaveBeenCalledWith('hello', expect.any(Object));
  });

  it('does not send on Shift+Enter', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/);

    fireEvent.change(textarea, { target: { value: 'hello' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(onSend).not.toHaveBeenCalled();
  });

  it('does not send empty message', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/);

    fireEvent.change(textarea, { target: { value: '   ' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSend).not.toHaveBeenCalled();
  });

  it('shows stop button when streaming', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={true} />);
    expect(screen.getByTitle('Stop generating')).toBeInTheDocument();
    expect(screen.queryByTitle('Send message')).not.toBeInTheDocument();
  });

  it('calls onStop when stop button clicked', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={true} />);
    fireEvent.click(screen.getByTitle('Stop generating'));
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('renders toolbar buttons (attach, thinking, web search)', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    expect(screen.getByTitle('Attach file')).toBeInTheDocument();
    expect(screen.getByTitle('Thinking mode OFF')).toBeInTheDocument();
    expect(screen.getByTitle('Web search OFF')).toBeInTheDocument();
    expect(screen.getByTestId('deep-research-toggle')).toBeInTheDocument();
  });

  it('toggles thinking mode on click', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);

    const btn = screen.getByTitle('Thinking mode OFF');
    fireEvent.click(btn);
    expect(screen.getByTitle('Thinking mode ON')).toBeInTheDocument();

    fireEvent.click(screen.getByTitle('Thinking mode ON'));
    expect(screen.getByTitle('Thinking mode OFF')).toBeInTheDocument();
  });

  it('toggles web search on click', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);

    const btn = screen.getByTitle('Web search OFF');
    fireEvent.click(btn);
    expect(screen.getByTitle('Web search ON')).toBeInTheDocument();
  });

  it('toggles deep research on click', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);

    const btn = screen.getByTestId('deep-research-toggle');
    expect(btn).toHaveAttribute('title', expect.stringContaining('OFF'));
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('title', expect.stringContaining('ON'));
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('title', expect.stringContaining('OFF'));
  });

  it('passes enableReasoning in onSend options when toggled on', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/);

    // Toggle thinking mode ON
    fireEvent.click(screen.getByTitle('Thinking mode OFF'));

    fireEvent.change(textarea, { target: { value: 'test' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSend).toHaveBeenCalledWith('test', expect.objectContaining({
      enableReasoning: true,
    }));
  });

  it('clears text after sending', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/) as HTMLTextAreaElement;

    fireEvent.change(textarea, { target: { value: 'hello' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(textarea.value).toBe('');
  });

  it('send button is disabled when textarea is empty', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const sendBtn = screen.getByTitle('Send message');
    expect(sendBtn).toBeDisabled();
  });

  it('send button is enabled when textarea has content', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    const textarea = screen.getByPlaceholderText(/Type a message/);
    fireEvent.change(textarea, { target: { value: 'hello' } });
    const sendBtn = screen.getByTitle('Send message');
    expect(sendBtn).not.toBeDisabled();
  });
});
