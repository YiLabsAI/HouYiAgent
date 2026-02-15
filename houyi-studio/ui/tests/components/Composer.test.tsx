/**
 * Tests for Composer component — specifically the P3-18 Deep Research toggle.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Composer } from '@/components/Chat/Composer';
import { useConsoleStore } from '@/stores/useConsoleStore';

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: vi.fn(),
}));

const mockConsoleStore = () => {
  const state = {
    showToast: vi.fn(),
  };
  const fn = useConsoleStore as unknown as ReturnType<typeof vi.fn>;
  fn.mockImplementation((selector?: (s: typeof state) => unknown) =>
    selector ? selector(state) : state,
  );
};

describe('Composer', () => {
  beforeEach(() => {
    mockConsoleStore();
  });

  it('renders Deep Research toggle button', () => {
    render(
      <Composer onSend={vi.fn()} onStop={vi.fn()} isStreaming={false} />,
    );
    const btn = screen.getByTestId('deep-research-toggle');
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('title', expect.stringContaining('Deep Research OFF'));
  });

  it('toggles Deep Research on/off', () => {
    render(
      <Composer onSend={vi.fn()} onStop={vi.fn()} isStreaming={false} />,
    );
    const btn = screen.getByTestId('deep-research-toggle');
    // Initially OFF
    expect(btn).toHaveAttribute('title', expect.stringContaining('OFF'));
    // Click to toggle ON
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('title', expect.stringContaining('ON'));
    // Click to toggle OFF again
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('title', expect.stringContaining('OFF'));
  });

  it('renders reasoning and web search toggles alongside deep research', () => {
    render(
      <Composer onSend={vi.fn()} onStop={vi.fn()} isStreaming={false} />,
    );
    expect(screen.getByTitle('Thinking mode OFF')).toBeInTheDocument();
    expect(screen.getByTitle('Web search OFF')).toBeInTheDocument();
    expect(screen.getByTestId('deep-research-toggle')).toBeInTheDocument();
  });

  it('sends message on enter', () => {
    const onSend = vi.fn();
    render(
      <Composer onSend={onSend} onStop={vi.fn()} isStreaming={false} />,
    );
    const textarea = screen.getByTestId('chat-input');
    fireEvent.change(textarea, { target: { value: 'Hello world' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });
    expect(onSend).toHaveBeenCalledWith(
      'Hello world',
      expect.objectContaining({}),
    );
  });
});
