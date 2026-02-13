import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { Composer } from '@/components/Chat/Composer';

describe('Composer', () => {
  let onSend: ReturnType<typeof vi.fn>;
  let onStop: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onSend = vi.fn();
    onStop = vi.fn();
  });

  it('renders textarea and send button', () => {
    render(<Composer onSend={onSend} onStop={onStop} isStreaming={false} />);
    expect(screen.getByPlaceholderText(/Type a message/)).toBeInTheDocument();
    expect(screen.getByTitle('Send message')).toBeInTheDocument();
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
