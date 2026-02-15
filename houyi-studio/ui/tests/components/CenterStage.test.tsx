/**
 * Tests for CenterStage overlay component (§7.5.3).
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { CenterStage } from '@/components/CenterStage';

describe('CenterStage', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    size: 'M' as const,
    title: 'Test Dialog',
    children: <div>Dialog content</div>,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ─── Rendering ─────────────────────────────────────────────────

  it('should render when isOpen is true', () => {
    render(<CenterStage {...defaultProps} />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText('Test Dialog')).toBeInTheDocument();
    expect(screen.getByText('Dialog content')).toBeInTheDocument();
  });

  it('should not render when isOpen is false', () => {
    render(<CenterStage {...defaultProps} isOpen={false} />);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('should render title in the header', () => {
    render(<CenterStage {...defaultProps} title="Skill Configuration" />);
    expect(screen.getByText('Skill Configuration')).toBeInTheDocument();
  });

  it('should render children in the body', () => {
    render(
      <CenterStage {...defaultProps}>
        <span>Custom content</span>
      </CenterStage>
    );
    expect(screen.getByText('Custom content')).toBeInTheDocument();
  });

  // ─── Size variants ─────────────────────────────────────────────

  it('should apply size S classes', () => {
    render(<CenterStage {...defaultProps} size="S" />);
    const panel = screen.getByTestId('center-stage-panel');
    expect(panel.getAttribute('data-size')).toBe('S');
    expect(panel.className).toContain('w-[480px]');
  });

  it('should apply size M classes', () => {
    render(<CenterStage {...defaultProps} size="M" />);
    const panel = screen.getByTestId('center-stage-panel');
    expect(panel.getAttribute('data-size')).toBe('M');
    expect(panel.className).toContain('w-[640px]');
  });

  it('should apply size L classes', () => {
    render(<CenterStage {...defaultProps} size="L" />);
    const panel = screen.getByTestId('center-stage-panel');
    expect(panel.getAttribute('data-size')).toBe('L');
    expect(panel.className).toContain('w-[90vw]');
    expect(panel.className).toContain('h-[85vh]');
  });

  // ─── Close behavior ────────────────────────────────────────────

  it('should call onClose when close button is clicked', () => {
    const onClose = vi.fn();
    render(<CenterStage {...defaultProps} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('center-stage-close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('should call onClose when Esc key is pressed', () => {
    const onClose = vi.fn();
    render(<CenterStage {...defaultProps} onClose={onClose} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('should not call onClose for non-Escape keys', () => {
    const onClose = vi.fn();
    render(<CenterStage {...defaultProps} onClose={onClose} />);
    fireEvent.keyDown(window, { key: 'Enter' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('should call onClose when backdrop is clicked (default)', () => {
    const onClose = vi.fn();
    render(<CenterStage {...defaultProps} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('center-stage-backdrop'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('should NOT call onClose when backdrop is clicked if closeOnBackdrop is false', () => {
    const onClose = vi.fn();
    render(<CenterStage {...defaultProps} onClose={onClose} closeOnBackdrop={false} />);
    fireEvent.click(screen.getByTestId('center-stage-backdrop'));
    expect(onClose).not.toHaveBeenCalled();
  });

  // ─── Accessibility ─────────────────────────────────────────────

  it('should have role=dialog and aria-modal', () => {
    render(<CenterStage {...defaultProps} />);
    const dialog = screen.getByRole('dialog');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.getAttribute('aria-label')).toBe('Test Dialog');
  });

  // ─── Esc cleanup on unmount ────────────────────────────────────

  it('should not call onClose after component unmounts', () => {
    const onClose = vi.fn();
    const { unmount } = render(<CenterStage {...defaultProps} onClose={onClose} />);
    unmount();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
  });

  // ─── Transition from closed to open ────────────────────────────

  it('should render when transitioning from closed to open', () => {
    const { rerender } = render(<CenterStage {...defaultProps} isOpen={false} />);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    rerender(<CenterStage {...defaultProps} isOpen={true} />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});
