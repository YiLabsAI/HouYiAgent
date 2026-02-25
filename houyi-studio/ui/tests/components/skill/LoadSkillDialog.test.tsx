/**
 * Tests for LoadSkillDialog — Center Stage M skill loading form.
 *
 * Tests cover: rendering, mode switching, input validation, submit behavior
 * (loading state), server result handling, and close behavior.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { LoadSkillDialog } from '@/components/panels/skill/LoadSkillDialog';

describe('LoadSkillDialog', () => {
  const defaultProps = {
    isOpen: true,
    onLoad: vi.fn(),
    onClose: vi.fn(),
    loadResult: null as { success: boolean; message: string } | null,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ─── Rendering ────────────────────────────────────────────────

  it('renders dialog when open', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    expect(screen.getByTestId('load-skill-dialog')).toBeInTheDocument();
    expect(screen.getAllByText('Load Skill').length).toBeGreaterThanOrEqual(1);
  });

  it('does not render when closed', () => {
    render(<LoadSkillDialog {...defaultProps} isOpen={false} />);
    expect(screen.queryByTestId('load-skill-dialog')).not.toBeInTheDocument();
  });

  // ─── Mode selector ───────────────────────────────────────────

  it('renders three source mode buttons', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    expect(screen.getByTestId('load-mode-file')).toBeInTheDocument();
    expect(screen.getByTestId('load-mode-url')).toBeInTheDocument();
    expect(screen.getByTestId('load-mode-directory')).toBeInTheDocument();
  });

  it('file mode is default', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    const input = screen.getByTestId('load-skill-source-input');
    expect(input.getAttribute('placeholder')).toContain('SKILL.md');
  });

  it('switches to URL mode', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('load-mode-url'));
    const input = screen.getByTestId('load-skill-source-input');
    expect(input.getAttribute('placeholder')).toContain('github.com');
  });

  it('switches to directory mode', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('load-mode-directory'));
    const input = screen.getByTestId('load-skill-source-input');
    expect(input.getAttribute('placeholder')).toContain('skills');
  });

  // ─── Submit behavior ──────────────────────────────────────────

  it('submit button is disabled when input is empty', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    expect(screen.getByTestId('load-skill-submit')).toBeDisabled();
  });

  it('submit button becomes enabled when input is filled', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: '/path/to/SKILL.md' },
    });
    expect(screen.getByTestId('load-skill-submit')).not.toBeDisabled();
  });

  it('calls onLoad and shows loading state on submit', () => {
    const onLoad = vi.fn();
    const onClose = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onLoad={onLoad} onClose={onClose} />);
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: '/skills/my-skill/SKILL.md' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));
    expect(onLoad).toHaveBeenCalledWith('/skills/my-skill/SKILL.md');
    // Should NOT close immediately — waits for server response
    expect(onClose).not.toHaveBeenCalled();
    // Shows loading state
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('calls onLoad with trimmed path', () => {
    const onLoad = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onLoad={onLoad} />);
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: '  /path/to/SKILL.md  ' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));
    expect(onLoad).toHaveBeenCalledWith('/path/to/SKILL.md');
  });

  // ─── Server response handling ───────────────────────────────

  it('auto-closes on successful load result', async () => {
    const onLoad = vi.fn();
    const onClose = vi.fn();
    const { rerender } = render(
      <LoadSkillDialog isOpen={true} onLoad={onLoad} onClose={onClose} loadResult={null} />,
    );
    // Submit to enter loading state
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: '/path/SKILL.md' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));
    expect(screen.getByText('Loading...')).toBeInTheDocument();

    // Simulate server success via prop change
    rerender(
      <LoadSkillDialog
        isOpen={true}
        onLoad={onLoad}
        onClose={onClose}
        loadResult={{ success: true, message: 'Skill "test" loaded' }}
      />,
    );

    // Auto-close after delay
    await waitFor(() => expect(onClose).toHaveBeenCalled(), { timeout: 2000 });
  });

  it('shows error on failed load result', () => {
    const onLoad = vi.fn();
    const onClose = vi.fn();
    const { rerender } = render(
      <LoadSkillDialog isOpen={true} onLoad={onLoad} onClose={onClose} loadResult={null} />,
    );
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: '/bad/path' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));

    // Simulate server error via prop change
    rerender(
      <LoadSkillDialog
        isOpen={true}
        onLoad={onLoad}
        onClose={onClose}
        loadResult={{ success: false, message: 'File not found: /bad/path' }}
      />,
    );

    expect(screen.getByText(/File not found/)).toBeInTheDocument();
    // Should NOT close on error
    expect(onClose).not.toHaveBeenCalled();
    // Button should be re-enabled (not showing "Loading...")
    const submitBtn = screen.getByTestId('load-skill-submit');
    expect(submitBtn).not.toBeDisabled();
  });

  // ─── Validation ───────────────────────────────────────────────

  it('shows error for empty path on submit', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: 'temp' },
    });
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: '' },
    });
    expect(screen.getByTestId('load-skill-submit')).toBeDisabled();
  });

  it('validates URL mode requires http/https', () => {
    const onLoad = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onLoad={onLoad} />);
    fireEvent.click(screen.getByTestId('load-mode-url'));
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: 'ftp://example.com/SKILL.md' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));
    expect(onLoad).not.toHaveBeenCalled();
    expect(screen.getByText(/URL must start with/)).toBeInTheDocument();
  });

  it('accepts valid URL', () => {
    const onLoad = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onLoad={onLoad} />);
    fireEvent.click(screen.getByTestId('load-mode-url'));
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: 'https://raw.githubusercontent.com/example/SKILL.md' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));
    expect(onLoad).toHaveBeenCalledWith('https://raw.githubusercontent.com/example/SKILL.md');
  });

  it('accepts GitHub blob URL (backend converts to raw)', () => {
    const onLoad = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onLoad={onLoad} />);
    fireEvent.click(screen.getByTestId('load-mode-url'));
    fireEvent.change(screen.getByTestId('load-skill-source-input'), {
      target: { value: 'https://github.com/user/repo/blob/main/SKILL.md' },
    });
    fireEvent.click(screen.getByTestId('load-skill-submit'));
    expect(onLoad).toHaveBeenCalledWith('https://github.com/user/repo/blob/main/SKILL.md');
  });

  // ─── Enter key submission ─────────────────────────────────────

  it('submits on Enter key in input field', () => {
    const onLoad = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onLoad={onLoad} />);
    const input = screen.getByTestId('load-skill-source-input');
    fireEvent.change(input, { target: { value: '/path/to/SKILL.md' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onLoad).toHaveBeenCalledWith('/path/to/SKILL.md');
  });

  // ─── Close behavior ───────────────────────────────────────────

  it('calls onClose when cancel clicked', () => {
    const onClose = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onClose={onClose} />);
    fireEvent.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn();
    render(<LoadSkillDialog {...defaultProps} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('center-stage-close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  // ─── Mode switching clears input ──────────────────────────────

  it('clears input when switching modes', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    const input = screen.getByTestId('load-skill-source-input');
    fireEvent.change(input, { target: { value: '/some/path' } });
    expect(input).toHaveValue('/some/path');

    fireEvent.click(screen.getByTestId('load-mode-url'));
    expect(input).toHaveValue('');
  });

  // ─── Info box ─────────────────────────────────────────────────

  it('shows supported formats info', () => {
    render(<LoadSkillDialog {...defaultProps} />);
    expect(screen.getByText(/SKILL.md.*YAML frontmatter/)).toBeInTheDocument();
    expect(screen.getByText(/simpleskill\.json.*JSON manifest/)).toBeInTheDocument();
    expect(screen.getByText(/GitHub blob URLs.*auto-converted/)).toBeInTheDocument();
    expect(screen.getByText(/Directory.*recursively scans/)).toBeInTheDocument();
  });
});
