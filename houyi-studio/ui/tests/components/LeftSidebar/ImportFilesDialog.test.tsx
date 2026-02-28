import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ImportFilesDialog } from '@/components/LeftSidebar/ImportFilesDialog';

describe('ImportFilesDialog', () => {
  const defaultProps = {
    isOpen: true,
    libraryId: 'lib_test',
    libraryName: 'Test Library',
    onImport: vi.fn(),
    onCancel: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    if (typeof localStorage !== 'undefined' && typeof localStorage.clear === 'function') {
      localStorage.clear();
    }
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ uploaded_paths: ['/tmp/uploaded.md'], errors: [] }),
      }),
    );
  });

  it('renders mode-specific browse controls', () => {
    render(<ImportFilesDialog {...defaultProps} />);
    expect(screen.getByTestId('import-browse-folder')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Files'));
    expect(screen.getByTestId('import-browse-file')).toBeInTheDocument();
    expect(screen.queryByTestId('import-browse-folder')).not.toBeInTheDocument();
  });

  it('accepts dropped plain-text path', async () => {
    render(<ImportFilesDialog {...defaultProps} />);

    const dropzone = screen.getByText(/drop local files\/folders/i).closest('div');
    expect(dropzone).not.toBeNull();

    fireEvent.drop(dropzone!, {
      dataTransfer: {
        getData: (key: string) => (key === 'text/plain' ? '/tmp/knowledge' : ''),
        items: [],
      },
    });

    const textarea = screen.getByRole('textbox');
    await waitFor(() => {
      expect(textarea).toHaveValue('/tmp/knowledge');
    });
  });

  it('submits imported paths', () => {
    const onImport = vi.fn();
    render(<ImportFilesDialog {...defaultProps} onImport={onImport} />);

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, {
      target: { value: '/tmp/knowledge/doc1.md\n/tmp/knowledge/doc2.md' },
    });

    fireEvent.click(screen.getByText('Import'));

    // Now requires confirmation
    expect(screen.getByText('Confirm Import')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Start Import'));

    expect(onImport).toHaveBeenCalledWith(['/tmp/knowledge/doc1.md', '/tmp/knowledge/doc2.md']);
  });
});
