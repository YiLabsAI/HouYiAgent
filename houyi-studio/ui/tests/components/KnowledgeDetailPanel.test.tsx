/**
 * Tests for KnowledgeDetailPanel Tier 1 component (P3-07 / §7.5.2).
 *
 * Covers:
 *   - Empty state when no library selected
 *   - Library metadata display (name, description, mode badge)
 *   - Document / chunk stats
 *   - Index status derivation (Empty / Not Indexed / Indexed)
 *   - Ingestion progress indicator
 *   - Configure / Rebuild action callbacks
 *   - Rebuild button disabled during active ingestion
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { KnowledgeDetailPanel } from '@/components/panels/KnowledgeDetailPanel';
import { useConsoleStore } from '@/stores/useConsoleStore';
import type { KnowledgeLibrary } from '@/types/ir';

vi.mock('@/stores/useConsoleStore', () => ({
  useConsoleStore: vi.fn(),
}));

const LIBRARY: KnowledgeLibrary = {
  library_id: 'lib-001',
  name: 'Project Docs',
  description: 'Internal project documentation',
  mode: 'indexed',
  knowledge_dir: '/data/docs',
  created_at: '2026-01-15T08:00:00Z',
  updated_at: '2026-02-01T12:00:00Z',
  doc_count: 42,
  chunk_count: 256,
  metadata: {},
};

const mockStore = (overrides: Record<string, unknown> = {}) => {
  const state: Record<string, unknown> = {
    selectedLibraryId: null,
    knowledgeLibraries: [],
    isIngesting: false,
    ingestLibraryId: null,
    ingestProgress: 0,
    ...overrides,
  };
  const fn = useConsoleStore as unknown as ReturnType<typeof vi.fn>;
  fn.mockImplementation((selector?: (s: typeof state) => unknown) =>
    selector ? selector(state) : state,
  );
};

describe('KnowledgeDetailPanel', () => {
  beforeEach(() => {
    mockStore();
  });

  // --- Empty state ---

  it('shows empty state when no library is selected', () => {
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText(/Select a knowledge library/i)).toBeInTheDocument();
  });

  it('shows empty state when selectedLibraryId has no match', () => {
    mockStore({ selectedLibraryId: 'nonexistent', knowledgeLibraries: [LIBRARY] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText(/Select a knowledge library/i)).toBeInTheDocument();
  });

  // --- Library metadata ---

  it('displays library name and description', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('Project Docs')).toBeInTheDocument();
    expect(screen.getByText('Internal project documentation')).toBeInTheDocument();
  });

  it('displays mode badge with correct label', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    render(<KnowledgeDetailPanel />);
    // "Indexed" appears as both mode badge and index status; verify at least 2
    const indexedElements = screen.getAllByText('Indexed');
    expect(indexedElements.length).toBeGreaterThanOrEqual(2);
  });

  it('displays agentic mode badge', () => {
    const agenticLib = { ...LIBRARY, mode: 'agentic' as const };
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [agenticLib] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('Agentic')).toBeInTheDocument();
  });

  // --- Stats ---

  it('displays document and chunk counts', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('256')).toBeInTheDocument();
    expect(screen.getByText('Documents')).toBeInTheDocument();
    expect(screen.getByText('Chunks')).toBeInTheDocument();
  });

  // --- Index status ---

  it('shows "Indexed" when both doc and chunk counts > 0', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    render(<KnowledgeDetailPanel />);
    // "Indexed" appears as badge AND as index status label
    const indexedElements = screen.getAllByText('Indexed');
    expect(indexedElements.length).toBeGreaterThanOrEqual(1);
  });

  it('shows "Not Indexed" when docs exist but chunks = 0', () => {
    const noChunks = { ...LIBRARY, chunk_count: 0 };
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [noChunks] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('Not Indexed')).toBeInTheDocument();
  });

  it('shows "Empty" when doc_count = 0', () => {
    const empty = { ...LIBRARY, doc_count: 0, chunk_count: 0 };
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [empty] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('Empty')).toBeInTheDocument();
  });

  // --- Ingestion progress ---

  it('shows ingestion progress bar when ingesting this library', () => {
    mockStore({
      selectedLibraryId: 'lib-001',
      knowledgeLibraries: [LIBRARY],
      isIngesting: true,
      ingestOperation: 'import',
      ingestLibraryId: 'lib-001',
      ingestProgress: 65,
    });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('Ingesting...')).toBeInTheDocument();
    expect(screen.getByText('65%')).toBeInTheDocument();
  });

  it('does NOT show ingestion progress when ingesting a different library', () => {
    mockStore({
      selectedLibraryId: 'lib-001',
      knowledgeLibraries: [LIBRARY],
      isIngesting: true,
      ingestOperation: 'import',
      ingestLibraryId: 'lib-other',
      ingestProgress: 40,
    });
    render(<KnowledgeDetailPanel />);
    expect(screen.queryByText('Ingesting...')).not.toBeInTheDocument();
  });

  // --- Action callbacks ---

  it('calls onConfigure with library_id when Configure button clicked', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    const onConfigure = vi.fn();
    render(<KnowledgeDetailPanel onConfigure={onConfigure} />);
    fireEvent.click(screen.getByText('Configure...'));
    expect(onConfigure).toHaveBeenCalledWith('lib-001');
  });

  it('calls onRebuildIndex with library_id when Rebuild button clicked', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    const onRebuildIndex = vi.fn();
    render(<KnowledgeDetailPanel onRebuildIndex={onRebuildIndex} />);
    fireEvent.click(screen.getByText('Rebuild'));
    expect(onRebuildIndex).toHaveBeenCalledWith('lib-001');
  });

  it('disables Rebuild button during active ingestion', () => {
    mockStore({
      selectedLibraryId: 'lib-001',
      knowledgeLibraries: [LIBRARY],
      isIngesting: true,
      ingestLibraryId: 'lib-001',
      ingestProgress: 50,
    });
    render(<KnowledgeDetailPanel />);
    const rebuildBtn = screen.getByText('Rebuild').closest('button');
    expect(rebuildBtn).toBeDisabled();
  });

  // --- Metadata display ---

  it('displays directory path', () => {
    mockStore({ selectedLibraryId: 'lib-001', knowledgeLibraries: [LIBRARY] });
    render(<KnowledgeDetailPanel />);
    expect(screen.getByText('/data/docs')).toBeInTheDocument();
  });
});
