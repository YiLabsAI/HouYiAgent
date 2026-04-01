import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryInbox } from '@/components/Memory/MemoryInbox';
import type { MemoryCandidate } from '@/stores/useMemoryStore';

const mockStore: Record<string, unknown> = {
  candidates: [] as MemoryCandidate[],
  records: [],
  config: { enabled: true, auto_extract: true },
  filter: 'pending',
  loading: false,
  error: null as string | null,
  setFilter: vi.fn(),
  fetchCandidates: vi.fn(),
  fetchRecords: vi.fn(),
  fetchConfig: vi.fn(),
  updateConfig: vi.fn(),
  approveCandidate: vi.fn(),
  rejectCandidate: vi.fn(),
  updateCandidate: vi.fn(),
  updateRecord: vi.fn(),
  deleteRecord: vi.fn(),
};

vi.mock('@/stores/useMemoryStore', () => ({
  useMemoryStore: vi.fn((selector?: (s: typeof mockStore) => unknown) =>
    selector ? selector(mockStore) : mockStore,
  ),
}));

describe('MemoryInbox', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.candidates = [];
    mockStore.records = [];
    mockStore.config = { enabled: true, auto_extract: true };
    mockStore.filter = 'pending';
    mockStore.loading = false;
    mockStore.error = null;
  });

  it('renders title', () => {
    render(<MemoryInbox />);
    expect(screen.getByRole('heading', { name: 'Memory' })).toBeInTheDocument();
    expect(screen.getByText('Review candidates and manage stored memories')).toBeInTheDocument();
  });

  it('renders filter tabs', () => {
    render(<MemoryInbox />);
    expect(screen.getByRole('button', { name: 'Pending' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Approved' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Rejected' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'All' })).toBeInTheDocument();
  });

  it('shows empty state', () => {
    render(<MemoryInbox />);
    expect(screen.getByText('No memory candidates')).toBeInTheDocument();
  });

  it('shows loading', () => {
    mockStore.loading = true;
    render(<MemoryInbox />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('shows error', () => {
    mockStore.error = 'failed';
    render(<MemoryInbox />);
    expect(screen.getByText('failed')).toBeInTheDocument();
  });

  it('renders candidate card', () => {
    mockStore.candidates = [
      {
        candidate_id: 'c1',
        content: 'Important fact',
        source_context: 'session-1',
        confidence: 0.9,
        suggested_tags: [],
        status: 'pending',
      },
    ];
    render(<MemoryInbox />);
    expect(screen.getByText('Important fact')).toBeInTheDocument();
  });

  it('shows candidate metadata', () => {
    mockStore.candidates = [
      {
        candidate_id: 'c1',
        content: 'Important fact',
        memory_type: 'preference',
        source_context: 'turn:3',
        confidence: 0.87,
        suggested_tags: ['pinned'],
        status: 'pending',
      },
    ];
    render(<MemoryInbox />);
    expect(screen.getByText('turn 3')).toBeInTheDocument();
    expect(screen.getByText('87%')).toBeInTheDocument();
    expect(screen.getByText('preference')).toBeInTheDocument();
    expect(screen.getByText('pinned')).toBeInTheDocument();
  });

  it('clicking filter calls setFilter', () => {
    render(<MemoryInbox />);
    fireEvent.click(screen.getByRole('button', { name: 'Approved' }));
    expect(mockStore.setFilter).toHaveBeenCalledWith('approved');
  });

  it('fetchCandidates called on mount', () => {
    render(<MemoryInbox />);
    expect(mockStore.fetchCandidates).toHaveBeenCalled();
    expect(mockStore.fetchRecords).toHaveBeenCalled();
    expect(mockStore.fetchConfig).toHaveBeenCalled();
  });

  it('renders Records tab', () => {
    render(<MemoryInbox />);
    const recordsTab = screen.getByRole('button', { name: /Records/ });
    expect(recordsTab).toBeInTheDocument();
  });

  it('switches to Records view', () => {
    render(<MemoryInbox />);
    fireEvent.click(screen.getByRole('button', { name: /Records/ }));
    expect(screen.getByText('No stored memories')).toBeInTheDocument();
  });

  it('does not render config toggles (managed in Global Settings)', () => {
    render(<MemoryInbox />);
    expect(screen.queryByText('Auto-extract')).not.toBeInTheDocument();
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
  });
});
