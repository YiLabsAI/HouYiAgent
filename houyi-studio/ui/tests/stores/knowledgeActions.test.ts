/**
 * Tests for knowledgeActions store actions.
 *
 * TDD: These tests define expected behavior. Some will fail initially (RED),
 * then we fix the code to make them pass (GREEN).
 */

import { describe, it, expect, vi } from 'vitest';

// Mock zustand store pattern similar to toastActions.test.ts
function createMockStore() {
  let state: any = {
    toasts: [],
    toastKeys: {},
    knowledgeLibraries: [],
    selectedLibraryId: null,
    knowledgeSearchResults: null,
    knowledgeSearchLoading: false,
    ws: null,
    sessionId: 'test-session',
  };

  const set = (partial: any) => {
    const next = typeof partial === 'function' ? partial(state) : partial;
    state = { ...state, ...next };
  };

  const get = () => state;

  return { state: () => state, set, get };
}

describe('knowledgeActions', () => {
  describe('K-001: createKnowledgeLibrary should NOT show toast on send', () => {
    it('should only show toast when library is actually created (on event), not when command is sent', async () => {
      const { state, set, get } = createMockStore();

      // Import and create actions
      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);

      // Merge actions into state
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      // Call createKnowledgeLibrary
      state().createKnowledgeLibrary({
        name: 'Test KB',
        description: 'Test description',
        mode: 'auto',
        knowledge_dir: './knowledge',
      });

      // Should NOT show toast immediately
      // The toast should only be shown when the backend responds with success
      expect(state().showToast).not.toHaveBeenCalled();

      // Command should be sent
      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'create_knowledge_library',
          name: 'Test KB',
        })
      );
    });

    it('should show exactly ONE success toast when library is added', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);

      Object.assign(state(), actions);
      const showToastMock = vi.fn();
      state().showToast = showToastMock;

      // Simulate backend response: addKnowledgeLibrary is called
      state().addKnowledgeLibrary({
        library_id: 'lib-123',
        name: 'Test KB',
        description: '',
        mode: 'auto',
        knowledge_dir: './knowledge',
        created_at: new Date().toISOString(),
      });

      // Should show exactly ONE toast
      expect(showToastMock).toHaveBeenCalledTimes(1);
      expect(showToastMock).toHaveBeenCalledWith(
        expect.stringContaining('Test KB'),
        'success'
      );
    });
  });

  describe('addKnowledgeLibrary', () => {
    it('should add library to state', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();

      const library = {
        library_id: 'lib-123',
        name: 'Test KB',
        description: '',
        mode: 'auto' as const,
        knowledge_dir: './knowledge',
        created_at: new Date().toISOString(),
      };

      state().addKnowledgeLibrary(library);

      expect(state().knowledgeLibraries).toHaveLength(1);
      expect(state().knowledgeLibraries[0].library_id).toBe('lib-123');
    });
  });

  describe('createKnowledgeLibrary duplicate name validation', () => {
    it('should show error toast and NOT send command when name already exists', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      const showToastMock = vi.fn();
      const sendCommandMock = vi.fn();
      state().showToast = showToastMock;
      state().sendCommand = sendCommandMock;

      // Pre-populate with an existing library
      state().knowledgeLibraries = [
        {
          library_id: 'lib-123',
          name: 'Existing KB',
          description: '',
          mode: 'auto',
          knowledge_dir: './knowledge',
          created_at: new Date().toISOString(),
        },
      ];

      // Try to create library with same name
      state().createKnowledgeLibrary({
        name: 'Existing KB',
        description: 'Test description',
        mode: 'auto',
        knowledge_dir: './knowledge',
      });

      // Should show error toast
      expect(showToastMock).toHaveBeenCalledTimes(1);
      expect(showToastMock).toHaveBeenCalledWith(
        expect.stringContaining('already exists'),
        'error'
      );

      // Should NOT send command
      expect(sendCommandMock).not.toHaveBeenCalled();
    });

    it('should be case-insensitive when checking for duplicates', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      const showToastMock = vi.fn();
      const sendCommandMock = vi.fn();
      state().showToast = showToastMock;
      state().sendCommand = sendCommandMock;

      state().knowledgeLibraries = [
        {
          library_id: 'lib-123',
          name: 'My Knowledge Base',
          description: '',
          mode: 'auto',
          knowledge_dir: './knowledge',
          created_at: new Date().toISOString(),
        },
      ];

      // Try with different case
      state().createKnowledgeLibrary({
        name: 'MY KNOWLEDGE BASE',
        description: 'Test',
        mode: 'auto',
        knowledge_dir: './knowledge',
      });

      // Should show error (case-insensitive match)
      expect(showToastMock).toHaveBeenCalledWith(
        expect.stringContaining('already exists'),
        'error'
      );
      expect(sendCommandMock).not.toHaveBeenCalled();
    });
  });

  describe('deleteKnowledgeLibrary', () => {
    it('should send delete command without showing toast', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().deleteKnowledgeLibrary('lib-123');

      // Should send command
      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'delete_knowledge_library',
          library_id: 'lib-123',
        })
      );

      // Should NOT show toast until backend confirms
      expect(state().showToast).not.toHaveBeenCalled();
    });
  });

  describe('removeKnowledgeLibrary', () => {
    it('should remove library from state and show success toast', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();

      // Pre-populate with a library
      state().knowledgeLibraries = [
        {
          library_id: 'lib-123',
          name: 'Test KB',
          description: '',
          mode: 'auto',
          knowledge_dir: './knowledge',
          created_at: new Date().toISOString(),
        },
      ];

      state().removeKnowledgeLibrary('lib-123');

      expect(state().knowledgeLibraries).toHaveLength(0);
      expect(state().showToast).toHaveBeenCalledTimes(1);
      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('deleted'),
        'success'
      );
    });
  });

  describe('searchKnowledge', () => {
    it('should send search command with correct parameters', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();
      state().selectedLibraryId = 'lib-123';

      state().searchKnowledge('test query', 'lib-123', 'indexed', 10);

      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'search_knowledge',
          query: 'test query',
          library_id: 'lib-123',
          mode: 'indexed',
          top_k: 10,
        })
      );
      expect(state().isSearchingKnowledge).toBe(true);
    });

    it('should show error toast for empty query', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().searchKnowledge('   ');

      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('enter a search query'),
        'error'
      );
      expect(state().sendCommand).not.toHaveBeenCalled();
    });
  });

  describe('editKnowledgeLibrary', () => {
    it('should send update command with updates', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().sendCommand = vi.fn();

      state().editKnowledgeLibrary('lib-123', { name: 'New Name', description: 'New desc' });

      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'update_knowledge_library',
          library_id: 'lib-123',
          updates: { name: 'New Name', description: 'New desc' },
        })
      );
    });
  });

  describe('rebuildKnowledgeIndex', () => {
    it('should send rebuild command with incremental=false by default', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().rebuildKnowledgeIndex('lib-123');

      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'rebuild_knowledge_index',
          library_id: 'lib-123',
          incremental: false,
        })
      );
      expect(state().isIngesting).toBe(true);
      expect(state().ingestLibraryId).toBe('lib-123');
    });

    it('should send rebuild command with incremental=true when specified', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().rebuildKnowledgeIndex('lib-123', true);

      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'rebuild_knowledge_index',
          library_id: 'lib-123',
          incremental: true,
        })
      );
    });

    it('should show error toast when no libraryId provided', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().rebuildKnowledgeIndex('');

      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('select a library'),
        'error'
      );
      expect(state().sendCommand).not.toHaveBeenCalled();
    });
  });

  describe('ingestKnowledgeFiles', () => {
    it('should send ingest command with paths', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().ingestKnowledgeFiles('lib-123', ['./docs', './readme.md']);

      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'ingest_knowledge_files',
          library_id: 'lib-123',
          paths: ['./docs', './readme.md'],
        })
      );
      expect(state().isIngesting).toBe(true);
    });

    it('should show error toast for empty paths', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();

      state().ingestKnowledgeFiles('lib-123', []);

      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('specify files'),
        'error'
      );
      expect(state().sendCommand).not.toHaveBeenCalled();
    });
  });

  describe('cancelIngest', () => {
    it('should send cancel command and clear ingest state', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().sendCommand = vi.fn();
      state().ingestLibraryId = 'lib-123';
      state().isIngesting = true;

      state().cancelIngest();

      expect(state().sendCommand).toHaveBeenCalledWith(
        expect.objectContaining({
          command_type: 'cancel_ingest',
          library_id: 'lib-123',
        })
      );
      expect(state().isIngesting).toBe(false);
      expect(state().ingestLibraryId).toBe(null);
      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('cancelled'),
        'info'
      );
    });

    it('should do nothing when no ingest in progress', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().sendCommand = vi.fn();
      state().ingestLibraryId = null;

      state().cancelIngest();

      expect(state().sendCommand).not.toHaveBeenCalled();
    });
  });

  describe('handleIngestProgress', () => {
    it('should update ingest progress state', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);

      state().handleIngestProgress({
        library_id: 'lib-123',
        progress: 50,
        current_file: 'doc.md',
        files_processed: 5,
        total_files: 10,
      });

      expect(state().ingestProgress).toBe(50);
      expect(state().ingestCurrentFile).toBe('doc.md');
      expect(state().ingestFilesProcessed).toBe(5);
      expect(state().ingestTotalFiles).toBe(10);
    });
  });

  describe('handleIngestComplete', () => {
    it('should clear ingest state and show success toast on success', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().isIngesting = true;
      state().ingestLibraryId = 'lib-123';
      state().ingestLibraryName = 'Test Library';
      state().ingestPaths = ['/path/to/docs'];

      state().handleIngestComplete({
        library_id: 'lib-123',
        success: true,
        stats: { files_processed: 10, chunks_created: 50, files_skipped: 2 },
        message: '',
      });

      expect(state().isIngesting).toBe(false);
      expect(state().ingestLibraryId).toBe(null);
      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('10 files'),
        'success'
      );
      // Should include skipped count
      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('2 skipped'),
        'success'
      );
    });

    it('should show error toast on failure', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);
      state().showToast = vi.fn();
      state().isIngesting = true;

      state().handleIngestComplete({
        library_id: 'lib-123',
        success: false,
        stats: {},
        message: 'Directory not found',
      });

      expect(state().isIngesting).toBe(false);
      expect(state().showToast).toHaveBeenCalledWith(
        expect.stringContaining('Directory not found'),
        'error'
      );
    });
  });

  describe('updateKnowledgeLibrary', () => {
    it('should update library in state', async () => {
      const { state, set, get } = createMockStore();

      const { createKnowledgeActions } = await import('@/stores/storeActions/knowledgeActions');
      const actions = createKnowledgeActions(set, get);
      Object.assign(state(), actions);

      state().knowledgeLibraries = [
        {
          library_id: 'lib-123',
          name: 'Old Name',
          description: '',
          mode: 'auto',
          knowledge_dir: './knowledge',
          created_at: new Date().toISOString(),
        },
      ];

      state().updateKnowledgeLibrary({
        library_id: 'lib-123',
        name: 'New Name',
        description: 'Updated',
        mode: 'indexed',
        knowledge_dir: './knowledge',
        created_at: new Date().toISOString(),
      });

      expect(state().knowledgeLibraries[0].name).toBe('New Name');
      expect(state().knowledgeLibraries[0].description).toBe('Updated');
      expect(state().knowledgeLibraries[0].mode).toBe('indexed');
    });
  });
});
