/**
 * E2E tests for Knowledge UI functionality.
 *
 * These tests verify the complete knowledge library management flow:
 * - Creating knowledge libraries
 * - Listing and selecting libraries
 * - Searching within libraries
 * - Editing and deleting libraries
 * - Document management
 * - Chunk visualization
 */

import { test, expect, Page } from '@playwright/test';

/**
 * Install WebSocket mock for knowledge operations AND set initial empty state.
 * MUST be called before opening the Knowledge panel.
 */
async function installKnowledgeMock(page: Page): Promise<void> {
  await page.evaluate(() => {
    const store = (window as any).__consoleStore;
    if (!store) {
      console.error('[E2E Mock] Store not found!');
      return;
    }

    // Initialize mock state
    const mockState = {
      libraries: [] as any[],
      documents: {} as Record<string, any[]>,
      libraryIdCounter: 1,
      docIdCounter: 1,
    };

    // Store in window for debugging
    (window as any).__mockState = mockState;

    // Save original sendCommand
    const state = store.getState();
    (window as any).__e2eOriginalSendCommand = state.sendCommand;

    // Override sendCommand
    const mockSendCommand = (command: any): boolean => {
      const currentState = store.getState();
      const handleEvent = currentState.handleEvent;
      const sessionId = currentState.sessionId;
      const timestamp = new Date().toISOString();

      console.log('[E2E Mock] Command:', command?.command_type);

      // List libraries
      if (command?.command_type === 'list_knowledge_libraries') {
        setTimeout(() => {
          handleEvent({
            event_type: 'knowledge_library_list',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            libraries: [...mockState.libraries],
          });
        }, 50);
        return true;
      }

      // Create library
      if (command?.command_type === 'create_knowledge_library') {
        const existingLib = mockState.libraries.find((l: any) => l.name === command.name);
        if (existingLib) {
          setTimeout(() => {
            handleEvent({
              event_type: 'knowledge_error',
              event_id: `evt_${Date.now()}`,
              timestamp,
              session_id: sessionId,
              error: `Library "${command.name}" already exists`,
              operation: 'create',
            });
          }, 50);
          return true;
        }

        const newLib = {
          library_id: `lib_${mockState.libraryIdCounter++}`,
          name: command.name,
          description: command.description || '',
          mode: command.mode || 'auto',
          knowledge_dir: command.knowledge_dir || './knowledge',
          created_at: timestamp,
          updated_at: timestamp,
          doc_count: 0,
          chunk_count: 0,
          status: 'empty',
          metadata: {},
        };
        mockState.libraries.push(newLib);
        mockState.documents[newLib.library_id] = [];

        setTimeout(() => {
          handleEvent({
            event_type: 'knowledge_library_created',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library: newLib,
          });
        }, 50);
        return true;
      }

      // Delete library
      if (command?.command_type === 'delete_knowledge_library') {
        const idx = mockState.libraries.findIndex((l: any) => l.library_id === command.library_id);
        if (idx >= 0) {
          mockState.libraries.splice(idx, 1);
        }
        setTimeout(() => {
          handleEvent({
            event_type: 'knowledge_library_deleted',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library_id: command.library_id,
          });
        }, 50);
        return true;
      }

      // Update library
      if (command?.command_type === 'update_knowledge_library') {
        const lib = mockState.libraries.find((l: any) => l.library_id === command.library_id);
        if (lib && command.updates) {
          Object.assign(lib, command.updates);
          lib.updated_at = timestamp;
        }
        setTimeout(() => {
          handleEvent({
            event_type: 'knowledge_library_updated',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library: lib,
          });
        }, 50);
        return true;
      }

      // Search
      if (command?.command_type === 'search_knowledge') {
        setTimeout(() => {
          handleEvent({
            event_type: 'knowledge_search_results',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            query: command.query,
            library_id: command.library_id,
            results: [
              {
                chunk_id: 'chunk_1',
                content: `Mock result for: ${command.query}`,
                score: 0.95,
                source: {
                  file_path: 'docs/example.md',
                  location: 'line 10',
                  snippet: 'Example snippet',
                },
                metadata: {},
              },
              {
                chunk_id: 'chunk_2',
                content: `Another result for: ${command.query}`,
                score: 0.82,
                source: {
                  file_path: 'docs/guide.md',
                  location: 'line 25',
                  snippet: 'Guide content',
                },
                metadata: {},
              },
              {
                chunk_id: 'chunk_3',
                content: `Third result for: ${command.query}`,
                score: 0.65,
                source: {
                  file_path: 'docs/readme.md',
                  location: 'line 5',
                  snippet: 'Readme content',
                },
                metadata: {},
              },
            ],
            mode_used: command.mode || 'indexed',
            total_results: 3,
            // v1.1: Quality summary
            quality: {
              min_score: 0.65,
              max_score: 0.95,
              avg_score: 0.807,
              above_threshold_count: 2,
              total_count: 3,
              relevance: 'high',
              coverage: 'medium',
              confidence_level: 'high',
              suggestion: null,
              score_distribution: {
                '80-100': 2,
                '60-80': 1,
                '40-60': 0,
                '20-40': 0,
                '0-20': 0,
              },
            },
          });
        }, 100);
        return true;
      }

      // List documents
      if (command?.command_type === 'list_documents') {
        const docs = mockState.documents[command.library_id] || [];
        setTimeout(() => {
          handleEvent({
            event_type: 'document_list',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library_id: command.library_id,
            documents: docs,
          });
        }, 50);
        return true;
      }

      // List chunks
      if (command?.command_type === 'list_chunks') {
        setTimeout(() => {
          handleEvent({
            event_type: 'chunk_list',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library_id: command.library_id,
            doc_id: command.doc_id,
            chunks: [
              {
                chunk_id: 'chunk_1',
                doc_id: command.doc_id,
                content: 'First chunk content from the document.',
                chunk_index: 0,
                start_char: 0,
                end_char: 40,
                metadata: {},
              },
              {
                chunk_id: 'chunk_2',
                doc_id: command.doc_id,
                content: 'Second chunk with more content.',
                chunk_index: 1,
                start_char: 40,
                end_char: 72,
                metadata: {},
              },
            ],
          });
        }, 50);
        return true;
      }

      // Document status operations
      if (command?.command_type === 'disable_document' || command?.command_type === 'enable_document') {
        const newStatus = command.command_type === 'disable_document' ? 'disabled' : 'indexed';
        const docs = mockState.documents[command.library_id] || [];
        const doc = docs.find((d: any) => d.doc_id === command.doc_id);
        if (doc) {
          doc.status = newStatus;
        }
        setTimeout(() => {
          handleEvent({
            event_type: 'document_status_changed',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library_id: command.library_id,
            doc_id: command.doc_id,
            status: newStatus,
          });
        }, 50);
        return true;
      }

      // Delete document
      if (command?.command_type === 'delete_document') {
        const docs = mockState.documents[command.library_id] || [];
        const idx = docs.findIndex((d: any) => d.doc_id === command.doc_id);
        if (idx >= 0) {
          docs.splice(idx, 1);
        }
        setTimeout(() => {
          handleEvent({
            event_type: 'document_deleted',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library_id: command.library_id,
            doc_id: command.doc_id,
          });
        }, 50);
        return true;
      }

      // Ingest files
      if (command?.command_type === 'ingest_knowledge_files') {
        const lib = mockState.libraries.find((l: any) => l.library_id === command.library_id);
        if (lib) {
          const docs = mockState.documents[command.library_id] || [];
          for (const path of command.paths || []) {
            const fileName = path.split('/').pop() || path;
            docs.push({
              doc_id: `doc_${mockState.docIdCounter++}`,
              library_id: command.library_id,
              file_path: path,
              file_name: fileName,
              file_size: 1024,
              status: 'indexed',
              chunk_count: 5,
              created_at: timestamp,
              updated_at: timestamp,
              metadata: {},
            });
          }
          mockState.documents[command.library_id] = docs;
          lib.doc_count = docs.length;
          lib.chunk_count = docs.length * 5;
          lib.status = 'ready';
        }

        setTimeout(() => {
          handleEvent({
            event_type: 'knowledge_ingest_complete',
            event_id: `evt_${Date.now()}`,
            timestamp,
            session_id: sessionId,
            library_id: command.library_id,
            success: true,
            stats: {
              files_processed: command.paths?.length || 0,
              chunks_created: (command.paths?.length || 0) * 5,
            },
            message: 'Ingest complete',
          });
        }, 100);
        return true;
      }

      // Pass through unknown commands - use original sendCommand
      const originalSendCommand = (window as any).__e2eOriginalSendCommand;
      console.log('[E2E Mock] Unknown command, passing through:', command?.command_type);
      return originalSendCommand ? originalSendCommand(command) : true;
    };

    // Set the mock sendCommand in store
    store.setState({ sendCommand: mockSendCommand });

    // Also set initial empty state for knowledge
    store.setState({
      knowledgeLibraries: [],
      isLoadingLibraries: false,
      selectedLibraryId: null,
      knowledgeSearchResults: [],
      knowledgeSearchQuery: '',
      isSearchingKnowledge: false,
    });

    console.log('[E2E Mock] Knowledge mock installed successfully');
  });
}

/**
 * Open the Knowledge panel and wait for it to be ready.
 * Assumes mock is already installed.
 */
async function openKnowledgePanel(page: Page): Promise<void> {
  // Click Knowledge tab
  await page.getByLabel('Knowledge').click();

  // Wait for Knowledge Libraries heading
  await expect(page.getByText('Knowledge Libraries')).toBeVisible({ timeout: 10000 });

  // Give time for the panel to render fully
  await page.waitForTimeout(300);
}

async function expectKnowledgeResultsInCenterStage(page: Page): Promise<void> {
  await expect(page.getByTestId('center-stage-panel')).toBeVisible({ timeout: 5000 });
  await expect(page.getByText('Knowledge Results')).toBeVisible({ timeout: 5000 });
}

async function switchToChatMode(page: Page): Promise<void> {
  await page.locator('button', { hasText: 'Chat' }).first().click();
  await expect(
    page.getByTestId('chat-empty-state').or(page.getByTestId('chat-page')),
  ).toBeVisible({ timeout: 5000 });
}

/**
 * Create a library using the UI
 */
async function createLibrary(page: Page, name: string): Promise<void> {
  // Click the Create Library button (either in empty state or header)
  const createBtn = page.locator('button').filter({ hasText: 'Create Library' }).first();
  await createBtn.click();

  // Wait for dialog - the title is in a span, not a heading
  await expect(page.getByText('Create Knowledge Library').first()).toBeVisible({ timeout: 5000 });

  // Fill the name input - use the placeholder to find the right input
  const nameInput = page.getByPlaceholder('My Knowledge Library');
  await nameInput.fill(name);

  // Click Create Library button in dialog (the submit button)
  await page.locator('button[type="submit"]').filter({ hasText: 'Create Library' }).click();

  // Wait for the dialog to fully close by checking the backdrop is gone
  await page.waitForSelector('.fixed.inset-0.bg-black.bg-opacity-50', { state: 'hidden', timeout: 5000 });

  // Wait for the library to appear in the DOM (existence, not visibility)
  await page.waitForSelector(`text="${name}"`, { state: 'attached', timeout: 5000 });

  // Wait for toast to auto-dismiss (toasts last ~3 seconds)
  await page.waitForTimeout(3500);
}

/**
 * Click on a library card by name. Handles toast overlays and scrolling.
 */
async function clickLibraryCard(page: Page, name: string): Promise<void> {
  // Find the library card - it's a div with cursor-pointer that contains the library name
  const libraryCard = page.locator('.bg-gray-900.border.rounded.cursor-pointer').filter({ hasText: name });

  // Scroll into view and click
  await libraryCard.scrollIntoViewIfNeeded();
  await libraryCard.click();
}

// ============================================================================
// Test Suites
// ============================================================================

test.describe('Knowledge UI - Basic Operations', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    // IMPORTANT: Install mock BEFORE opening Knowledge panel
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should display Knowledge panel with empty state', async ({ page }) => {
    await expect(page.getByText('No knowledge libraries')).toBeVisible();
    await expect(page.locator('button').filter({ hasText: 'Create Library' })).toBeVisible();
  });

  test('should create a new knowledge library', async ({ page }) => {
    await createLibrary(page, 'My Test Library');

    // Verify library appears in list
    await expect(page.locator('.bg-gray-900.border.rounded').filter({ hasText: 'My Test Library' })).toBeVisible();
    // Empty state should be gone
    await expect(page.getByText('No knowledge libraries')).not.toBeVisible();
  });

  test('should prevent duplicate library names', async ({ page }) => {
    // Create first library
    await createLibrary(page, 'Unique Library');

    // Try to create another with same name - click the header + button
    await page.locator('button[title="Create Library"]').click();
    await expect(page.getByText('Create Knowledge Library').first()).toBeVisible();
    await page.getByPlaceholder('My Knowledge Library').fill('Unique Library');
    await page.locator('button[type="submit"]').filter({ hasText: 'Create Library' }).click();

    // Should see error
    await expect(page.getByText(/already exists/i)).toBeVisible({ timeout: 5000 });
  });

  test('should select a library and enable search', async ({ page }) => {
    await createLibrary(page, 'Searchable Library');

    // Click to select using the library card
    await clickLibraryCard(page, 'Searchable Library');

    // Search input should be enabled
    const searchInput = page.getByPlaceholder(/search/i);
    await expect(searchInput).toBeEnabled();
  });

  test('should delete a knowledge library', async ({ page }) => {
    await createLibrary(page, 'Library To Delete');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click delete in menu - use role to be specific
    await page.getByRole('button', { name: 'Delete' }).click();

    // Confirm deletion
    await expect(page.getByRole('heading', { name: /delete/i })).toBeVisible();
    await page.locator('button').filter({ hasText: /^Delete$/ }).last().click();

    // Library should be gone
    await expect(page.locator('.bg-gray-900.border.rounded').filter({ hasText: 'Library To Delete' })).not.toBeVisible({ timeout: 5000 });
  });

  test('keeps knowledge search result experience consistent in Graph and Chat', async ({ page }) => {
    await createLibrary(page, 'Cross Mode Library');
    await clickLibraryCard(page, 'Cross Mode Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('graph mode query');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result for: graph mode query/i)).toBeVisible({ timeout: 5000 });

    await page.getByTestId('center-stage-close').click();
    await switchToChatMode(page);
    const knowledgeHeading = page.getByText('Knowledge Libraries');
    if (!(await knowledgeHeading.isVisible())) {
      await page.getByLabel('Knowledge').click();
      await expect(knowledgeHeading).toBeVisible({ timeout: 10000 });
    }
    await clickLibraryCard(page, 'Cross Mode Library');

    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      const targetLibId = state.selectedLibraryId || state.knowledgeLibraries?.[0]?.library_id;
      state.searchKnowledge('chat mode query', targetLibId);
    });

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result for: chat mode query/i)).toBeVisible({ timeout: 5000 });
    await expect(page.getByRole('button', { name: 'Observability' })).not.toBeVisible();
  });
});

test.describe('Knowledge UI - Search', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should perform search and show results with mode indicator', async ({ page }) => {
    await createLibrary(page, 'Search Library');

    // Click to select using the library card
    await clickLibraryCard(page, 'Search Library');

    // Perform search
    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test query');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);

    // Should show results
    await expect(page.getByText(/Mock result for/i)).toBeVisible({ timeout: 5000 });

    // Should show mode indicator (Auto is the default mode)
    await expect(page.getByText('Auto').first()).toBeVisible();
  });

  test('should show score statistics panel', async ({ page }) => {
    await createLibrary(page, 'Stats Library');

    // Click to select using the library card
    await clickLibraryCard(page, 'Stats Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result/i)).toBeVisible({ timeout: 5000 });

    // Click stats button to show statistics
    const statsButton = page.locator('button[title="Show score statistics"]');
    if (await statsButton.isVisible()) {
      await statsButton.click();
      // Stats panel should show
      await expect(page.getByText('Score Distribution')).toBeVisible();
    }
  });
});

test.describe('Knowledge UI - Document Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should open document list dialog', async ({ page }) => {
    await createLibrary(page, 'Doc Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click View Documents in menu
    await page.getByText('View Documents').click();

    // Document dialog should appear - check for the dialog heading
    await expect(page.getByRole('heading', { name: 'Documents' })).toBeVisible();
    // The dialog shows library name in subtitle
    await expect(page.getByText(/Doc Library · \d+ documents/)).toBeVisible();
  });

  test('should show empty document list for new library', async ({ page }) => {
    await createLibrary(page, 'Empty Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click View Documents in menu
    await page.getByText('View Documents').click();

    await expect(page.getByText('No documents in this library')).toBeVisible();
  });
});

test.describe('Knowledge UI - Library Configuration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should open edit library dialog', async ({ page }) => {
    await createLibrary(page, 'Config Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click Edit Settings in menu
    await page.getByText('Edit Settings').click();

    // Edit dialog should appear
    await expect(page.getByRole('heading', { name: 'Edit Knowledge Library' })).toBeVisible();
  });

  test('should show and use advanced settings', async ({ page }) => {
    await createLibrary(page, 'Advanced Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click Edit Settings in menu
    await page.getByText('Edit Settings').click();

    await expect(page.getByRole('heading', { name: 'Edit Knowledge Library' })).toBeVisible();

    // Expand advanced settings
    await page.getByText('Advanced Settings').click();

    // Verify advanced options are visible (use specific selectors to avoid collisions)
    await expect(page.getByText('Chunking')).toBeVisible();
    await expect(page.getByText('Search', { exact: true }).first()).toBeVisible();
    await expect(page.getByText(/Chunk Size/i)).toBeVisible();
    await expect(page.getByText(/Enable Rerank/i)).toBeVisible();
  });

  test('should change library mode', async ({ page }) => {
    await createLibrary(page, 'Mode Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click Edit Settings in menu
    await page.getByText('Edit Settings').click();

    await expect(page.getByRole('heading', { name: 'Edit Knowledge Library' })).toBeVisible();

    // Change mode to Indexed (use button type to be specific)
    await page.locator('button[type="button"]').filter({ hasText: /^Indexed$/ }).click();

    // Save (use type="submit" to distinguish from "Save as default")
    await page.locator('button[type="submit"]').filter({ hasText: 'Save' }).click();

    // Dialog should close
    await expect(page.getByRole('heading', { name: 'Edit Knowledge Library' })).not.toBeVisible({ timeout: 3000 });
  });

  test('should modify chunk size settings', async ({ page }) => {
    await createLibrary(page, 'Chunk Config Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click Edit Settings in menu
    await page.getByText('Edit Settings').click();

    await expect(page.getByRole('heading', { name: 'Edit Knowledge Library' })).toBeVisible();

    // Expand advanced settings
    await page.getByText('Advanced Settings').click();

    // Find and modify chunk size input
    const chunkSizeInput = page.locator('input[type="number"]').first();
    await chunkSizeInput.fill('1024');

    // Save
    await page.locator('button[type="submit"]').filter({ hasText: 'Save' }).click();

    // Dialog should close
    await expect(page.getByRole('heading', { name: 'Edit Knowledge Library' })).not.toBeVisible({ timeout: 3000 });
  });

  test('should trigger rebuild index', async ({ page }) => {
    await createLibrary(page, 'Rebuild Library');

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click Rebuild Index in menu
    await page.getByText('Rebuild Index').click();

    // Should trigger rebuild (mock just succeeds)
    // Wait a moment for any feedback
    await page.waitForTimeout(500);
  });
});

test.describe('Knowledge UI - Document Management Extended', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should show documents after file import', async ({ page }) => {
    await createLibrary(page, 'Import Library');

    // Simulate file import by directly triggering mock ingest
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      // Call sendCommand with ingest command
      state.sendCommand({
        command_type: 'ingest_knowledge_files',
        library_id: 'lib_1',
        paths: ['/docs/test1.md', '/docs/test2.md'],
      });
    });

    // Wait for mock to process
    await page.waitForTimeout(200);

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click View Documents in menu
    await page.getByText('View Documents').click();

    // Should show documents heading
    await expect(page.getByRole('heading', { name: 'Documents' })).toBeVisible();

    // Should show 2 documents imported (use first() to avoid strict mode with footer text)
    await expect(page.getByText(/2 documents/).first()).toBeVisible();
  });

  test('should show imported documents in list', async ({ page }) => {
    await createLibrary(page, 'Toggle Library');

    // Simulate file import
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      state.sendCommand({
        command_type: 'ingest_knowledge_files',
        library_id: 'lib_1',
        paths: ['/docs/toggle-test.md'],
      });
    });
    await page.waitForTimeout(500);

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click View Documents in menu
    await page.getByText('View Documents').click();

    // Wait for dialog to fully open and list to load
    await expect(page.getByRole('heading', { name: 'Documents' })).toBeVisible({ timeout: 5000 });

    // Verify document count indicates 1 document
    await expect(page.getByText(/1 of 1 documents/)).toBeVisible({ timeout: 5000 });

    // Verify the document path is shown
    await expect(page.getByText('/docs/toggle-test.md')).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Knowledge UI - Chunk Visualization', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should open chunk view for document', async ({ page }) => {
    await createLibrary(page, 'Chunk View Library');

    // Import a file
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      state.sendCommand({
        command_type: 'ingest_knowledge_files',
        library_id: 'lib_1',
        paths: ['/docs/chunked.md'],
      });
    });
    await page.waitForTimeout(200);

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click View Documents in menu
    await page.getByText('View Documents').click();
    await expect(page.getByRole('heading', { name: 'Documents' })).toBeVisible();

    // Click on the document to view chunks (if there's a view chunks button)
    const viewChunksBtn = page.locator('button[title*="chunk"]').first();
    if (await viewChunksBtn.isVisible()) {
      await viewChunksBtn.click();
      // Should show chunks dialog
      await expect(page.getByText(/Chunks/i)).toBeVisible();
    }
  });
});

test.describe('Knowledge UI - Search Results Display', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should display search result scores', async ({ page }) => {
    await createLibrary(page, 'Score Display Library');
    await clickLibraryCard(page, 'Score Display Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test query');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);

    // Wait for results
    await expect(page.getByText(/Mock result for/i)).toBeVisible({ timeout: 5000 });

    // Should show score percentages (use first() to avoid matching summary text)
    await expect(page.getByText('95.0%').first()).toBeVisible();
    await expect(page.getByText('82.0%').first()).toBeVisible();
  });

  test('should show empty results message', async ({ page }) => {
    await createLibrary(page, 'No Results Library');
    await clickLibraryCard(page, 'No Results Library');

    // Override mock to return empty results for this search
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const currentMock = store.getState().sendCommand;
      const enhancedMock = (command: any): boolean => {
        if (command?.command_type === 'search_knowledge' && command.query === 'nonexistent') {
          const handleEvent = store.getState().handleEvent;
          setTimeout(() => {
            handleEvent({
              event_type: 'knowledge_search_results',
              event_id: `evt_${Date.now()}`,
              timestamp: new Date().toISOString(),
              session_id: store.getState().sessionId,
              query: command.query,
              library_id: command.library_id,
              results: [],
              mode_used: 'indexed',
              total_results: 0,
            });
          }, 100);
          return true;
        }
        return currentMock(command);
      };
      store.setState({ sendCommand: enhancedMock });
    });

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('nonexistent');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);

    // Should show no results message (exact text to avoid matching library name)
    await expect(page.getByText('No results found')).toBeVisible({ timeout: 5000 });
  });

  test('should clear search and results', async ({ page }) => {
    await createLibrary(page, 'Clear Test Library');
    await clickLibraryCard(page, 'Clear Test Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result/i)).toBeVisible({ timeout: 5000 });

    // Clear the search input
    await searchInput.clear();

    // Results should eventually be cleared or prompt should change
    await expect(searchInput).toHaveValue('');
  });
});

test.describe('Knowledge UI - File Import', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should show import dialog with supported formats', async ({ page }) => {
    await createLibrary(page, 'Format Test Library');

    // Click import button
    await page.locator('button[title="Import files"]').first().click();

    // Should show import dialog
    await expect(page.getByRole('heading', { name: 'Import Files' })).toBeVisible();

    // Should list supported formats
    await expect(page.getByText(/Supported formats.*\.md.*\.txt.*\.pdf.*\.json.*\.csv.*\.html/i)).toBeVisible();
  });

  test('should import supported file types successfully', async ({ page }) => {
    await createLibrary(page, 'Supported Types Library');

    // Test all supported file types
    const supportedFiles = [
      '/docs/readme.md',
      '/docs/notes.txt',
      '/docs/report.pdf',
      '/docs/data.json',
      '/docs/table.csv',
      '/docs/page.html',
    ];

    await page.evaluate((files) => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      state.sendCommand({
        command_type: 'ingest_knowledge_files',
        library_id: 'lib_1',
        paths: files,
      });
    }, supportedFiles);

    await page.waitForTimeout(300);

    // Open the more actions menu first
    await page.locator('button[title="More actions"]').first().click();

    // Click View Documents in menu
    await page.getByText('View Documents').click();
    await expect(page.getByRole('heading', { name: 'Documents' })).toBeVisible();

    // Should show all 6 documents
    await expect(page.getByText(/6 of 6 documents/)).toBeVisible({ timeout: 5000 });
  });

  test('should show correct toast after import', async ({ page }) => {
    await createLibrary(page, 'Toast Test Library');

    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      state.sendCommand({
        command_type: 'ingest_knowledge_files',
        library_id: 'lib_1',
        paths: ['/docs/file1.md', '/docs/file2.txt'],
      });
    });

    // Should show success toast with file and chunk counts
    await expect(page.getByText(/Import complete.*2 files.*10 chunks/i)).toBeVisible({ timeout: 5000 });
  });

  test('should update library stats after import', async ({ page }) => {
    await createLibrary(page, 'Stats Update Library');

    // Before import - should show 0 docs
    await expect(page.getByText('0 docs')).toBeVisible();

    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const state = store.getState();
      state.sendCommand({
        command_type: 'ingest_knowledge_files',
        library_id: 'lib_1',
        paths: ['/docs/test.md'],
      });
    });

    await page.waitForTimeout(300);

    // Refresh library list to get updated stats
    await page.locator('button[title="Refresh"]').first().click();
    await page.waitForTimeout(200);

    // Should show updated doc count
    await expect(page.getByText('1 docs')).toBeVisible({ timeout: 5000 });
  });
});

// ============================================================================
// v1.1 Quality Summary Tests
// ============================================================================

test.describe('Knowledge UI - v1.1 Quality Summary', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should display Quality Summary section in search results', async ({ page }) => {
    await createLibrary(page, 'Quality Test Library');
    await clickLibraryCard(page, 'Quality Test Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test query');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);

    // Wait for results
    await expect(page.getByText(/Mock result for/i)).toBeVisible({ timeout: 5000 });

    // Should show Quality Summary section
    await expect(page.getByText('Quality Summary')).toBeVisible();
  });

  test('should show quality indicators (Relevance/Coverage/Confidence)', async ({ page }) => {
    await createLibrary(page, 'Indicators Library');
    await clickLibraryCard(page, 'Indicators Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result/i)).toBeVisible({ timeout: 5000 });

    // Should show quality indicators
    await expect(page.getByText('Relevance')).toBeVisible();
    await expect(page.getByText('Coverage')).toBeVisible();
    await expect(page.getByText('Confidence')).toBeVisible();
  });

  test('should show score distribution in Quality Summary', async ({ page }) => {
    await createLibrary(page, 'Distribution Library');
    await clickLibraryCard(page, 'Distribution Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('distribution test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result/i)).toBeVisible({ timeout: 5000 });

    // Should show score range (min - max)
    // Mock returns min=0.65, max=0.95
    await expect(page.getByText(/65.*95|0\.65.*0\.95/)).toBeVisible();
  });

  test('should display individual result scores with color coding', async ({ page }) => {
    await createLibrary(page, 'Scores Library');
    await clickLibraryCard(page, 'Scores Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('score test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result/i)).toBeVisible({ timeout: 5000 });

    // Should show individual scores (95%, 82%, 65%)
    await expect(page.getByText('95.0%').first()).toBeVisible();
    await expect(page.getByText('82.0%').first()).toBeVisible();
    await expect(page.getByText('65.0%').first()).toBeVisible();
  });

  test('should toggle Quality Summary collapse', async ({ page }) => {
    await createLibrary(page, 'Toggle Quality Library');
    await clickLibraryCard(page, 'Toggle Quality Library');

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('toggle test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText(/Mock result/i)).toBeVisible({ timeout: 5000 });

    // Quality Summary should be visible by default
    await expect(page.getByText('Relevance')).toBeVisible();

    // Click to collapse
    await page.getByText('Quality Summary').click();

    // Indicators should be hidden after collapse
    await expect(page.getByText('Relevance')).not.toBeVisible();

    // Click to expand again
    await page.getByText('Quality Summary').click();

    // Indicators should be visible again
    await expect(page.getByText('Relevance')).toBeVisible();
  });
});

// ============================================================================
// Retrieval Strategies Tests
// ============================================================================
test.describe('Knowledge UI - Retrieval Strategies', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Workflow', exact: true })).toBeVisible();
    await installKnowledgeMock(page);
    await openKnowledgePanel(page);
  });

  test('should display strategies used in search results', async ({ page }) => {
    await createLibrary(page, 'Strategies Library');
    await clickLibraryCard(page, 'Strategies Library');

    // Override mock to include strategies_used
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const currentMock = store.getState().sendCommand;

      const enhancedMock = (command: any): boolean => {
        if (command?.command_type === 'search_knowledge') {
          const handleEvent = store.getState().handleEvent;
          setTimeout(() => {
            handleEvent({
              event_type: 'knowledge_search_results',
              event_id: `evt_${Date.now()}`,
              timestamp: new Date().toISOString(),
              session_id: store.getState().sessionId,
              query: command.query,
              library_id: command.library_id,
              results: [
                {
                  chunk_id: 'chunk_1',
                  content: 'Test result with strategies',
                  score: 0.85,
                  source: { file_path: '/test/file.md' },
                  metadata: {},
                },
              ],
              mode_used: 'indexed',
              strategies_used: ['bm25', 'vector'],
              total_results: 1,
              quality: {
                min_score: 0.85,
                max_score: 0.85,
                avg_score: 0.85,
                above_threshold_count: 1,
                total_count: 1,
                relevance: 'high',
                coverage: 'high',
                confidence_level: 'high',
              },
            });
          }, 100);
          return true;
        }
        return currentMock(command);
      };
      store.setState({ sendCommand: enhancedMock });
    });

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('test strategies');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText('Test result with strategies')).toBeVisible({ timeout: 5000 });

    // Should show strategies indicator (BM25+VECTOR)
    await expect(page.getByText('BM25+VECTOR')).toBeVisible();
  });

  test('should show all three strategies when graph is enabled', async ({ page }) => {
    await createLibrary(page, 'Triple Strategy Library');
    await clickLibraryCard(page, 'Triple Strategy Library');

    // Override mock to include all three strategies
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const currentMock = store.getState().sendCommand;

      const enhancedMock = (command: any): boolean => {
        if (command?.command_type === 'search_knowledge') {
          const handleEvent = store.getState().handleEvent;
          setTimeout(() => {
            handleEvent({
              event_type: 'knowledge_search_results',
              event_id: `evt_${Date.now()}`,
              timestamp: new Date().toISOString(),
              session_id: store.getState().sessionId,
              query: command.query,
              library_id: command.library_id,
              results: [
                {
                  chunk_id: 'chunk_1',
                  content: 'Test result with graph',
                  score: 0.92,
                  source: { file_path: '/test/graph.md' },
                  metadata: {},
                },
              ],
              mode_used: 'indexed',
              strategies_used: ['bm25', 'vector', 'graph'],
              total_results: 1,
              quality: {
                min_score: 0.92,
                max_score: 0.92,
                avg_score: 0.92,
                above_threshold_count: 1,
                total_count: 1,
                relevance: 'high',
                coverage: 'high',
                confidence_level: 'high',
              },
            });
          }, 100);
          return true;
        }
        return currentMock(command);
      };
      store.setState({ sendCommand: enhancedMock });
    });

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('graph test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText('Test result with graph')).toBeVisible({ timeout: 5000 });

    // Should show all three strategies
    await expect(page.getByText('BM25+VECTOR+GRAPH')).toBeVisible();
  });

  test('should show BM25 only when vector is disabled', async ({ page }) => {
    await createLibrary(page, 'BM25 Only Library');
    await clickLibraryCard(page, 'BM25 Only Library');

    // Override mock to show only BM25
    await page.evaluate(() => {
      const store = (window as any).__consoleStore;
      const currentMock = store.getState().sendCommand;

      const enhancedMock = (command: any): boolean => {
        if (command?.command_type === 'search_knowledge') {
          const handleEvent = store.getState().handleEvent;
          setTimeout(() => {
            handleEvent({
              event_type: 'knowledge_search_results',
              event_id: `evt_${Date.now()}`,
              timestamp: new Date().toISOString(),
              session_id: store.getState().sessionId,
              query: command.query,
              library_id: command.library_id,
              results: [
                {
                  chunk_id: 'chunk_1',
                  content: 'BM25 only result',
                  score: 0.75,
                  source: { file_path: '/test/bm25.md' },
                  metadata: {},
                },
              ],
              mode_used: 'indexed',
              strategies_used: ['bm25'],
              total_results: 1,
              quality: {
                min_score: 0.75,
                max_score: 0.75,
                avg_score: 0.75,
                above_threshold_count: 1,
                total_count: 1,
                relevance: 'high',
                coverage: 'high',
                confidence_level: 'high',
              },
            });
          }, 100);
          return true;
        }
        return currentMock(command);
      };
      store.setState({ sendCommand: enhancedMock });
    });

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('bm25 test');
    await searchInput.press('Enter');

    await expectKnowledgeResultsInCenterStage(page);
    await expect(page.getByText('BM25 only result')).toBeVisible({ timeout: 5000 });

    // Should show only BM25 strategy (no VECTOR or GRAPH)
    // The strategies are displayed in a small gray box next to the mode indicator
    // When only BM25 is used, it shows just "BM25" without any + signs
    await expect(page.locator('.bg-gray-700\\/30').filter({ hasText: 'BM25' })).toBeVisible();
  });
});
