/**
 * Tests for primaryMode / sidebarTab state decoupling in useConsoleStore.
 *
 * Covers:
 *   - Default values
 *   - setPrimaryMode with automatic sidebarTab remapping
 *   - setSidebarTab with constraint enforcement
 *   - Edge cases: rapid switching, no-op, invalid combinations
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

// We need a fresh store for each test to avoid shared state leakage.
const loadStoreFresh = async () => {
  vi.resetModules();
  return await import('@/stores/useConsoleStore');
};

describe('useConsoleStore — primaryMode & sidebarTab', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  // ---------------------------------------------------------------------------
  // Defaults
  // ---------------------------------------------------------------------------

  describe('default values', () => {
    it('starts with primaryMode = graph and sidebarTab = workflow', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      const state = useConsoleStore.getState();
      expect(state.primaryMode).toBe('graph');
      expect(state.sidebarTab).toBe('workflow');
    });
  });

  // ---------------------------------------------------------------------------
  // setPrimaryMode
  // ---------------------------------------------------------------------------

  describe('setPrimaryMode', () => {
    it('graph → chat remaps workflow to conversations', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      const state = useConsoleStore.getState();
      expect(state.primaryMode).toBe('chat');
      expect(state.sidebarTab).toBe('conversations');
    });

    it('chat → graph remaps conversations to workflow', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      // Set to chat first
      useConsoleStore.getState().setPrimaryMode('chat');
      expect(useConsoleStore.getState().sidebarTab).toBe('conversations');
      // Back to graph
      useConsoleStore.getState().setPrimaryMode('graph');
      const state = useConsoleStore.getState();
      expect(state.primaryMode).toBe('graph');
      expect(state.sidebarTab).toBe('workflow');
    });

    it('graph → chat preserves knowledge tab', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('knowledge');
      useConsoleStore.getState().setPrimaryMode('chat');
      expect(useConsoleStore.getState().sidebarTab).toBe('knowledge');
    });

    it('graph → chat preserves skills tab', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('skills');
      useConsoleStore.getState().setPrimaryMode('chat');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
    });

    it('chat → graph preserves knowledge tab', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setSidebarTab('knowledge');
      useConsoleStore.getState().setPrimaryMode('graph');
      expect(useConsoleStore.getState().sidebarTab).toBe('knowledge');
    });

    it('chat → graph preserves skills tab', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setSidebarTab('skills');
      useConsoleStore.getState().setPrimaryMode('graph');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
    });

    it('no-op when setting same mode (graph → graph)', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('knowledge');
      useConsoleStore.getState().setPrimaryMode('graph');
      // Should not change sidebarTab
      expect(useConsoleStore.getState().sidebarTab).toBe('knowledge');
    });

    it('no-op when setting same mode (chat → chat)', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setSidebarTab('skills');
      useConsoleStore.getState().setPrimaryMode('chat');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
    });
  });

  // ---------------------------------------------------------------------------
  // setSidebarTab
  // ---------------------------------------------------------------------------

  describe('setSidebarTab', () => {
    it('sets workflow tab in graph mode', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('knowledge');
      useConsoleStore.getState().setSidebarTab('workflow');
      expect(useConsoleStore.getState().sidebarTab).toBe('workflow');
    });

    it('sets knowledge tab in graph mode', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('knowledge');
      expect(useConsoleStore.getState().sidebarTab).toBe('knowledge');
    });

    it('sets skills tab in graph mode', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('skills');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
    });

    it('rejects conversations tab in graph mode (constraint guard)', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('conversations');
      // Should silently reject — stay on workflow
      expect(useConsoleStore.getState().sidebarTab).toBe('workflow');
    });

    it('sets conversations tab in chat mode', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setSidebarTab('knowledge');
      useConsoleStore.getState().setSidebarTab('conversations');
      expect(useConsoleStore.getState().sidebarTab).toBe('conversations');
    });

    it('rejects workflow tab in chat mode (constraint guard)', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      // Currently on conversations
      useConsoleStore.getState().setSidebarTab('workflow');
      // Should silently reject — stay on conversations
      expect(useConsoleStore.getState().sidebarTab).toBe('conversations');
    });

    it('sets knowledge tab in chat mode', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setSidebarTab('knowledge');
      expect(useConsoleStore.getState().sidebarTab).toBe('knowledge');
    });

    it('sets skills tab in chat mode', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setSidebarTab('skills');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
    });
  });

  // ---------------------------------------------------------------------------
  // Rapid switching / edge cases
  // ---------------------------------------------------------------------------

  describe('rapid switching edge cases', () => {
    it('rapid graph→chat→graph→chat ends in correct state', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setPrimaryMode('chat');
      useConsoleStore.getState().setPrimaryMode('graph');
      useConsoleStore.getState().setPrimaryMode('chat');
      expect(useConsoleStore.getState().primaryMode).toBe('chat');
      expect(useConsoleStore.getState().sidebarTab).toBe('conversations');
    });

    it('rapid tab switching respects constraints', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      useConsoleStore.getState().setSidebarTab('skills');
      useConsoleStore.getState().setSidebarTab('conversations'); // rejected
      useConsoleStore.getState().setSidebarTab('knowledge');
      useConsoleStore.getState().setSidebarTab('workflow');
      expect(useConsoleStore.getState().sidebarTab).toBe('workflow');
    });

    it('mode switch + tab switch interleave preserves consistency', async () => {
      const { useConsoleStore } = await loadStoreFresh();
      // graph + skills
      useConsoleStore.getState().setSidebarTab('skills');
      // switch to chat → skills preserved
      useConsoleStore.getState().setPrimaryMode('chat');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
      // try to set workflow (invalid in chat) → rejected
      useConsoleStore.getState().setSidebarTab('workflow');
      expect(useConsoleStore.getState().sidebarTab).toBe('skills');
      // set conversations (valid in chat)
      useConsoleStore.getState().setSidebarTab('conversations');
      // switch to graph → conversations remapped to workflow
      useConsoleStore.getState().setPrimaryMode('graph');
      expect(useConsoleStore.getState().sidebarTab).toBe('workflow');
    });
  });
});
