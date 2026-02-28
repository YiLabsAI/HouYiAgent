/**
 * Tests for Secondary Sidebar content routing state:
 *   - selectedSkillId / selectSkill
 *   - getSecondaryContentMode derivation
 */
import { describe, expect, it, beforeEach } from 'vitest';
import { useConsoleStore } from '@/stores/useConsoleStore';

describe('Secondary Sidebar routing state', () => {
  beforeEach(() => {
    // Reset store to defaults
    useConsoleStore.setState({
      primaryMode: 'graph',
      sidebarTab: 'workflow',
      selectedNodeId: null,
      selectedSkillId: null,
      selectedLibraryId: null,
    });
  });

  // ─── selectedSkillId ───────────────────────────────────────────

  describe('selectSkill', () => {
    it('should default to null', () => {
      expect(useConsoleStore.getState().selectedSkillId).toBeNull();
    });

    it('should set selectedSkillId', () => {
      useConsoleStore.getState().selectSkill('calculator');
      expect(useConsoleStore.getState().selectedSkillId).toBe('calculator');
    });

    it('should clear selectedSkillId when set to null', () => {
      useConsoleStore.getState().selectSkill('calculator');
      useConsoleStore.getState().selectSkill(null);
      expect(useConsoleStore.getState().selectedSkillId).toBeNull();
    });
  });

  // ─── getSecondaryContentMode ───────────────────────────────────

  describe('getSecondaryContentMode', () => {
    it('should return "empty" when nothing is selected', () => {
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('empty');
    });

    // Priority 1: node in graph mode
    it('should return "node" when a node is selected in graph mode', () => {
      useConsoleStore.setState({ selectedNodeId: 'llm_1' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('node');
    });

    it('should NOT return "node" when a node is selected in chat mode', () => {
      useConsoleStore.setState({ primaryMode: 'chat', sidebarTab: 'conversations', selectedNodeId: 'llm_1' });
      // Chat mode doesn't show node properties (no DAG)
      expect(useConsoleStore.getState().getSecondaryContentMode()).not.toBe('node');
    });

    // Priority 2: skill selected
    it('should return "skill" when skill is selected and sidebar is skills tab', () => {
      useConsoleStore.setState({ sidebarTab: 'skills', selectedSkillId: 'web-search' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('skill');
    });

    it('should return "node" over "skill" when both node and skill are selected (graph)', () => {
      useConsoleStore.setState({ primaryMode: 'graph', sidebarTab: 'workflow', selectedNodeId: 'llm_1', selectedSkillId: 'web-search' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('node');
    });

    // Priority 3: knowledge library selected
    it('should return "knowledge" when library is selected and sidebarTab is knowledge', () => {
      useConsoleStore.setState({ selectedLibraryId: 'kb-1', sidebarTab: 'knowledge' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('knowledge');
    });

    it('should NOT return "knowledge" when library is selected but sidebarTab is NOT knowledge', () => {
      useConsoleStore.setState({ selectedLibraryId: 'kb-1', sidebarTab: 'workflow' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).not.toBe('knowledge');
    });

    // Priority 4: conversation settings in chat mode
    it('should return "conversation" in chat mode with conversations tab', () => {
      useConsoleStore.setState({ primaryMode: 'chat', sidebarTab: 'conversations' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('conversation');
    });

    it('should NOT return "conversation" in graph mode', () => {
      useConsoleStore.setState({ primaryMode: 'graph', sidebarTab: 'workflow' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).not.toBe('conversation');
    });

    it('should NOT return "conversation" in chat mode with skills tab', () => {
      useConsoleStore.setState({ primaryMode: 'chat', sidebarTab: 'skills' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('empty');
    });

    // Complex interactions
    it('should return "node" even when conversation tab is active (graph priority)', () => {
      useConsoleStore.setState({ primaryMode: 'graph', sidebarTab: 'workflow', selectedNodeId: 'tool_1' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('node');
    });

    it('should return "empty" in chat knowledge tab with no library', () => {
      useConsoleStore.setState({ primaryMode: 'chat', sidebarTab: 'knowledge' });
      expect(useConsoleStore.getState().getSecondaryContentMode()).toBe('empty');
    });
  });
});
