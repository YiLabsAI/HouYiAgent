/**
 * Tests for ActivityBar with mode-aware tab rendering.
 *
 * Covers:
 *   - Graph mode: renders Workflow + Knowledge + Skills
 *   - Chat mode: renders Conversations + Knowledge + Skills
 *   - Active tab highlighting
 *   - Tab click callbacks
 *   - Settings button always present
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ActivityBar } from '@/components/ActivityBar';

describe('ActivityBar', () => {
  // ---------------------------------------------------------------------------
  // Graph mode rendering
  // ---------------------------------------------------------------------------

  describe('graph mode', () => {
    it('renders Workflow, Knowledge, Skills tabs', () => {
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="workflow"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Workflow')).toBeInTheDocument();
      expect(screen.getByTitle('Knowledge')).toBeInTheDocument();
      expect(screen.getByTitle('Skills')).toBeInTheDocument();
      // Should NOT render Conversations in graph mode
      expect(screen.queryByTitle('Conversations')).not.toBeInTheDocument();
    });

    it('highlights Workflow tab when active', () => {
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="workflow"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      const workflowBtn = screen.getByTitle('Workflow');
      expect(workflowBtn.className).toContain('bg-gray-700');
      const knowledgeBtn = screen.getByTitle('Knowledge');
      expect(knowledgeBtn.className).not.toContain('bg-gray-700');
    });

    it('highlights Knowledge tab when active', () => {
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="knowledge"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Knowledge').className).toContain('bg-gray-700');
      expect(screen.getByTitle('Workflow').className).not.toContain('bg-gray-700');
    });

    it('highlights Skills tab when active', () => {
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="skills"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Skills').className).toContain('bg-gray-700');
    });
  });

  // ---------------------------------------------------------------------------
  // Chat mode rendering
  // ---------------------------------------------------------------------------

  describe('chat mode', () => {
    it('renders Conversations, Knowledge, Skills tabs', () => {
      render(
        <ActivityBar
          primaryMode="chat"
          sidebarTab="conversations"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Conversations')).toBeInTheDocument();
      expect(screen.getByTitle('Knowledge')).toBeInTheDocument();
      expect(screen.getByTitle('Skills')).toBeInTheDocument();
      // Should NOT render Workflow in chat mode
      expect(screen.queryByTitle('Workflow')).not.toBeInTheDocument();
    });

    it('highlights Conversations tab when active', () => {
      render(
        <ActivityBar
          primaryMode="chat"
          sidebarTab="conversations"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Conversations').className).toContain('bg-gray-700');
    });

    it('highlights Knowledge tab in chat mode when active', () => {
      render(
        <ActivityBar
          primaryMode="chat"
          sidebarTab="knowledge"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Knowledge').className).toContain('bg-gray-700');
      expect(screen.getByTitle('Conversations').className).not.toContain('bg-gray-700');
    });
  });

  // ---------------------------------------------------------------------------
  // Tab click callbacks
  // ---------------------------------------------------------------------------

  describe('tab click callbacks', () => {
    it('calls onSelectTab("workflow") in graph mode', () => {
      const onSelectTab = vi.fn();
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="knowledge"
          onSelectTab={onSelectTab}
          onOpenSettings={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByTitle('Workflow'));
      expect(onSelectTab).toHaveBeenCalledWith('workflow');
    });

    it('calls onSelectTab("conversations") in chat mode', () => {
      const onSelectTab = vi.fn();
      render(
        <ActivityBar
          primaryMode="chat"
          sidebarTab="knowledge"
          onSelectTab={onSelectTab}
          onOpenSettings={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByTitle('Conversations'));
      expect(onSelectTab).toHaveBeenCalledWith('conversations');
    });

    it('calls onSelectTab("knowledge") from either mode', () => {
      const onSelectTab = vi.fn();
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="workflow"
          onSelectTab={onSelectTab}
          onOpenSettings={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByTitle('Knowledge'));
      expect(onSelectTab).toHaveBeenCalledWith('knowledge');
    });

    it('calls onSelectTab("skills") from either mode', () => {
      const onSelectTab = vi.fn();
      render(
        <ActivityBar
          primaryMode="chat"
          sidebarTab="conversations"
          onSelectTab={onSelectTab}
          onOpenSettings={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByTitle('Skills'));
      expect(onSelectTab).toHaveBeenCalledWith('skills');
    });
  });

  // ---------------------------------------------------------------------------
  // Settings button
  // ---------------------------------------------------------------------------

  describe('settings button', () => {
    it('renders Settings button in graph mode', () => {
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="workflow"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Settings')).toBeInTheDocument();
    });

    it('renders Settings button in chat mode', () => {
      render(
        <ActivityBar
          primaryMode="chat"
          sidebarTab="conversations"
          onSelectTab={vi.fn()}
          onOpenSettings={vi.fn()}
        />,
      );
      expect(screen.getByTitle('Settings')).toBeInTheDocument();
    });

    it('calls onOpenSettings when clicked', () => {
      const onOpenSettings = vi.fn();
      render(
        <ActivityBar
          primaryMode="graph"
          sidebarTab="workflow"
          onSelectTab={vi.fn()}
          onOpenSettings={onOpenSettings}
        />,
      );
      fireEvent.click(screen.getByTitle('Settings'));
      expect(onOpenSettings).toHaveBeenCalledTimes(1);
    });
  });
});
