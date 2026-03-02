import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

const storeState: Record<string, any> = {
  toasts: [],
  removeToast: vi.fn(),
  bottomPanelTab: 'observability',
  setBottomPanelTab: vi.fn(),
  setRunSettingsOpen: vi.fn(),
  primaryMode: 'graph',
  sidebarTab: 'workflow',
  isSearchingKnowledge: false,
  knowledgeSearchQuery: '',
  setPrimaryMode: vi.fn(),
  setSidebarTab: vi.fn(),
  selectKnowledgeLibrary: vi.fn(),
  connect: vi.fn(),
  disconnect: vi.fn(),
  sendCommand: vi.fn(),
  registerSkillEventHandler: vi.fn(() => vi.fn()),
  sessionId: 'session-test',
};

vi.mock('@/stores/useConsoleStore', () => {
  const useConsoleStore = vi.fn((selector?: (s: typeof storeState) => unknown) =>
    selector ? selector(storeState as any) : (storeState as any),
  );
  (useConsoleStore as any).getState = () => storeState;
  return { useConsoleStore };
});

vi.mock('@/stores/useThemeStore', () => ({
  useThemeStore: vi.fn(() => null),
}));

vi.mock('@/components/LeftSidebar/useSkillsLogic', () => ({
  useSkillsLogic: vi.fn(() => ({
    skills: [],
    isLoadingList: false,
    selectedSkill: null,
    selectSkill: vi.fn(),
    refreshSkills: vi.fn(),
    skillDetail: null,
    skillMetrics: null,
    isLoadingDetail: false,
    unloadSkill: vi.fn(),
    removeSkillFromDisk: vi.fn(),
    configureSkill: vi.fn(),
    dryRunSkill: vi.fn(),
    clearDryRunResult: vi.fn(),
    loadSkill: vi.fn(),
    loadResult: null,
    dryRunResult: null,
  })),
}));

vi.mock('@/components/Header', () => ({
  Header: () => <div data-testid="header" />,
}));

vi.mock('@/components/ActivityBar', () => ({
  ActivityBar: () => <div data-testid="activity-bar" />,
}));

vi.mock('@/components/LeftSidebar/index', () => ({
  LeftSidebar: () => <div data-testid="left-sidebar" />,
}));

vi.mock('@/components/RightSidebar', () => ({
  RightSidebar: ({ isCollapsed }: { isCollapsed: boolean }) => (
    <div data-testid={isCollapsed ? 'right-sidebar-collapsed' : 'right-sidebar-open'} />
  ),
}));

vi.mock('@/components/BottomPanel', () => ({
  BottomPanel: ({ activeTab }: { activeTab?: string }) => (
    <div data-testid="bottom-panel">bottom:{activeTab}</div>
  ),
}));

vi.mock('@/components/CenterStage', () => ({
  CenterStage: ({ isOpen, title, children }: { isOpen: boolean; title: string; children: React.ReactNode }) =>
    isOpen ? (
      <div data-testid="center-stage">
        <div data-testid="center-stage-title">{title}</div>
        {children}
      </div>
    ) : null,
}));

vi.mock('@/components/panels/KnowledgeResultsPanel', () => ({
  KnowledgeResultsPanel: () => <div data-testid="knowledge-results-panel" />,
}));

vi.mock('@/components/DAGCanvas', () => ({ DAGCanvas: () => <div data-testid="dag-canvas" /> }));
vi.mock('@/components/Chat', () => ({ ChatPage: () => <div data-testid="chat-page" /> }));
vi.mock('@/components/Chat/ConversationSettingsDrawer', () => ({ ConversationSettingsDrawer: () => null }));
vi.mock('@/components/Chat/GlobalSettingsPage', () => ({ GlobalSettingsPage: () => null }));
vi.mock('@/components/Chat/SearchModal', () => ({ SearchModal: () => null }));
vi.mock('@/components/Chat/BookmarkModal', () => ({ BookmarkModal: () => null }));
vi.mock('@/components/Toast', () => ({ ToastContainer: () => null }));
vi.mock('@/components/RunSettingsDrawer', () => ({ RunSettingsDrawer: () => null }));
vi.mock('@/components/panels/ObsFullView', () => ({ ObsFullView: () => null }));
vi.mock('@/components/panels/skill/SkillConfigDialog', () => ({ SkillConfigDialog: () => null }));
vi.mock('@/components/panels/skill/DryRunDialog', () => ({ DryRunDialog: () => null }));
vi.mock('@/components/panels/skill/LoadSkillDialog', () => ({ LoadSkillDialog: () => null }));

import App from '@/App';

describe('App layout routing', () => {
  beforeEach(() => {
    storeState.primaryMode = 'graph';
    storeState.sidebarTab = 'workflow';
    storeState.bottomPanelTab = 'observability';
    storeState.isSearchingKnowledge = false;
    storeState.knowledgeSearchQuery = '';
  });

  it('shows BottomPanel in graph mode and hides it in chat mode', () => {
    const { rerender } = render(<App />);
    expect(screen.getByTestId('bottom-panel')).toBeInTheDocument();

    storeState.primaryMode = 'chat';
    rerender(<App />);
    expect(screen.queryByTestId('bottom-panel')).not.toBeInTheDocument();
  });

  it('opens Knowledge Results CenterStage when a knowledge search starts in graph mode', () => {
    const { rerender } = render(<App />);
    expect(screen.queryByTestId('center-stage')).not.toBeInTheDocument();

    storeState.isSearchingKnowledge = true;
    storeState.knowledgeSearchQuery = 'rag query';
    rerender(<App />);

    expect(screen.getByTestId('center-stage')).toBeInTheDocument();
    expect(screen.getByTestId('center-stage-title').textContent).toBe('Knowledge Results');
    expect(screen.getByTestId('knowledge-results-panel')).toBeInTheDocument();
  });

  it('opens Knowledge Results CenterStage when a knowledge search starts in chat mode', () => {
    storeState.primaryMode = 'chat';
    const { rerender } = render(<App />);
    expect(screen.queryByTestId('bottom-panel')).not.toBeInTheDocument();

    storeState.isSearchingKnowledge = true;
    storeState.knowledgeSearchQuery = 'chat kb query';
    rerender(<App />);

    expect(screen.getByTestId('center-stage')).toBeInTheDocument();
    expect(screen.getByTestId('knowledge-results-panel')).toBeInTheDocument();
  });
});
