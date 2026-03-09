import React from 'react';
import { Header } from './components/Header';
import { ActivityBar } from './components/ActivityBar';
import { LeftSidebar } from './components/LeftSidebar/index';
import { RightSidebar } from './components/RightSidebar';
import { BottomPanel } from './components/BottomPanel';
import { DAGCanvas } from './components/DAGCanvas';
import { ChatPage } from './components/Chat';
import { ConversationSettingsDrawer } from './components/Chat/ConversationSettingsDrawer';
import { GlobalSettingsPage } from './components/Chat/GlobalSettingsPage';
import { SearchModal } from './components/Chat/SearchModal';
import { BookmarkModal } from './components/Chat/BookmarkModal';
import { ToastContainer } from './components/Toast';
import { RunSettingsDrawer } from './components/RunSettingsDrawer';
import { CenterStage } from './components/CenterStage';
import { useConsoleStore } from './stores/useConsoleStore';
import { useThemeStore } from './stores/useThemeStore';
import { useChatStore } from './stores/useChatStore';
import { ObsFullView } from './components/panels/ObsFullView';
import { KnowledgeResultsPanel } from './components/panels/KnowledgeResultsPanel';
import { useSkillsLogic } from './components/LeftSidebar/useSkillsLogic';
import { SkillConfigDialog } from './components/panels/skill/SkillConfigDialog';
import { DryRunDialog } from './components/panels/skill/DryRunDialog';
import { LoadSkillDialog } from './components/panels/skill/LoadSkillDialog';
import type { SkillConfigValues } from './components/LeftSidebar/useSkillsLogic';

// Module-level session ID: survives Vite HMR but resets on full page reload.
// This is the correct persistence scope — not sessionStorage (persists across
// refreshes) and not a simple const (doesn't survive HMR).
let _currentSessionId: string | null = null;

const DEFAULT_LEFT_WIDTH = 208;
const MIN_LEFT_WIDTH = 180;
const MAX_LEFT_WIDTH = 420;

const DEFAULT_RIGHT_WIDTH = 320;
const MIN_RIGHT_WIDTH = 200;
const MAX_RIGHT_WIDTH = 500;

const DEFAULT_BOTTOM_HEIGHT = 280;
const MIN_BOTTOM_HEIGHT = 120;
const MAX_BOTTOM_HEIGHT = 600;

function App() {
  const { toasts, removeToast, bottomPanelTab, setBottomPanelTab } = useConsoleStore();
  const setRunSettingsOpen = useConsoleStore((state) => state.setRunSettingsOpen);
  const primaryMode = useConsoleStore((s) => s.primaryMode);
  const sidebarTab = useConsoleStore((s) => s.sidebarTab);
  const isSearchingKnowledge = useConsoleStore((s) => s.isSearchingKnowledge);
  const knowledgeSearchQuery = useConsoleStore((s) => s.knowledgeSearchQuery);
  const setPrimaryMode = useConsoleStore((s) => s.setPrimaryMode);
  const setSidebarTab = useConsoleStore((s) => s.setSidebarTab);
  // Ensure theme store is initialized at app boot (applies theme class to <html>).
  // Subscribe to a no-op selector so the App component does NOT re-render
  // when the theme changes — CSS custom properties handle the visual switch.
  useThemeStore(() => null);
  const [leftCollapsed, setLeftCollapsed] = React.useState(false);
  const [leftWidth, setLeftWidth] = React.useState(DEFAULT_LEFT_WIDTH);
  const [rightCollapsed, setRightCollapsed] = React.useState(true);
  const [rightWidth, setRightWidth] = React.useState(DEFAULT_RIGHT_WIDTH);
  const [bottomCollapsed, setBottomCollapsed] = React.useState(false);
  const [showObsFullView, setShowObsFullView] = React.useState(false);
  const [showChatSettings, setShowChatSettings] = React.useState(false);
  const [showGlobalSettings, setShowGlobalSettings] = React.useState(false);
  const [showSearch, setShowSearch] = React.useState(false);
  const [showBookmarks, setShowBookmarks] = React.useState(false);
  const [showSkillConfig, setShowSkillConfig] = React.useState(false);
  const [showDryRun, setShowDryRun] = React.useState(false);
  const [showLoadSkill, setShowLoadSkill] = React.useState(false);
  const [configureKnowledgeLibraryId, setConfigureKnowledgeLibraryId] = React.useState<string | null>(null);
  // Center Stage L state for Panel expand and knowledge search results.
  const [centerStageTab, setCenterStageTab] = React.useState<
    'observability' | 'checkpoints' | 'context' | 'logs' | 'knowledge' | null
  >(null);
  // Shared checkpoint selection state between BottomPanel and CenterStage
  const [sharedCheckedCheckpoints, setSharedCheckedCheckpoints] = React.useState<Array<{ checkpointId: string; executionId: string }>>([]);
  const [bottomHeight, setBottomHeight] = React.useState(DEFAULT_BOTTOM_HEIGHT);
  const bottomHeightRef = React.useRef(bottomHeight);
  React.useEffect(() => { bottomHeightRef.current = bottomHeight; }, [bottomHeight]);
  const isResizingBottomRef = React.useRef(false);
  const startYRef = React.useRef(0);
  const startHeightRef = React.useRef(DEFAULT_BOTTOM_HEIGHT);

  const leftWidthRef = React.useRef(leftWidth);
  React.useEffect(() => {
    leftWidthRef.current = leftWidth;
  }, [leftWidth]);

  const isResizingRef = React.useRef(false);
  const startXRef = React.useRef(0);
  const startWidthRef = React.useRef(DEFAULT_LEFT_WIDTH);

  const rightWidthRef = React.useRef(rightWidth);
  React.useEffect(() => { rightWidthRef.current = rightWidth; }, [rightWidth]);
  const isResizingRightRef = React.useRef(false);
  const startXRightRef = React.useRef(0);
  const startWidthRightRef = React.useRef(DEFAULT_RIGHT_WIDTH);

  React.useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (isResizingRef.current) {
        const delta = e.clientX - startXRef.current;
        const next = Math.min(MAX_LEFT_WIDTH, Math.max(MIN_LEFT_WIDTH, startWidthRef.current + delta));
        setLeftWidth(next);
      }
      if (isResizingRightRef.current) {
        // Dragging left increases width (Secondary Sidebar is on the right edge)
        const delta = startXRightRef.current - e.clientX;
        const next = Math.min(MAX_RIGHT_WIDTH, Math.max(MIN_RIGHT_WIDTH, startWidthRightRef.current + delta));
        setRightWidth(next);
      }
      if (isResizingBottomRef.current) {
        const delta = startYRef.current - e.clientY;
        const next = Math.min(MAX_BOTTOM_HEIGHT, Math.max(MIN_BOTTOM_HEIGHT, startHeightRef.current + delta));
        setBottomHeight(next);
      }
    };
    const onMouseUp = () => {
      if (isResizingRef.current) {
        isResizingRef.current = false;
        document.body.classList.remove('select-none');
        document.body.style.cursor = '';
      }
      if (isResizingRightRef.current) {
        isResizingRightRef.current = false;
        document.body.classList.remove('select-none');
        document.body.style.cursor = '';
      }
      if (isResizingBottomRef.current) {
        isResizingBottomRef.current = false;
        document.body.classList.remove('select-none');
        document.body.style.cursor = '';
      }
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  const handleStartResizeBottom = (e: React.MouseEvent<HTMLDivElement>) => {
    isResizingBottomRef.current = true;
    startYRef.current = e.clientY;
    startHeightRef.current = bottomHeightRef.current;
    document.body.classList.add('select-none');
    document.body.style.cursor = 'row-resize';
  };

  const handleSelectSidebarTab = React.useCallback(
    (tab: import('./stores/useConsoleStore').SidebarTab) => {
      if (!leftCollapsed && tab === sidebarTab) {
        setLeftCollapsed(true);
        return;
      }
      setSidebarTab(tab);
      setLeftCollapsed(false);
    },
    [leftCollapsed, sidebarTab, setSidebarTab],
  );

  const handleSetPrimaryMode = React.useCallback(
    (mode: import('./stores/useConsoleStore').PrimaryMode) => {
      setPrimaryMode(mode);
      // Close Run Settings panel when leaving Graph mode
      if (mode === 'chat') {
        setRunSettingsOpen(false);
      }
    },
    [setPrimaryMode, setRunSettingsOpen],
  );

  const handleToggleLeftCollapsed = () => {
    setLeftCollapsed((prev) => !prev);
  };

  const handleResetLeftWidth = () => {
    setLeftWidth(DEFAULT_LEFT_WIDTH);
  };

  const handleOpenBottomPanel = (
    tab: 'observability' | 'checkpoints' | 'context' | 'logs',
  ) => {
    setBottomPanelTab(tab);
    setBottomCollapsed(false);
  };

  const handleConfigureKnowledge = React.useCallback((libraryId: string) => {
    const store = useConsoleStore.getState();
    store.selectKnowledgeLibrary(libraryId);
    store.setSidebarTab('knowledge');
    setLeftCollapsed(false);
    setConfigureKnowledgeLibraryId(libraryId);
  }, []);

  const showGraphBottomPanel = primaryMode === 'graph';

  const handleStartResizeRight = (e: React.MouseEvent<HTMLDivElement>) => {
    isResizingRightRef.current = true;
    startXRightRef.current = e.clientX;
    startWidthRightRef.current = rightWidthRef.current;
    document.body.classList.add('select-none');
    document.body.style.cursor = 'col-resize';
  };

  const handleResetRightWidth = () => {
    setRightWidth(DEFAULT_RIGHT_WIDTH);
  };

  const handleStartResizeLeft = (e: React.MouseEvent<HTMLDivElement>) => {
    isResizingRef.current = true;
    startXRef.current = e.clientX;
    startWidthRef.current = leftWidthRef.current;
    document.body.classList.add('select-none');
    document.body.style.cursor = 'col-resize';
  };

  // Session ID strategy:
  // - HMR re-mount (dev): reuse the same session (module-level flag survives HMR)
  // - Page refresh (F5): generate a new session (module-level flag is reset)
  // - Tab backgrounding: ReconnectingWebSocket keeps the same session alive
  // - Backend restart: server_boot_id mismatch triggers reload (websocket.ts)
  //
  // We use a module-level variable (not sessionStorage) because sessionStorage
  // persists across page refreshes, which would cause stale session reuse.
  // Module-level variables survive Vite HMR but are reset on full page reload.
  const sessionIdRef = React.useRef(() => {
    if (!_currentSessionId) {
      _currentSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    return _currentSessionId;
  });

  React.useEffect(() => {
    const sid = sessionIdRef.current();
    console.log('[App] Session ID:', sid);
    const { connect: c } = useConsoleStore.getState();
    c(sid);
    // Delay disconnect to survive React StrictMode's mount→unmount→remount cycle.
    // If the component re-mounts within 100ms (StrictMode), the disconnect is cancelled.
    return () => {
      const timer = setTimeout(() => {
        useConsoleStore.getState().disconnect();
      }, 100);
      // Store timer so next mount can cancel it
      (window as any).__houyi_ws_disconnect_timer = timer;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cancel pending disconnect if we re-mounted quickly (StrictMode)
  React.useEffect(() => {
    const pending = (window as any).__houyi_ws_disconnect_timer;
    if (pending) {
      clearTimeout(pending);
      delete (window as any).__houyi_ws_disconnect_timer;
    }
  }, []);

  React.useEffect(() => {
    if (import.meta.env.MODE !== 'production') {
      (window as any).__consoleStore = useConsoleStore;
      (window as any).__chatStore = useChatStore;
    }
  }, []);

  // Keep knowledge search results presentation consistent in both Graph and Chat:
  // open Center Stage L when a new search starts.
  const wasSearchingKnowledgeRef = React.useRef(false);
  React.useEffect(() => {
    if (isSearchingKnowledge && !wasSearchingKnowledgeRef.current && knowledgeSearchQuery.trim()) {
      setCenterStageTab('knowledge');
    }
    wasSearchingKnowledgeRef.current = isSearchingKnowledge;
  }, [isSearchingKnowledge, knowledgeSearchQuery]);

  // Skills logic
  const sendCommand = React.useCallback(
    (cmd: Record<string, unknown>) => useConsoleStore.getState().sendCommand(cmd),
    [],
  );
  const registerSkillEventHandler = React.useCallback(
    (eventType: string, handler: (event: unknown) => void) =>
      useConsoleStore.getState().registerSkillEventHandler(eventType, handler),
    [],
  );
  const sessionId = useConsoleStore((s) => s.sessionId);
  const skillsLogic = useSkillsLogic(sessionId, sendCommand, registerSkillEventHandler);

  // Cmd+K / Ctrl+K shortcut for global search
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowSearch((v) => !v);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-gray-50">
      <Header
        onOpenBottomPanel={handleOpenBottomPanel}
        primaryMode={primaryMode}
        onSetPrimaryMode={handleSetPrimaryMode}
        onOpenGlobalSettings={() => setShowGlobalSettings(true)}
        onOpenSearch={() => setShowSearch(true)}
        onOpenBookmarks={() => setShowBookmarks(true)}
      />

      <div className="flex-1 flex overflow-hidden">
        <ActivityBar
          primaryMode={primaryMode}
          sidebarTab={sidebarTab}
          onSelectTab={handleSelectSidebarTab}
          onOpenSettings={() => {
            // Unified settings entry point: mode-aware
            if (primaryMode === 'graph') {
              setRunSettingsOpen(true);
            } else {
              setShowGlobalSettings(true);
            }
          }}
        />

        {!leftCollapsed && (
          <>
            <div
              className="shrink-0"
              style={{ width: leftWidth }}
            >
              <LeftSidebar
                activeTab={sidebarTab}
                isCollapsed={false}
                onToggleCollapse={handleToggleLeftCollapsed}
                onResetWidth={handleResetLeftWidth}
                onOpenChatSettings={() => setShowChatSettings(true)}
                onOpenGlobalSettings={() => setShowGlobalSettings(true)}
                skills={skillsLogic.skills}
                isLoadingSkills={skillsLogic.isLoadingList}
                selectedSkill={skillsLogic.selectedSkill}
                onSelectSkill={(name) => {
                  skillsLogic.selectSkill(name);
                  if (primaryMode === 'graph') {
                    // Sync to global store so Secondary Sidebar can route
                    useConsoleStore.getState().selectSkill(name);
                    // Expand Secondary Sidebar to show skill detail
                    setRightCollapsed(false);
                  }
                }}
                onRefreshSkills={skillsLogic.refreshSkills}
                onLoadSkill={() => setShowLoadSkill(true)}
                configureKnowledgeLibraryId={configureKnowledgeLibraryId}
                onKnowledgeConfigureHandled={() => setConfigureKnowledgeLibraryId(null)}
              />
            </div>
            <div
              className="w-px shrink-0 cursor-col-resize bg-gray-700 hover:bg-blue-500 transition-colors"
              onMouseDown={handleStartResizeLeft}
              title="Resize sidebar"
            />
          </>
        )}

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 flex flex-col overflow-hidden" style={{ display: primaryMode === 'chat' ? 'flex' : 'none' }}>
            <ChatPage />
          </div>
          <div className="flex-1 flex flex-col overflow-hidden" style={{ display: primaryMode === 'graph' ? 'flex' : 'none' }}>
            <DAGCanvas />
          </div>

          {showGraphBottomPanel && !bottomCollapsed && (
            <div
              className="h-px shrink-0 cursor-row-resize bg-gray-700 hover:bg-blue-500 transition-colors"
              onMouseDown={handleStartResizeBottom}
              title="Resize panel"
            />
          )}
          {showGraphBottomPanel && (
            <BottomPanel
              isCollapsed={bottomCollapsed}
              onToggleCollapse={() => setBottomCollapsed(!bottomCollapsed)}
              activeTab={bottomPanelTab}
              onTabChange={setBottomPanelTab}
              height={bottomHeight}
              onExpandTab={(tab) => setCenterStageTab(tab)}
              checkedCheckpoints={sharedCheckedCheckpoints}
              onCheckedCheckpointsChange={setSharedCheckedCheckpoints}
            />
          )}
        </div>

        {!rightCollapsed && (
          <>
            <div
              className="w-px shrink-0 cursor-col-resize bg-gray-700 hover:bg-blue-500 transition-colors"
              onMouseDown={handleStartResizeRight}
              onDoubleClick={handleResetRightWidth}
              title="Resize properties panel (double-click to reset)"
            />
            <div className="shrink-0" style={{ width: rightWidth }}>
              <RightSidebar
                isCollapsed={false}
                onToggleCollapse={() => setRightCollapsed(true)}
                skillDetail={skillsLogic.skillDetail}
                skillMetrics={skillsLogic.skillMetrics}
                isLoadingSkillDetail={skillsLogic.isLoadingDetail}
                onConfigureSkill={() => setShowSkillConfig(true)}
                onDryRunSkill={() => setShowDryRun(true)}
                onUnloadSkill={skillsLogic.unloadSkill}
                onRemoveSkillFromDisk={skillsLogic.removeSkillFromDisk}
                onOpenChatSettings={() => setShowChatSettings(true)}
                onConfigureKnowledge={handleConfigureKnowledge}
                onRebuildKnowledgeIndex={(libId) => useConsoleStore.getState().rebuildKnowledgeIndex(libId, true)}
              />
            </div>
          </>
        )}
        {rightCollapsed && (
          <RightSidebar
            isCollapsed={true}
            onToggleCollapse={() => setRightCollapsed(false)}
          />
        )}
      </div>

      <ObsFullView isOpen={showObsFullView} onClose={() => setShowObsFullView(false)} />

      {/* Expanded panel content in large overlay */}
      {centerStageTab !== null && (
        <CenterStage
          isOpen={true}
          onClose={() => setCenterStageTab(null)}
          size="L"
          title={
            centerStageTab === 'checkpoints' ? 'Checkpoints' :
            centerStageTab === 'observability' ? 'Observability' :
            centerStageTab === 'logs' ? 'Logs' :
            centerStageTab === 'context' ? 'Context' :
            centerStageTab === 'knowledge' ? 'Knowledge Results' :
            'Panel'
          }
        >
          {centerStageTab === 'knowledge' ? (
            <KnowledgeResultsPanel className="h-full" />
          ) : (
            <div className="h-full">
              <BottomPanel
                isCollapsed={false}
                onToggleCollapse={() => setCenterStageTab(null)}
                activeTab={centerStageTab}
                onTabChange={(tab) => setCenterStageTab(tab)}
                fillHeight
                checkedCheckpoints={sharedCheckedCheckpoints}
                onCheckedCheckpointsChange={setSharedCheckedCheckpoints}
              />
            </div>
          )}
        </CenterStage>
      )}

      <ToastContainer toasts={toasts} onRemove={removeToast} />
      <RunSettingsDrawer />
      <ConversationSettingsDrawer
        isOpen={showChatSettings}
        onClose={() => setShowChatSettings(false)}
        onOpenGlobalSettings={() => setShowGlobalSettings(true)}
      />
      <GlobalSettingsPage
        isOpen={showGlobalSettings}
        onClose={() => setShowGlobalSettings(false)}
      />
      <SearchModal
        isOpen={showSearch}
        onClose={() => setShowSearch(false)}
      />
      <BookmarkModal
        isOpen={showBookmarks}
        onClose={() => setShowBookmarks(false)}
      />
      {skillsLogic.skillDetail && (
        <SkillConfigDialog
          isOpen={showSkillConfig}
          detail={skillsLogic.skillDetail}
          onSave={(config: SkillConfigValues) => {
            const skillName = skillsLogic.selectedSkill;
            if (skillName) {
              skillsLogic.configureSkill(skillName, config);
            }
            setShowSkillConfig(false);
          }}
          onCancel={() => setShowSkillConfig(false)}
        />
      )}

      {/* Dry-run Center Stage M dialog */}
      {skillsLogic.skillDetail && (
        <DryRunDialog
          isOpen={showDryRun}
          detail={skillsLogic.skillDetail}
          dryRunResult={skillsLogic.dryRunResult}
          onExecute={(toolName, input, live, options) => {
            const skillName = skillsLogic.selectedSkill;
            if (skillName) {
              skillsLogic.dryRunSkill(skillName, toolName, input, live, options);
            }
          }}
          onClose={() => {
            setShowDryRun(false);
            skillsLogic.clearDryRunResult();
          }}
          onClearResult={skillsLogic.clearDryRunResult}
        />
      )}

      {/* Load Skill Center Stage M dialog */}
      <LoadSkillDialog
        isOpen={showLoadSkill}
        onLoad={(source, installStrategy) => {
          skillsLogic.loadSkill(source, installStrategy);
        }}
        onClose={() => setShowLoadSkill(false)}
        loadResult={skillsLogic.loadResult}
      />
    </div>
  );
}

export default App;
