import React from 'react';
import { Header } from './components/Header';
import { ActivityBar } from './components/ActivityBar';
import { LeftSidebar } from './components/LeftSidebar/index';
import { RightSidebar } from './components/RightSidebar';
import { BottomPanel } from './components/BottomPanel';
import { DAGCanvas } from './components/DAGCanvas';
import { ToastContainer } from './components/Toast';
import { RunSettingsDrawer } from './components/RunSettingsDrawer';
import { useConsoleStore } from './stores/useConsoleStore';
import { ObsFullView } from './components/panels/ObsFullView';

// Module-level session ID: survives Vite HMR but resets on full page reload.
// This is the correct persistence scope — not sessionStorage (persists across
// refreshes) and not a simple const (doesn't survive HMR).
let _currentSessionId: string | null = null;

const DEFAULT_LEFT_WIDTH = 208;
const MIN_LEFT_WIDTH = 180;
const MAX_LEFT_WIDTH = 420;

const DEFAULT_BOTTOM_HEIGHT = 280;
const MIN_BOTTOM_HEIGHT = 120;
const MAX_BOTTOM_HEIGHT = 600;

function App() {
  const { toasts, removeToast, bottomPanelTab, setBottomPanelTab } = useConsoleStore();
  const setRunSettingsOpen = useConsoleStore((state) => state.setRunSettingsOpen);
  const [leftCollapsed, setLeftCollapsed] = React.useState(false);
  const [leftWidth, setLeftWidth] = React.useState(DEFAULT_LEFT_WIDTH);
  const [activeLeftTab, setActiveLeftTab] = React.useState<
    'workflow' | 'chat' | 'knowledge' | 'skills'
  >('workflow');
  const [rightCollapsed, setRightCollapsed] = React.useState(false);
  const [bottomCollapsed, setBottomCollapsed] = React.useState(false);
  const [showObsFullView, setShowObsFullView] = React.useState(false);
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

  React.useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (isResizingRef.current) {
        const delta = e.clientX - startXRef.current;
        const next = Math.min(MAX_LEFT_WIDTH, Math.max(MIN_LEFT_WIDTH, startWidthRef.current + delta));
        setLeftWidth(next);
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

  const handleSelectLeftTab = (
    tab: 'workflow' | 'chat' | 'knowledge' | 'skills',
  ) => {
    if (!leftCollapsed && tab === activeLeftTab) {
      setLeftCollapsed(true);
      return;
    }
    setActiveLeftTab(tab);
    setLeftCollapsed(false);
  };

  const handleToggleLeftCollapsed = () => {
    setLeftCollapsed((prev) => !prev);
  };

  const handleResetLeftWidth = () => {
    setLeftWidth(DEFAULT_LEFT_WIDTH);
  };

  const handleOpenBottomPanel = (
    tab: 'observability' | 'checkpoints' | 'context' | 'logs' | 'compare',
  ) => {
    // NOTE(core): Centralized entry for opening BottomPanel tabs (e.g. run-toolbar shortcuts in Header).
    setBottomPanelTab(tab);
    setBottomCollapsed(false);
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
    const { connect: c, disconnect: d } = useConsoleStore.getState();
    c(sid);
    return () => d();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    if (import.meta.env.MODE !== 'production') {
      (window as any).__consoleStore = useConsoleStore;
    }
  }, []);

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      <Header
        onOpenBottomPanel={handleOpenBottomPanel}
        onSelectLeftTab={handleSelectLeftTab}
        activeLeftTab={activeLeftTab}
        onOpenObs={() => setShowObsFullView(true)}
      />

      <div className="flex-1 flex overflow-hidden">
        <ActivityBar
          activeTab={activeLeftTab}
          onSelectTab={handleSelectLeftTab}
          onOpenSettings={() => setRunSettingsOpen(true)}
        />

        {!leftCollapsed && (
          <>
            <div
              className="shrink-0"
              style={{ width: leftWidth }}
            >
              <LeftSidebar
                activeTab={activeLeftTab as any}
                isCollapsed={false}
                onToggleCollapse={handleToggleLeftCollapsed}
                onResetWidth={handleResetLeftWidth}
              />
            </div>
            <div
              className="w-1 shrink-0 cursor-col-resize bg-gray-900 hover:bg-gray-700"
              onMouseDown={handleStartResizeLeft}
              title="Resize sidebar"
            />
          </>
        )}

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <DAGCanvas />
          </div>

          {!bottomCollapsed && (
            <div
              className="h-1 shrink-0 cursor-row-resize bg-gray-900 hover:bg-gray-700"
              onMouseDown={handleStartResizeBottom}
              title="Resize panel"
            />
          )}
          <BottomPanel
            isCollapsed={bottomCollapsed}
            onToggleCollapse={() => setBottomCollapsed(!bottomCollapsed)}
            activeTab={bottomPanelTab}
            onTabChange={setBottomPanelTab}
            height={bottomHeight}
          />
        </div>

        <RightSidebar
          isCollapsed={rightCollapsed}
          onToggleCollapse={() => setRightCollapsed(!rightCollapsed)}
        />
      </div>

      {showObsFullView && (
        <ObsFullView onClose={() => setShowObsFullView(false)} />
      )}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
      <RunSettingsDrawer />
    </div>
  );
}

export default App;
