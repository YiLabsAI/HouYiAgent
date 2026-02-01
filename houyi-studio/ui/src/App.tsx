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

const DEFAULT_LEFT_WIDTH = 208;
const MIN_LEFT_WIDTH = 180;
const MAX_LEFT_WIDTH = 420;

function App() {
  const { connect, disconnect, toasts, removeToast } = useConsoleStore();
  const setRunSettingsOpen = useConsoleStore((state) => state.setRunSettingsOpen);
  const [leftCollapsed, setLeftCollapsed] = React.useState(false);
  const [leftWidth, setLeftWidth] = React.useState(DEFAULT_LEFT_WIDTH);
  const [activeLeftTab, setActiveLeftTab] = React.useState<
    'workflow' | 'chat' | 'knowledge' | 'skills'
  >('workflow');
  const [rightCollapsed, setRightCollapsed] = React.useState(false);
  const [bottomCollapsed, setBottomCollapsed] = React.useState(false);
  const [bottomActiveTab, setBottomActiveTab] = React.useState<
    'timeline' | 'checkpoints' | 'context' | 'logs' | 'verification' | 'compare'
  >('timeline');

  const leftWidthRef = React.useRef(leftWidth);
  React.useEffect(() => {
    leftWidthRef.current = leftWidth;
  }, [leftWidth]);

  const isResizingRef = React.useRef(false);
  const startXRef = React.useRef(0);
  const startWidthRef = React.useRef(DEFAULT_LEFT_WIDTH);

  React.useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isResizingRef.current) return;
      const delta = e.clientX - startXRef.current;
      const next = Math.min(MAX_LEFT_WIDTH, Math.max(MIN_LEFT_WIDTH, startWidthRef.current + delta));
      setLeftWidth(next);
    };
    const onMouseUp = () => {
      if (!isResizingRef.current) return;
      isResizingRef.current = false;
      document.body.classList.remove('select-none');
      document.body.style.cursor = '';
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

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
    tab: 'timeline' | 'checkpoints' | 'context' | 'logs' | 'verification' | 'compare',
  ) => {
    // NOTE(core): Centralized entry for opening BottomPanel tabs (e.g. run-toolbar shortcuts in Header).
    setBottomActiveTab(tab);
    setBottomCollapsed(false);
  };

  const handleStartResizeLeft = (e: React.MouseEvent<HTMLDivElement>) => {
    isResizingRef.current = true;
    startXRef.current = e.clientX;
    startWidthRef.current = leftWidthRef.current;
    document.body.classList.add('select-none');
    document.body.style.cursor = 'col-resize';
  };

  React.useEffect(() => {
    // Generate unique session ID for each page load (independent workspaces)
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    console.log('[App] Created new session ID:', sessionId);
    connect(sessionId);
    return () => disconnect();
  }, [connect, disconnect]);

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

          <BottomPanel
            isCollapsed={bottomCollapsed}
            onToggleCollapse={() => setBottomCollapsed(!bottomCollapsed)}
            activeTab={bottomActiveTab}
            onTabChange={setBottomActiveTab}
          />
        </div>

        <RightSidebar
          isCollapsed={rightCollapsed}
          onToggleCollapse={() => setRightCollapsed(!rightCollapsed)}
        />
      </div>

      <ToastContainer toasts={toasts} onRemove={removeToast} />
      <RunSettingsDrawer />
    </div>
  );
}

export default App;
