import React from 'react';
import { useLeftSidebarLogic } from './useLeftSidebarLogic';
import { NodePalette } from './NodePalette';
import { WorkflowManagement } from './WorkflowManagement';
import { ExecutionControls } from './ExecutionControls';
import { SaveWorkflowDialog } from './SaveWorkflowDialog';
import { LoadWorkflowDialog } from './LoadWorkflowDialog';
import { KnowledgePanel } from './KnowledgePanel';
import { KnowledgeConfigDialog } from './KnowledgeConfigDialog';
import { KnowledgeSearch } from './KnowledgeSearch';
import { ConversationRail } from '../Chat/ConversationRail';
import { useChatStore } from '@/stores/useChatStore';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface LeftSidebarProps {
  activeTab: 'workflow' | 'chat' | 'knowledge' | 'skills';
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  onResetWidth: () => void;
  onOpenChatSettings?: (conversationId: string) => void;
  onOpenGlobalSettings?: () => void;
}

export const LeftSidebar: React.FC<LeftSidebarProps> = ({
  activeTab,
  isCollapsed,
  onToggleCollapse,
  onResetWidth: _onResetWidth,
  onOpenChatSettings,
  // onOpenGlobalSettings is available via props but currently only used in Header
}) => {
  const logic = useLeftSidebarLogic();
  const [isKnowledgeDialogOpen, setIsKnowledgeDialogOpen] = React.useState(false);
  void _onResetWidth; // available via double-click on drag handle

  const sidebarTitle = React.useMemo(() => {
    switch (activeTab) {
      case 'workflow':
        return 'Workflow';
      case 'chat':
        return 'Chat';
      case 'knowledge':
        return 'Knowledge';
      case 'skills':
        return 'Skills';
      default:
        return 'Workflow';
    }
  }, [activeTab]);

  const viewExecution = logic.getViewExecution ? logic.getViewExecution() : logic.currentExecution;

  if (isCollapsed) {
    return (
      <div className="bg-gray-800 border-r border-gray-700 flex flex-col transition-all duration-300 ease-in-out w-8">
        <div className="flex items-center justify-center h-full">
          <button
            onClick={onToggleCollapse}
            className="p-1 hover:bg-gray-700 rounded text-gray-400"
            title="Expand sidebar"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    );
  }

  const chatConversations = useChatStore((s) => s.conversations);
  const chatActiveId = useChatStore((s) => s.activeConversationId);
  const chatIsLoading = useChatStore((s) => s.isLoadingList);
  const chatFetch = useChatStore((s) => s.fetchConversations);
  const chatCreate = useChatStore((s) => s.createConversation);
  const chatLoad = useChatStore((s) => s.loadConversation);
  const chatDelete = useChatStore((s) => s.deleteConversation);
  const chatUpdate = useChatStore((s) => s.updateConversation);

  // Fetch conversations when chat tab is first shown
  const chatFetchedRef = React.useRef(false);
  React.useEffect(() => {
    if (activeTab === 'chat' && !chatFetchedRef.current) {
      chatFetchedRef.current = true;
      chatFetch();
    }
  }, [activeTab, chatFetch]);

  const renderKnowledgeEntry = () => {
    return (
      <>
        <KnowledgePanel onOpenCreateDialog={() => setIsKnowledgeDialogOpen(true)} />
        <KnowledgeSearch />
      </>
    );
  };

  const renderSkillsEntry = () => {
    return (
      <div className="p-3 text-xs text-gray-300">
        <div className="text-xs font-semibold text-gray-200">Skills</div>
        <div className="text-[11px] text-gray-500 mt-1">MVP placeholder</div>
        <div className="mt-3 bg-gray-900 border border-gray-700 rounded p-2">
          <div className="text-[11px] text-gray-400">Installed</div>
          <div className="text-[11px] text-gray-500 mt-1">No skills installed</div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-800 border-r border-gray-700 flex flex-col transition-all duration-300 ease-in-out min-h-0 h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-700 flex items-center justify-between shrink-0">
        <h2 className="text-sm font-semibold text-gray-200">{sidebarTitle}</h2>
        <button
          onClick={onToggleCollapse}
          className="p-1 hover:bg-gray-700 rounded text-gray-400"
          title="Collapse sidebar"
          aria-label="Collapse sidebar"
          type="button"
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar pb-6 flex flex-col">
        {activeTab === 'workflow' && (
          <>
            {/* Node Palette */}
            <NodePalette />

            {/* Execution Controls */}
            <ExecutionControls
              currentExecution={viewExecution}
              viewMode={logic.viewMode}
              isStarting={logic.isStarting}
              hasNodes={logic.hasNodes}
              onStartExecution={logic.handleStartExecution}
              onPauseExecution={logic.handlePauseExecution}
              onResumeExecution={logic.handleResumeExecution}
              onAbortExecution={logic.handleStopExecution}
              runSettingsSummary={logic.runSettingsSummary}
              onOpenRunSettings={logic.handleOpenRunSettings}
            />

            {/* Workflow Management */}
            <WorkflowManagement
              nodes={logic.nodes}
              workflows={logic.workflows}
              isLoadingWorkflows={logic.isLoadingWorkflows}
              onSaveWorkflow={logic.handleSaveWorkflow}
              onLoadWorkflow={logic.handleLoadWorkflow}
              onLoadWorkflowByName={logic.handleLoadWorkflowByName}
              onRefreshWorkflows={logic.handleRefreshWorkflows}
              onExportToFile={logic.handleExportToFile}
            />
          </>
        )}

        {activeTab === 'chat' && (
          <ConversationRail
            conversations={chatConversations}
            activeConversationId={chatActiveId}
            isLoading={chatIsLoading}
            onSelect={chatLoad}
            onCreate={() => chatCreate()}
            onDelete={chatDelete}
            onOpenSettings={(id) => {
              chatLoad(id);
              onOpenChatSettings?.(id);
            }}
            onToggleBookmark={(id, bookmarked) => chatUpdate(id, { bookmarked })}
          />
        )}

        {activeTab === 'knowledge' && renderKnowledgeEntry()}
        {activeTab === 'skills' && renderSkillsEntry()}
      </div>

      {/* Save Workflow Dialog */}
      <SaveWorkflowDialog
        isOpen={logic.showSaveDialog}
        onClose={() => logic.setShowSaveDialog(false)}
        onSave={logic.handleSaveWorkflowWithName}
      />

      {/* Load Workflow Dialog */}
      <LoadWorkflowDialog
        isOpen={logic.showLoadDialog}
        onClose={() => logic.setShowLoadDialog(false)}
        onLoad={logic.handleLoadWorkflowWithName}
        workflows={logic.workflows}
        isLoadingWorkflows={logic.isLoadingWorkflows}
        onRefreshWorkflows={logic.handleRefreshWorkflows}
      />

      {logic.showRestoreDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-gray-800 border border-gray-700 rounded-lg shadow-xl w-[420px] max-w-[90vw]">
            <div className="p-4 border-b border-gray-700">
              <h3 className="text-sm font-semibold text-gray-200">Restore Checkpoint</h3>
              <p className="text-xs text-gray-400 mt-1">
                Choose how to replay after restoring.
              </p>
            </div>

            <div className="p-4 space-y-3">
              <button
                onClick={() => logic.handleConfirmRestoreDialog('deterministic')}
                className="w-full p-2 bg-gray-700 hover:bg-gray-600 rounded text-sm text-gray-50 transition-colors"
              >
                Deterministic replay (use recorded output)
              </button>

              <button
                onClick={() => logic.handleConfirmRestoreDialog('fresh')}
                className="w-full p-2 bg-blue-600 hover:bg-blue-700 rounded text-sm text-white transition-colors"
              >
                Fresh replay (re-run nodes with current settings)
              </button>
            </div>

            <div className="p-3 border-t border-gray-700 flex justify-end">
              <button
                onClick={logic.handleCancelRestoreDialog}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Knowledge Config Dialog */}
      <KnowledgeConfigDialog
        isOpen={isKnowledgeDialogOpen}
        onClose={() => setIsKnowledgeDialogOpen(false)}
      />
    </div>
  );
};
