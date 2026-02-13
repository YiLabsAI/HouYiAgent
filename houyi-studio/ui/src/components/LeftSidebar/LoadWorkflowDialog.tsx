import React, { useEffect, useState } from 'react';
import { WorkflowList } from './WorkflowList';

interface LoadWorkflowDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onLoad: (name: string) => void;
  workflows: { name: string; saved_at: string; nodes_count: number; edges_count?: number }[];
  isLoadingWorkflows: boolean;
  onRefreshWorkflows: () => void;
}

export const LoadWorkflowDialog: React.FC<LoadWorkflowDialogProps> = ({
  isOpen,
  onClose,
  onLoad,
  workflows,
  isLoadingWorkflows,
  onRefreshWorkflows,
}) => {
  const [workflowName, setWorkflowName] = useState('');

  const refreshRef = React.useRef(onRefreshWorkflows);
  const wasOpenRef = React.useRef(false);

  useEffect(() => {
    refreshRef.current = onRefreshWorkflows;
  }, [onRefreshWorkflows]);

  useEffect(() => {
    const wasOpen = wasOpenRef.current;
    wasOpenRef.current = isOpen;
    if (!wasOpen && isOpen) {
      refreshRef.current();
    }
  }, [isOpen]);

  const handleLoad = () => {
    if (workflowName.trim()) {
      onLoad(workflowName.trim());
      setWorkflowName('');
      onClose();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLoad();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-[420px] shadow-xl border border-gray-700">
        <h2 className="text-xl font-bold mb-4 text-gray-50">Load Workflow</h2>

        <div className="mb-4">
          <WorkflowList
            workflows={workflows}
            onLoadWorkflow={(name) => {
              setWorkflowName(name);
              onLoad(name);
              setWorkflowName('');
              onClose();
            }}
            isLoading={isLoadingWorkflows}
          />
        </div>

        <input
          type="text"
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Enter workflow name to load..."
          className="w-full px-4 py-2 bg-gray-700 text-gray-50 rounded border border-gray-600 focus:border-purple-500 focus:outline-none mb-4"
          autoFocus
        />

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-gray-50 rounded transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleLoad}
            disabled={!workflowName.trim()}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded transition-colors"
          >
            Load
          </button>
        </div>
      </div>
    </div>
  );
};
