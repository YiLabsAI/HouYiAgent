import React, { useState } from 'react';
import { WorkflowList } from './WorkflowList';

interface Workflow {
  name: string;
  saved_at: string;
  nodes_count: number;
  edges_count?: number;
}

interface WorkflowManagementProps {
  nodes: any[];
  workflows: Workflow[];
  isLoadingWorkflows: boolean;
  onSaveWorkflow: () => void;
  onLoadWorkflow: () => void;
  onLoadWorkflowByName: (name: string) => void;
  onExportToFile: () => void;
  onRefreshWorkflows: () => void;
}

export const WorkflowManagement: React.FC<WorkflowManagementProps> = ({
  nodes,
  workflows,
  isLoadingWorkflows,
  onSaveWorkflow,
  onLoadWorkflow,
  onLoadWorkflowByName,
  onExportToFile,
  onRefreshWorkflows,
}) => {
  const [showList, setShowList] = useState(false);

  return (
    <div className="p-3 border-t border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-gray-400">Workflow Management</h3>
        <button
          onClick={() => {
            setShowList(!showList);
            if (!showList) {
              onRefreshWorkflows();
            }
          }}
          className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
          title={showList ? "Hide workflow list" : "Show workflow list"}
        >
          {showList ? '▼' : '▶'}
        </button>
      </div>

      {showList && (
        <WorkflowList
          workflows={workflows}
          onLoadWorkflow={onLoadWorkflowByName}
          isLoading={isLoadingWorkflows}
        />
      )}

      <div className="space-y-2 mt-2">
        {/* Save Workflow */}
        <button
          onClick={onSaveWorkflow}
          disabled={!nodes || nodes.length === 0}
          className="w-full h-10 px-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
          title="Save workflow with a name for later use"
        >
          <span>💾</span>
          <span className="min-w-0 overflow-hidden text-ellipsis">Save Workflow</span>
        </button>

        {/* Load Workflow */}
        <button
          onClick={onLoadWorkflow}
          className="w-full h-10 px-3 bg-purple-600 hover:bg-purple-700 rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
          title="Load a previously saved workflow"
        >
          <span>📂</span>
          <span className="min-w-0 overflow-hidden text-ellipsis">Load Workflow</span>
        </button>

        {/* Export to File */}
        <button
          onClick={onExportToFile}
          disabled={!nodes || nodes.length === 0}
          className="w-full h-10 px-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 active:scale-95 whitespace-nowrap"
          title="Export workflow to JSON file"
        >
          <span>📥</span>
          <span className="min-w-0 overflow-hidden text-ellipsis">Export to File</span>
        </button>
      </div>
    </div>
  );
};
