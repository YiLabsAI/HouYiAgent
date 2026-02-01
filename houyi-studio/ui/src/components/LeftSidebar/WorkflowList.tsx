import React from 'react';

interface Workflow {
  name: string;
  saved_at: string;
  nodes_count: number;
  edges_count?: number;
}

interface WorkflowListProps {
  workflows: Workflow[];
  onLoadWorkflow: (name: string) => void;
  isLoading: boolean;
}

export const WorkflowList: React.FC<WorkflowListProps> = ({
  workflows,
  onLoadWorkflow,
  isLoading,
}) => {
  if (isLoading) {
    return (
      <div className="text-xs text-gray-400 italic p-2">
        Loading workflows...
      </div>
    );
  }

  if (workflows.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic p-2">
        No saved workflows yet
      </div>
    );
  }

  return (
    <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
      {workflows.map((workflow) => (
        <div
          key={workflow.name}
          onClick={() => {
            if ((workflow.nodes_count ?? 0) === 0) return;
            onLoadWorkflow(workflow.name);
          }}
          className={`p-2 rounded transition-colors ${
            (workflow.nodes_count ?? 0) === 0
              ? 'bg-gray-800 border border-red-500/30 text-gray-400 cursor-not-allowed'
              : 'bg-gray-700 hover:bg-gray-600 cursor-pointer'
          }`}
          title={(workflow.nodes_count ?? 0) === 0 ? 'This workflow is empty/corrupt (0 nodes). Please delete or fix the file.' : workflow.name}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-200 truncate">
              📄 {workflow.name}
            </span>
            <span className="text-xs text-gray-400">
              {workflow.nodes_count}n / {workflow.edges_count ?? 0}e
            </span>
          </div>
          {(workflow.nodes_count ?? 0) === 0 ? (
            <div className="text-[10px] text-red-300 mt-0.5">Empty workflow (cannot load)</div>
          ) : null}
          <div className="text-xs text-gray-500 mt-0.5">
            {new Date(workflow.saved_at).toLocaleDateString()}
          </div>
        </div>
      ))}
    </div>
  );
};
