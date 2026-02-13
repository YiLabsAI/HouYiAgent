import React, { useState } from 'react';

interface SaveWorkflowDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string) => void;
}

export const SaveWorkflowDialog: React.FC<SaveWorkflowDialogProps> = ({
  isOpen,
  onClose,
  onSave,
}) => {
  const [workflowName, setWorkflowName] = useState('');

  const handleSave = () => {
    if (workflowName.trim()) {
      onSave(workflowName.trim());
      setWorkflowName('');
      onClose();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-96 shadow-xl border border-gray-700">
        <h2 className="text-xl font-bold mb-4 text-gray-50">Save Workflow</h2>

        <input
          type="text"
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Enter workflow name..."
          className="w-full px-4 py-2 bg-gray-700 text-gray-50 rounded border border-gray-600 focus:border-blue-500 focus:outline-none mb-4"
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
            onClick={handleSave}
            disabled={!workflowName.trim()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};
