import React from 'react';

const NODE_TYPES = [
  { type: 'LLM', icon: '🤖', label: 'LLM' },
  { type: 'Tool', icon: '🔧', label: 'Tool' },
  { type: 'Verify', icon: '✓', label: 'Verify' },
  { type: 'Logic', icon: '🧠', label: 'Logic' },
  { type: 'Route', icon: '🔀', label: 'Route' },
];

export const NodePalette: React.FC = () => {
  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="p-3 border-b border-gray-700">
      <h3 className="text-xs font-semibold text-gray-400 mb-2">Node Types</h3>
      <div className="grid grid-cols-2 gap-2">
        {NODE_TYPES.map((node) => (
          <div
            key={node.type}
            draggable
            onDragStart={(e) => onDragStart(e, node.type)}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded cursor-move text-center transition-all duration-200 active:scale-95"
            title={`Drag to add ${node.label} node`}
          >
            <div className="text-lg mb-1">{node.icon}</div>
            <div className="text-xs text-gray-200">{node.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
