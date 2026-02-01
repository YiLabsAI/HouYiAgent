/**
 * DAG Canvas component using React Flow
 */

import React from 'react';
import ReactFlow, {
  Background,
  Connection,
  useNodesState,
  useEdgesState,
  OnConnect,
  NodeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { LLMNode } from './nodes/LLMNode';
import { ToolNode } from './nodes/ToolNode';
import { VerifyNode } from './nodes/VerifyNode';
import { LogicNode } from './nodes/LogicNode';
import { RouteNode } from './nodes/RouteNode';
import { useConsoleStore } from '../stores/useConsoleStore';

const nodeTypes: NodeTypes = {
  llm: LLMNode,
  tool: ToolNode,
  verify: VerifyNode,
  logic: LogicNode,
  route: RouteNode,
};

export const DAGCanvas: React.FC = () => {
  const {
    nodes,
    edges,
    addNode,
    addEdge: storeAddEdge,
    selectNode,
    deleteNode,
    deleteEdge,
    selectedNodeId,
    updateNodePosition,
    currentExecution,
    liveExecution,
    checkpointExecution,
    viewMode
  } = useConsoleStore();

  const viewExecution = React.useMemo(() => {
    return viewMode === 'checkpoint' ? checkpointExecution : liveExecution || currentExecution;
  }, [viewMode, checkpointExecution, liveExecution, currentExecution]);

  // Determine if canvas is editable based on execution status
  const isEditable = !viewExecution ||
    viewExecution.status === 'paused' ||
    viewExecution.status === 'completed' ||
    viewExecution.status === 'aborted' ||
    viewExecution.status === 'failed';

  const [localNodes, setLocalNodes, onNodesChange] = useNodesState(nodes);
  const [localEdges, setLocalEdges, onEdgesChange] = useEdgesState(edges);
  const [reactFlowInstance, setReactFlowInstance] = React.useState<any>(null);
  const [isFlowReady, setIsFlowReady] = React.useState(false);
  const [selectedEdgeId, setSelectedEdgeId] = React.useState<string | null>(null);

  // Sync store nodes to local state (when store changes)
  React.useEffect(() => {
    setLocalNodes(nodes);
  }, [nodes, setLocalNodes]);

  React.useEffect(() => {
    setLocalEdges(edges);
  }, [edges, setLocalEdges]);

  const onConnect: OnConnect = React.useCallback(
    (connection: Connection) => {
      if (connection.source && connection.target) {
        storeAddEdge(connection.source, connection.target);
      }
    },
    [storeAddEdge]
  );

  const onNodeClick = React.useCallback(
    (_event: React.MouseEvent, node: any) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  const onDrop = React.useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const nodeType = event.dataTransfer.getData('application/reactflow');
      if (!nodeType || !reactFlowInstance) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addNode(nodeType, position);
    },
    [reactFlowInstance, addNode]
  );

  const onDragOver = React.useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onNodeDragStop = React.useCallback(
    (_event: React.MouseEvent, node: any) => {
      // Sync node position back to store
      updateNodePosition(node.id, node.position);
    },
    [updateNodePosition]
  );

  const onEdgeClick = React.useCallback(
    (_event: React.MouseEvent, edge: any) => {
      setSelectedEdgeId(edge.id);
      selectNode(null); // Deselect node when edge is selected
    },
    [selectNode]
  );

  const onPaneClick = React.useCallback(() => {
    setSelectedEdgeId(null);
    selectNode(null);
  }, [selectNode]);

  // Handle keyboard events for deletion
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only delete if not typing in an input/textarea
      const target = event.target as HTMLElement;
      const isInputField = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';

      console.log('[DAGCanvas] Key pressed:', {
        key: event.key,
        target: target.tagName,
        isInputField,
        selectedNodeId,
        selectedEdgeId
      });

      if (isInputField) {
        console.log('[DAGCanvas] Input field detected, skipping delete');
        return; // Don't delete nodes when typing
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault(); // Prevent default browser behavior

        if (selectedEdgeId) {
          console.log('[DAGCanvas] Deleting edge:', selectedEdgeId);
          deleteEdge(selectedEdgeId);
          setSelectedEdgeId(null);
        } else if (selectedNodeId) {
          console.log('[DAGCanvas] Deleting node:', selectedNodeId);
          deleteNode(selectedNodeId);
        }
      }
    };

    // Use capture phase to intercept events before React Flow
    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [selectedEdgeId, selectedNodeId, deleteEdge, deleteNode]);

  // Update edge styles to show selection
  const styledEdges = React.useMemo(() => {
    return localEdges.map((edge) => ({
      ...edge,
      style: {
        ...edge.style,
        stroke: edge.id === selectedEdgeId ? '#3b82f6' : '#6b7280',
        strokeWidth: edge.id === selectedEdgeId ? 3 : 2,
      },
      animated: edge.id === selectedEdgeId,
    }));
  }, [localEdges, selectedEdgeId]);

  return (
    <div
      className="w-full h-full bg-gray-950 relative"
      data-testid="dag-canvas"
      data-ready={isFlowReady ? 'true' : 'false'}
      onDrop={isEditable ? onDrop : undefined}
      onDragOver={isEditable ? onDragOver : undefined}
    >
      {/* Execution status overlay */}
      {viewExecution && viewExecution.status === 'running' && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 bg-yellow-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
          <span className="text-sm font-medium">Execution Running - Canvas Locked</span>
        </div>
      )}
      {viewExecution && viewExecution.status === 'paused' && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
          <span className="text-sm font-medium">⏸ Paused - You can edit the workflow</span>
        </div>
      )}
      <ReactFlow
        nodes={localNodes}
        edges={styledEdges}
        onNodesChange={isEditable ? onNodesChange : undefined}
        onEdgesChange={isEditable ? onEdgesChange : undefined}
        onConnect={isEditable ? onConnect : undefined}
        onNodeClick={onNodeClick}
        onNodeDragStop={isEditable ? onNodeDragStop : undefined}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        onInit={(instance) => {
          setReactFlowInstance(instance);
          setIsFlowReady(true);
        }}
        nodeTypes={nodeTypes}
        deleteKeyCode={null}
        nodesDraggable={isEditable}
        nodesConnectable={isEditable}
        elementsSelectable={true}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        {/* <MiniMap /> */}
      </ReactFlow>
    </div>
  );
};
