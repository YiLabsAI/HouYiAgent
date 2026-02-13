import type { Node, Edge } from 'reactflow';

import { DEFAULT_MODEL } from '../../constants/models';

const DEFAULT_LLM_CONFIG = {
  model: DEFAULT_MODEL,
  max_tokens: 2000,
  prompt: 'Hello, how can I help you?',
};

type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

type Position = { x: number; y: number };

type NodeData = {
  label?: string;
  nodeType?: string;
  config?: Record<string, any>;
  inputs?: Record<string, any>;
  outputs?: Record<string, any>;
  metadata?: Record<string, any>;
};

export const createNodeActions = (set: StoreSet, get: StoreGet) => ({
  addNode: (nodeType: string, position: Position) => {
    console.log('[Store] addNode called, nodeType:', nodeType, 'position:', position);
    const state = get();
    const normalizedType = nodeType.toLowerCase();
    const existingCount = state.nodes.filter((node: any) => node.type === normalizedType).length;
    const nodeId = `${normalizedType}_${existingCount + 1}`;
    console.log('[Store] Creating node with ID:', nodeId);

    const defaultConfig = normalizedType === 'llm' ? { ...DEFAULT_LLM_CONFIG } : {};

    const newNode: Node = {
      id: nodeId,
      type: normalizedType,
      position,
      data: {
        label: `${nodeType} ${existingCount + 1}`,
        nodeType,
        config: defaultConfig,
        inputs: {},
        outputs: {},
        metadata: {
          label: `${nodeType} ${existingCount + 1}`,
        },
      } as NodeData,
    };

    set((prev: any) => ({
      nodes: [...prev.nodes, newNode],
    }));

    get().sendPatchPlan([
      {
        action: 'add_node',
        node: {
          node_id: nodeId,
          node_type: nodeType,
          position,
          config: defaultConfig,
          inputs: {},
          outputs: {},
          metadata: {
            label: `${nodeType} ${existingCount + 1}`,
          },
        },
      },
    ]);
  },

  updateNode: (nodeId: string, updates: Partial<NodeData>) => {
    let mergedData: NodeData | null = null;

    set((state: any) => ({
      nodes: state.nodes.map((node: any) => {
        if (node.id === nodeId) {
          mergedData = { ...node.data, ...updates };
          return { ...node, data: mergedData };
        }
        return node;
      }),
    }));

    if (mergedData) {
      get().sendPatchPlan([
        {
          action: 'update_node',
          node_id: nodeId,
          node: mergedData,
        },
      ]);
    }
  },

  updateNodePosition: (nodeId: string, position: Position) => {
    set((state: any) => ({
      nodes: state.nodes.map((node: any) =>
        node.id === nodeId ? { ...node, position } : node,
      ),
    }));

    get().sendPatchPlan([
      {
        action: 'update_node',
        node_id: nodeId,
        node: { position },
      },
    ]);
  },

  deleteNode: (nodeId: string) => {
    set((state: any) => ({
      nodes: state.nodes.filter((node: any) => node.id !== nodeId),
      edges: state.edges.filter(
        (edge: any) => edge.source !== nodeId && edge.target !== nodeId,
      ),
    }));

    get().sendPatchPlan([
      {
        action: 'delete_node',
        node_id: nodeId,
      },
    ]);
  },

  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
  },

  addEdge: (source: string, target: string) => {
    const edgeId = `${source}-${target}`;
    const newEdge: Edge = {
      id: edgeId,
      source,
      target,
      type: 'default',
    };

    set((state: any) => ({
      edges: [...state.edges, newEdge],
    }));

    get().sendPatchPlan([
      {
        action: 'add_edge',
        edge: {
          edge_id: edgeId,
          source,
          target,
          edge_type: 'default',
          metadata: {},
        },
      },
    ]);
  },

  deleteEdge: (edgeId: string) => {
    set((state: any) => ({
      edges: state.edges.filter((edge: any) => edge.id !== edgeId),
    }));

    get().sendPatchPlan([
      {
        action: 'delete_edge',
        edge_id: edgeId,
      },
    ]);
  },
});
