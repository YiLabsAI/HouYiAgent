type StoreSet = (partial: any) => void;
type StoreGet = () => any;

export const createWorkflowActions = (set: StoreSet, get: StoreGet) => ({
  saveWorkflow: (workflowName: string) => {
    const { nodes, edges, currentPlan } = get();
    if (!nodes || nodes.length === 0) {
      get().showToast('No workflow to save (canvas is empty)', 'error');
      return;
    }
    const planId = currentPlan?.plan_id || `plan_${Date.now()}`;
    const layoutPositions = nodes.reduce((acc: Record<string, { x: number; y: number }>, node: any) => {
      const nodeId = node.node_id ?? node.id ?? '';
      if (nodeId) {
        acc[nodeId] = node.position ?? { x: 0, y: 0 };
      }
      return acc;
    }, {});
    const plan = {
      plan_id: planId,
      version: currentPlan?.version ?? 1,
      nodes: nodes.map((node: any) => ({
        node_id: node.node_id ?? node.id ?? '',
        node_type: node.node_type?.value ?? node.node_type ?? node.type ?? 'llm',
        position: node.position ?? { x: 0, y: 0 },
        config: node.config ?? node.data?.config ?? {},
        inputs: node.inputs ?? node.data?.inputs ?? {},
        outputs: node.outputs ?? node.data?.outputs ?? {},
        metadata: node.metadata ?? node.data?.metadata ?? {},
      })),
      edges: edges.map((edge: any) => ({
        edge_id: edge.edge_id ?? edge.id ?? `${edge.source}-${edge.target}`,
        source_node_id: edge.source_node_id ?? edge.source,
        target_node_id: edge.target_node_id ?? edge.target,
        metadata: edge.metadata ?? edge.data?.metadata ?? {},
      })),
      entry_node_id: nodes[0]?.id ?? nodes[0]?.node_id ?? '',
      metadata: currentPlan?.metadata ?? { source: 'console_ui' },
      layout: {
        positions: layoutPositions,
      },
    };

    const command = {
      command_type: 'save_workflow',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      workflow_name: workflowName,
      plan,
    };
    get().sendCommand(command);
    console.log('[Store] Saving workflow:', workflowName, 'with', nodes.length, 'nodes');
    get().showToast(`Workflow "${workflowName}" saved successfully!`, 'success');
  },

  loadWorkflow: (workflowName: string) => {
    const state = get();
    if (state.loadingWorkflowName && state.loadingWorkflowName === workflowName) {
      return;
    }

    const catalog = get().workflows || [];
    const meta = catalog.find((w: any) => w?.name === workflowName);
    if (meta && (meta.nodes_count ?? 0) === 0) {
      get().showToast(
        `Workflow "${workflowName}" has 0 nodes (empty or corrupt). Please delete or fix the workflow file.`,
        'error',
      );
      return;
    }

    // Reset view/execution state so previously completed node statuses don't leak into the loaded workflow.
    set({
      viewMode: 'live',
      selectedCheckpointKey: null,
      checkpointExecution: null,
      liveExecution: null,
      currentExecution: null,
      checkpoints: [],
      selectedNodeId: null,
      nodes: (get().nodes || []).map((node: any) => ({
        ...node,
        data: {
          ...node.data,
          status: 'pending',
        },
      })),
    });

    const command = {
      command_type: 'load_workflow',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
      workflow_name: workflowName,
    };
    get().sendCommand(command);
    console.log('[Store] Loading workflow:', workflowName);
    set({ loadingWorkflowName: workflowName });
  },

  requestWorkflows: () => {
    set({ isLoadingWorkflows: true });
    const command = {
      command_type: 'list_workflows',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId,
    };
    get().sendCommand(command);
    console.log('[Store] Requesting workflow list');
  },
});
