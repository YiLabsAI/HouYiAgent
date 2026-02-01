import { useState, useEffect } from 'react';
import { useConsoleStore } from '../../stores/useConsoleStore';

export const useLeftSidebarLogic = () => {
  const store = useConsoleStore();
  const [isStarting, setIsStarting] = useState(false);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [showRestoreDialog, setShowRestoreDialog] = useState(false);
  const [restoreCheckpointId, setRestoreCheckpointId] = useState<string | null>(null);
  const [restoreExecutionId, setRestoreExecutionId] = useState<string | null>(null);
  const [startBaselineExecutionId, setStartBaselineExecutionId] = useState<string | null>(null);
  const [startBaselineStatus, setStartBaselineStatus] = useState<string | null>(null);

  const liveExecutionForCommands = store.liveExecution || store.currentExecution;

  const hasNodes = Boolean(
    (store.nodes && store.nodes.length > 0)
      || (store.currentPlan?.nodes && store.currentPlan.nodes.length > 0)
  );
  const runSettingsSummary = store.runSettings.enable_tool_calls
    ? `Tool calls enabled · Tools: ${store.runSettings.tool_names.length || 'all'} · Choice: ${store.runSettings.tool_choice ?? 'auto'} · Max: ${store.runSettings.max_tool_calls}`
    : 'Run settings: default · Tool calls disabled';

  const validatePlanNodes = (plan: any) => {
    const errors: string[] = [];
    const nodes = plan?.nodes ?? [];
    nodes.forEach((node: any) => {
      const rawType = node.node_type?.value ?? node.node_type ?? node.type ?? '';
      const nodeType = typeof rawType === 'string' ? rawType.toLowerCase() : '';
      if (nodeType !== 'tool') return;
      const config = node.config ?? node.data?.config ?? {};
      const metadata = node.metadata ?? node.data?.metadata ?? {};
      const toolName = config.tool_name || metadata.tool_name || metadata.skill_name;
      if (!toolName) {
        const nodeId = node.node_id ?? node.id ?? 'unknown';
        errors.push(`Tool node ${nodeId} is missing tool_name`);
      }
    });
    return errors;
  };

  const buildPlanPayload = () => {
    const planId = store.currentPlan?.plan_id || `plan_${Date.now()}`;
    const storeNodes = store.nodes || [];
    const storeEdges = store.edges || [];
    const useStoreNodes = storeNodes.length > 0;
    const rawNodes = useStoreNodes ? storeNodes : (store.currentPlan?.nodes || []);
    const rawEdges = useStoreNodes ? storeEdges : (store.currentPlan?.edges || []);

    const nodes = rawNodes.map((node: any) => {
      const nodeId = node.node_id ?? node.id ?? '';
      const rawType = node.node_type?.value ?? node.node_type ?? node.type ?? 'llm';
      const nodeType = typeof rawType === 'string' ? rawType.toLowerCase() : 'llm';
      const position = node.position || { x: 0, y: 0 };
      const config = node.config ?? node.data?.config ?? {};
      const inputs = node.inputs ?? node.data?.inputs ?? {};
      const outputs = node.outputs ?? node.data?.outputs ?? {};
      const metadata = { ...(node.metadata ?? node.data?.metadata ?? {}) };

      if (node.data?.label && metadata.label == null) {
        metadata.label = node.data.label;
      }

      return {
        node_id: nodeId,
        node_type: nodeType,
        position,
        config,
        inputs,
        outputs,
        metadata,
      };
    });

    const edges = rawEdges.map((edge: any) => {
      const source = edge.source ?? edge.source_node_id;
      const target = edge.target ?? edge.target_node_id;
      const edgeId = edge.id ?? edge.edge_id ?? `${source}-${target}`;
      return {
        id: edgeId,
        source,
        target,
        metadata: edge.metadata ?? edge.data?.metadata ?? {},
      };
    });

    return {
      planId,
      plan: {
        plan_id: planId,
        version: store.currentPlan?.version ?? 1,
        nodes,
        edges,
        entry_node_id: nodes[0]?.node_id ?? '',
        metadata: store.currentPlan?.metadata ?? { source: 'console_ui' },
      },
    };
  };

  // Debug: Log state changes
  useEffect(() => {
    console.log('[LeftSidebar] State update:', {
      viewMode: store.viewMode,
      hasCurrentExecution: !!store.currentExecution,
      currentExecutionStatus: store.currentExecution?.status,
      currentExecutionId: store.currentExecution?.execution_id,
    });
  }, [store.viewMode, store.currentExecution]);

  useEffect(() => {
    if (!isStarting) return;

    const executionId = store.currentExecution?.execution_id ?? null;
    const status = store.currentExecution?.status ?? null;
    const executionIdChanged = executionId && executionId !== startBaselineExecutionId;
    const statusChanged = status && status !== startBaselineStatus;
    const unlockStatuses = ['running', 'failed', 'aborted', 'completed'];
    const canUnlock = status ? unlockStatuses.includes(status) : false;

    if (canUnlock && (executionIdChanged || statusChanged)) {
      setIsStarting(false);
      setStartBaselineExecutionId(null);
      setStartBaselineStatus(null);
    }
  }, [
    isStarting,
    startBaselineExecutionId,
    startBaselineStatus,
    store.currentExecution?.execution_id,
    store.currentExecution?.status,
  ]);

  useEffect(() => {
    if (!isStarting) return;
    if (store.connectionStatus === 'connected') return;
    setIsStarting(false);
    setStartBaselineExecutionId(null);
    setStartBaselineStatus(null);
  }, [isStarting, store.connectionStatus]);

  const handleStartExecution = () => {
    console.log('[LeftSidebar] Start execution clicked');
    console.log('[LeftSidebar] Current plan:', store.currentPlan);
    console.log('[LeftSidebar] Session ID:', store.sessionId);
    console.log('[LeftSidebar] WebSocket:', store.ws ? 'connected' : 'not connected');

    if (store.connectionStatus !== 'connected' || !store.ws) {
      store.showToastOnce('ws-command', 'Backend not connected. Please start the server.', 'error');
      return;
    }

    const { planId, plan } = buildPlanPayload();
    if (!plan.nodes.length) {
      console.error('[LeftSidebar] No plan available to execute');
      store.showToast('Add at least one node to start execution.', 'error');
      return;
    }

    const validationErrors = validatePlanNodes(plan);
    if (validationErrors.length > 0) {
      store.showToast(
        `Failed to start: ${validationErrors.join('; ')}`,
        'error',
      );
      return;
    }

    setStartBaselineExecutionId(store.currentExecution?.execution_id ?? null);
    setStartBaselineStatus(store.currentExecution?.status ?? null);
    setIsStarting(true);

    // Backend expects full plan object in inputs
    const startCommand = {
      command_type: 'start_execution',
      command_id: `cmd_${Date.now()}`,
      session_id: store.sessionId,
      plan_id: planId,
      inputs: {
        plan,
        run_settings: store.runSettings,
      },
    };

    console.log('[LeftSidebar] Sending start command:', startCommand);
    store.sendCommand(startCommand);

  };

  const handleStopExecution = () => {
    if (!liveExecutionForCommands) return;

    const stopCommand = {
      command_type: 'abort',
      command_id: `cmd_${Date.now()}`,
      session_id: store.sessionId,
      execution_id: liveExecutionForCommands.execution_id,
    };

    console.log('[LeftSidebar] Sending stop command:', stopCommand);
    store.sendCommand(stopCommand);
  };

  const handlePauseExecution = () => {
    if (!liveExecutionForCommands) return;

    const pauseCommand = {
      command_type: 'pause',
      command_id: `cmd_${Date.now()}`,
      session_id: store.sessionId,
      execution_id: liveExecutionForCommands.execution_id,
    };

    console.log('[LeftSidebar] Sending pause command:', pauseCommand);
    store.sendCommand(pauseCommand);
  };

  const handleResumeExecution = () => {
    if (!liveExecutionForCommands) return;

    const resumeCommand = {
      command_type: 'resume',
      command_id: `cmd_${Date.now()}`,
      session_id: store.sessionId,
      execution_id: liveExecutionForCommands.execution_id,
    };

    console.log('[LeftSidebar] Sending resume command:', resumeCommand);
    store.sendCommand(resumeCommand);
  };

  const handleExitCheckpointView = () => {
    useConsoleStore.getState().exitCheckpointView();
  };

  const handleOpenRunSettings = () => {
    store.setRunSettingsOpen(true);
  };

  const handleRestoreCheckpoint = (checkpointId: string, executionId?: string) => {
    console.log('[LeftSidebar] Restore checkpoint:', { checkpointId, executionId });

    setRestoreCheckpointId(checkpointId);
    setRestoreExecutionId(executionId || null);
    setShowRestoreDialog(true);
  };

  const handleCancelRestoreDialog = () => {
    setShowRestoreDialog(false);
    setRestoreCheckpointId(null);
    setRestoreExecutionId(null);
  };

  const handleConfirmRestoreDialog = (replayMode: 'fresh' | 'deterministic') => {
    if (!restoreCheckpointId) return;

    const checkpoint = (store.checkpoints || []).find((cp: any) => {
      if (restoreExecutionId) {
        return cp.checkpoint_id === restoreCheckpointId && cp.execution_id === restoreExecutionId;
      }
      return cp.checkpoint_id === restoreCheckpointId;
    });

    if (replayMode === 'fresh') {
      useConsoleStore.getState().clearCurrentExecutionOutputsForFreshReplay();
    }

    useConsoleStore.getState().prepareRestoreFromCheckpoint({
      checkpointId: restoreCheckpointId,
      executionId: restoreExecutionId,
      planId: store.currentPlan?.plan_id,
    });

    const restoreCommand = {
      command_type: 'restore_checkpoint',
      command_id: `cmd_${Date.now()}`,
      session_id: store.sessionId,
      execution_id: restoreExecutionId || checkpoint?.execution_id || undefined,
      checkpoint_id: restoreCheckpointId,
      replay_mode: replayMode,
    };

    console.log('[LeftSidebar] Sending restore command:', restoreCommand);
    store.sendCommand(restoreCommand);
    store.showToast(`Restoring from checkpoint ${restoreCheckpointId.slice(0, 8)}...`, 'info');
    setShowRestoreDialog(false);
    setRestoreCheckpointId(null);
    setRestoreExecutionId(null);
  };

  const handleSaveWorkflow = () => {
    if (!store.nodes || store.nodes.length === 0) return;
    setShowSaveDialog(true);
  };

  const handleSaveWorkflowWithName = (name: string) => {
    useConsoleStore.getState().saveWorkflow(name);
  };

  const handleLoadWorkflow = () => {
    setShowLoadDialog(true);
  };

  const handleLoadWorkflowWithName = (name: string) => {
    useConsoleStore.getState().loadWorkflow(name);
  };

  const handleRefreshWorkflows = async () => {
    store.requestWorkflows();
  };

  const handleLoadWorkflowByName = (name: string) => {
    useConsoleStore.getState().loadWorkflow(name);
  };

  const handleExportToFile = () => {
    if (!store.nodes || store.nodes.length === 0) return;

    const btn = document.activeElement as HTMLButtonElement;
    if (btn) btn.blur();

    const planData = {
      plan_id: store.currentPlan?.plan_id || `plan_${Date.now()}`,
      version: store.currentPlan?.version || 1,
      nodes: store.nodes.map((node) => ({
        node_id: node.id,
        node_type: node.type,
        position: node.position,
        config: node.data?.config || {},
      })),
      edges: store.edges.map((edge) => ({
        edge_id: edge.id,
        source_node_id: edge.source,
        target_node_id: edge.target,
      })),
      entry_node_id: store.nodes[0]?.id || '',
      metadata: { source: 'console_ui', exported_at: new Date().toISOString() },
    };

    const blob = new Blob([JSON.stringify(planData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return {
    ...store,
    isStarting,
    workflows: store.workflows,
    isLoadingWorkflows: store.isLoadingWorkflows,
    showSaveDialog,
    setShowSaveDialog,
    showLoadDialog,
    setShowLoadDialog,
    showRestoreDialog,
    restoreCheckpointId,
    hasNodes,
    runSettingsSummary,
    handleStartExecution,
    handlePauseExecution,
    handleResumeExecution,
    handleStopExecution,
    handleExitCheckpointView,
    handleOpenRunSettings,
    handleRestoreCheckpoint,
    handleCancelRestoreDialog,
    handleConfirmRestoreDialog,
    handleSaveWorkflow,
    handleSaveWorkflowWithName,
    handleLoadWorkflow,
    handleLoadWorkflowWithName,
    handleLoadWorkflowByName,
    handleRefreshWorkflows,
    handleExportToFile,
  };
};
