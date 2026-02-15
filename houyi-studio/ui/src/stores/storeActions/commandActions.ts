type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

export const createCommandActions = (_set: StoreSet, get: StoreGet) => ({
  sendCommand: (command: any): boolean => {
    const { ws } = get();
    console.log('[Store] sendCommand called, ws:', ws ? 'exists' : 'null', 'command:', command);
    if (command?.command_type) {
      const detailParts: string[] = [];
      if (command.execution_id) detailParts.push(`execution=${command.execution_id}`);
      if (command.checkpoint_id) detailParts.push(`checkpoint=${command.checkpoint_id}`);
      if (command.replay_mode) detailParts.push(`mode=${command.replay_mode}`);
      get().addActivityLog({
        message: `Command sent: ${command.command_type}`,
        detail: detailParts.length > 0 ? detailParts.join(' · ') : undefined,
      });
    }
    if (!ws) {
      console.error('[Store] WebSocket not initialized, cannot send command');
      get().showToastOnce('backend-connection', 'Backend not connected. Please start the server.', 'error');
      return false;
    }
    // Always delegate to ws.sendCommand() — it queues commands when the
    // connection is still opening, and sends them once connected.  The
    // old code checked isConnected() first and silently dropped commands
    // sent during the CONNECTING phase, causing skills & other features
    // that fire on mount to never receive a response.
    ws.sendCommand(command);
    return true;
  },

  sendPatchPlan: (patches: any[]) => {
    const { currentPlan } = get();
    const command = {
      command_type: 'patch_plan',
      command_id: `cmd_${Date.now()}`,
      session_id: get().sessionId || 'session-123',
      execution_id: get().executionId || '',
      base_version: currentPlan?.version || 0,
      patches,
    };
    get().sendCommand(command);
    console.log('[Store] Sent patch_plan:', patches.length, 'patches');
  },
});
