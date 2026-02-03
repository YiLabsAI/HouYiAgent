import { createToastActions } from '@/stores/storeActions/toastActions';
import { createCommandActions } from '@/stores/storeActions/commandActions';

describe('toastActions.showToastOnce', () => {
  it('should be atomic and avoid duplicates when called multiple times for the same key', () => {
    let state: any = {
      toasts: [],
      toastKeys: {},
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;

    const actions = createToastActions(set, get);

    actions.showToastOnce('ws-command', 'Backend not connected. Please start the server.', 'error');
    actions.showToastOnce('ws-command', 'Backend not connected. Please start the server.', 'error');
    actions.showToastOnce('ws-command', 'Backend not connected. Please start the server.', 'error');

    expect(state.toasts).toHaveLength(1);
    expect(Object.keys(state.toastKeys)).toEqual(['ws-command']);
  });

  it('should avoid duplicate backend-connection toasts when sending commands while offline', () => {
    let state: any = {
      toasts: [],
      toastKeys: {},
      activityLogs: [],
      ws: null,
    };

    const set = (partial: any) => {
      const next = typeof partial === 'function' ? partial(state) : partial;
      state = { ...state, ...next };
    };

    const get = () => state;

    Object.assign(state, createToastActions(set, get));
    Object.assign(state, createCommandActions(set, get));

    state.sendCommand({ command_type: 'list_workflows', command_id: 'cmd1' });
    state.sendCommand({ command_type: 'list_workflows', command_id: 'cmd2' });
    state.sendCommand({ command_type: 'list_workflows', command_id: 'cmd3' });

    expect(state.toasts.filter((t: any) => t.type === 'error')).toHaveLength(1);
    expect(Object.keys(state.toastKeys)).toEqual(['backend-connection']);
  });
});
