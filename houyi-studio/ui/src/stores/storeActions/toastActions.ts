type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

type ActivityLog = {
  id: string;
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  detail?: string;
};

type ActivityLogInput = Partial<Omit<ActivityLog, 'id' | 'timestamp'>> & {
  message: string;
  timestamp?: string;
};

export const createToastActions = (set: StoreSet, get: StoreGet) => ({
  showToast: (message: string, type: 'success' | 'error' | 'info') => {
    const id = `toast_${Date.now()}`;
    set((state: any) => ({
      toasts: [...state.toasts, { id, message, type }],
    }));
  },

  showToastOnce: (key: string, message: string, type: 'success' | 'error' | 'info') => {
    const existing = get().toastKeys[key];
    if (existing) return;
    const id = `toast_${Date.now()}`;
    set((state: any) => ({
      toasts: [...state.toasts, { id, message, type }],
      toastKeys: { ...state.toastKeys, [key]: id },
    }));
  },

  removeToast: (id: string) => {
    set((state: any) => ({
      toasts: state.toasts.filter((t: any) => t.id !== id),
      toastKeys: Object.fromEntries(
        Object.entries(state.toastKeys).filter(([, value]) => value !== id),
      ),
    }));
  },

  removeToastByKey: (key: string) => {
    const id = get().toastKeys[key];
    if (!id) return;
    set((state: any) => {
      const { [key]: _removed, ...rest } = state.toastKeys;
      return {
        toasts: state.toasts.filter((t: any) => t.id !== id),
        toastKeys: rest,
      };
    });
  },

  addActivityLog: (log: ActivityLogInput) => {
    const entry: ActivityLog = {
      id: `log_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      timestamp: log.timestamp ?? new Date().toISOString(),
      level: log.level ?? 'info',
      message: log.message,
      detail: log.detail,
    };

    set((state: any) => ({
      activityLogs: [...state.activityLogs.slice(-199), entry],
    }));
  },
});
