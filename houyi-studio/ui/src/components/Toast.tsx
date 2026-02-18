import React from 'react';

interface ToastProps {
  message: string;
  type: 'success' | 'error' | 'info';
  onClose: () => void;
}

// Auto-dismiss times by type (in ms)
const AUTO_DISMISS_TIMES = {
  success: 2000,  // Success messages dismiss quickly
  error: 5000,    // Errors stay longer for user to read
  info: 3000,     // Info is in between
};

export const Toast: React.FC<ToastProps> = ({ message, type, onClose }) => {
  // Use a ref so the auto-dismiss timer is not reset when onClose
  // changes identity (e.g. parent re-renders without useCallback).
  const onCloseRef = React.useRef(onClose);
  onCloseRef.current = onClose;

  React.useEffect(() => {
    const timer = setTimeout(() => onCloseRef.current(), AUTO_DISMISS_TIMES[type]);
    return () => clearTimeout(timer);
  }, [type]);

  const bgColor = {
    success: 'bg-green-600',
    error: 'bg-red-600',
    info: 'bg-blue-600',
  }[type];

  const icon = {
    success: '✓',
    error: '✗',
    info: 'ℹ',
  }[type];

  return (
    <div className={`${bgColor} text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 animate-slide-in text-sm`}>
      <span className="text-base">{icon}</span>
      <span className="font-medium">{message}</span>
      <button
        onClick={onClose}
        className="ml-2 text-white hover:text-gray-200 opacity-70 hover:opacity-100"
      >
        ×
      </button>
    </div>
  );
};

interface ToastContainerProps {
  toasts: Array<{ id: string; message: string; type: 'success' | 'error' | 'info' }>;
  onRemove: (id: string) => void;
}

// Maximum number of toasts to display at once
const MAX_VISIBLE_TOASTS = 3;

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onRemove }) => {
  // Only show the most recent toasts, oldest first (so newest appears at bottom)
  const visibleToasts = toasts.slice(-MAX_VISIBLE_TOASTS);
  const hiddenCount = toasts.length - visibleToasts.length;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {hiddenCount > 0 && (
        <div className="text-xs text-gray-400 text-right px-2">
          +{hiddenCount} more
        </div>
      )}
      {visibleToasts.map((toast) => (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          onClose={() => onRemove(toast.id)}
        />
      ))}
    </div>
  );
};
