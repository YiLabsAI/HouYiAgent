/**
 * ConfirmModal: reusable confirmation dialog for destructive actions.
 *
 * Displays a centered modal with title, description, and confirm/cancel buttons.
 * Used globally for delete confirmations and other irreversible operations.
 */
import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface ConfirmModalProps {
  isOpen: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'warning' | 'info';
  onConfirm: () => void;
  onCancel: () => void;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'danger',
  onConfirm,
  onCancel,
}) => {
  if (!isOpen) return null;

  const confirmColors = {
    danger: 'bg-red-600 hover:bg-red-700 focus:ring-red-500',
    warning: 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500',
    info: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500',
  };

  const iconColors = {
    danger: 'text-red-400',
    warning: 'text-yellow-400',
    info: 'text-blue-400',
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/60 z-[60]" onClick={onCancel} />

      {/* Modal */}
      <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
        <div
          className="bg-gray-800 border border-gray-700 rounded-xl shadow-2xl w-[400px] max-w-[90vw]"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="p-5">
            <div className="flex items-start gap-3">
              <div className={`shrink-0 mt-0.5 ${iconColors[variant]}`}>
                <AlertTriangle size={20} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
                <p className="text-[12px] text-gray-400 mt-1 leading-relaxed">{description}</p>
              </div>
            </div>
          </div>

          <div className="px-5 py-3 border-t border-gray-700 flex justify-end gap-2">
            <button
              onClick={onCancel}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-[12px] text-gray-300 transition-colors"
              type="button"
            >
              {cancelLabel}
            </button>
            <button
              onClick={onConfirm}
              className={`px-3 py-1.5 rounded-lg text-[12px] text-white transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 ${confirmColors[variant]}`}
              type="button"
              autoFocus
            >
              {confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};
