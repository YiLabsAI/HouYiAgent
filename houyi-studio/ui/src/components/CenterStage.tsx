/**
 * Unified overlay component used by modal-style views.
 *
 * Supports three sizes (S / M / L) and common behaviors:
 *   - semi-transparent backdrop
 *   - close on Esc
 *   - optional close on backdrop click
 */
import React from 'react';
import { X } from 'lucide-react';

export type CenterStageSize = 'S' | 'M' | 'L';

export interface CenterStageProps {
  /** Whether the overlay is open. */
  isOpen: boolean;
  /** Called when the overlay should close (Esc, backdrop click, close button). */
  onClose: () => void;
  /** Size variant: S (480px), M (640px), L (90vw x 85vh). */
  size: CenterStageSize;
  /** Title displayed in the header bar. */
  title: string;
  /** Content to render inside the overlay. */
  children: React.ReactNode;
  /** Whether clicking the backdrop closes the overlay. Default: true. */
  closeOnBackdrop?: boolean;
}

// ─── Size → CSS class mapping ────────────────────────────────────
const SIZE_CLASSES: Record<CenterStageSize, string> = {
  S: 'w-[480px] max-h-[80vh]',
  M: 'w-[640px] max-h-[80vh]',
  L: 'w-[90vw] h-[85vh]',
};

export const CenterStage: React.FC<CenterStageProps> = ({
  isOpen,
  onClose,
  size,
  title,
  children,
  closeOnBackdrop = true,
}) => {
  const contentRef = React.useRef<HTMLDivElement>(null);

  // ─── Esc key handler ─────────────────────────────────────────
  React.useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // ─── Focus trap: focus the content on open ───────────────────
  React.useEffect(() => {
    if (isOpen && contentRef.current) {
      contentRef.current.focus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 transition-opacity"
        data-testid="center-stage-backdrop"
        onClick={closeOnBackdrop ? onClose : undefined}
      />

      {/* Content panel */}
      <div
        ref={contentRef}
        tabIndex={-1}
        className={`relative bg-gray-800 border border-gray-600 rounded-lg shadow-2xl flex flex-col overflow-hidden ${SIZE_CLASSES[size]}`}
        data-testid="center-stage-panel"
        data-size={size}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 shrink-0">
          <h2 className="text-sm font-semibold text-gray-200 truncate">{title}</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
            title="Close"
            data-testid="center-stage-close"
          >
            <X size={16} />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto p-4">
          {children}
        </div>
      </div>
    </div>
  );
};
