import React, { useState, useCallback } from 'react';
import { useConsoleStore } from '@/stores/useConsoleStore';
import { ExecutionLineageTree } from './ExecutionLineageTree';
import { TimelineWaterfall } from './TimelineWaterfall';
import { MetricsPanel } from './MetricsPanel';

export interface ObsFullViewProps {
  onClose: () => void;
}

/**
 * Full-screen observability overlay.
 *
 * Layout (split-pane):
 *   Left:   ExecutionLineageTree (execution DAG selector)
 *   Right:  TimelineWaterfall (selected execution's span waterfall)
 *   Bottom: MetricsPanel (aggregated metrics for selected execution)
 *
 * Triggered from the Logs tab "Expand" button. Occupies 90% viewport as overlay.
 */
export const ObsFullView: React.FC<ObsFullViewProps> = ({ onClose }) => {
  const { spanStore, currentExecution, liveExecution } = useConsoleStore();
  const getSpanTree = useConsoleStore((s) => s.getSpanTree);

  const defaultExecId = liveExecution?.execution_id || currentExecution?.execution_id || null;
  const [selectedExecId, setSelectedExecId] = useState<string | null>(null);
  const effectiveExecId = selectedExecId || defaultExecId;

  const availableExecIds = React.useMemo(() => {
    return Object.keys(spanStore).sort().reverse();
  }, [spanStore]);

  const handleSelect = useCallback((eid: string) => {
    setSelectedExecId(eid);
  }, []);

  // Left panel width (resizable via drag)
  const [leftWidth, setLeftWidth] = useState(260);
  const [dragging, setDragging] = useState(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
    const startX = e.clientX;
    const startWidth = leftWidth;

    const handleMouseMove = (ev: MouseEvent) => {
      const delta = ev.clientX - startX;
      setLeftWidth(Math.max(180, Math.min(500, startWidth + delta)));
    };
    const handleMouseUp = () => {
      setDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [leftWidth]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-2xl flex flex-col"
        style={{ width: '92vw', height: '90vh' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-200">Observability Full View</span>
            {effectiveExecId && (
              <span className="text-[10px] text-gray-500 font-mono">
                {effectiveExecId.length > 24 ? effectiveExecId.slice(0, 12) + '…' + effectiveExecId.slice(-6) : effectiveExecId}
              </span>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white rounded transition-colors"
          >
            Close
          </button>
        </div>

        {/* Main content: split pane */}
        <div className="flex flex-1 min-h-0 overflow-hidden">
          {/* Left: Execution Tree */}
          <div
            className="shrink-0 overflow-y-auto border-r border-gray-700 p-2"
            style={{ width: leftWidth }}
          >
            <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-medium px-1">
              Execution Tree
            </div>
            <ExecutionLineageTree
              executionIds={availableExecIds}
              activeExecutionId={effectiveExecId}
              getSpanTree={getSpanTree}
              onSelect={handleSelect}
            />
          </div>

          {/* Drag handle */}
          <div
            className={`w-1 shrink-0 cursor-col-resize hover:bg-blue-500/50 transition-colors ${dragging ? 'bg-blue-500/50' : 'bg-transparent'}`}
            onMouseDown={handleMouseDown}
          />

          {/* Right: Waterfall + Metrics */}
          <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
            {/* Waterfall (takes most space) */}
            <div className="flex-1 min-h-0 overflow-auto p-2">
              {effectiveExecId ? (
                <TimelineWaterfall executionId={effectiveExecId} />
              ) : (
                <div className="text-gray-500 text-center py-12 text-xs">
                  Select an execution from the tree to view spans
                </div>
              )}
            </div>

            {/* Metrics (bottom strip) */}
            <div className="shrink-0 border-t border-gray-700 p-2 max-h-[200px] overflow-y-auto">
              <MetricsPanel executionId={effectiveExecId || undefined} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
