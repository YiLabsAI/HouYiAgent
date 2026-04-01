import React, { useState } from 'react';
import { useResearchStore } from '@/stores/useResearchStore';
import { PlanEditor } from './PlanEditor';
import { ProgressPanel } from './ProgressPanel';
import { ReportViewer } from './ReportViewer';
import { Search, Loader2 } from 'lucide-react';

export const DeepResearchWorkspace: React.FC = () => {
  const { phase, sessionId, plan, progress, report, error, loading, createSession, confirmAndExecute, cancelSession, reset } = useResearchStore();
  const [query, setQuery] = useState('');

  React.useEffect(() => {
    if (sessionId) {
      window.history.replaceState(null, '', `#/research/${sessionId}`);
    }
  }, [sessionId]);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    await createSession(query.trim());
  };

  return (
    <div className="flex-1 overflow-y-auto bg-gray-900 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header with back */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-100">Deep Research</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {phase === 'input' && 'Enter your research topic to begin'}
              {phase === 'planning' && 'Review and edit the research plan'}
              {phase === 'executing' && 'Research in progress...'}
              {phase === 'report' && 'Research complete'}
            </p>
          </div>
          {phase !== 'input' && (
            <button
              type="button"
              onClick={reset}
              className="text-xs text-gray-500 hover:text-gray-300 px-3 py-1.5 rounded border border-gray-700 hover:border-gray-600 transition-colors"
            >
              New Research
            </button>
          )}
        </div>

        {error && (
          <div className="px-4 py-3 rounded-lg bg-red-900/30 border border-red-700/50 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Phase: Input */}
        {phase === 'input' && (
          <div className="space-y-4">
            <div className="relative">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to research? Be specific for better results..."
                className="w-full h-32 px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit();
                }}
              />
            </div>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!query.trim() || loading}
                className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
                Start Research
              </button>
            </div>
          </div>
        )}

        {/* Phase: Planning */}
        {phase === 'planning' && plan && (
          <PlanEditor
            plan={plan}
            onConfirm={confirmAndExecute}
            loading={loading}
          />
        )}

        {/* Phase: Executing */}
        {phase === 'executing' && (
          <ProgressPanel
            progress={progress}
            events={useResearchStore.getState().events}
            onCancel={cancelSession}
            error={error}
          />
        )}

        {/* Phase: Report */}
        {phase === 'report' && report && (
          <ReportViewer report={report} plan={plan} />
        )}
      </div>
    </div>
  );
};
