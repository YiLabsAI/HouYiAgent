import React, { useState } from 'react';
import { useResearchStore } from '@/stores/useResearchStore';
import { PlanEditor } from './PlanEditor';
import { ProgressPanel } from './ProgressPanel';
import { ReportViewer } from './ReportViewer';
import { Search, Loader2 } from 'lucide-react';

/** Matches backend `ResearchDepth`: drives planner (sub-question count), search rounds, and report pipeline. */
export type ResearchDepthOption = 'quick' | 'standard' | 'deep';

const DEPTH_CARDS: Array<{
  value: ResearchDepthOption;
  title: string;
  subtitle: string;
  meta: string;
}> = [
  {
    value: 'quick',
    title: 'Quick',
    subtitle: 'Fast answer, lighter validation',
    meta: '~2-3 sub-questions',
  },
  {
    value: 'standard',
    title: 'Standard',
    subtitle: 'Balanced speed and coverage',
    meta: '~4-6 sub-questions',
  },
  {
    value: 'deep',
    title: 'Deep',
    subtitle: 'Broad exploration with strict checks',
    meta: '~6-9 sub-questions',
  },
];

export const DeepResearchWorkspace: React.FC = () => {
  const { phase, sessionId, plan, progress, report, searchResults, sessions, error, loading, createSession, confirmAndExecute, cancelSession, reset, events } = useResearchStore();
  const [query, setQuery] = useState('');
  const [depth, setDepth] = useState<ResearchDepthOption>('standard');
  const prevPhaseRef = React.useRef(phase);
  const [justCompleted, setJustCompleted] = React.useState(false);
  const showErrorBanner = !!error && phase !== 'executing' && !loading;

  React.useEffect(() => {
    if (prevPhaseRef.current === 'executing' && phase === 'report') {
      setJustCompleted(true);
    } else if (phase !== 'report') {
      setJustCompleted(false);
    }
    prevPhaseRef.current = phase;
  }, [phase]);

  React.useEffect(() => {
    if (sessionId) {
      window.history.replaceState(null, '', `#/research/${sessionId}`);
    }
  }, [sessionId]);

  React.useEffect(() => {
    if (phase !== 'input' || !sessionId || query.trim()) {
      return;
    }
    const session = sessions.find((s) => s.run_id === sessionId);
    if (session?.query) {
      setQuery(session.query);
    }
  }, [phase, sessionId, query, sessions]);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    await createSession(query.trim(), { depth });
  };

  return (
    <div className="flex-1 min-h-0 overflow-y-auto bg-gray-900 p-6">
      <div className="max-w-4xl mx-auto space-y-6 pb-8">
        {/* Header with back */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-100">Deep Research</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {phase === 'input' && 'Enter your research topic to begin'}
              {phase === 'planning' && 'Review and edit the research plan'}
              {phase === 'executing' && 'Research in progress...'}
              {phase === 'report' && !error && 'Research complete'}
              {phase === 'report' && error && 'Research incomplete — partial results available'}
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

        {showErrorBanner && (
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
            <div className="space-y-2">
              <p className="text-xs text-gray-400">Research depth</p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                {DEPTH_CARDS.map((item) => {
                  const active = item.value === depth;
                  return (
                    <button
                      key={item.value}
                      type="button"
                      onClick={() => setDepth(item.value)}
                      aria-pressed={active}
                      className={`text-left rounded-xl border px-3 py-2.5 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500/70 ${active
                        ? 'border-purple-500 bg-purple-500/10 shadow-[0_0_0_1px_rgba(168,85,247,0.35)]'
                        : 'border-gray-700 bg-gray-800/70 hover:border-gray-500'
                      }`}
                    >
                      <div className={`text-sm font-medium ${active ? 'text-purple-300' : 'text-gray-200'}`}>
                        {item.title}
                      </div>
                      <div className={`mt-0.5 text-[11px] ${active ? 'text-purple-200/90' : 'text-gray-400'}`}>
                        {item.subtitle}
                      </div>
                      <div className={`mt-1 text-[10px] uppercase tracking-wide ${active ? 'text-purple-300/80' : 'text-gray-500'}`}>
                        {item.meta}
                      </div>
                    </button>
                  );
                })}
              </div>
              <p className="text-[11px] text-gray-500 leading-relaxed">
                Depth controls planning breadth and validation intensity. You can still edit the plan before running.
              </p>
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

        {phase === 'planning' && !plan && !loading && (
          <div className="rounded-xl border border-amber-700/50 bg-amber-900/20 p-6 space-y-3">
            <p className="text-sm text-amber-200">Plan is unavailable for this session.</p>
            <p className="text-xs text-amber-300/80">
              This usually means the run was interrupted before plan generation completed.
            </p>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={reset}
                className="px-4 py-2 text-xs rounded border border-amber-600/60 text-amber-200 hover:bg-amber-700/20 transition-colors"
              >
                Start New Research
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

        {phase === 'planning' && !plan && loading && (
          <div className="rounded-xl border border-gray-700/50 bg-gray-800/30 p-6">
            <div className="flex items-center gap-3">
              <Loader2 size={16} className="text-purple-400 animate-spin" />
              <div className="space-y-1">
                <p className="text-sm text-gray-200">Generating research plan...</p>
                <p className="text-xs text-gray-500">
                  Analyzing the query and decomposing it into sub-questions.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Phase: Executing */}
        {phase === 'executing' && (
          <ProgressPanel
            progress={progress}
            events={events}
            subQuestions={plan?.sub_questions}
            onCancel={cancelSession}
            error={error}
          />
        )}

        {/* Phase: Report */}
        {phase === 'report' && report && (
          <ReportViewer
            report={report}
            plan={plan}
            animate={justCompleted}
            onRetry={() => {
              if (plan) confirmAndExecute();
            }}
          />
        )}

        {/* Partial results for failed sessions (no report but have search data) */}
        {phase === 'report' && !report && searchResults && searchResults.length > 0 && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-300">
                Partial Search Results ({searchResults.length} sub-questions completed)
              </h3>
              <button
                type="button"
                onClick={() => { if (plan) confirmAndExecute(); }}
                className="text-xs px-3 py-1.5 rounded border border-purple-600 text-purple-400 hover:bg-purple-600/20 transition-colors"
              >
                Retry Research
              </button>
            </div>
            {searchResults.map((sr) => {
              const question = plan?.sub_questions?.find(q => q.question_id === sr.question_id);
              return (
                <div key={sr.question_id} className="bg-gray-800/60 rounded-lg border border-gray-700/50 p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-gray-200">
                      {question?.question || sr.question_id}
                    </h4>
                    <span className="text-xs text-gray-500">
                      {sr.sources?.length || 0} sources
                    </span>
                  </div>
                  {sr.summary && (
                    <p className="text-xs text-gray-400 leading-relaxed">{sr.summary}</p>
                  )}
                  {sr.sources && sr.sources.length > 0 && (
                    <ul className="space-y-1 mt-2">
                      {sr.sources.slice(0, 5).map((s, i) => (
                        <li key={i} className="text-xs text-gray-500 flex items-start gap-1.5">
                          <span className="text-gray-600 mt-0.5">•</span>
                          <a
                            href={s.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400/70 hover:text-blue-300 truncate"
                          >
                            {s.title || s.url}
                          </a>
                        </li>
                      ))}
                      {sr.sources.length > 5 && (
                        <li className="text-xs text-gray-600 pl-3">
                          +{sr.sources.length - 5} more sources
                        </li>
                      )}
                    </ul>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
