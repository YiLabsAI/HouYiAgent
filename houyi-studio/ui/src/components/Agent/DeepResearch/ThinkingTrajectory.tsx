import React, { useMemo, useState } from 'react';
import type { SSEEvent, SubQuestion } from '@/stores/useResearchStore';
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Loader2,
  Search as SearchIcon,
  AlertCircle,
  FileText,
  ShieldCheck,
  Sparkles,
  Globe,
  ExternalLink,
} from 'lucide-react';

interface Props {
  events: SSEEvent[];
  subQuestions?: SubQuestion[];
}

interface SourceItem {
  title: string;
  url: string;
  snippet: string;
  query?: string;
}

interface SearchQueryGroup {
  round: number;
  queries: string[];
}

interface QuestionGroup {
  questionId: string;
  label: string;
  events: SSEEvent[];
  status: 'pending' | 'searching' | 'completed' | 'failed';
  sources: SourceItem[];
  searchQueries: SearchQueryGroup[];
}

function groupByQuestion(
  events: SSEEvent[],
  subQuestions?: SubQuestion[],
): { questionGroups: QuestionGroup[]; pipelineEvents: SSEEvent[]; pipelinePhase: string | null } {
  const sqMap = new Map(
    (subQuestions || []).map((sq) => [sq.question_id, sq.question]),
  );

  const groups = new Map<string, QuestionGroup>();
  const pipeline: SSEEvent[] = [];
  let pipelinePhase: string | null = null;

  for (const evt of events) {
    const qid =
      (evt.payload.question_id as string) ||
      (evt.payload.step_id as string) ||
      '';

    // Pipeline step events (report_generation) are not sub-question searches
    const isPipelineStep = qid === 'report_generation';

    const isQuestionEvent =
      qid &&
      !isPipelineStep &&
      (evt.event_type.startsWith('research.step_') ||
        evt.event_type === 'research.source_found' ||
        evt.event_type === 'research.search_queries' ||
        evt.event_type === 'research.agent_spawned' ||
        evt.event_type === 'research.agent_completed');

    if (evt.event_type === 'research.pipeline_phase') {
      pipelinePhase = (evt.payload.phase as string) || null;
    }

    if (isQuestionEvent) {
      if (!groups.has(qid)) {
        groups.set(qid, {
          questionId: qid,
          label: sqMap.get(qid) || (evt.payload.step as string) || qid,
          events: [],
          status: 'pending',
          sources: [],
          searchQueries: [],
        });
      }
      const g = groups.get(qid)!;
      g.events.push(evt);

      if (evt.event_type === 'research.step_started') g.status = 'searching';
      if (evt.event_type === 'research.step_completed') {
        g.status = (evt.payload.failed as boolean) ? 'failed' : 'completed';
      }
      if (evt.event_type === 'research.source_found') {
        g.sources.push({
          title: (evt.payload.title as string) || '',
          url: (evt.payload.url as string) || '',
          snippet: (evt.payload.snippet as string) || '',
          query: (evt.payload.query as string) || undefined,
        });
      }
      if (evt.event_type === 'research.search_queries') {
        g.searchQueries.push({
          round: (evt.payload.round as number) || 0,
          queries: (evt.payload.queries as string[]) || [],
        });
      }
      if (
        evt.event_type === 'research.agent_completed' &&
        (evt.payload.status as string) === 'failed'
      )
        g.status = 'failed';
    } else {
      pipeline.push(evt);
      if (evt.event_type === 'research.failed') {
        for (const g of groups.values()) {
          if (g.status === 'searching') g.status = 'failed';
        }
      }
    }
  }

  // Once report phase has started, all search questions must be done
  const reportPhaseStarted = pipeline.some(
    (e) =>
      e.event_type === 'research.intermediate_report' ||
      e.event_type === 'research.report_section' ||
      e.event_type === 'research.completed' ||
      e.event_type === 'research.pipeline_phase',
  );
  if (reportPhaseStarted) {
    for (const g of groups.values()) {
      if (g.status === 'searching') g.status = 'completed';
    }
  }

  return {
    questionGroups: Array.from(groups.values()),
    pipelineEvents: pipeline,
    pipelinePhase,
  };
}

const statusIcon = (status: QuestionGroup['status']) => {
  switch (status) {
    case 'searching':
      return <Loader2 size={14} className="text-purple-400 animate-spin" />;
    case 'completed':
      return <CheckCircle2 size={14} className="text-green-400" />;
    case 'failed':
      return <AlertCircle size={14} className="text-red-400" />;
    default:
      return <div className="w-3.5 h-3.5 rounded-full border border-gray-600" />;
  }
};

const pipelineIcon = (type: string) => {
  if (type.includes('plan_confirmed')) return <CheckCircle2 size={12} className="text-green-400" />;
  if (type.includes('intermediate')) return <FileText size={12} className="text-blue-400" />;
  if (type.includes('conflict')) return <AlertCircle size={12} className="text-amber-400" />;
  if (type.includes('pipeline_phase')) return <Loader2 size={12} className="text-purple-400 animate-spin" />;
  if (type.includes('validation')) return <ShieldCheck size={12} className="text-green-400" />;
  if (type.includes('quality')) return <Sparkles size={12} className="text-purple-400" />;
  if (type.includes('report_section')) return <FileText size={12} className="text-purple-400" />;
  if (type.includes('completed')) return <CheckCircle2 size={12} className="text-green-400" />;
  if (type.includes('failed') || type.includes('cancelled')) return <AlertCircle size={12} className="text-red-400" />;
  return <Loader2 size={12} className="text-gray-400 animate-spin" />;
};

const PIPELINE_PHASE_LABELS: Record<string, string> = {
  conflict_detection: 'Detecting conflicts...',
  report_generation: 'Writing report sections...',
  url_validation: 'Validating URLs...',
  validation: 'Validating sections...',
  quality_evaluation: 'Evaluating quality...',
};

const pipelineLabel = (evt: SSEEvent): string => {
  if (evt.event_type === 'research.pipeline_phase') {
    const phase = (evt.payload.phase as string) || '';
    return PIPELINE_PHASE_LABELS[phase] || phase;
  }
  if (evt.event_type === 'research.intermediate_report')
    return `Intermediate report: ${(evt.payload.question_id as string) || ''}`;
  if (evt.event_type === 'research.conflict_detected')
    return `Conflict: ${(evt.payload.agent_a as string) || ''} vs ${(evt.payload.agent_b as string) || ''}`;
  if (evt.event_type === 'research.validation_issues')
    return `Validation: ${(evt.payload.sections_flagged as number) || 0} sections flagged`;
  if (evt.event_type === 'research.quality_evaluated') return 'Quality evaluation complete';
  if (evt.event_type === 'research.report_section')
    return `Writing: ${((evt.payload.chunk as Record<string, unknown>)?.title as string) || 'section'}`;
  if (evt.event_type === 'research.completed') return 'Research completed';
  if (evt.event_type === 'research.failed') return (evt.payload.error as string) || 'Research failed';
  if (evt.event_type === 'research.query_refined')
    return `Query refined: ${(evt.payload.refined as string) || ''}`;
  if (evt.event_type === 'memory.candidate_extracted')
    return `Extracted ${(evt.payload.count as number) || 0} memory candidates`;
  return evt.event_type.replace('research.', '');
};

const QuestionGroupItem: React.FC<{ group: QuestionGroup }> = ({ group }) => {
  const [expanded, setExpanded] = useState(false);
  const sourceCount = group.sources.length;

  return (
    <div className="border border-gray-700/40 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2.5 px-3 py-2 text-left hover:bg-gray-800/40 transition-colors"
      >
        {statusIcon(group.status)}
        <span className="text-xs text-gray-300 flex-1 truncate">{group.label}</span>
        {sourceCount > 0 && (
          <span className="flex items-center gap-1 text-[10px] text-gray-500 bg-gray-800/60 px-1.5 py-0.5 rounded">
            <Globe size={10} />
            {sourceCount} sources
          </span>
        )}
        {group.status === 'searching' && sourceCount === 0 && (
          <span className="text-[10px] text-purple-400/70 animate-pulse">searching...</span>
        )}
        {expanded ? (
          <ChevronDown size={12} className="text-gray-500 shrink-0" />
        ) : (
          <ChevronRight size={12} className="text-gray-500 shrink-0" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-gray-700/30 bg-gray-900/30">
          {/* Search queries */}
          {group.searchQueries.length > 0 && (
            <div className="px-3 py-2 space-y-1.5">
              {group.searchQueries.map((qg) => (
                <div key={qg.round} className="space-y-1">
                  <div className="text-[10px] text-gray-600 uppercase tracking-wider">
                    Round {qg.round}
                  </div>
                  {qg.queries.map((q, i) => (
                    <div key={i} className="flex items-start gap-2 text-[11px]">
                      <SearchIcon size={10} className="text-purple-400/70 mt-0.5 shrink-0" />
                      <span className="text-gray-400 italic">"{q}"</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Found sources */}
          {sourceCount > 0 && (
            <div className="px-3 py-2 space-y-1 border-t border-gray-700/20">
              <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-1">
                Found {sourceCount} sources
              </div>
              {group.sources.map((src, i) => (
                <div key={i} className="flex items-start gap-2 text-[11px] group">
                  <Globe size={10} className="text-blue-400/70 mt-0.5 shrink-0" />
                  <div className="min-w-0 flex-1">
                    {src.url ? (
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-300 hover:text-purple-300 transition-colors truncate block"
                      >
                        {src.title || src.url}
                        <ExternalLink size={8} className="inline ml-1 opacity-0 group-hover:opacity-60" />
                      </a>
                    ) : (
                      <span className="text-gray-400 truncate block">{src.title || 'source'}</span>
                    )}
                    {src.snippet && (
                      <span className="text-gray-600 text-[10px] line-clamp-1 block">{src.snippet}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Fallback: show raw events if no structured data */}
          {sourceCount === 0 && group.searchQueries.length === 0 && (
            <div className="px-3 py-2 space-y-1">
              {group.events.map((evt) => (
                <div key={evt.event_id} className="flex items-start gap-2 text-[11px]">
                  <span className="text-gray-500 shrink-0 w-4 text-right">
                    #{evt.sequence}
                  </span>
                  <span className="text-gray-400">
                    {evt.event_type.replace('research.', '')}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const ThinkingTrajectory: React.FC<Props> = ({ events, subQuestions }) => {
  const { questionGroups, pipelineEvents } = useMemo(
    () => groupByQuestion(events, subQuestions),
    [events, subQuestions],
  );

  if (events.length === 0) {
    return (
      <div className="rounded-xl border border-gray-700/50 bg-gray-800/30 p-6">
        <div className="flex items-center justify-center gap-2">
          <Loader2 size={14} className="text-purple-400 animate-spin" />
          <p className="text-xs text-gray-500">Waiting for research events...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-700/50 bg-gray-800/30 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-gray-700/50 flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400">Thinking Trajectory</span>
        <span className="text-[10px] text-gray-600">{events.length} events</span>
      </div>

      <div className="p-3 space-y-2 max-h-[32rem] overflow-y-auto">
        {/* Question groups */}
        {questionGroups.length > 0 && (
          <div className="space-y-1.5">
            {questionGroups.map((g) => (
              <QuestionGroupItem key={g.questionId} group={g} />
            ))}
          </div>
        )}

        {/* Pipeline events (intermediate reports, conflicts, validation, etc.) */}
        {pipelineEvents.length > 0 && (() => {
          const visible = pipelineEvents.filter(
            (e) =>
              !(e.event_type.startsWith('research.step_') &&
                (e.payload.step_id as string) === 'report_generation'),
          );
          const lastPhaseSeq = Math.max(
            0,
            ...visible
              .filter((e) => e.event_type === 'research.pipeline_phase')
              .map((e) => e.sequence),
          );
          return visible.length > 0 ? (
            <div className="space-y-1 pt-1 border-t border-gray-700/30">
              <span className="text-[10px] text-gray-600 uppercase tracking-wider">
                Pipeline
              </span>
              {visible.map((evt) => {
                const isPastPhase =
                  evt.event_type === 'research.pipeline_phase' &&
                  evt.sequence < lastPhaseSeq;
                const icon = isPastPhase
                  ? <CheckCircle2 size={12} className="text-green-400" />
                  : pipelineIcon(evt.event_type);
                return (
                  <div key={evt.event_id} className="flex items-start gap-2 text-xs">
                    <div className="pt-0.5 shrink-0">{icon}</div>
                    <span className="text-gray-400">{pipelineLabel(evt)}</span>
                  </div>
                );
              })}
            </div>
          ) : null;
        })()}
      </div>
    </div>
  );
};
