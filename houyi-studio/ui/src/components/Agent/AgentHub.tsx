/**
 * AgentHub — landing page for the Agent workspace mode.
 *
 * Renders agent capability cards from the server Agent Types API,
 * recent research sessions from the Research API, and handles
 * navigation to the Deep Research workspace and Memory Inbox.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { AgentCard } from './AgentCard';
import { DeepResearchWorkspace } from './DeepResearch/Workspace';
import { MemoryInbox } from '../Memory/MemoryInbox';
import { useResearchStore } from '@/stores/useResearchStore';
import { ArrowLeft, Brain, Trash2 } from 'lucide-react';

type AgentView = 'hub' | 'deep_research' | 'memory_inbox';

const PAGE_SIZE = 10;

function pushHash(path: string) {
  const target = path ? `#${path}` : window.location.pathname;
  if (window.location.hash === (path ? `#${path}` : '')) return;
  window.history.pushState(null, '', target);
}

function parseHash(): { view: AgentView; sessionId?: string } {
  const hash = window.location.hash.replace(/^#/, '');
  const sessionMatch = /^\/research\/([a-zA-Z0-9_-]+)/.exec(hash);
  if (sessionMatch) return { view: 'deep_research', sessionId: sessionMatch[1] };
  if (hash === '/research') return { view: 'deep_research' };
  if (hash === '/memory') return { view: 'memory_inbox' };
  return { view: 'hub' };
}

interface AgentType {
  id: string;
  name: string;
  description: string;
  icon: string;
  available: boolean;
}

const statusColors: Record<string, string> = {
  executing: 'bg-blue-500 animate-pulse',
  generating_report: 'bg-blue-500 animate-pulse',
  completed: 'bg-green-500',
  failed: 'bg-red-500',
  cancelled: 'bg-gray-500',
  planning: 'bg-yellow-500',
  plan_ready: 'bg-yellow-500',
  draft: 'bg-yellow-500',
};

const statusLabels: Record<string, string> = {
  executing: 'Running',
  generating_report: 'Writing report',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  planning: 'Planning',
  plan_ready: 'Plan ready',
  draft: 'Draft',
};

export const AgentHub: React.FC = () => {
  const [view, setView] = useState<AgentView>('hub');
  const [agentTypes, setAgentTypes] = useState<AgentType[]>([]);
  const [page, setPage] = useState(0);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const { sessions, fetchSessions, openSession, deleteSession, reset } = useResearchStore();
  const initializedRef = React.useRef(false);

  // Sync view state from hash (used on mount and on popstate)
  const syncFromHash = useCallback(async () => {
    const parsed = parseHash();
    if (parsed.view === 'deep_research' && parsed.sessionId) {
      await openSession(parsed.sessionId);
      setView('deep_research');
    } else {
      if (parsed.view === 'hub') {
        useResearchStore.getState().disconnectSSE();
        reset();
      }
      setView(parsed.view);
    }
  }, [openSession, reset]);

  useEffect(() => {
    fetch('/api/agents/types')
      .then((r) => r.json())
      .then((data) => setAgentTypes(data.types || []))
      .catch(() => {});
    fetchSessions();

    if (!initializedRef.current) {
      initializedRef.current = true;
      syncFromHash();
    }
  }, [fetchSessions, syncFromHash]);

  // Listen for browser back/forward to update view
  useEffect(() => {
    const onPopState = () => {
      syncFromHash();
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, [syncFromHash]);

  // Auto-refresh session list while any session is still executing
  useEffect(() => {
    if (view !== 'hub') return;
    const hasRunning = sessions.some((s) => s.status === 'executing' || s.status === 'generating_report');
    if (!hasRunning) return;
    const interval = setInterval(fetchSessions, 5000);
    return () => clearInterval(interval);
  }, [view, sessions, fetchSessions]);

  const navigate = (v: AgentView, sessionId?: string) => {
    setView(v);
    if (v === 'deep_research' && sessionId) {
      pushHash(`/research/${sessionId}`);
    } else if (v === 'deep_research') {
      pushHash('/research');
    } else if (v === 'memory_inbox') {
      pushHash('/memory');
    } else {
      pushHash('');
    }
  };

  const handleCardClick = (id: string) => {
    if (id === 'deep_research') {
      reset();
      navigate('deep_research');
    }
  };

  const handleOpenSession = async (sessionId: string) => {
    reset();
    await openSession(sessionId);
    navigate('deep_research', sessionId);
  };

  const handleBackToHub = () => {
    useResearchStore.getState().disconnectSSE();
    reset();
    fetchSessions();
    navigate('hub');
  };

  if (view === 'deep_research') {
    return (
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800">
          <button
            type="button"
            onClick={handleBackToHub}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            <ArrowLeft size={14} /> Agent Hub
          </button>
        </div>
        <DeepResearchWorkspace />
      </div>
    );
  }

  if (view === 'memory_inbox') {
    return (
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800">
          <button
            type="button"
            onClick={() => navigate('hub')}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            <ArrowLeft size={14} /> Agent Hub
          </button>
        </div>
        <MemoryInbox />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-gray-900 p-8">
      <div className="max-w-4xl mx-auto space-y-10">
        <div>
          <h2 className="text-xl font-semibold text-gray-100 mb-1">Agent Hub</h2>
          <p className="text-sm text-gray-500">Choose an agent to start a new session</p>
        </div>

        {/* Agent cards grid */}
        <div className="grid grid-cols-2 gap-4">
          {agentTypes.length > 0
            ? agentTypes.map((at) => (
                <AgentCard
                  key={at.id}
                  name={at.name}
                  description={at.description}
                  icon={at.icon}
                  active={at.available}
                  onClick={() => handleCardClick(at.id)}
                />
              ))
            : (
              <>
                <AgentCard name="Deep Research" description="Multi-agent deep research with iterative search, verification, and structured report generation." icon="🔬" active onClick={() => handleCardClick('deep_research')} />
                <AgentCard name="Code Analyst" description="Analyze codebases, find bugs, suggest improvements, and generate documentation." icon="💻" />
                <AgentCard name="Personal Office" description="Manage schedules, draft emails, organize files, and handle daily productivity workflows." icon="📋" />
                <AgentCard name="Data Analysis" description="AI-Powered Automation for every decision, connect data and produce insights." icon="📊" />
              </>
            )}
        </div>

        {/* Memory Inbox quick access */}
        <button
          type="button"
          onClick={() => navigate('memory_inbox')}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-800/40 border border-gray-700/50 hover:border-purple-500/50 transition-colors text-left group"
        >
          <Brain size={18} className="text-purple-400 shrink-0" />
          <div className="flex-1">
            <div className="text-sm text-gray-200 group-hover:text-purple-300 transition-colors">Memory Inbox</div>
            <div className="text-xs text-gray-500">Review and manage extracted memory candidates</div>
          </div>
        </button>

        {/* Recent sessions */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-400">Recent Sessions</h3>
            {sessions.length > PAGE_SIZE && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  className="text-xs text-gray-500 hover:text-gray-300 disabled:opacity-30 disabled:cursor-default transition-colors"
                >
                  Prev
                </button>
                <span className="text-xs text-gray-600">
                  {page + 1}/{Math.ceil(sessions.length / PAGE_SIZE)}
                </span>
                <button
                  type="button"
                  disabled={(page + 1) * PAGE_SIZE >= sessions.length}
                  onClick={() => setPage((p) => p + 1)}
                  className="text-xs text-gray-500 hover:text-gray-300 disabled:opacity-30 disabled:cursor-default transition-colors"
                >
                  Next
                </button>
              </div>
            )}
          </div>
          {sessions.length === 0 ? (
            <p className="text-xs text-gray-600">No recent sessions</p>
          ) : (
            <div className="space-y-2">
              {sessions.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map((session) => (
                <div
                  key={session.run_id}
                  className="flex items-center gap-2 group"
                >
                  {session.status !== 'executing' && session.status !== 'generating_report' ? (
                    pendingDeleteId === session.run_id ? (
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); deleteSession(session.run_id); setPendingDeleteId(null); }}
                          className="px-1.5 py-0.5 text-[10px] bg-red-600 hover:bg-red-500 text-white rounded transition-colors"
                        >
                          Confirm
                        </button>
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); setPendingDeleteId(null); }}
                          className="px-1.5 py-0.5 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); setPendingDeleteId(session.run_id); }}
                        className="p-1.5 text-gray-700 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all shrink-0"
                        title="Delete session"
                      >
                        <Trash2 size={13} />
                      </button>
                    )
                  ) : (
                    <div className="w-[25px]" />
                  )}
                  <button
                    type="button"
                    onClick={() => handleOpenSession(session.run_id)}
                    className="flex-1 flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-800/40 border border-gray-700/50 hover:border-gray-600 transition-colors text-left min-w-0"
                  >
                    <div
                      className={`w-2 h-2 rounded-full shrink-0 ${statusColors[session.status] || 'bg-gray-500'}`}
                      title={statusLabels[session.status] || session.status}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-gray-200 truncate">{session.query || `Run ${session.run_id.slice(0, 8)}`}</div>
                      <div className="text-xs text-gray-500">Deep Research • {statusLabels[session.status] || session.status}</div>
                    </div>
                    {session.created_at && (
                      <span className="text-xs text-gray-600 shrink-0">{session.created_at}</span>
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
