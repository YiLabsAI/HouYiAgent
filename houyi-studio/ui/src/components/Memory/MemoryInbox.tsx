import React, { useEffect, useState, useCallback } from 'react';
import { useMemoryStore } from '@/stores/useMemoryStore';
import type { MemoryCandidate, MemoryRecord } from '@/stores/useMemoryStore';
import { Check, X, Pencil, Tag, Brain, Database, Trash2 } from 'lucide-react';

type TabView = 'candidates' | 'records';

const FILTER_TABS = [
  { key: 'pending' as const, label: 'Pending' },
  { key: 'approved' as const, label: 'Approved' },
  { key: 'rejected' as const, label: 'Rejected' },
  { key: 'all' as const, label: 'All' },
];

const typeBadgeClass: Record<string, string> = {
  fact: 'bg-blue-900/40 text-blue-300 border-blue-700/50',
  preference: 'bg-purple-900/40 text-purple-300 border-purple-700/50',
  constraint: 'bg-orange-900/40 text-orange-300 border-orange-700/50',
  profile: 'bg-green-900/40 text-green-300 border-green-700/50',
  episodic: 'bg-yellow-900/40 text-yellow-300 border-yellow-700/50',
};

const statusBadgeClass: Record<string, string> = {
  pending: 'bg-yellow-900/40 text-yellow-300 border-yellow-700/50',
  approved: 'bg-green-900/40 text-green-300 border-green-700/50',
  rejected: 'bg-red-900/40 text-red-300 border-red-700/50',
  merged: 'bg-blue-900/40 text-blue-300 border-blue-700/50',
};

function formatSource(ctx: string): string {
  if (!ctx) return '';
  if (ctx.startsWith('turn:')) return `turn ${ctx.slice(5)}`;
  if (ctx.length > 20) return ctx.slice(0, 18) + '…';
  return ctx;
}

const CandidateCard: React.FC<{
  candidate: MemoryCandidate;
  onApprove: () => void;
  onReject: () => void;
  onEdit: (content: string) => void;
}> = ({ candidate, onApprove, onReject, onEdit }) => {
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(candidate.content);
  const [confirmReject, setConfirmReject] = useState(false);

  const handleSave = () => {
    onEdit(editContent);
    setEditing(false);
  };

  const handleReject = () => {
    onReject();
    setConfirmReject(false);
  };

  return (
    <div className="px-4 py-3 rounded-lg bg-gray-800/40 border border-gray-700/50 hover:border-gray-600 transition-colors group">
      <div className="flex items-start gap-3">
        <Brain size={16} className="text-purple-400 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0">
          {editing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-20 px-3 py-2 bg-gray-900 border border-gray-600 rounded text-xs text-gray-200 focus:outline-none focus:border-purple-500 resize-none"
              />
              <div className="flex gap-2">
                <button type="button" onClick={handleSave} className="text-xs px-2.5 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded transition-colors">Save</button>
                <button type="button" onClick={() => setEditing(false)} className="text-xs px-2.5 py-1 text-gray-400 hover:text-gray-200 border border-gray-700 rounded transition-colors">Cancel</button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-200 whitespace-pre-wrap">{candidate.content}</p>
          )}

          <div className="flex items-center flex-wrap gap-x-2 gap-y-1 mt-2">
            <span className={`text-[10px] px-1.5 py-0.5 rounded border ${statusBadgeClass[candidate.status] || 'text-gray-400 border-gray-700'}`}>
              {candidate.status}
            </span>
            {candidate.memory_type && (
              <span className={`text-[10px] px-1.5 py-0.5 rounded border ${typeBadgeClass[candidate.memory_type] || 'text-gray-400 border-gray-700'}`}>
                {candidate.memory_type}
              </span>
            )}
            {candidate.source_context && (
              <span className="text-[10px] text-gray-600" title={candidate.source_context}>
                {formatSource(candidate.source_context)}
              </span>
            )}
            <span className="text-[10px] text-gray-600" title="Extraction confidence score">
              {(candidate.confidence * 100).toFixed(0)}%
            </span>
            {candidate.suggested_tags.length > 0 && (
              <div className="flex items-center gap-1">
                <Tag size={10} className="text-gray-600" />
                {candidate.suggested_tags.map((t) => (
                  <span key={t} className="text-[10px] text-gray-500">{t}</span>
                ))}
              </div>
            )}
          </div>
        </div>

        {candidate.status === 'pending' && (
          <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
            <button type="button" onClick={onApprove} className="p-1.5 rounded hover:bg-green-900/40 text-gray-500 hover:text-green-400 transition-colors" title="Approve">
              <Check size={14} />
            </button>
            {confirmReject ? (
              <div className="flex items-center gap-1">
                <button type="button" onClick={handleReject} className="px-1.5 py-0.5 text-[10px] bg-red-600 hover:bg-red-500 text-white rounded transition-colors">Reject</button>
                <button type="button" onClick={() => setConfirmReject(false)} className="px-1.5 py-0.5 text-[10px] text-gray-500 hover:text-gray-300 transition-colors">Cancel</button>
              </div>
            ) : (
              <button type="button" onClick={() => setConfirmReject(true)} className="p-1.5 rounded hover:bg-red-900/40 text-gray-500 hover:text-red-400 transition-colors" title="Reject">
                <X size={14} />
              </button>
            )}
            <button type="button" onClick={() => setEditing(true)} className="p-1.5 rounded hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors" title="Edit">
              <Pencil size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

const RecordCard: React.FC<{
  record: MemoryRecord;
  onUpdate: (content: string, tags?: string[]) => void;
  onDelete: () => void;
}> = ({ record, onUpdate, onDelete }) => {
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(record.content);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const handleSave = () => {
    onUpdate(editContent);
    setEditing(false);
  };

  return (
    <div className="px-4 py-3 rounded-lg bg-gray-800/40 border border-gray-700/50 hover:border-gray-600 transition-colors group">
      <div className="flex items-start gap-3">
        <Database size={16} className="text-blue-400 mt-0.5 shrink-0" />
        <div className="flex-1 min-w-0">
          {editing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-20 px-3 py-2 bg-gray-900 border border-gray-600 rounded text-xs text-gray-200 focus:outline-none focus:border-purple-500 resize-none"
              />
              <div className="flex gap-2">
                <button type="button" onClick={handleSave} className="text-xs px-2.5 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded transition-colors">Save</button>
                <button type="button" onClick={() => { setEditing(false); setEditContent(record.content); }} className="text-xs px-2.5 py-1 text-gray-400 hover:text-gray-200 border border-gray-700 rounded transition-colors">Cancel</button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-200 whitespace-pre-wrap">{record.content}</p>
          )}

          <div className="flex items-center flex-wrap gap-x-2 gap-y-1 mt-2">
            {record.memory_type && (
              <span className={`text-[10px] px-1.5 py-0.5 rounded border ${typeBadgeClass[record.memory_type] || 'text-gray-400 border-gray-700'}`}>
                {record.memory_type}
              </span>
            )}
            <span className="text-[10px] text-gray-600">{record.scope}</span>
            {record.confidence != null && (
              <span className="text-[10px] text-gray-600" title="Confidence score">
                {(record.confidence * 100).toFixed(0)}%
              </span>
            )}
            {record.tags && record.tags.length > 0 && (
              <div className="flex items-center gap-1">
                <Tag size={10} className="text-gray-600" />
                {record.tags.map((t) => (
                  <span key={t} className="text-[10px] text-gray-500">{t}</span>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
          <button type="button" onClick={() => setEditing(true)} className="p-1.5 rounded hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors" title="Edit">
            <Pencil size={14} />
          </button>
          {confirmDelete ? (
            <div className="flex items-center gap-1">
              <button type="button" onClick={onDelete} className="px-1.5 py-0.5 text-[10px] bg-red-600 hover:bg-red-500 text-white rounded transition-colors">Confirm</button>
              <button type="button" onClick={() => setConfirmDelete(false)} className="px-1.5 py-0.5 text-[10px] text-gray-500 hover:text-gray-300 transition-colors">Cancel</button>
            </div>
          ) : (
            <button type="button" onClick={() => setConfirmDelete(true)} className="p-1.5 rounded hover:bg-red-900/40 text-gray-500 hover:text-red-400 transition-colors" title="Delete">
              <Trash2 size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export const MemoryInbox: React.FC = () => {
  const {
    candidates, records, filter, loading, error,
    setFilter, fetchCandidates, fetchRecords, fetchConfig,
    approveCandidate, rejectCandidate, updateCandidate,
    updateRecord, deleteRecord,
  } = useMemoryStore();
  const [tab, setTab] = useState<TabView>('candidates');

  useEffect(() => {
    fetchCandidates();
    fetchRecords();
    fetchConfig();
  }, [fetchCandidates, fetchRecords, fetchConfig]);

  const handleApprove = useCallback((id: string) => {
    approveCandidate(id);
  }, [approveCandidate]);

  const handleReject = useCallback((id: string) => {
    rejectCandidate(id);
  }, [rejectCandidate]);

  return (
    <div className="flex-1 overflow-y-auto bg-gray-900 p-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <div>
          <h2 className="text-lg font-semibold text-gray-100">Memory</h2>
          <p className="text-xs text-gray-500 mt-0.5">Review candidates and manage stored memories</p>
        </div>

        {error && (
          <div className="px-4 py-3 rounded-lg bg-red-900/30 border border-red-700/50 text-sm text-red-300">{error}</div>
        )}

        <div className="flex gap-1 p-1 bg-gray-800/60 rounded-lg w-fit">
          <button
            type="button"
            onClick={() => setTab('candidates')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md transition-colors ${
              tab === 'candidates' ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <Brain size={12} /> Inbox ({candidates.length})
          </button>
          <button
            type="button"
            onClick={() => { setTab('records'); fetchRecords(); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md transition-colors ${
              tab === 'records' ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            <Database size={12} /> Records ({records.length})
          </button>
        </div>

        {tab === 'candidates' && (
          <>
            <div className="flex gap-1 p-1 bg-gray-800/60 rounded-lg w-fit">
              {FILTER_TABS.map((ft) => (
                <button
                  key={ft.key}
                  type="button"
                  onClick={() => setFilter(ft.key)}
                  className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                    filter === ft.key
                      ? 'bg-gray-700 text-gray-200'
                      : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {ft.label}
                </button>
              ))}
            </div>

            {loading ? (
              <div className="text-center py-8 text-xs text-gray-600">Loading...</div>
            ) : candidates.length === 0 ? (
              <div className="text-center py-12">
                <Brain size={32} className="text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-500">No memory candidates</p>
                <p className="text-xs text-gray-600 mt-1">Candidates appear after Deep Research or Chat conversations extract memories</p>
              </div>
            ) : (
              <div className="space-y-2">
                {candidates.map((c) => (
                  <CandidateCard
                    key={c.candidate_id}
                    candidate={c}
                    onApprove={() => handleApprove(c.candidate_id)}
                    onReject={() => handleReject(c.candidate_id)}
                    onEdit={(content) => updateCandidate(c.candidate_id, content)}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {tab === 'records' && (
          <>
            {loading ? (
              <div className="text-center py-8 text-xs text-gray-600">Loading...</div>
            ) : records.length === 0 ? (
              <div className="text-center py-12">
                <Database size={32} className="text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-500">No stored memories</p>
                <p className="text-xs text-gray-600 mt-1">Approve candidates from the Inbox to create stored memory records</p>
              </div>
            ) : (
              <div className="space-y-2">
                {records.map((r) => (
                  <RecordCard
                    key={r.record_id}
                    record={r}
                    onUpdate={(content) => updateRecord(r.record_id, content)}
                    onDelete={() => deleteRecord(r.record_id)}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
