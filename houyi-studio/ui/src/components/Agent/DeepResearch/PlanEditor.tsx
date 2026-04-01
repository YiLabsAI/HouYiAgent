import React, { useState } from 'react';
import type { ResearchPlan } from '@/stores/useResearchStore';
import { useResearchStore } from '@/stores/useResearchStore';
import { Plus, Trash2, Play, GripVertical, ChevronUp, ChevronDown } from 'lucide-react';

interface Props {
  plan: ResearchPlan;
  onConfirm: () => void;
  loading: boolean;
}

export const PlanEditor: React.FC<Props> = ({ plan, onConfirm, loading }) => {
  const { editPlan } = useResearchStore();
  const [newQuestion, setNewQuestion] = useState('');

  const handleAdd = async () => {
    if (!newQuestion.trim()) return;
    await editPlan([{ op: 'add', target_question: newQuestion.trim() }]);
    setNewQuestion('');
  };

  const handleDelete = async (questionId: string) => {
    await editPlan([{ op: 'delete', question_id: questionId }]);
  };

  const handleUpdate = async (questionId: string, text: string) => {
    await editPlan([{ op: 'update', question_id: questionId, target_question: text }]);
  };

  const handlePriority = async (questionId: string, priority: number) => {
    await editPlan([{ op: 'set_priority', question_id: questionId, new_priority: Math.max(1, Math.min(5, priority)) }]);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">
          Research Plan <span className="text-gray-600 font-normal">v{plan.version}</span>
        </h3>
        <span className="text-xs text-gray-500">{plan.sub_questions.length} sub-questions</span>
      </div>

      {/* Query display */}
      <div className="px-4 py-3 rounded-lg bg-gray-800/60 border border-gray-700/50">
        <span className="text-xs text-gray-500 uppercase tracking-wider">Topic</span>
        <p className="text-sm text-gray-200 mt-1">{plan.query}</p>
      </div>

      {/* Sub-questions */}
      <div className="space-y-2">
        {plan.sub_questions.map((sq, idx) => (
          <div
            key={sq.question_id}
            className="group flex items-start gap-3 px-4 py-3 rounded-lg bg-gray-800/40 border border-gray-700/50 hover:border-gray-600 transition-colors"
          >
            <div className="flex items-center gap-1 pt-0.5 text-gray-600">
              <GripVertical size={14} />
              <span className="text-xs w-4 text-center">{idx + 1}</span>
            </div>

            <div className="flex-1 min-w-0">
              <input
                type="text"
                defaultValue={sq.question}
                onBlur={(e) => {
                  if (e.target.value !== sq.question) handleUpdate(sq.question_id, e.target.value);
                }}
                className="w-full bg-transparent text-sm text-gray-200 border-none outline-none focus:ring-0"
              />
            </div>

            <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" title="Priority (1–5): higher = searched first">
              <button type="button" onClick={() => handlePriority(sq.question_id, sq.priority + 1)} className="p-1 text-gray-500 hover:text-gray-300" title="Increase priority">
                <ChevronUp size={14} />
              </button>
              <span className="text-xs text-gray-500 w-3 text-center" title={`Priority ${sq.priority}`}>{sq.priority}</span>
              <button type="button" onClick={() => handlePriority(sq.question_id, sq.priority - 1)} className="p-1 text-gray-500 hover:text-gray-300" title="Decrease priority">
                <ChevronDown size={14} />
              </button>
              <button type="button" onClick={() => handleDelete(sq.question_id)} className="p-1 text-gray-500 hover:text-red-400 ml-1" title="Remove">
                <Trash2 size={14} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Add new */}
      <div className="flex gap-2">
        <input
          type="text"
          value={newQuestion}
          onChange={(e) => setNewQuestion(e.target.value)}
          placeholder="Add a sub-question..."
          className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-purple-500"
          onKeyDown={(e) => { if (e.key === 'Enter') handleAdd(); }}
        />
        <button
          type="button"
          onClick={handleAdd}
          disabled={!newQuestion.trim()}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-400 hover:text-gray-200 border border-gray-700 hover:border-gray-600 rounded-lg disabled:opacity-40 transition-colors"
        >
          <Plus size={14} /> Add
        </button>
      </div>

      {/* Outline preview */}
      {plan.outline.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Report Outline</h4>
          <div className="space-y-1">
            {plan.outline.map((sec, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-gray-400 px-3 py-1.5 rounded bg-gray-800/30">
                <span className="text-gray-600">{i + 1}.</span>
                <span>{sec.title}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Confirm */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onConfirm}
          disabled={loading || plan.sub_questions.length === 0}
          className="flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <Play size={16} />
          Confirm &amp; Execute
        </button>
      </div>
    </div>
  );
};
