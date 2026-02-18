/**
 * Skill Configuration Dialog.
 *
 * Allows users to adjust runtime settings for a skill:
 * - Policy action (Allow / Require Consent / Deny)
 * - Auto-invoke toggle (whether LLM can auto-trigger)
 *
 * Changes are sent via WebSocket and applied at runtime.
 * They do not modify the SKILL.md file.
 */
import React, { useState, useEffect } from 'react';
import { Settings } from 'lucide-react';
import type { SkillDetail } from '../../types/websocket';

export interface SkillConfigValues {
  policy_action: string;
  auto_invoke: boolean;
}

export interface SkillConfigDialogProps {
  isOpen: boolean;
  detail: SkillDetail;
  onSave: (config: SkillConfigValues) => void;
  onCancel: () => void;
}

const POLICY_OPTIONS = [
  {
    value: 'allow',
    label: 'Allow',
    description: 'LLM can invoke freely without user approval',
    color: 'text-green-400',
  },
  {
    value: 'allow_with_consent',
    label: 'Require Consent',
    description: 'User must approve each invocation before execution',
    color: 'text-yellow-400',
  },
  {
    value: 'deny',
    label: 'Deny',
    description: 'Skill is blocked from invocation entirely',
    color: 'text-red-400',
  },
] as const;

export const SkillConfigDialog: React.FC<SkillConfigDialogProps> = ({
  isOpen,
  detail,
  onSave,
  onCancel,
}) => {
  const [policyAction, setPolicyAction] = useState(
    detail.policy?.default_action ?? 'allow',
  );
  const [autoInvoke, setAutoInvoke] = useState(true);

  // Sync when detail changes or dialog opens
  useEffect(() => {
    if (isOpen && detail) {
      setPolicyAction(detail.policy?.default_action ?? 'allow');
      setAutoInvoke(detail.policy?.model_auto_invoke !== false);
    }
  }, [isOpen, detail]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave({ policy_action: policyAction, auto_invoke: autoInvoke });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onCancel();
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSave();
  };

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/60 z-[60]" onClick={onCancel} />

      {/* Dialog */}
      <div
        className="fixed inset-0 z-[60] flex items-center justify-center p-4"
        onKeyDown={handleKeyDown}
        role="dialog"
        aria-modal="true"
        aria-labelledby="skill-config-title"
      >
        <div
          className="bg-gray-800 border border-gray-700 rounded-xl shadow-2xl w-[440px] max-w-[90vw]"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center gap-2 px-5 py-4 border-b border-gray-700">
            <Settings size={18} className="text-gray-400" />
            <div>
              <h3 id="skill-config-title" className="text-sm font-semibold text-gray-200">
                Configure Skill
              </h3>
              <p className="text-[11px] text-gray-500 mt-0.5">
                {detail.display_name || detail.name}
                {detail.version && ` v${detail.version}`}
              </p>
            </div>
          </div>

          {/* Body */}
          <div className="px-5 py-4 space-y-5">
            {/* Policy Action */}
            <div>
              <label className="block text-[11px] font-medium text-gray-400 mb-2 uppercase tracking-wide">
                Invocation Policy
              </label>
              <div className="space-y-1.5">
                {POLICY_OPTIONS.map((opt) => (
                  <label
                    key={opt.value}
                    className={`flex items-start gap-3 p-2.5 rounded-lg border cursor-pointer transition-colors ${
                      policyAction === opt.value
                        ? 'border-blue-500/50 bg-blue-900/20'
                        : 'border-gray-700 hover:border-gray-600 bg-gray-900/30'
                    }`}
                  >
                    <input
                      type="radio"
                      name="policy_action"
                      value={opt.value}
                      checked={policyAction === opt.value}
                      onChange={() => {
                        setPolicyAction(opt.value);
                        setAutoInvoke(opt.value !== 'deny');
                      }}
                      className="mt-0.5 accent-blue-500"
                    />
                    <div>
                      <span className={`text-[12px] font-medium ${opt.color}`}>
                        {opt.label}
                      </span>
                      <p className="text-[11px] text-gray-500 mt-0.5">{opt.description}</p>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Auto-invoke Toggle */}
            <div>
              <label className="block text-[11px] font-medium text-gray-400 mb-2 uppercase tracking-wide">
                Model Auto-invoke
              </label>
              <label className="flex items-center justify-between p-2.5 rounded-lg border border-gray-700 bg-gray-900/30 cursor-pointer hover:border-gray-600 transition-colors">
                <div>
                  <span className="text-[12px] font-medium text-gray-200">
                    Allow LLM to trigger automatically
                  </span>
                  <p className="text-[11px] text-gray-500 mt-0.5">
                    When disabled, skill can only be invoked manually via dry-run
                  </p>
                </div>
                <div
                  className={`relative w-9 h-5 rounded-full transition-colors ${
                    autoInvoke ? 'bg-blue-600' : 'bg-gray-600'
                  }`}
                  onClick={() => {
                    const next = !autoInvoke;
                    setAutoInvoke(next);
                    if (!next && policyAction === 'allow') {
                      setPolicyAction('deny');
                    } else if (next && policyAction === 'deny') {
                      setPolicyAction('allow');
                    }
                  }}
                >
                  <div
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                      autoInvoke ? 'translate-x-4' : 'translate-x-0.5'
                    }`}
                  />
                </div>
              </label>
            </div>

            {/* Info box */}
            <div className="text-[11px] text-gray-500 bg-gray-900/40 rounded-lg p-2.5 border border-gray-700/50">
              Changes apply at runtime and do not modify the SKILL.md file.
              Restart the server to reset to defaults.
            </div>
          </div>

          {/* Footer */}
          <div className="px-5 py-3 border-t border-gray-700 flex justify-end gap-2">
            <button
              onClick={onCancel}
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-[12px] text-gray-300 transition-colors"
              type="button"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-[12px] text-white transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800"
              type="button"
            >
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </>
  );
};
