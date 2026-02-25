/**
 * Skill detail panel shown in the right sidebar.
 *
 * Displays metadata, policy, permissions, hooks, tools, and metrics for the
 * selected skill. Provides actions for configure, dry-run, and unload.
 */
import React, { useState } from 'react';
import type { SkillDetail, SkillMetricsData } from '../../types/websocket';
import { ConfirmDialog } from '../common/ConfirmDialog';

// ─── Certification badge colors ──────────────────────────────────
const CERT_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  unverified: { bg: 'bg-gray-700', text: 'text-gray-400', label: 'Unverified' },
  bronze: { bg: 'bg-amber-900/50', text: 'text-amber-400', label: 'Bronze' },
  silver: { bg: 'bg-gray-500/30', text: 'text-gray-300', label: 'Silver' },
  gold: { bg: 'bg-yellow-900/50', text: 'text-yellow-400', label: 'Gold' },
};

// ─── Policy action display ───────────────────────────────────────
const POLICY_STYLES: Record<string, { color: string; label: string }> = {
  allow: { color: 'text-green-400', label: 'Allow' },
  allow_with_consent: { color: 'text-yellow-400', label: 'Allow with Consent' },
  deny: { color: 'text-red-400', label: 'Deny' },
};

// ─── Side effect display ─────────────────────────────────────────
const SIDE_EFFECT_STYLES: Record<string, { color: string; icon: string }> = {
  none: { color: 'text-green-400', icon: '○' },
  network: { color: 'text-yellow-400', icon: '◐' },
  filesystem: { color: 'text-orange-400', icon: '◑' },
  exec: { color: 'text-red-400', icon: '●' },
  mixed: { color: 'text-red-400', icon: '●' },
};

export interface SkillDetailPanelProps {
  detail: SkillDetail | null;
  metrics: SkillMetricsData | null;
  isLoading: boolean;
  onConfigure?: () => void;
  onDryRun?: () => void;
  onUnload?: (skillName: string) => void;
}

export const SkillDetailPanel: React.FC<SkillDetailPanelProps> = ({
  detail,
  metrics,
  isLoading,
  onConfigure,
  onDryRun,
  onUnload,
}) => {
  const [showUnloadConfirm, setShowUnloadConfirm] = useState(false);

  if (isLoading) {
    return (
      <div className="p-4 text-center text-gray-400 text-sm">
        <div className="animate-pulse">Loading skill detail...</div>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="p-4 text-center text-gray-400 text-sm">
        Select a skill from the Skills list to view its detail.
      </div>
    );
  }

  const cert = CERT_STYLES[detail.certification] ?? CERT_STYLES.unverified;
  const policyAction = detail.policy?.default_action ?? 'allow';
  const policyStyle = POLICY_STYLES[policyAction] ?? POLICY_STYLES.allow;
  const sideEffectStyle = SIDE_EFFECT_STYLES[detail.side_effect] ?? SIDE_EFFECT_STYLES.none;

  return (
    <div className="flex flex-col gap-3 p-3 text-xs">
      {/* ─── Header ─────────────────────────────────────────── */}
      <div className="flex items-start justify-between" data-testid="skill-header">
        <div>
          <div className="text-sm font-semibold text-gray-200">
            {detail.display_name || detail.name}
          </div>
          <div className="text-gray-500 mt-0.5">
            {detail.version && detail.version !== '0.0.0' && (
              <span>v{detail.version}</span>
            )}
            {detail.author && <span className="ml-2">by {detail.author}</span>}
          </div>
        </div>
        <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${cert.bg} ${cert.text}`}>
          {cert.label}
        </span>
      </div>

      {detail.description && (
        <p className="text-gray-400 leading-relaxed">{detail.description}</p>
      )}

      {/* ─── Policy ─────────────────────────────────────────── */}
      <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-policy">
        <div className="text-gray-500 font-medium mb-1">POLICY</div>
        <div className="flex items-center gap-2">
          <span className={policyStyle.color}>{policyStyle.label}</span>
          <span className="text-gray-600">|</span>
          <span className={sideEffectStyle.color}>
            {sideEffectStyle.icon} {detail.side_effect}
          </span>
        </div>
      </div>

      {/* ─── Permissions ────────────────────────────────────── */}
      {detail.permissions.length > 0 && (
        <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-permissions">
          <div className="text-gray-500 font-medium mb-1">PERMISSIONS</div>
          <div className="space-y-0.5">
            {detail.permissions.map((perm) => (
              <div key={perm.name} className="flex items-center gap-1">
                <span className={perm.is_sensitive ? 'text-red-400' : 'text-gray-400'}>
                  {perm.is_sensitive ? '[!]' : '[ ]'}
                </span>
                <span className="text-gray-300">{perm.name}</span>
                {perm.description && (
                  <span className="text-gray-600 truncate">{perm.description}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─── Hooks ──────────────────────────────────────────── */}
      {detail.hooks.length > 0 && (
        <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-hooks">
          <div className="text-gray-500 font-medium mb-1">HOOKS</div>
          <div className="flex flex-wrap gap-1">
            {detail.hooks.map((hook) => (
              <span
                key={hook}
                className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-300 text-[10px]"
              >
                {hook}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ─── Tools ──────────────────────────────────────────── */}
      <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-tools">
        <div className="text-gray-500 font-medium mb-1">
          TOOLS ({detail.tools.length})
        </div>
        {detail.tools.length === 0 ? (
          <div className="text-gray-600">No tools defined</div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {detail.tools.map((tool) => (
              <span
                key={tool.name}
                className="px-1.5 py-0.5 bg-gray-700 rounded text-gray-300 text-[10px]"
                title={tool.description ? `${tool.name}: ${tool.description}` : tool.name}
              >
                {tool.name}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* ─── Metrics ────────────────────────────────────────── */}
      <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-metrics">
        <div className="text-gray-500 font-medium mb-1">METRICS</div>
        {metrics && metrics.total_calls > 0 ? (
          <div className="grid grid-cols-3 gap-2">
            <div>
              <div className="text-gray-500">Calls</div>
              <div className="text-gray-200 font-medium">{metrics.total_calls}</div>
            </div>
            <div>
              <div className="text-gray-500">Success</div>
              <div className="text-green-400 font-medium">
                {(metrics.success_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-gray-500">Avg Latency</div>
              <div className="text-gray-200 font-medium">{metrics.avg_latency_ms}ms</div>
            </div>
          </div>
        ) : (
          <div className="text-[11px] text-gray-600">
            No invocations yet. Metrics are collected from Chat tool calls, not Dry-run.
          </div>
        )}
      </div>

      {/* ─── Action bar ─────────────────────────────────────── */}
      <div className="flex items-center gap-2 pt-1 border-t border-gray-700" data-testid="skill-actions">
        <button
          type="button"
          onClick={onConfigure}
          className="flex-1 px-2 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 text-[11px] rounded transition-colors"
        >
          Configure...
        </button>
        <button
          type="button"
          onClick={() => onDryRun?.()}
          className="flex-1 px-2 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[11px] rounded transition-colors"
          data-testid="skill-dry-run-button"
        >
          Dry-run
        </button>
        <button
          type="button"
          onClick={() => setShowUnloadConfirm(true)}
          className="px-2 py-1.5 bg-red-900/50 hover:bg-red-800/60 text-red-300 text-[11px] rounded transition-colors"
        >
          Unload
        </button>
      </div>

      {/* ─── Unload confirmation dialog ──────────────────────── */}
      <ConfirmDialog
        isOpen={showUnloadConfirm}
        title="Unload Skill"
        message="Are you sure you want to unload this skill? It will no longer be available for invocations until reloaded."
        itemName={detail.display_name || detail.name}
        confirmText="Unload"
        cancelText="Cancel"
        variant="danger"
        onConfirm={() => {
          setShowUnloadConfirm(false);
          onUnload?.(detail.name);
        }}
        onCancel={() => setShowUnloadConfirm(false)}
      />
    </div>
  );
};
