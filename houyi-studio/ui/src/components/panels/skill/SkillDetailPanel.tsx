/**
 * Skill detail panel shown in the right sidebar.
 *
 * Displays metadata, policy, permissions, hooks, tools, and metrics for the
 * selected skill. Provides actions for configure, dry-run, and unload.
 */
import React, { useState } from 'react';
import type { SkillDetail, SkillMetricsData } from '../../../types/websocket';
import { ConfirmDialog } from '../../common/ConfirmDialog';
import { MarkdownRenderer } from '../../Chat/MarkdownRenderer';

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

// ─── Integration level display ──────────────────────────────────
const INTEGRATION_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  metadata: { bg: 'bg-gray-700', text: 'text-gray-400', label: 'Metadata' },
  schema: { bg: 'bg-blue-900/40', text: 'text-blue-300', label: 'Schema' },
  executable: { bg: 'bg-green-900/40', text: 'text-green-300', label: 'Executable' },
};

// ─── Runtime status display ─────────────────────────────────────
const RUNTIME_STATUS_STYLES: Record<string, { color: string; icon: string; label: string }> = {
  ready: { color: 'text-green-400', icon: '●', label: 'Ready' },
  degraded: { color: 'text-yellow-400', icon: '◐', label: 'Degraded' },
  unavailable: { color: 'text-red-400', icon: '○', label: 'Unavailable' },
};

// ─── Side effect display ─────────────────────────────────────────
const SIDE_EFFECT_STYLES: Record<string, { color: string; icon: string }> = {
  none: { color: 'text-green-400', icon: '○' },
  network: { color: 'text-yellow-400', icon: '◐' },
  filesystem: { color: 'text-orange-400', icon: '◑' },
  exec: { color: 'text-red-400', icon: '●' },
  mixed: { color: 'text-red-400', icon: '●' },
};

const normalizeWorkspacePlaceholder = (text: string): string => text.replace(/\$\{WORKSPACE\}/g, 'workspace');

const parsePermissionText = (text: string): { title: string; targets: string[] } => {
  const normalized = normalizeWorkspacePlaceholder(text).trim();
  const fileAccessMatch = normalized.match(/^(Read|Write)\s+files?\s+(from|to):\s*(.+)$/i);
  if (!fileAccessMatch) {
    return { title: normalized, targets: [] };
  }

  const verb = fileAccessMatch[1].toLowerCase() === 'read' ? 'Read files' : 'Write files';
  const targets = fileAccessMatch[3]
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

  return {
    title: verb,
    targets,
  };
};

const formatHookLabel = (hook: string): string => {
  const match = hook.match(/^([^:]+):(.*?)(?:\s+\(([^)]+)\))?$/);
  const eventRaw = match?.[1]?.trim() || hook;
  const matcherRaw = match?.[2]?.trim() || '';

  const eventLabelMap: Record<string, string> = {
    PreToolUse: 'Before tool use',
    PostToolUse: 'After tool use',
    Stop: 'Before stop',
  };

  const eventLabel = eventLabelMap[eventRaw] || eventRaw;
  if (!matcherRaw) {
    return eventLabel;
  }

  const matcherLabel = matcherRaw === '*' || matcherRaw === '.*'
    ? 'all tools'
    : matcherRaw.split('|').map((part) => part.trim()).filter(Boolean).join(', ');

  return `${eventLabel} · ${matcherLabel}`;
};

export interface SkillDetailPanelProps {
  detail: SkillDetail | null;
  metrics: SkillMetricsData | null;
  isLoading: boolean;
  onConfigure?: () => void;
  onDryRun?: () => void;
  onUnload?: (skillName: string) => void;
  onRemoveFromDisk?: (skillName: string) => void;
}

export const SkillDetailPanel: React.FC<SkillDetailPanelProps> = ({
  detail,
  metrics,
  isLoading,
  onConfigure,
  onDryRun,
  onUnload,
  onRemoveFromDisk,
}) => {
  const [showUnloadConfirm, setShowUnloadConfirm] = useState(false);
  const [showRemoveConfirm, setShowRemoveConfirm] = useState(false);
  const [showRemoveFinalConfirm, setShowRemoveFinalConfirm] = useState(false);
  const [showDangerActions, setShowDangerActions] = useState(false);
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);

  if (isLoading && !detail) {
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
  const integrationStyle = INTEGRATION_STYLES[detail.capability_tier ?? 'metadata'] ?? INTEGRATION_STYLES.metadata;
  const runtimeStatusStyle = RUNTIME_STATUS_STYLES[detail.runtime_status ?? 'unavailable'] ?? RUNTIME_STATUS_STYLES.unavailable;
  const versionLabel = `v${detail.version || '0.0.0'}`;
  const hasVersion = Boolean(detail.version && detail.version !== '0.0.0');
  const instructionsLength = detail.instructions_length ?? (detail.instructions?.length ?? 0);
  const hasInstructions = Boolean(detail.instructions && detail.instructions.trim().length > 0);
  const canRemoveFromDisk = !detail.is_core && (detail.source || 'local') !== 'builtin';
  const hookSpecs = detail.hook_specs ?? [];
  const description = detail.description ?? '';
  const isLongDescription = description.length > 260 || description.split('\n').length > 5;
  const descriptionPreview = isLongDescription
    ? `${description.slice(0, 220).trimEnd()}...`
    : description;
  const normalizedFrontmatter = {
    name: detail.name,
    description: detail.description ?? '',
    user_invocable: detail.policy?.default_action !== 'deny',
    allowed_tools: detail.tools.map((tool) => tool.name),
    hooks: hookSpecs,
    metadata: {
      version: detail.version,
      capability_tier: detail.capability_tier ?? 'metadata',
      runtime_status: detail.runtime_status ?? 'unavailable',
      runtime_binding: detail.runtime_binding ?? 'none',
      is_external_alias: Boolean(detail.is_external_alias),
      alias_target: detail.alias_target ?? null,
    },
  };

  return (
    <div className="flex flex-col gap-3 p-3 text-xs">
      {/* ─── Header ─────────────────────────────────────────── */}
      <div className="flex items-start justify-between" data-testid="skill-header">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm font-semibold text-gray-200">
              {detail.display_name || detail.name}
            </div>
            {hasVersion && (
              <span className="inline-flex items-center h-6 px-2 rounded bg-gray-700/60 text-[10px] text-gray-300" data-testid="skill-version-chip">
                {versionLabel}
              </span>
            )}
            {detail.is_core && (
              <span
                className="inline-flex items-center h-6 px-2 rounded bg-cyan-900/40 text-[10px] text-cyan-300 border border-cyan-700/60"
                data-testid="skill-core-chip"
                title="Host core protected skill"
              >
                CORE
              </span>
            )}
            {detail.is_external_alias && (
              <span
                className="inline-flex items-center h-6 px-2 rounded bg-amber-900/40 text-[10px] text-amber-300 border border-amber-700/60"
                data-testid="skill-external-alias-chip"
                title={detail.alias_target ? `External alias of core skill: ${detail.alias_target}` : 'External alias skill'}
              >
                {detail.alias_target ? `EXT → ${detail.alias_target}` : 'EXT'}
              </span>
            )}
            <span className={`inline-flex items-center h-6 px-2 rounded text-[10px] font-medium ${cert.bg} ${cert.text}`}>
              {cert.label}
            </span>
          </div>
          <div className="text-gray-500 mt-0.5">
            {detail.author ? <span>by {detail.author}</span> : hasVersion ? <span>Version {versionLabel}</span> : null}
            {isLoading && (
              <span className="ml-2 text-[10px] text-gray-500" data-testid="skill-detail-loading-indicator">
                Refreshing...
              </span>
            )}
          </div>
        </div>
      </div>

      {description && (
        <div className="bg-gray-900/40 border border-gray-700/40 rounded p-2" data-testid="skill-description">
          {isLongDescription ? (
            <div>
              {!descriptionExpanded ? (
                <div className="text-gray-300 text-[12px] leading-relaxed">{descriptionPreview}</div>
              ) : (
                <div className="text-gray-300 text-[12px] leading-relaxed markdown-body">
                  <MarkdownRenderer content={description} />
                </div>
              )}
              <button
                type="button"
                onClick={() => setDescriptionExpanded((v) => !v)}
                className="mt-1 text-[11px] text-cyan-300 hover:text-cyan-200"
                data-testid="skill-description-more"
              >
                {descriptionExpanded ? 'Show less' : 'Show more'}
              </button>
            </div>
          ) : (
            <div className="text-gray-300 text-[12px] leading-relaxed markdown-body">
              <MarkdownRenderer content={description} />
            </div>
          )}
        </div>
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

      {/* ─── Capability ──────────────────────────────────────── */}
      <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-capability">
        <div className="text-gray-500 font-medium mb-1">CAPABILITY</div>
        <div className="flex items-center gap-3">
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${integrationStyle.bg} ${integrationStyle.text}`}>
            {integrationStyle.label}
          </span>
          <span className={`flex items-center gap-1 ${runtimeStatusStyle.color}`}>
            <span>{runtimeStatusStyle.icon}</span>
            <span>{runtimeStatusStyle.label}</span>
          </span>
        </div>
        <div className="mt-1.5 text-[10px] text-gray-400" data-testid="skill-runtime-binding">
          Binding: <span className="font-mono text-gray-300">{detail.runtime_binding ?? 'none'}</span>
          {instructionsLength > 0 && (
            <>
              {' · '}Instructions loaded: <span className="font-mono text-gray-300">{instructionsLength}</span> chars
            </>
          )}
        </div>
        {detail.runtime_status === 'unavailable' && (
          <div className="mt-1.5 text-[10px] text-red-400/80">
            Skill is not executable. Missing runtime adapter or core executor binding.
          </div>
        )}
        {detail.runtime_status === 'degraded' && (
          <div className="mt-1.5 text-[10px] text-yellow-400/80">
            Skill has schema but no executor. Dry-run validation is available but live execution is not.
          </div>
        )}
      </div>

      {/* ─── Permissions ────────────────────────────────────── */}
      {detail.permissions.length > 0 && (
        <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-permissions">
          <div className="text-gray-500 font-medium mb-1">PERMISSIONS</div>
          <div className="space-y-0.5">
            {detail.permissions.map((perm) => {
              const parsed = parsePermissionText(perm.name);
              return (
                <div key={perm.name} className="flex items-start gap-1">
                  <span className={perm.is_sensitive ? 'text-red-400' : 'text-gray-400'}>
                    {perm.is_sensitive ? '[!]' : '[ ]'}
                  </span>
                  <div className="min-w-0">
                    <div className="text-gray-300 break-all">{parsed.title}</div>
                    {parsed.targets.length > 0 && (
                      <ul className="mt-0.5 space-y-0.5 text-gray-500">
                        {parsed.targets.map((target) => (
                          <li key={target} className="break-all">• {target}</li>
                        ))}
                      </ul>
                    )}
                    {perm.description && perm.description !== perm.name && (
                      <div className="text-gray-600 break-all">{normalizeWorkspacePlaceholder(perm.description)}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-frontmatter-normalized">
        <div className="text-gray-500 font-medium mb-1">FRONTMATTER (NORMALIZED)</div>
        <div className="text-[11px] text-gray-400 mb-1">
          HouYi renders core fields in structured cards, and shows this normalized frontmatter view for 1:1 comparison with native SKILL.md style.
        </div>
        <details className="group" data-testid="skill-frontmatter-more">
          <summary className="cursor-pointer text-[11px] text-cyan-300 hover:text-cyan-200 select-none">
            Show normalized frontmatter
          </summary>
          <pre className="mt-1.5 max-h-40 overflow-auto rounded bg-gray-950/70 border border-gray-700/50 p-2 text-[10px] text-gray-300 whitespace-pre-wrap">
            {JSON.stringify(normalizedFrontmatter, null, 2)}
          </pre>
        </details>
      </div>

      {hasInstructions && (
        <div className="bg-gray-900/60 border border-gray-700 rounded p-2" data-testid="skill-instructions">
          <div className="text-gray-500 font-medium mb-1">INSTRUCTIONS</div>
          <div className="text-[11px] text-gray-400 mb-1">
            Prompt body loaded from SKILL.md ({instructionsLength} chars)
          </div>
          <details className="group" data-testid="skill-instructions-more">
            <summary className="cursor-pointer text-[11px] text-cyan-300 hover:text-cyan-200 select-none">
              Show full instructions
            </summary>
            <pre className="mt-1.5 max-h-32 overflow-auto rounded bg-gray-950/70 border border-gray-700/50 p-2 text-[10px] text-gray-300 whitespace-pre-wrap">
              {detail.instructions}
            </pre>
          </details>
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
                title={hook}
              >
                {formatHookLabel(hook)}
              </span>
            ))}
          </div>
          {hookSpecs.length > 0 && (
            <details className="mt-2 group" data-testid="skill-hook-specs-more">
              <summary className="cursor-pointer text-[11px] text-cyan-300 hover:text-cyan-200 select-none">
                Show hook specs ({hookSpecs.length})
              </summary>
              <div className="mt-1.5 space-y-1.5">
                {hookSpecs.map((hook, idx) => (
                  <div key={`${hook.event}-${hook.matcher}-${idx}`} className="rounded border border-gray-700/60 bg-gray-800/40 p-1.5 text-[10px]">
                    <div className="text-gray-300">
                      <span className="font-mono">{hook.event}</span>
                      {' · '}
                      <span className="text-gray-400">{hook.matcher}</span>
                      {' · '}
                      <span className="text-gray-400">{hook.type}</span>
                    </div>
                    {hook.command && (
                      <div className="mt-1 text-gray-400 break-all">
                        command: <span className="font-mono text-gray-300">{hook.command}</span>
                      </div>
                    )}
                    {hook.handler && (
                      <div className="mt-1 text-gray-400 break-all">
                        handler: <span className="font-mono text-gray-300">{hook.handler}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </details>
          )}
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
      <div className="space-y-2 pt-1 border-t border-gray-700" data-testid="skill-actions">
        <button
          type="button"
          onClick={onConfigure}
          className="w-full px-2 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 text-[11px] rounded transition-colors"
        >
          Configure...
        </button>
        <button
          type="button"
          onClick={() => onDryRun?.()}
          className="w-full px-2 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-[11px] rounded transition-colors"
          data-testid="skill-dry-run-button"
        >
          Dry-run
        </button>

        <button
          type="button"
          onClick={() => setShowDangerActions((prev) => !prev)}
          className="w-full px-2 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-[11px] rounded transition-colors"
          data-testid="skill-more-actions-button"
        >
          {showDangerActions ? 'Hide destructive actions' : 'More actions'}
        </button>

        {showDangerActions && (
          <div className="space-y-1 rounded border border-gray-700/80 bg-gray-900/40 p-1.5" data-testid="skill-danger-actions">
            <button
              type="button"
              onClick={() => {
                setShowDangerActions(false);
                setShowUnloadConfirm(true);
              }}
              className="w-full px-2 py-1.5 bg-red-900/50 hover:bg-red-800/60 text-red-300 text-[11px] rounded transition-colors"
            >
              Unload
            </button>
            {canRemoveFromDisk && (
              <button
                type="button"
                onClick={() => {
                  setShowDangerActions(false);
                  setShowRemoveConfirm(true);
                }}
                className="w-full px-2 py-1.5 bg-red-950/70 hover:bg-red-900/70 text-red-200 text-[11px] rounded transition-colors"
                data-testid="skill-remove-disk-button"
              >
                Remove from disk
              </button>
            )}
          </div>
        )}
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

      <ConfirmDialog
        isOpen={showRemoveConfirm}
        title="Remove Skill from Disk"
        message="This will delete managed local package links/data for this skill (both ~/.houyi/skills and ~/.houyi/sources/local if present). Continue?"
        itemName={detail.display_name || detail.name}
        confirmText="Continue"
        cancelText="Cancel"
        variant="danger"
        onConfirm={() => {
          setShowRemoveConfirm(false);
          setShowRemoveFinalConfirm(true);
        }}
        onCancel={() => setShowRemoveConfirm(false)}
      />

      <ConfirmDialog
        isOpen={showRemoveFinalConfirm}
        title="Confirm Permanent Removal"
        message="Final confirmation: remove this managed skill package from disk now?"
        itemName={detail.display_name || detail.name}
        confirmText="Remove Permanently"
        cancelText="Cancel"
        variant="danger"
        onConfirm={() => {
          setShowRemoveFinalConfirm(false);
          onRemoveFromDisk?.(detail.name);
        }}
        onCancel={() => setShowRemoveFinalConfirm(false)}
      />
    </div>
  );
};
