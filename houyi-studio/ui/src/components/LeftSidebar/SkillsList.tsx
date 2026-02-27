import React from 'react';
import { Package, RefreshCw, ChevronRight, Shield, Zap, AlertTriangle, Check, Plus } from 'lucide-react';
import type { SkillSummary } from '../../types/websocket';

interface SkillsListProps {
  skills: SkillSummary[];
  isLoading: boolean;
  selectedSkill: string | null;
  onSelectSkill: (skillName: string) => void;
  onRefresh: () => void;
  onLoadSkill?: () => void;
}

const PolicyBadge: React.FC<{ action: string }> = ({ action }) => {
  switch (action) {
    case 'allow':
      return (
        <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] bg-green-900/50 text-green-400" title="Allowed">
          <Check size={10} />
        </span>
      );
    case 'allow_with_consent':
      return (
        <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] bg-yellow-900/50 text-yellow-400" title="Requires consent">
          <AlertTriangle size={10} />
        </span>
      );
    case 'deny':
      return (
        <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] bg-red-900/50 text-red-400" title="Denied">
          <Shield size={10} />
        </span>
      );
    default:
      return null;
  }
};

const sourceLabel: Record<string, string> = {
  builtin: 'builtin',
  community: 'community',
  third_party: 'third_party',
  local: 'local',
};

const sourceColor: Record<string, string> = {
  builtin: 'bg-emerald-900/40 text-emerald-300 border-emerald-700/60',
  community: 'bg-indigo-900/40 text-indigo-300 border-indigo-700/60',
  third_party: 'bg-amber-900/40 text-amber-300 border-amber-700/60',
  local: 'bg-gray-800/70 text-gray-300 border-gray-600/60',
};

const SourceBadge: React.FC<{ source?: string; isCore?: boolean }> = ({ source, isCore }) => {
  const key = source || 'local';
  const display = isCore && key === 'builtin' ? 'host' : (sourceLabel[key] || key);
  const title = isCore && key === 'builtin'
    ? 'Source: builtin (host core)'
    : `Source: ${sourceLabel[key] || key}`;

  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] border ${sourceColor[key] || sourceColor.local}`}
      title={title}
    >
      {display}
    </span>
  );
};

const CoreBadge: React.FC = () => (
  <span
    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-cyan-900/40 text-cyan-300 border border-cyan-700/60"
    title="Host core protected skill"
  >
    <Shield size={10} />
    core
  </span>
);

const SideEffectBadge: React.FC<{ effect: string }> = ({ effect }) => {
  if (effect === 'none') return null;

  const colors = {
    network: 'bg-blue-900/50 text-blue-400',
    filesystem: 'bg-purple-900/50 text-purple-400',
    exec: 'bg-red-900/50 text-red-400',
  };

  return (
    <span
      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] ${colors[effect as keyof typeof colors] || 'bg-gray-700 text-gray-400'}`}
      title={`Side effect: ${effect}`}
    >
      <Zap size={10} />
      {effect}
    </span>
  );
};

const runtimeStatusColor: Record<string, string> = {
  ready: 'text-green-400',
  degraded: 'text-yellow-400',
  unavailable: 'text-red-400',
};

const runtimeStatusIcon: Record<string, string> = {
  ready: '●',
  degraded: '◐',
  unavailable: '○',
};

const RuntimeBadge: React.FC<{ capabilityTier?: string; runtimeStatus?: string }> = ({
  capabilityTier,
  runtimeStatus,
}) => {
  const level = capabilityTier || 'metadata';
  const status = runtimeStatus || 'unavailable';
  const color = runtimeStatusColor[status] || runtimeStatusColor.unavailable;
  const icon = runtimeStatusIcon[status] || runtimeStatusIcon.unavailable;

  return (
    <span
      className={`inline-flex items-center gap-0.5 text-[10px] ${color}`}
      title={`Integration: ${level} · Runtime: ${status}`}
    >
      {icon} {level}
    </span>
  );
};

const CertificationBadge: React.FC<{ level: string }> = ({ level }) => {
  switch (level) {
    case 'gold':
      return <span className="text-[10px]" title="Gold certified">&#x1F947;</span>;
    case 'silver':
      return <span className="text-[10px]" title="Silver certified">&#x1F948;</span>;
    case 'bronze':
      return <span className="text-[10px]" title="Bronze certified">&#x1F949;</span>;
    case 'unverified':
    default:
      return (
        <span className="text-[10px] px-1 py-0.5 rounded bg-gray-700 text-gray-500" title="Unverified">
          ?
        </span>
      );
  }
};

export const SkillsList: React.FC<SkillsListProps> = ({
  skills,
  isLoading,
  selectedSkill,
  onSelectSkill,
  onRefresh,
  onLoadSkill,
}) => {
  const coreSkills = skills.filter((s) => s.is_core);
  const builtinSkills = skills.filter((s) => !s.is_core && (s.source || 'local') === 'builtin');
  const externalSkills = skills.filter((s) => !s.is_core && (s.source || 'local') !== 'builtin');

  const renderSection = (title: string, items: SkillSummary[]) => {
    if (items.length === 0) return null;
    return (
      <div className="space-y-1" data-testid={`skills-group-${title.toLowerCase()}`}>
        <div className="px-1 pt-2 pb-1 text-[10px] uppercase tracking-wide text-gray-500">
          {title} ({items.length})
        </div>
        {items.map((skill) => (
          <button
            key={skill.name}
            onClick={() => onSelectSkill(skill.name)}
            className={`w-full text-left p-2 rounded transition-colors ${
              selectedSkill === skill.name
                ? 'bg-blue-900/50 border border-blue-700'
                : 'bg-gray-900 border border-gray-700 hover:border-gray-600'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2 min-w-0">
                <Package size={14} className="text-gray-400 shrink-0" />
                <div className="min-w-0">
                  <div className="text-xs font-medium text-gray-200 truncate">
                    {skill.display_name}
                  </div>
                  {skill.description && (
                    <div className="text-[10px] text-gray-500 truncate mt-0.5">
                      {skill.description}
                    </div>
                  )}
                </div>
              </div>
              <ChevronRight size={14} className="text-gray-500 shrink-0 mt-0.5" />
            </div>

            {/* Badges Row */}
            <div className="flex items-center gap-1.5 mt-2 flex-wrap">
              {skill.is_core && <CoreBadge />}
              {skill.is_external_alias && (
                <span
                  className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] border bg-amber-900/40 text-amber-300 border-amber-700/60"
                  title={skill.alias_target ? `External alias of core skill: ${skill.alias_target}` : 'External alias skill'}
                >
                  {skill.alias_target ? `ext→${skill.alias_target}` : 'ext'}
                </span>
              )}
              <SourceBadge source={skill.source} isCore={skill.is_core} />
              <PolicyBadge action={skill.policy_action} />
              <SideEffectBadge effect={skill.side_effect} />
              <CertificationBadge level={skill.certification} />
              <RuntimeBadge capabilityTier={skill.capability_tier} runtimeStatus={skill.runtime_status} />
              {skill.tools.length > 0 && (
                <span className="text-[10px] text-gray-500">
                  {skill.tools.length} tool{skill.tools.length > 1 ? 's' : ''}
                </span>
              )}
            </div>
          </button>
        ))}
      </div>
    );
  };

  return (
    <div className="p-3">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs font-semibold text-gray-200">Registered Skills</div>
        <div className="flex items-center gap-1">
          <button
            onClick={onLoadSkill}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
            title="Load skill"
            data-testid="load-skill-button"
          >
            <Plus size={14} />
          </button>
          <button
            onClick={onRefresh}
            className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-gray-200 transition-colors"
            title="Refresh skills"
            disabled={isLoading}
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Legend */}
      {skills.length > 0 && (
        <div className="mb-3 pb-2 border-b border-gray-700" data-testid="skills-policy-legend">
          <div className="text-[10px] text-gray-500 mb-1.5">Policy Legend</div>
          <div className="flex flex-wrap gap-2 text-[10px]">
            <span className="flex items-center gap-1 text-gray-400">
              <Check size={10} className="text-green-400" /> Allow
            </span>
            <span className="flex items-center gap-1 text-gray-400">
              <AlertTriangle size={10} className="text-yellow-400" /> Consent
            </span>
            <span className="flex items-center gap-1 text-gray-400">
              <Shield size={10} className="text-red-400" /> Deny
            </span>
          </div>
        </div>
      )}

      {/* Skills List */}
      {isLoading ? (
        <div className="text-xs text-gray-500 text-center py-4">Loading skills...</div>
      ) : skills.length === 0 ? (
        <div className="bg-gray-900 border border-gray-700 rounded p-3 text-center">
          <Package size={24} className="mx-auto text-gray-600 mb-2" />
          <div className="text-xs text-gray-500">No skills registered</div>
          <div className="text-[10px] text-gray-600 mt-1">
            Skills will appear here when loaded
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {renderSection('Core', coreSkills)}
          {renderSection('Builtin', builtinSkills)}
          {renderSection('External', externalSkills)}
        </div>
      )}
    </div>
  );
};

export default SkillsList;
