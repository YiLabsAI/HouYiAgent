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
        <div className="space-y-1">
          {skills.map((skill) => (
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
                <PolicyBadge action={skill.policy_action} />
                <SideEffectBadge effect={skill.side_effect} />
                <CertificationBadge level={skill.certification} />
                {skill.tools.length > 0 && (
                  <span className="text-[10px] text-gray-500">
                    {skill.tools.length} tool{skill.tools.length > 1 ? 's' : ''}
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Legend */}
      {skills.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-700">
          <div className="text-[10px] text-gray-500 mb-2">Policy Legend</div>
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
    </div>
  );
};

export default SkillsList;
