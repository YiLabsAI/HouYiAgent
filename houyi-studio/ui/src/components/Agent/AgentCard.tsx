/**
 * AgentCard — card component for the Agent Hub grid.
 *
 * Active agents show a clickable card; "coming soon" agents are visually
 * dimmed with a badge overlay.
 */

import React from 'react';

export interface AgentCardProps {
  name: string;
  description: string;
  icon: string;
  active?: boolean;
  onClick?: () => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({
  name,
  description,
  icon,
  active = false,
  onClick,
}) => {
  return (
    <button
      type="button"
      onClick={active ? onClick : undefined}
      disabled={!active}
      className={`relative flex flex-col items-start gap-3 p-6 rounded-xl border text-left transition-all ${
        active
          ? 'bg-gray-800/60 border-gray-600 hover:border-purple-500 hover:bg-gray-800 cursor-pointer shadow-sm hover:shadow-purple-500/10'
          : 'bg-gray-800/30 border-gray-700/50 cursor-not-allowed opacity-60'
      }`}
    >
      {!active && (
        <span className="absolute top-3 right-3 text-[10px] font-medium uppercase tracking-wider text-gray-500 bg-gray-700/60 px-2 py-0.5 rounded-full">
          Coming Soon
        </span>
      )}
      <div className="text-3xl">{icon}</div>
      <div>
        <h3 className={`text-sm font-semibold ${active ? 'text-gray-100' : 'text-gray-400'}`}>
          {name}
        </h3>
        <p className="text-xs text-gray-500 mt-1 leading-relaxed">{description}</p>
      </div>
    </button>
  );
};
