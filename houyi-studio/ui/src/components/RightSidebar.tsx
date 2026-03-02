/**
 * Contextual detail panel on the right side.
 *
 * Routes content by selection context:
 *   - selected node      -> NodePropertiesPanel
 *   - selected skill     -> SkillDetailPanel
 *   - selected knowledge -> KnowledgeDetailPanel
 *   - chat conversations -> ConversationSettingsPanel
 *   - none selected      -> empty state
 */
import React, { useRef, useLayoutEffect } from 'react';
import { useConsoleStore } from '../stores/useConsoleStore';
import type { SecondaryContentMode } from '../stores/useConsoleStore';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { NodePropertiesPanel } from './panels/NodePropertiesPanel';
import { SkillDetailPanel } from './panels/skill/SkillDetailPanel';
import { KnowledgeDetailPanel } from './panels/KnowledgeDetailPanel';
import { ConversationSettingsPanel } from './panels/ConversationSettingsPanel';

// ─── Scroll position cache (per content mode) ───────────────────
const scrollPositionCache = new Map<SecondaryContentMode, number>();

interface RightSidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  /** Skill detail data from useSkillsLogic (for SkillDetailPanel). */
  skillDetail?: import('../types/websocket').SkillDetail | null;
  skillMetrics?: import('../types/websocket').SkillMetricsData | null;
  isLoadingSkillDetail?: boolean;
  onConfigureSkill?: () => void;
  onDryRunSkill?: () => void;
  onUnloadSkill?: (skillName: string) => void;
  onRemoveSkillFromDisk?: (skillName: string) => void;
  /** Called when user clicks [Full Settings...] in ConversationSettingsPanel. */
  onOpenChatSettings?: () => void;
  /** Called when user clicks [Configure...] in KnowledgeDetailPanel. */
  onConfigureKnowledge?: (libraryId: string) => void;
  /** Called when user clicks [Rebuild Index] in KnowledgeDetailPanel. */
  onRebuildKnowledgeIndex?: (libraryId: string) => void;
}

// ─── Content mode → header title mapping ─────────────────────────
const HEADER_TITLES: Record<SecondaryContentMode, string> = {
  node: 'Properties',
  skill: 'Skill Detail',
  knowledge: 'Knowledge Detail',
  conversation: 'Conversation Settings',
  empty: 'Details',
};

const EmptyState: React.FC = () => {
  const primaryMode = useConsoleStore((s) => s.primaryMode);
  return (
    <div className="text-center text-gray-400 text-sm mt-8">
      {primaryMode === 'graph'
        ? 'Select a node, skill, or knowledge library to view details'
        : 'Select a skill or knowledge library to view details'}
    </div>
  );
};

// ─── Content router ──────────────────────────────────────────────
interface SecondaryContentProps {
  mode: SecondaryContentMode;
  skillDetail?: import('../types/websocket').SkillDetail | null;
  skillMetrics?: import('../types/websocket').SkillMetricsData | null;
  isLoadingSkillDetail?: boolean;
  onConfigureSkill?: () => void;
  onDryRunSkill?: () => void;
  onUnloadSkill?: (skillName: string) => void;
  onRemoveSkillFromDisk?: (skillName: string) => void;
  onOpenChatSettings?: () => void;
  onConfigureKnowledge?: (libraryId: string) => void;
  onRebuildKnowledgeIndex?: (libraryId: string) => void;
}

const SecondaryContent: React.FC<SecondaryContentProps> = ({
  mode,
  skillDetail,
  skillMetrics,
  isLoadingSkillDetail,
  onConfigureSkill,
  onDryRunSkill,
  onUnloadSkill,
  onRemoveSkillFromDisk,
  onOpenChatSettings,
  onConfigureKnowledge,
  onRebuildKnowledgeIndex,
}) => {
  switch (mode) {
    case 'node':
      return <NodePropertiesPanel />;
    case 'skill':
      return (
        <SkillDetailPanel
          detail={skillDetail ?? null}
          metrics={skillMetrics ?? null}
          isLoading={isLoadingSkillDetail ?? false}
          onConfigure={onConfigureSkill}
          onDryRun={onDryRunSkill}
          onUnload={onUnloadSkill}
          onRemoveFromDisk={onRemoveSkillFromDisk}
        />
      );
    case 'knowledge':
      return (
        <KnowledgeDetailPanel
          onConfigure={onConfigureKnowledge}
          onRebuildIndex={onRebuildKnowledgeIndex}
        />
      );
    case 'conversation':
      return <ConversationSettingsPanel onOpenFullSettings={onOpenChatSettings} />;
    case 'empty':
    default:
      return <EmptyState />;
  }
};

// ─── Main component ──────────────────────────────────────────────
export const RightSidebar: React.FC<RightSidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
  skillDetail,
  skillMetrics,
  isLoadingSkillDetail,
  onConfigureSkill,
  onDryRunSkill,
  onUnloadSkill,
  onRemoveSkillFromDisk,
  onOpenChatSettings,
  onConfigureKnowledge,
  onRebuildKnowledgeIndex,
}) => {
  const contentMode = useConsoleStore((s) => s.getSecondaryContentMode());
  const title = HEADER_TITLES[contentMode];

  const scrollRef = useRef<HTMLDivElement>(null);

  // ─── Scroll position save/restore ──────────────────────────────
  //
  // Key insight: useLayoutEffect cleanup runs BEFORE the DOM is
  // updated for the next render.  This means scrollRef.current still
  // has the old content and a correct scrollTop.  The effect body
  // runs AFTER the DOM has the new content, so we restore there.
  useLayoutEffect(() => {
    // Restore: schedule after browser has laid out the new content.
    const raf = requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollPositionCache.get(contentMode) ?? 0;
      }
    });

    return () => {
      // Save: the DOM still has the CURRENT mode's content here.
      cancelAnimationFrame(raf);
      if (scrollRef.current) {
        scrollPositionCache.set(contentMode, scrollRef.current.scrollTop);
      }
    };
  }, [contentMode]);

  return (
    <div className={`bg-gray-800 border-l border-gray-700 flex flex-col h-full ${
      isCollapsed ? 'w-8' : 'w-full'
    }`}>
      {isCollapsed ? (
        <div className="flex items-center justify-center h-full">
          <button
            onClick={onToggleCollapse}
            className="p-1 hover:bg-gray-700 rounded text-gray-400"
            title="Expand sidebar"
          >
            <ChevronLeft size={16} />
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between p-3 border-b border-gray-700">
            <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
            <button
              onClick={onToggleCollapse}
              className="p-1 hover:bg-gray-700 rounded text-gray-400"
              title="Collapse sidebar"
            >
              <ChevronRight size={16} />
            </button>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto" data-testid="right-sidebar-scroll-container">
            <SecondaryContent
              mode={contentMode}
              skillDetail={skillDetail}
              skillMetrics={skillMetrics}
              isLoadingSkillDetail={isLoadingSkillDetail}
              onConfigureSkill={onConfigureSkill}
              onDryRunSkill={onDryRunSkill}
              onUnloadSkill={onUnloadSkill}
              onRemoveSkillFromDisk={onRemoveSkillFromDisk}
              onOpenChatSettings={onOpenChatSettings}
              onConfigureKnowledge={onConfigureKnowledge}
              onRebuildKnowledgeIndex={onRebuildKnowledgeIndex}
            />
          </div>
        </>
      )}
    </div>
  );
};
