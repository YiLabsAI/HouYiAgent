import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { SkillsList } from '@/components/LeftSidebar/SkillsList';
import type { SkillSummary } from '@/types/websocket';

const MOCK_SKILLS: SkillSummary[] = [
  {
    name: 'web_search',
    display_name: 'Web Search',
    description: 'Host search tool',
    tools: ['web_search'],
    policy_action: 'allow',
    side_effect: 'network',
    certification: 'gold',
    is_core: true,
    source: 'builtin',
  },
  {
    name: 'planner',
    display_name: 'Planner',
    description: 'Builtin planner',
    tools: ['plan'],
    policy_action: 'allow_with_consent',
    side_effect: 'none',
    certification: 'silver',
    is_core: false,
    source: 'builtin',
  },
  {
    name: 'ext__planning-with-files',
    display_name: 'Planning with Files',
    description: 'External planner',
    tools: ['read', 'write'],
    policy_action: 'allow_with_consent',
    side_effect: 'filesystem',
    certification: 'unverified',
    is_core: false,
    source: 'third_party',
  },
];

describe('SkillsList', () => {
  it('groups skills into Core, Builtin, and External sections', () => {
    render(
      <SkillsList
        skills={MOCK_SKILLS}
        isLoading={false}
        selectedSkill={null}
        onSelectSkill={vi.fn()}
        onRefresh={vi.fn()}
        onLoadSkill={vi.fn()}
      />,
    );

    expect(screen.getByTestId('skills-group-core')).toHaveTextContent('Core (1)');
    expect(screen.getByTestId('skills-group-builtin')).toHaveTextContent('Builtin (1)');
    expect(screen.getByTestId('skills-group-external')).toHaveTextContent('External (1)');
  });

  it('shows source badge and emits selection callback', () => {
    const onSelectSkill = vi.fn();
    render(
      <SkillsList
        skills={MOCK_SKILLS}
        isLoading={false}
        selectedSkill={null}
        onSelectSkill={onSelectSkill}
        onRefresh={vi.fn()}
        onLoadSkill={vi.fn()}
      />,
    );

    expect(screen.getByText('host')).toBeInTheDocument();
    expect(screen.getByText('builtin')).toBeInTheDocument();
    expect(screen.getByText('third_party')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Planning with Files'));
    expect(onSelectSkill).toHaveBeenCalledWith('ext__planning-with-files');
  });

  it('renders policy legend above skill groups', () => {
    render(
      <SkillsList
        skills={MOCK_SKILLS}
        isLoading={false}
        selectedSkill={null}
        onSelectSkill={vi.fn()}
        onRefresh={vi.fn()}
        onLoadSkill={vi.fn()}
      />,
    );

    const legend = screen.getByTestId('skills-policy-legend');
    const coreGroup = screen.getByTestId('skills-group-core');
    const relation = legend.compareDocumentPosition(coreGroup);
    expect((relation & Node.DOCUMENT_POSITION_FOLLOWING) !== 0).toBe(true);
  });
});
