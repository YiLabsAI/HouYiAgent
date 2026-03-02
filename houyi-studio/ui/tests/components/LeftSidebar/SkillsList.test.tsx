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
    source_group: 'superpowers',
  },
  {
    name: 'ext__systematic-debugging',
    display_name: 'Systematic Debugging',
    description: 'Debug workflow',
    tools: ['read', 'run'],
    policy_action: 'allow',
    side_effect: 'exec',
    certification: 'unverified',
    is_core: false,
    source: 'third_party',
    source_group: 'superpowers',
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
    expect(screen.getByTestId('skills-group-external')).toHaveTextContent('External (2)');
    expect(screen.getByTestId('skills-subgroup-superpowers')).toBeInTheDocument();
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
    expect(screen.getAllByText('third_party').length).toBeGreaterThanOrEqual(1);

    fireEvent.click(screen.getByText('Planning with Files'));
    expect(onSelectSkill).toHaveBeenCalledWith('ext__planning-with-files');
  });

  it('keeps rendering existing skills while refresh loading is true', () => {
    render(
      <SkillsList
        skills={MOCK_SKILLS}
        isLoading={true}
        selectedSkill={null}
        onSelectSkill={vi.fn()}
        onRefresh={vi.fn()}
        onLoadSkill={vi.fn()}
      />,
    );

    expect(screen.getByText('Web Search')).toBeInTheDocument();
    expect(screen.queryByText('Loading skills...')).not.toBeInTheDocument();
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

  it('keeps superpowers and using-superpowers as separate subgroups when both are present', () => {
    const skills = [
      ...MOCK_SKILLS,
      {
        name: 'using-superpowers',
        display_name: 'using-superpowers',
        description: 'Local entrypoint',
        tools: ['run'],
        policy_action: 'allow',
        side_effect: 'none',
        certification: 'unverified',
        is_core: false,
        source: 'local',
        source_group: 'using-superpowers',
      } as SkillSummary,
    ];

    render(
      <SkillsList
        skills={skills}
        isLoading={false}
        selectedSkill={null}
        onSelectSkill={vi.fn()}
        onRefresh={vi.fn()}
        onLoadSkill={vi.fn()}
      />,
    );

    expect(screen.getByTestId('skills-subgroup-superpowers')).toBeInTheDocument();
    expect(screen.getByTestId('skills-subgroup-using-superpowers')).toBeInTheDocument();
  });

  it('supports collapsing and expanding external subgroup', () => {
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

    expect(screen.getByText('Planning with Files')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /superpowers/i }));
    expect(screen.queryByText('Planning with Files')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /superpowers/i }));
    expect(screen.getByText('Planning with Files')).toBeInTheDocument();
  });
});
