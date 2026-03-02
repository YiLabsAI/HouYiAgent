/**
 * Tests for SkillDetailPanel Tier 1 component (§7.10.3).
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { SkillDetailPanel } from '@/components/panels/skill/SkillDetailPanel';
import type { SkillDetail, SkillMetricsData } from '@/types/websocket';

const createDetail = (overrides: Partial<SkillDetail> = {}): SkillDetail => ({
  name: 'planning-with-files',
  display_name: 'Planning with Files',
  description: 'A skill for file-based planning operations.',
  version: '1.0.0',
  author: 'HouYi',
  tools: [
    { name: 'Read', description: 'Read a file' },
    { name: 'Write', description: 'Write a file' },
    { name: 'Edit', description: 'Edit a file' },
  ],
  permissions: [
    { name: 'file:read', description: '/workspace/**', is_sensitive: false },
    { name: 'file:write', description: '/workspace/**', is_sensitive: true },
  ],
  policy: { default_action: 'allow_with_consent' },
  hooks: ['PreToolUse', 'PostToolUse', 'Stop'],
  certification: 'gold',
  side_effect: 'filesystem',
  ...overrides,
  is_core: overrides.is_core ?? false,
});

const createMetrics = (overrides: Partial<SkillMetricsData> = {}): SkillMetricsData => ({
  skill_name: 'planning-with-files',
  total_calls: 47,
  success_count: 45,
  failure_count: 2,
  avg_latency_ms: 23,
  p50_latency_ms: 18,
  p99_latency_ms: 120,
  success_rate: 0.957,
  last_invoked: '2026-02-03T10:00:00Z',
  ...overrides,
});

describe('SkillDetailPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ─── Loading state ─────────────────────────────────────────────

  it('should show loading state', () => {
    render(<SkillDetailPanel detail={null} metrics={null} isLoading={true} />);
    expect(screen.getByText('Loading skill detail...')).toBeInTheDocument();
  });

  it('should keep current detail visible while refreshing', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={true} />);
    expect(screen.getByText('Planning with Files')).toBeInTheDocument();
    expect(screen.getByTestId('skill-detail-loading-indicator')).toHaveTextContent('Refreshing...');
    expect(screen.queryByText('Loading skill detail...')).not.toBeInTheDocument();
  });

  // ─── Empty state ───────────────────────────────────────────────

  it('should show empty state when no detail', () => {
    render(<SkillDetailPanel detail={null} metrics={null} isLoading={false} />);
    expect(screen.getByText(/Select a skill/)).toBeInTheDocument();
  });

  // ─── Header section ────────────────────────────────────────────

  it('should render skill name and version', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    expect(screen.getByText('Planning with Files')).toBeInTheDocument();
    expect(screen.getByText(/v1\.0\.0/)).toBeInTheDocument();
    expect(screen.getByText(/by HouYi/)).toBeInTheDocument();
  });

  it('should render core badge when skill is core', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ is_core: true })}
        metrics={null}
        isLoading={false}
      />,
    );
    expect(screen.getByTestId('skill-core-chip')).toBeInTheDocument();
    expect(screen.getByTestId('skill-core-chip')).toHaveTextContent('CORE');
  });

  it('should render external alias badge when skill is ext alias', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ is_external_alias: true, alias_target: 'planning-with-files' })}
        metrics={null}
        isLoading={false}
      />,
    );
    const chip = screen.getByTestId('skill-external-alias-chip');
    expect(chip).toHaveTextContent('EXT → planning-with-files');
  });

  it('should render certification badge', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    expect(screen.getByText('Gold')).toBeInTheDocument();
  });

  it('should render unverified badge by default', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ certification: 'unverified' })}
        metrics={null}
        isLoading={false}
      />
    );
    expect(screen.getByText('Unverified')).toBeInTheDocument();
  });

  it('should render description', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    expect(screen.getByText('A skill for file-based planning operations.')).toBeInTheDocument();
  });

  it('should auto-collapse long description and render markdown after expand', () => {
    const longMarkdown = [
      'Get weather summary with modes:',
      '',
      '- **Coordinates mode**: `lat` + `lon`',
      '- **City mode**: `city` + optional `country`',
      '- Supports provider fallback',
      '- Returns readable weather text',
      '',
      'Extra context for display quality checks.',
    ].join('\n');

    render(
      <SkillDetailPanel
        detail={createDetail({ description: longMarkdown })}
        metrics={null}
        isLoading={false}
      />,
    );

    const desc = screen.getByTestId('skill-description-more');
    expect(desc).toBeInTheDocument();
    expect(desc).toHaveTextContent('Show more');
    fireEvent.click(desc);
    expect(screen.getByTestId('skill-description')).toHaveTextContent('Coordinates mode');
    expect(screen.getByTestId('skill-description-more')).toHaveTextContent('Show less');
  });

  it('should render normalized frontmatter view', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const block = screen.getByTestId('skill-frontmatter-normalized');
    expect(block).toHaveTextContent('FRONTMATTER (NORMALIZED)');
    expect(block).toHaveTextContent('planning-with-files');
    expect(block).toHaveTextContent('allowed_tools');
  });

  // ─── Policy section ────────────────────────────────────────────

  it('should render policy and side effect', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const policySection = screen.getByTestId('skill-policy');
    expect(policySection).toHaveTextContent('Allow with Consent');
    expect(policySection).toHaveTextContent('filesystem');
  });

  // ─── Permissions section ───────────────────────────────────────

  it('should render permissions with sensitivity indicators', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const permSection = screen.getByTestId('skill-permissions');
    expect(permSection).toHaveTextContent('file:read');
    expect(permSection).toHaveTextContent('file:write');
    // Sensitive permission marker
    expect(permSection).toHaveTextContent('[!]');
    // Non-sensitive permission marker
    expect(permSection).toHaveTextContent('[ ]');
  });

  it('should render file permissions in user-friendly format', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({
          permissions: [
            {
              name: 'Read files from: ${WORKSPACE}/**/*.md, ${WORKSPACE}/.plan/**',
              description: 'Read files from: ${WORKSPACE}/**/*.md, ${WORKSPACE}/.plan/**',
              is_sensitive: true,
            },
          ],
        })}
        metrics={null}
        isLoading={false}
      />,
    );

    const permSection = screen.getByTestId('skill-permissions');
    expect(permSection).toHaveTextContent('Read files');
    expect(permSection).toHaveTextContent('workspace/**/*.md');
    expect(permSection).toHaveTextContent('workspace/.plan/**');
    expect(permSection).not.toHaveTextContent('${WORKSPACE}');
  });

  it('should not render permissions section when empty', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ permissions: [] })}
        metrics={null}
        isLoading={false}
      />
    );
    expect(screen.queryByTestId('skill-permissions')).not.toBeInTheDocument();
  });

  it('should not render ad-hoc template sections block', () => {
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
      />
    );
    expect(screen.queryByTestId('skill-template-sections')).not.toBeInTheDocument();
  });

  // ─── Hooks section ─────────────────────────────────────────────

  it('should render hooks list', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const hooksSection = screen.getByTestId('skill-hooks');
    expect(hooksSection).toHaveTextContent('Before tool use');
    expect(hooksSection).toHaveTextContent('After tool use');
    expect(hooksSection).toHaveTextContent('Before stop');
  });

  it('should render hooks in human-readable labels', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({
          hooks: [
            'PreToolUse:Write|Edit|StrReplace (handler)',
            'PostToolUse:.* (handler)',
            'Stop:* (handler)',
          ],
        })}
        metrics={null}
        isLoading={false}
      />,
    );

    const hooksSection = screen.getByTestId('skill-hooks');
    expect(hooksSection).toHaveTextContent('Before tool use · Write, Edit, StrReplace');
    expect(hooksSection).toHaveTextContent('After tool use · all tools');
    expect(hooksSection).toHaveTextContent('Before stop · all tools');
  });

  it('should not render hooks section when empty', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ hooks: [] })}
        metrics={null}
        isLoading={false}
      />
    );
    expect(screen.queryByTestId('skill-hooks')).not.toBeInTheDocument();
  });

  // ─── Tools section ─────────────────────────────────────────────

  it('should render tools with count', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const toolsSection = screen.getByTestId('skill-tools');
    expect(toolsSection).toHaveTextContent('TOOLS (3)');
    expect(toolsSection).toHaveTextContent('Read');
    expect(toolsSection).toHaveTextContent('Write');
    expect(toolsSection).toHaveTextContent('Edit');
  });

  it('should show "No tools defined" when tools are empty', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ tools: [] })}
        metrics={null}
        isLoading={false}
      />
    );
    expect(screen.getByText('No tools defined')).toBeInTheDocument();
  });

  it('should display tool chips as informational labels', () => {
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
      />
    );
    const toolsSection = screen.getByTestId('skill-tools');
    expect(toolsSection).toHaveTextContent('Read');
    expect(toolsSection).toHaveTextContent('Write');
    expect(toolsSection).toHaveTextContent('Edit');
  });

  // ─── Metrics section ───────────────────────────────────────────

  it('should render metrics when available', () => {
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={createMetrics()}
        isLoading={false}
      />
    );
    const metricsSection = screen.getByTestId('skill-metrics');
    expect(metricsSection).toHaveTextContent('47');
    expect(metricsSection).toHaveTextContent('95.7%');
    expect(metricsSection).toHaveTextContent('23ms');
  });

  it('should show "No metrics available" when metrics are null', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    expect(screen.getByText('No invocations yet. Metrics are collected from Chat tool calls, not Dry-run.')).toBeInTheDocument();
  });

  // ─── Action buttons ────────────────────────────────────────────

  it('should render Configure, Dry-run, and More actions buttons', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const actions = screen.getByTestId('skill-actions');
    expect(actions).toHaveTextContent('Configure...');
    expect(actions).toHaveTextContent('Dry-run');
    expect(actions).toHaveTextContent('More actions');
    expect(screen.queryByText('Unload')).not.toBeInTheDocument();
    expect(screen.queryByText('Remove from disk')).not.toBeInTheDocument();
  });

  it('should hide remove-from-disk action for core skills', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ is_core: true, source: 'builtin' as any })}
        metrics={null}
        isLoading={false}
      />,
    );
    fireEvent.click(screen.getByTestId('skill-more-actions-button'));
    expect(screen.queryByTestId('skill-remove-disk-button')).not.toBeInTheDocument();
  });

  it('should trigger onConfigure when Configure is clicked', () => {
    const onConfigure = vi.fn();
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
        onConfigure={onConfigure}
      />
    );
    fireEvent.click(screen.getByText('Configure...'));
    expect(onConfigure).toHaveBeenCalledTimes(1);
  });

  it('should trigger onDryRun (no args) when Dry-run button is clicked', () => {
    const onDryRun = vi.fn();
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
        onDryRun={onDryRun}
      />
    );
    fireEvent.click(screen.getByText('Dry-run'));
    expect(onDryRun).toHaveBeenCalledTimes(1);
    expect(onDryRun).toHaveBeenCalledWith();
  });

  it('should show confirmation dialog when Unload is clicked', () => {
    const onUnload = vi.fn();
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
        onUnload={onUnload}
      />
    );
    fireEvent.click(screen.getByTestId('skill-more-actions-button'));

    // Click Unload — should NOT immediately call onUnload
    fireEvent.click(screen.getByText('Unload'));
    expect(onUnload).not.toHaveBeenCalled();

    // Confirmation dialog should appear
    expect(screen.getByText('Unload Skill')).toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to unload this skill/)).toBeInTheDocument();
    // The dialog shows the skill name in a highlighted box
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveTextContent('Planning with Files');
  });

  it('should trigger onUnload only after confirming the dialog', () => {
    const onUnload = vi.fn();
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
        onUnload={onUnload}
      />
    );
    // Open confirm dialog
    fireEvent.click(screen.getByTestId('skill-more-actions-button'));
    fireEvent.click(screen.getByText('Unload'));

    // Click the confirm button inside the dialog (labeled "Unload")
    const confirmButtons = screen.getAllByText('Unload');
    // The second "Unload" button is the confirm button in the dialog
    fireEvent.click(confirmButtons[confirmButtons.length - 1]);
    expect(onUnload).toHaveBeenCalledWith('planning-with-files');
  });

  it('should not trigger onUnload when cancel is clicked in the dialog', () => {
    const onUnload = vi.fn();
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
        onUnload={onUnload}
      />
    );
    // Open confirm dialog
    fireEvent.click(screen.getByTestId('skill-more-actions-button'));
    fireEvent.click(screen.getByText('Unload'));

    // Click Cancel
    fireEvent.click(screen.getByText('Cancel'));
    expect(onUnload).not.toHaveBeenCalled();

    // Dialog should be closed
    expect(screen.queryByText('Unload Skill')).not.toBeInTheDocument();
  });

  it('should require two confirmations before removing from disk', () => {
    const onRemoveFromDisk = vi.fn();
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
        onRemoveFromDisk={onRemoveFromDisk}
      />,
    );

    fireEvent.click(screen.getByTestId('skill-more-actions-button'));
    fireEvent.click(screen.getByTestId('skill-remove-disk-button'));
    expect(screen.getByText('Remove Skill from Disk')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Continue'));
    expect(screen.getByText('Confirm Permanent Removal')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Remove Permanently'));
    expect(onRemoveFromDisk).toHaveBeenCalledWith('planning-with-files');
  });

  // ─── Capability section ──────────────────────────────────────────

  it('should render capability section with integration level and runtime status', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ capability_tier: 'executable', runtime_status: 'ready' })}
        metrics={null}
        isLoading={false}
      />
    );
    const capSection = screen.getByTestId('skill-capability');
    expect(capSection).toHaveTextContent('CAPABILITY');
    expect(capSection).toHaveTextContent('Executable');
    expect(capSection).toHaveTextContent('Ready');
  });

  it('should show degraded warning when runtime_status is degraded', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ capability_tier: 'schema', runtime_status: 'degraded' })}
        metrics={null}
        isLoading={false}
      />
    );
    const capSection = screen.getByTestId('skill-capability');
    expect(capSection).toHaveTextContent('Schema');
    expect(capSection).toHaveTextContent('Degraded');
    expect(capSection).toHaveTextContent('has schema but no executor');
  });

  it('should show unavailable warning when runtime_status is unavailable', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ capability_tier: 'metadata', runtime_status: 'unavailable' })}
        metrics={null}
        isLoading={false}
      />
    );
    const capSection = screen.getByTestId('skill-capability');
    expect(capSection).toHaveTextContent('Metadata');
    expect(capSection).toHaveTextContent('Unavailable');
    expect(capSection).toHaveTextContent('not executable');
  });

  it('should render runtime binding and instruction length metadata', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({
          capability_tier: 'executable',
          runtime_status: 'ready',
          runtime_binding: 'prompt_instructions',
          instructions_length: 1234,
        })}
        metrics={null}
        isLoading={false}
      />,
    );
    const capSection = screen.getByTestId('skill-capability');
    expect(capSection).toHaveTextContent('Binding:');
    expect(capSection).toHaveTextContent('prompt_instructions');
    expect(capSection).toHaveTextContent('Instructions loaded: 1234 chars');
  });

  it('should render instructions panel when instructions exist', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({
          instructions: 'Step 1\nStep 2',
          instructions_length: 12,
        })}
        metrics={null}
        isLoading={false}
      />,
    );
    expect(screen.getByTestId('skill-instructions')).toBeInTheDocument();
    expect(screen.getByText(/Prompt body loaded from SKILL.md/)).toBeInTheDocument();
    expect(screen.getByText(/Step 1/)).toBeInTheDocument();
  });

  it('should render hook specs when provided', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({
          hooks: ['PreToolUse:Write|Edit (command)'],
          hook_specs: [
            {
              event: 'PreToolUse',
              type: 'command',
              matcher: 'Write|Edit',
              command: 'echo hello',
              handler: null,
            },
          ],
        })}
        metrics={null}
        isLoading={false}
      />,
    );

    const hooksSection = screen.getByTestId('skill-hooks');
    expect(hooksSection).toHaveTextContent('PreToolUse');
    expect(hooksSection).toHaveTextContent('Write|Edit');
    expect(hooksSection).toHaveTextContent('command:');
    expect(hooksSection).toHaveTextContent('echo hello');
  });

  it('should default to metadata/unavailable when fields are absent', () => {
    render(
      <SkillDetailPanel
        detail={createDetail()}
        metrics={null}
        isLoading={false}
      />
    );
    const capSection = screen.getByTestId('skill-capability');
    expect(capSection).toHaveTextContent('Metadata');
    expect(capSection).toHaveTextContent('Unavailable');
  });

  // ─── Edge cases ────────────────────────────────────────────────

  it('should handle missing display_name gracefully', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ display_name: '' })}
        metrics={null}
        isLoading={false}
      />
    );
    // Falls back to name
    expect(screen.getByText('planning-with-files')).toBeInTheDocument();
  });

  it('should handle unknown certification level', () => {
    render(
      <SkillDetailPanel
        detail={createDetail({ certification: 'expert' as any })}
        metrics={null}
        isLoading={false}
      />
    );
    // Falls back to Unverified styling
    expect(screen.getByText('Unverified')).toBeInTheDocument();
  });
});
