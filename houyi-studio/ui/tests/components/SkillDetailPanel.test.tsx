/**
 * Tests for SkillDetailPanel Tier 1 component (§7.10.3).
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { SkillDetailPanel } from '@/components/panels/SkillDetailPanel';
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

  // ─── Hooks section ─────────────────────────────────────────────

  it('should render hooks list', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const hooksSection = screen.getByTestId('skill-hooks');
    expect(hooksSection).toHaveTextContent('PreToolUse');
    expect(hooksSection).toHaveTextContent('PostToolUse');
    expect(hooksSection).toHaveTextContent('Stop');
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
    expect(screen.getByText('No metrics available yet')).toBeInTheDocument();
  });

  // ─── Action buttons ────────────────────────────────────────────

  it('should render Configure, Dry-run, and Unload buttons', () => {
    render(<SkillDetailPanel detail={createDetail()} metrics={null} isLoading={false} />);
    const actions = screen.getByTestId('skill-actions');
    expect(actions).toHaveTextContent('Configure...');
    expect(actions).toHaveTextContent('Dry-run');
    expect(actions).toHaveTextContent('Unload');
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
    fireEvent.click(screen.getByText('Unload'));

    // Click Cancel
    fireEvent.click(screen.getByText('Cancel'));
    expect(onUnload).not.toHaveBeenCalled();

    // Dialog should be closed
    expect(screen.queryByText('Unload Skill')).not.toBeInTheDocument();
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
