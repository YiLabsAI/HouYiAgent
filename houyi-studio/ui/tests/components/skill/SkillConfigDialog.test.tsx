/**
 * Tests for SkillConfigDialog component.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { SkillConfigDialog } from '@/components/panels/skill/SkillConfigDialog';
import type { SkillDetail } from '@/types/websocket';

const createDetail = (overrides: Partial<SkillDetail> = {}): SkillDetail => ({
  name: 'web_search',
  display_name: 'Web Search',
  description: 'Search the web.',
  version: '1.0.0',
  author: 'HouYi',
  tools: [{ name: 'web_search', description: 'Search' }],
  permissions: [],
  policy: { default_action: 'allow', model_auto_invoke: true },
  hooks: [],
  certification: 'unverified',
  side_effect: 'network',
  ...overrides,
});

describe('SkillConfigDialog', () => {
  let onSave: ReturnType<typeof vi.fn>;
  let onCancel: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onSave = vi.fn();
    onCancel = vi.fn();
  });

  it('should not render when isOpen is false', () => {
    render(
      <SkillConfigDialog
        isOpen={false}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    expect(screen.queryByText('Configure Skill')).not.toBeInTheDocument();
  });

  it('should render when isOpen is true', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    expect(screen.getByText('Configure Skill')).toBeInTheDocument();
    expect(screen.getByText(/Web Search/)).toBeInTheDocument();
  });

  it('should show three policy options', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    expect(screen.getByText('Allow')).toBeInTheDocument();
    expect(screen.getByText('Require Consent')).toBeInTheDocument();
    expect(screen.getByText('Deny')).toBeInTheDocument();
  });

  it('should pre-select the current policy', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail({ policy: { default_action: 'deny' } })}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    const denyRadio = screen.getByDisplayValue('deny');
    expect(denyRadio).toBeChecked();
  });

  it('should call onSave with updated values when Save is clicked', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    // Change policy to deny
    fireEvent.click(screen.getByDisplayValue('deny'));

    // Click save
    fireEvent.click(screen.getByText('Save Changes'));

    expect(onSave).toHaveBeenCalledTimes(1);
    const savedValues = onSave.mock.calls[0][0];
    expect(savedValues.policy_action).toBe('deny');
    expect(typeof savedValues.auto_invoke).toBe('boolean');
  });

  it('should call onCancel when Cancel is clicked', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    fireEvent.click(screen.getByText('Cancel'));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('should show auto-invoke toggle', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    expect(screen.getByText('Allow LLM to trigger automatically')).toBeInTheDocument();
  });

  it('should show runtime-only info box', () => {
    render(
      <SkillConfigDialog
        isOpen={true}
        detail={createDetail()}
        onSave={onSave}
        onCancel={onCancel}
      />,
    );
    expect(
      screen.getByText(/Changes apply at runtime and do not modify the SKILL\.md file/),
    ).toBeInTheDocument();
  });
});
