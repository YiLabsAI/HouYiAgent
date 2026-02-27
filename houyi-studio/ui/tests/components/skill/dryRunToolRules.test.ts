import { describe, expect, it } from 'vitest';

import type { DryRunSchemaField } from '@/components/panels/skill/dryRun/SkillFieldInput';
import {
  applyToolSpecificValidation,
  getToolFieldRules,
  isPlanningWithFilesTool,
  normalizeToolInput,
  shouldShowActionHint,
} from '@/components/panels/skill/dryRun/dryRunToolRules';

const createField = (overrides: Partial<DryRunSchemaField> = {}): DryRunSchemaField => ({
  name: 'task',
  type: 'string',
  title: 'Task',
  description: 'Task title',
  required: false,
  nullable: false,
  ...overrides,
});

describe('dryRunToolRules', () => {
  it('detects planning tool names including external alias', () => {
    expect(isPlanningWithFilesTool('planning-with-files')).toBe(true);
    expect(isPlanningWithFilesTool('ext__planning-with-files')).toBe(true);
    expect(isPlanningWithFilesTool('web_search')).toBe(false);
  });

  it('keeps non-planning fields visible and preserves required flag', () => {
    const optionalField = createField({ required: false });
    const requiredField = createField({ required: true });

    expect(getToolFieldRules('web_search', optionalField, {})).toEqual({
      visible: true,
      required: false,
    });
    expect(getToolFieldRules('web_search', requiredField, {})).toEqual({
      visible: true,
      required: true,
    });
  });

  it('applies planning action-driven visibility and required rules', () => {
    expect(getToolFieldRules('planning-with-files', createField({ name: 'action' }), {})).toEqual({
      visible: true,
      required: true,
    });

    expect(getToolFieldRules('planning-with-files', createField({ name: 'task' }), {})).toEqual({
      visible: false,
      required: false,
    });

    expect(
      getToolFieldRules(
        'planning-with-files',
        createField({ name: 'task' }),
        { action: 'create' },
      ),
    ).toEqual({
      visible: true,
      required: true,
    });

    expect(
      getToolFieldRules(
        'planning-with-files',
        createField({ name: 'subtask_index' }),
        { action: 'update' },
      ),
    ).toEqual({
      visible: true,
      required: true,
    });
  });

  it('validates planning conditional requirements', () => {
    const errorsCreate: Record<string, string> = {};
    applyToolSpecificValidation('planning-with-files', { action: 'create' }, errorsCreate);
    expect(errorsCreate).toEqual({
      task: 'Task is required when action is create',
    });

    const errorsUpdate: Record<string, string> = {};
    applyToolSpecificValidation('ext__planning-with-files', { action: 'update' }, errorsUpdate);
    expect(errorsUpdate).toEqual({
      subtask_index: 'subtask_index is required when action is update',
    });
  });

  it('normalizes planning create task input and keeps non-planning untouched', () => {
    const normalizedPlanning = normalizeToolInput('planning-with-files', {
      action: 'create',
      task: '  Build adapter  ',
    });
    expect(normalizedPlanning).toEqual({
      action: 'create',
      task: 'Build adapter',
    });

    const nonPlanningInput = {
      action: 'create',
      task: '  Keep spaces  ',
    };
    const normalizedNonPlanning = normalizeToolInput('web_search', nonPlanningInput);
    expect(normalizedNonPlanning).toBe(nonPlanningInput);
  });

  it('shows action hint only when planning action is not selected', () => {
    expect(shouldShowActionHint('planning-with-files', {})).toBe(true);
    expect(shouldShowActionHint('planning-with-files', { action: 'create' })).toBe(false);
    expect(shouldShowActionHint('web_search', {})).toBe(false);
  });
});
