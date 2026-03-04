import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { DryRunDialog } from '@/components/panels/skill/DryRunDialog';
import type { DryRunResultData } from '@/components/LeftSidebar/useSkillsLogic';
import type { SkillDetail } from '@/types/websocket';

const createPlanningDetail = (): SkillDetail => ({
  name: 'planning-with-files',
  display_name: 'Planning with Files',
  description: 'Plan tasks',
  version: '1.0.0',
  tools: [
    {
      name: 'planning-with-files',
      description: 'Plan tasks',
      input_schema: {
        type: 'object',
        properties: {
          action: { $ref: '#/$defs/ActionEnum' },
          task: { type: 'string' },
          subtasks: { type: 'string' },
          subtask_index: { $ref: '#/$defs/SubtaskIndex' },
          completed: { type: 'boolean', default: true },
        },
        required: ['action'],
        $defs: {
          ActionEnum: {
            type: 'string',
            enum: ['create', 'update', 'complete', 'status'],
          },
          SubtaskIndex: {
            type: 'integer',
            minimum: 0,
            maximum: 99,
          },
        },
      },
    },
  ],
  permissions: [],
  policy: { default_action: 'allow' },
  hooks: [],
  certification: 'gold',
  side_effect: 'filesystem',
  is_core: true,
  source: 'builtin',
});

const createPassResult = (): DryRunResultData => ({
  valid: true,
  schema_errors: [],
  policy_result: 'allow',
  capability_gaps: [],
  estimated_side_effects: [],
});

describe('DryRunDialog planning rules', () => {
  const defaultProps = {
    isOpen: true,
    detail: createPlanningDetail(),
    dryRunResult: null as DryRunResultData | null,
    onExecute: vi.fn(),
    onClose: vi.fn(),
    onClearResult: vi.fn(),
  };

  it('shows action-driven fields and required markers', () => {
    render(<DryRunDialog {...defaultProps} />);

    const action = screen.getByTestId('dry-run-input-action') as HTMLSelectElement;
    expect(action.tagName).toBe('SELECT');
    expect(action.value).toBe('create');
    expect(Array.from(action.options).map((o) => o.value)).toEqual([
      '',
      'create',
      'update',
      'complete',
      'status',
    ]);

    expect(screen.getByTestId('dry-run-input-task')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-input-subtasks')).toBeInTheDocument();
    expect(screen.queryByTestId('dry-run-input-subtask_index')).not.toBeInTheDocument();

    const taskField = screen.getByTestId('dry-run-input-task').closest('div');
    expect(taskField).toBeTruthy();
    expect(within(taskField as HTMLElement).getByTitle('Required')).toBeInTheDocument();

    fireEvent.change(action, { target: { value: 'update' } });
    const subtaskField = screen.getByTestId('dry-run-input-subtask_index').closest('div');
    expect(subtaskField).toBeTruthy();
    expect(within(subtaskField as HTMLElement).getByTitle('Required')).toBeInTheDocument();
  });

  it('clears required error after selecting action', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);

    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(screen.getByText('Task is required when action is create')).toBeInTheDocument();

    fireEvent.change(screen.getByTestId('dry-run-input-task'), {
      target: { value: 'Create plan task' },
    });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'planning-with-files',
      expect.objectContaining({ action: 'create', task: 'Create plan task' }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('blocks create when task is missing', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);

    fireEvent.change(screen.getByTestId('dry-run-input-action'), {
      target: { value: 'create' },
    });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).not.toHaveBeenCalled();
    expect(screen.getByText('Task is required when action is create')).toBeInTheDocument();
  });

  it('blocks update when subtask_index is missing', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);

    fireEvent.change(screen.getByTestId('dry-run-input-action'), {
      target: { value: 'update' },
    });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).not.toHaveBeenCalled();
    expect(screen.getByText('subtask_index is required when action is update')).toBeInTheDocument();
  });

  it('shows conditional validation errors in JSON mode', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);

    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '{"action":"create"}' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).not.toHaveBeenCalled();
    expect(screen.getByTestId('dry-run-json-validation-errors')).toHaveTextContent('Task is required when action is create');

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '{"action":"update"}' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(screen.getByTestId('dry-run-json-validation-errors')).toHaveTextContent('subtask_index is required when action is update');
  });

  it('clears previous result when toggling live mode', () => {
    const onClearResult = vi.fn();
    render(
      <DryRunDialog
        {...defaultProps}
        onClearResult={onClearResult}
        dryRunResult={createPassResult()}
      />,
    );

    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    expect(onClearResult).toHaveBeenCalled();
  });
});
