/**
 * Tests for DryRunDialog — CI/CD pipeline-style verification timeline.
 *
 * Coverage:
 *   - Required field validation ALWAYS enforced (no silent bypass)
 *   - Boolean fields: only true/false (no "not set" third option)
 *   - No programmer-type labels (no "nullable", no "(string)")
 *   - Live mode toggle sends `live` flag
 *   - Pipeline verification stages: registration, schema, policy, side effects, hooks, LLM
 *   - Progressive reveal animation
 *   - Collapsible input section
 *   - Mode switch clears results
 */
import { render, screen, fireEvent, within, act } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

import { DryRunDialog } from '@/components/panels/DryRunDialog';
import type { SkillDetail } from '@/types/websocket';
import type { DryRunResultData } from '@/components/LeftSidebar/useSkillsLogic';

const createDetail = (overrides: Partial<SkillDetail> = {}): SkillDetail => ({
  name: 'web_search',
  display_name: 'Web Search',
  description: 'Search the web',
  version: '1.0.0',
  tools: [
    {
      name: 'web_search',
      description: 'Search the web for information',
      input_schema: {
        type: 'object',
        properties: {
          query: { type: 'string', title: 'Query', description: 'Search query' },
          max_results: { type: 'integer', title: 'Max Results', description: 'Maximum results', default: 3 },
          provider: { anyOf: [{ type: 'string' }, { type: 'null' }], title: 'Provider', description: 'Search provider' },
          use_cache: { anyOf: [{ type: 'boolean' }, { type: 'null' }], title: 'Use Cache', description: 'Enable caching', default: null },
          mode: { anyOf: [{ type: 'string' }, { type: 'null' }], title: 'Mode', description: 'Search mode' },
        },
        required: ['query'],
      },
    },
  ],
  permissions: [],
  policy: { default_action: 'allow' },
  hooks: [],
  certification: 'gold',
  side_effect: 'network',
  ...overrides,
});

const createMultiToolDetail = (): SkillDetail => ({
  ...createDetail(),
  name: 'planning-with-files',
  display_name: 'Planning with Files',
  tools: [
    { name: 'Read', description: 'Read a file' },
    { name: 'Write', description: 'Write a file' },
    { name: 'Edit', description: 'Edit a file' },
  ],
});

const createPassResult = (): DryRunResultData => ({
  valid: true,
  schema_errors: [],
  policy_result: 'allow',
  capability_gaps: [],
  estimated_side_effects: [],
});

const createFailResult = (): DryRunResultData => ({
  valid: false,
  schema_errors: ['Missing required field: query'],
  policy_result: 'deny',
  capability_gaps: ['network access unavailable'],
  estimated_side_effects: ['network'],
});

/** Advance all timers to reveal every progressive stage. */
const revealAllStages = () => {
  act(() => { vi.advanceTimersByTime(3000); });
};

describe('DryRunDialog', () => {
  const defaultProps = {
    isOpen: true,
    detail: createDetail(),
    dryRunResult: null as DryRunResultData | null,
    onExecute: vi.fn(),
    onClose: vi.fn(),
    onClearResult: vi.fn(),
  };

  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ─── Rendering ────────────────────────────────────────────────

  it('renders dialog when open', () => {
    render(<DryRunDialog {...defaultProps} />);
    expect(screen.getByTestId('dry-run-dialog')).toBeInTheDocument();
    expect(screen.getByText(/Dry-run: Web Search/)).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(<DryRunDialog {...defaultProps} isOpen={false} />);
    expect(screen.queryByTestId('dry-run-dialog')).not.toBeInTheDocument();
  });

  it('renders form/json mode toggle', () => {
    render(<DryRunDialog {...defaultProps} />);
    expect(screen.getByTestId('dry-run-mode-form')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-mode-json')).toBeInTheDocument();
  });

  it('renders execute button', () => {
    render(<DryRunDialog {...defaultProps} />);
    expect(screen.getByTestId('dry-run-execute')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-execute')).toHaveTextContent('Execute Dry-run');
  });

  // ─── Single tool: no tool selector ────────────────────────────

  it('shows tool info instead of selector for single-tool skill', () => {
    render(<DryRunDialog {...defaultProps} />);
    expect(screen.queryByTestId('dry-run-tool-select')).not.toBeInTheDocument();
    expect(screen.getByText('web_search')).toBeInTheDocument();
  });

  // ─── Multi tool: tool selector ────────────────────────────────

  it('shows tool selector for multi-tool skill', () => {
    render(<DryRunDialog {...defaultProps} detail={createMultiToolDetail()} />);
    expect(screen.getByTestId('dry-run-tool-select')).toBeInTheDocument();
    const select = screen.getByTestId('dry-run-tool-select');
    expect(select).toHaveValue('Read');
  });

  it('allows switching tools in selector', () => {
    render(<DryRunDialog {...defaultProps} detail={createMultiToolDetail()} />);
    const select = screen.getByTestId('dry-run-tool-select');
    fireEvent.change(select, { target: { value: 'Write' } });
    expect(select).toHaveValue('Write');
  });

  // ─── Form mode ────────────────────────────────────────────────

  it('renders form fields from input_schema', () => {
    render(<DryRunDialog {...defaultProps} />);
    expect(screen.getByTestId('dry-run-form-inputs')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-input-query')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-input-max_results')).toBeInTheDocument();
  });

  it('shows required indicator for required fields', () => {
    render(<DryRunDialog {...defaultProps} />);
    const formInputs = screen.getByTestId('dry-run-form-inputs');
    expect(within(formInputs).getByText('*')).toBeInTheDocument();
  });

  it('shows "no input parameters" message when tool has no schema', () => {
    const detail = createDetail({
      tools: [{ name: 'simple_tool', description: 'No params' }],
    });
    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByText(/No input parameters/)).toBeInTheDocument();
  });

  // ─── Required field validation (CRITICAL FIX) ─────────────────

  it('blocks execution when required field is empty', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).not.toHaveBeenCalled();
    expect(screen.getByText('This field is required')).toBeInTheDocument();
  });

  it('clears validation error when required field is filled', () => {
    render(<DryRunDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(screen.getByText('This field is required')).toBeInTheDocument();
    fireEvent.change(screen.getByTestId('dry-run-input-query'), {
      target: { value: 'test' },
    });
    expect(screen.queryByText('This field is required')).not.toBeInTheDocument();
  });

  it('clears form validation errors when switching to JSON mode', () => {
    render(<DryRunDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(screen.getByText('This field is required')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    fireEvent.click(screen.getByTestId('dry-run-mode-form'));
    expect(screen.queryByText('This field is required')).not.toBeInTheDocument();
  });

  it('allows execution when required field is filled', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.change(screen.getByTestId('dry-run-input-query'), {
      target: { value: 'test query' },
    });
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith('web_search', expect.objectContaining({ query: 'test query' }), false);
  });

  it('does NOT bypass validation for empty form (no "availability check" shortcut)', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).not.toHaveBeenCalled();
  });

  // ─── Boolean fields: no "not set" option ────────────────────────

  it('boolean field has only true/false options (no "skip" or "not set")', () => {
    render(<DryRunDialog {...defaultProps} />);
    const select = screen.getByTestId('dry-run-input-use_cache') as HTMLSelectElement;
    const options = Array.from(select.options).map(o => o.text);
    expect(options).toEqual(['true', 'false']);
    expect(options.some(o => o.includes('skip'))).toBe(false);
    expect(options.some(o => o.includes('not set'))).toBe(false);
  });

  // ─── No programmer labels ──────────────────────────────────────

  it('does not show "(string, nullable)" programmer labels', () => {
    render(<DryRunDialog {...defaultProps} />);
    const formInputs = screen.getByTestId('dry-run-form-inputs');
    expect(formInputs.textContent).not.toContain('(string, nullable)');
    expect(formInputs.textContent).not.toContain('(string)');
    expect(formInputs.textContent).not.toContain('nullable');
  });

  it('shows "optional" label for non-required fields', () => {
    render(<DryRunDialog {...defaultProps} />);
    const formInputs = screen.getByTestId('dry-run-form-inputs');
    expect(within(formInputs).getAllByText('optional').length).toBeGreaterThanOrEqual(1);
  });

  // ─── Default values ────────────────────────────────────────────

  it('shows default value in placeholder for optional fields', () => {
    render(<DryRunDialog {...defaultProps} />);
    const maxResultsInput = screen.getByTestId('dry-run-input-max_results') as HTMLInputElement;
    expect(maxResultsInput.placeholder).toBe('Default: 3');
  });

  // ─── JSON mode ────────────────────────────────────────────────

  it('switches to JSON mode', () => {
    render(<DryRunDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    expect(screen.getByTestId('dry-run-json-input')).toBeInTheDocument();
    expect(screen.queryByTestId('dry-run-form-inputs')).not.toBeInTheDocument();
  });

  it('shows JSON textarea with default empty object', () => {
    render(<DryRunDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    const textarea = screen.getByTestId('dry-run-json-input').querySelector('textarea');
    expect(textarea).toHaveValue('{}');
  });

  // ─── Execute ──────────────────────────────────────────────────

  it('executes with filled form fields', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.change(screen.getByTestId('dry-run-input-query'), {
      target: { value: 'test query' },
    });
    fireEvent.change(screen.getByTestId('dry-run-input-max_results'), {
      target: { value: '5' },
    });
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith('web_search', {
      query: 'test query',
      max_results: 5,
    }, false);
  });

  it('executes with JSON input', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    const textarea = screen.getByTestId('dry-run-json-input').querySelector('textarea')!;
    fireEvent.change(textarea, { target: { value: '{"query": "hello"}' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith('web_search', { query: 'hello' }, false);
  });

  it('shows error for invalid JSON', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    const textarea = screen.getByTestId('dry-run-json-input').querySelector('textarea')!;
    fireEvent.change(textarea, { target: { value: '{ invalid }' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).not.toHaveBeenCalled();
    expect(screen.getByText(/Invalid JSON/)).toBeInTheDocument();
  });

  // ─── Live mode ────────────────────────────────────────────────

  it('renders live mode toggle', () => {
    render(<DryRunDialog {...defaultProps} />);
    expect(screen.getByTestId('dry-run-live-toggle')).toBeInTheDocument();
  });

  it('sends live=true when live mode is enabled', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.change(screen.getByTestId('dry-run-input-query'), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith('web_search', expect.objectContaining({ query: 'test' }), true);
  });

  it('changes button text in live mode', () => {
    render(<DryRunDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    expect(screen.getByTestId('dry-run-execute')).toHaveTextContent('Execute Live Dry-run');
  });

  // ─── Pipeline: verification pipeline display ────────────────────

  it('shows pipeline panel with ALL PASSED for valid result', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    expect(screen.getByTestId('dry-run-result-panel')).toBeInTheDocument();
    expect(screen.getByText('ALL PASSED')).toBeInTheDocument();
    expect(screen.getByText('Verification Pipeline')).toBeInTheDocument();
  });

  it('shows pipeline with FAILED header for invalid result', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createFailResult()} />);
    revealAllStages();
    const panel = screen.getByTestId('dry-run-result-panel');
    expect(within(panel).getByText('FAILED')).toBeInTheDocument();
  });

  it('shows all 5 core stages (registration, schema, policy, side-effects, hooks)', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    expect(screen.getByTestId('dry-run-stage-registration')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-stage-schema')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-stage-policy')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-stage-side-effects')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-stage-hooks')).toBeInTheDocument();
  });

  it('shows registration stage with skill name, tools, and certification', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-registration');
    expect(stage).toHaveTextContent('Skill Registration');
    expect(stage).toHaveTextContent(/Web Search/);
    expect(stage).toHaveTextContent('1 tool(s)');
    expect(stage).toHaveTextContent('gold');
    expect(stage).toHaveTextContent('web_search');
  });

  it('shows schema PASS for valid input', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-schema');
    expect(stage).toHaveTextContent('PASS');
    expect(stage).toHaveTextContent(/Input conforms to tool schema/);
  });

  it('shows schema FAIL with error details', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createFailResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-schema');
    expect(stage).toHaveTextContent('FAIL');
    expect(stage).toHaveTextContent(/Missing required field/);
  });

  it('shows policy stage — allow', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-policy');
    expect(stage).toHaveTextContent('PASS');
    expect(stage).toHaveTextContent(/Invocation allowed/);
  });

  it('shows policy stage — deny', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createFailResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-policy');
    expect(stage).toHaveTextContent('FAIL');
    expect(stage).toHaveTextContent(/denied by policy/);
  });

  it('shows side effects when declared', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createFailResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-side-effects');
    expect(stage).toHaveTextContent('WARN');
    expect(stage).toHaveTextContent('network');
  });

  it('shows no side effects for clean result', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-side-effects');
    expect(stage).toHaveTextContent('PASS');
    expect(stage).toHaveTextContent(/No side effects/);
  });

  it('shows hooks stage with SKIPPED when no hooks', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-hooks');
    expect(stage).toHaveTextContent('SKIPPED');
    expect(stage).toHaveTextContent(/No lifecycle hooks/);
  });

  it('shows hooks stage with PASS and hook names when hooks are registered', () => {
    const detail = createDetail({ hooks: ['PreToolUse', 'PostToolUse', 'Stop'] });
    render(<DryRunDialog {...defaultProps} detail={detail} dryRunResult={createPassResult()} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-hooks');
    expect(stage).toHaveTextContent('PASS');
    expect(stage).toHaveTextContent('3 hook(s) registered');
    expect(stage).toHaveTextContent('PreToolUse');
    expect(stage).toHaveTextContent('PostToolUse');
    expect(stage).toHaveTextContent('Stop');
  });

  it('shows capability gaps stage when present', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createFailResult()} />);
    revealAllStages();
    expect(screen.getByTestId('dry-run-stage-gaps')).toBeInTheDocument();
    const stage = screen.getByTestId('dry-run-stage-gaps');
    expect(stage).toHaveTextContent('WARN');
    expect(stage).toHaveTextContent(/network access unavailable/);
  });

  it('omits capability gaps stage when none', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    expect(screen.queryByTestId('dry-run-stage-gaps')).not.toBeInTheDocument();
  });

  // ─── Pipeline: LLM verification ──────────────────────────────

  it('shows LLM verification with progressive disclosure phases and response', () => {
    const result: DryRunResultData = {
      ...createPassResult(),
      llm_verification: {
        success: true,
        message: "LLM correctly called 'web_search'",
        tool_call: { name: 'web_search', arguments: { query: 'test' } },
        probe_prompt: "I need help with a task.",
        system_prompt: "You are a helpful assistant.",
        tool_definitions: [{
          type: 'function',
          function: {
            name: 'web_search',
            description: 'Search the web',
            parameters: {
              properties: { query: { type: 'string' }, max_results: { type: 'integer' } },
              required: ['query'],
            },
          },
        }],
        model_name: 'gemini-3-pro-preview',
        usage: { prompt_tokens: 50, completion_tokens: 10, total_tokens: 60 },
        phases: [
          { name: 'discovery', label: 'Skill Discovery', timestamp_ms: 0, status: 'pass', data: { skill_name: 'web_search', description: 'Search the web' } },
          { name: 'activation', label: 'Tool Activation', timestamp_ms: 2, status: 'pass', data: { tool_count: 1, tool_names: ['web_search'] } },
          { name: 'negotiation', label: 'LLM Negotiation', timestamp_ms: 5, status: 'pass', data: { system_prompt_length: 40, user_query: 'I need help with a task.' } },
          { name: 'execution', label: 'LLM Execution', timestamp_ms: 1200, status: 'pass', data: { model: 'gemini-3-pro-preview', latency_ms: 1195 } },
        ],
      },
    };
    render(<DryRunDialog {...defaultProps} dryRunResult={result} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-llm-verify');
    expect(stage).toHaveTextContent('PASS');
    const flow = screen.getByTestId('llm-verify-flow');
    // Phase timeline with timestamps
    expect(flow).toHaveTextContent('Skill Discovery');
    expect(flow).toHaveTextContent('Tool Activation');
    expect(flow).toHaveTextContent('LLM Negotiation');
    expect(flow).toHaveTextContent('LLM Execution');
    expect(flow).toHaveTextContent('t=0ms');
    expect(flow).toHaveTextContent('t=1200ms');
    // LLM Response section — tool call JSON
    expect(flow).toHaveTextContent('LLM Response');
    expect(flow).toHaveTextContent('"web_search"');
    // Usage stats (prove real API call)
    expect(flow).toHaveTextContent('prompt_tokens: 50');
    expect(flow).toHaveTextContent('total_tokens: 60');
    // Model badge + validation
    expect(flow).toHaveTextContent('gemini-3-pro-preview');
    expect(flow).toHaveTextContent(/LLM correctly called/);
  });

  it('shows LLM verification FAIL', () => {
    const result: DryRunResultData = {
      ...createPassResult(),
      llm_verification: {
        success: false,
        message: 'LLM did not call the expected tool',
      },
    };
    render(<DryRunDialog {...defaultProps} dryRunResult={result} />);
    revealAllStages();
    const stage = screen.getByTestId('dry-run-stage-llm-verify');
    expect(stage).toHaveTextContent('FAIL');
    expect(stage).toHaveTextContent(/did not call/);
  });

  it('shows raw LLM content when present', () => {
    const result: DryRunResultData = {
      ...createPassResult(),
      llm_verification: {
        success: true,
        message: "LLM correctly called 'web_search'",
        tool_call: { name: 'web_search', arguments: { query: 'test' } },
        raw_content: 'I will search for that.',
      },
    };
    render(<DryRunDialog {...defaultProps} dryRunResult={result} />);
    revealAllStages();
    const flow = screen.getByTestId('llm-verify-flow');
    expect(flow).toHaveTextContent('I will search for that.');
  });

  // ─── Pipeline: progressive reveal ────────────────────────────

  it('reveals pipeline stages progressively (not all at once)', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);

    // Before any timer tick — stages exist but are unrevealed (opacity-40)
    const regStage = screen.getByTestId('dry-run-stage-registration');
    expect(regStage.className).toContain('opacity-40');

    // After first tick (200ms) — registration reveals
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.getByTestId('dry-run-stage-registration').className).toContain('animate-stageReveal');
    expect(screen.getByTestId('dry-run-stage-schema').className).toContain('opacity-40');

    // After second tick — schema reveals
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.getByTestId('dry-run-stage-schema').className).toContain('animate-stageReveal');
    expect(screen.getByTestId('dry-run-stage-policy').className).toContain('opacity-40');
  });

  it('shows all stages as pending when executing (before result)', () => {
    const { rerender } = render(<DryRunDialog {...defaultProps} />);

    // Simulate execution by filling and clicking
    fireEvent.change(screen.getByTestId('dry-run-input-query'), { target: { value: 'test' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    // Pipeline should be visible with "Running..." header
    expect(screen.getByTestId('dry-run-result-panel')).toBeInTheDocument();
    expect(screen.getByText(/Running/)).toBeInTheDocument();

    // All stages should show as pending (opacity-40)
    const stages = screen.getAllByTestId(/^dry-run-stage-/);
    stages.forEach((stage) => {
      expect(stage.className).toContain('opacity-40');
    });
  });

  // ─── Collapsible input section ────────────────────────────────

  it('collapses input when result arrives and shows toggle', () => {
    const { rerender } = render(<DryRunDialog {...defaultProps} />);
    // Input should be visible initially
    expect(screen.getByTestId('dry-run-form-inputs')).toBeInTheDocument();

    // Re-render with result to trigger collapse
    rerender(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();

    // Input toggle should appear
    expect(screen.getByTestId('dry-run-input-toggle')).toBeInTheDocument();
    // Form should be hidden (collapsed)
    expect(screen.queryByTestId('dry-run-form-inputs')).not.toBeInTheDocument();
  });

  it('expands input section when toggle is clicked', () => {
    const { rerender } = render(<DryRunDialog {...defaultProps} />);
    rerender(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();

    // Input is collapsed
    expect(screen.queryByTestId('dry-run-form-inputs')).not.toBeInTheDocument();

    // Click toggle to expand
    fireEvent.click(screen.getByTestId('dry-run-input-toggle'));
    expect(screen.getByTestId('dry-run-form-inputs')).toBeInTheDocument();
  });

  // ─── Bug fix: results persist across mode switch ──────────────────

  it('preserves pipeline results when switching to JSON mode', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    expect(screen.getByTestId('dry-run-result-panel')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    // Pipeline should still be visible
    expect(screen.getByTestId('dry-run-result-panel')).toBeInTheDocument();
    expect(screen.getByText('ALL PASSED')).toBeInTheDocument();
  });

  it('preserves pipeline results when switching back to Form mode', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    fireEvent.click(screen.getByTestId('dry-run-mode-form'));
    expect(screen.getByTestId('dry-run-result-panel')).toBeInTheDocument();
    expect(screen.getByText('ALL PASSED')).toBeInTheDocument();
  });

  // ─── Close behavior ───────────────────────────────────────────

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn();
    render(<DryRunDialog {...defaultProps} onClose={onClose} />);
    fireEvent.click(screen.getByTestId('center-stage-close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
