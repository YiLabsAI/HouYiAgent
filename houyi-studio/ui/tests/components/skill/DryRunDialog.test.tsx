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

import { DryRunDialog } from '@/components/panels/skill/DryRunDialog';
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
  is_core: overrides.is_core ?? false,
});

const createCorePlanningToolDetail = (): SkillDetail => ({
  ...createPlanningToolDetail(),
  name: 'planning-with-files',
  display_name: 'Planning with Files (Core)',
  tools: [
    {
      ...createPlanningToolDetail().tools[0],
      name: 'planning-with-files',
    },
  ],
});

const createUsingSuperpowersDetail = (): SkillDetail => ({
  ...createDetail({
    name: 'using-superpowers',
    display_name: 'using-superpowers',
    runtime_binding: 'prompt_instructions',
    instructions_length: 1800,
    instructions: 'Invoke relevant skills before any response. Follow process skill order and announce selected skills.',
  }),
  tools: [
    {
      name: 'using-superpowers',
      description: 'Meta workflow discipline',
    },
  ],
});

const createFrontendDesignDetail = (): SkillDetail => ({
  ...createDetail({
    name: 'frontend-design',
    display_name: 'frontend-design',
    runtime_binding: 'prompt_instructions',
    instructions_length: 2100,
    instructions: 'Choose bold aesthetic direction before coding and avoid generic AI style choices.',
  }),
  tools: [
    {
      name: 'frontend-design',
      description: 'Frontend design workflow',
    },
  ],
});

const createSkillCreatorDetail = (): SkillDetail => ({
  ...createDetail({
    name: 'skill-creator',
    display_name: 'skill-creator',
    runtime_binding: 'prompt_instructions',
    instructions_length: 3600,
    instructions: 'Follow understand, plan, init, edit, package, and iterate workflow to build skills.',
  }),
  tools: [
    {
      name: 'skill-creator',
      description: 'Skill creation guide',
    },
  ],
});

const createNotebooklmDetail = (): SkillDetail => ({
  ...createDetail({
    name: 'notebooklm',
    display_name: 'notebooklm',
    runtime_binding: 'prompt_instructions',
    instructions_length: 5400,
    instructions: 'Always use scripts/run.py. Check auth_manager.py status first, then notebook_manager.py add/list and ask_question.py.',
  }),
  tools: [
    {
      name: 'notebooklm',
      description: 'NotebookLM workflow helper',
    },
  ],
});

const createRagSkillDetail = (): SkillDetail => ({
  ...createDetail({
    name: 'rag-skill',
    display_name: 'rag-skill',
    runtime_binding: 'prompt_instructions',
    instructions_length: 3200,
    instructions: 'Read data_structure.md in knowledge/. Use progressive retrieval. Do not read entire files.',
  }),
  tools: [
    {
      name: 'rag-skill',
      description: 'Knowledge retrieval helper',
    },
  ],
});

const createPlanningToolDetail = (): SkillDetail => ({
  ...createDetail({
    name: 'ext__planning-with-files',
    display_name: 'Planning with Files',
    runtime_binding: 'prompt_instructions',
    instructions_length: 5886,
  }),
  tools: [
    {
      name: 'ext__planning-with-files',
      description: 'Planning workflow skill',
      input_schema: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['create', 'update', 'complete', 'status'] },
          task: { type: 'string' },
          subtasks: { type: 'array', items: { type: 'string' } },
          subtask_index: { type: 'integer' },
          completed: { type: 'boolean' },
        },
        required: ['action'],
      },
    },
  ],
});

const createRefSchemaDetail = (): SkillDetail => ({
  ...createDetail(),
  name: 'ref-schema-tool',
  display_name: 'Ref Schema Tool',
  tools: [
    {
      name: 'ref-schema-tool',
      description: 'Ref-based schema tool',
      input_schema: {
        type: 'object',
        properties: {
          phase: { $ref: '#/$defs/PhaseEnum' },
          retry_count: { $ref: '#/$defs/RetryCount' },
        },
        required: ['phase'],
        $defs: {
          PhaseEnum: {
            type: 'string',
            enum: ['draft', 'running', 'done'],
          },
          RetryCount: {
            type: 'integer',
            minimum: 0,
            maximum: 5,
          },
        },
      },
    },
  ],
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

const createLocationDetail = (): SkillDetail => ({
  ...createDetail(),
  name: 'location',
  display_name: 'Location',
  tools: [
    {
      name: 'get_location',
      description: 'Get coordinates by city',
      input_schema: {
        type: 'object',
        properties: {
          city: { anyOf: [{ type: 'string' }, { type: 'null' }], title: 'city' },
        },
      },
    },
  ],
});

const createWeatherDetail = (): SkillDetail => ({
  ...createDetail(),
  name: 'weather',
  display_name: 'Weather',
  tools: [
    {
      name: 'get_weather',
      description: 'Get weather by coords or city',
      input_schema: {
        type: 'object',
        properties: {
          lat: { anyOf: [{ type: 'number' }, { type: 'null' }], title: 'lat' },
          lon: { anyOf: [{ type: 'number' }, { type: 'null' }], title: 'lon' },
          city: { anyOf: [{ type: 'string' }, { type: 'null' }], title: 'city' },
          country: { anyOf: [{ type: 'string' }, { type: 'null' }], title: 'country' },
          date: { anyOf: [{ type: 'string' }, { type: 'null' }], title: 'date' },
          provider: { type: 'string', title: 'provider', default: 'auto' },
        },
      },
    },
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
  const setNavigatorUserAgentDataPlatform = (platform?: string) => {
    if (!platform) {
      Reflect.deleteProperty(window.navigator, 'userAgentData');
      return;
    }
    Object.defineProperty(window.navigator, 'userAgentData', {
      value: { platform },
      configurable: true,
    });
  };

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
    setNavigatorUserAgentDataPlatform(undefined);
  });

  it('renders weather country/provider as dropdowns', () => {
    render(<DryRunDialog {...defaultProps} detail={createWeatherDetail()} />);

    const country = screen.getByTestId('dry-run-input-country') as HTMLSelectElement;
    const provider = screen.getByTestId('dry-run-input-provider') as HTMLSelectElement;
    const date = screen.getByTestId('dry-run-input-date') as HTMLSelectElement;

    expect(country.tagName).toBe('SELECT');
    expect(provider.tagName).toBe('SELECT');
    expect(date.tagName).toBe('SELECT');
    expect(Array.from(provider.options).map((o) => o.value)).toEqual(['auto', 'openmeteo', 'wttr']);
    expect(Array.from(date.options).map((o) => o.value)).toEqual(['', 'today', 'tomorrow', 'day_after_tomorrow']);
  });

  it('resolves $ref enum fields as dropdown and shows numeric ranges', () => {
    render(<DryRunDialog {...defaultProps} detail={createRefSchemaDetail()} />);

    const phase = screen.getByTestId('dry-run-input-phase') as HTMLSelectElement;
    const retryCount = screen.getByTestId('dry-run-input-retry_count') as HTMLInputElement;

    expect(phase.tagName).toBe('SELECT');
    expect(Array.from(phase.options).map((o) => o.value)).toEqual([
      '',
      'draft',
      'running',
      'done',
    ]);

    expect(retryCount.min).toBe('0');
    expect(retryCount.max).toBe('5');
    expect(screen.getByTestId('dry-run-range-retry_count')).toHaveTextContent('Range: 0 to 5');
  });

  it('clears previous result when toggling live mode', () => {
    const onClearResult = vi.fn();
    render(
      <DryRunDialog
        {...defaultProps}
        detail={createDetail()}
        dryRunResult={createPassResult()}
        onClearResult={onClearResult}
      />,
    );

    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    expect(onClearResult).toHaveBeenCalled();
  });

  it('renders web_search provider/mode as dropdowns and submits selected values', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} detail={createDetail()} />);

    const provider = screen.getByTestId('dry-run-input-provider') as HTMLSelectElement;
    const mode = screen.getByTestId('dry-run-input-mode') as HTMLSelectElement;

    expect(Array.from(provider.options).map((o) => o.value)).toEqual(['', 'ddg', 'serper', 'tavily', 'bocha', 'searxng']);
    expect(Array.from(mode.options).map((o) => o.value)).toEqual(['', 'search', 'browse']);

    fireEvent.change(screen.getByTestId('dry-run-input-query'), { target: { value: 'houyi' } });
    fireEvent.change(provider, { target: { value: 'ddg' } });
    fireEvent.change(mode, { target: { value: 'browse' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'web_search',
      expect.objectContaining({ query: 'houyi', provider: 'ddg', mode: 'browse' }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('renders single-tool description with collapsible markdown', () => {
    const longDescription = [
      'NotebookLM execution flow:',
      '',
      '- **Step 1**: `python scripts/run.py auth_manager.py status`',
      '- **Step 2**: `python scripts/run.py notebook_manager.py list`',
      '- **Step 3**: `python scripts/run.py ask_question.py --question "..."`',
      '',
      'Always use run.py wrapper.',
    ].join('\n');

    render(
      <DryRunDialog
        {...defaultProps}
        detail={{
          ...createNotebooklmDetail(),
          tools: [{ name: 'notebooklm', description: longDescription }],
        }}
      />,
    );

    const toggle = screen.getByTestId('dry-run-tool-description-more');
    expect(toggle).toHaveTextContent('Show more');
    fireEvent.click(toggle);
    expect(screen.getByTestId('dry-run-tool-description')).toHaveTextContent('auth_manager.py status');
    expect(screen.getByTestId('dry-run-tool-description-more')).toHaveTextContent('Show less');
  });

  it('renders multiple generic presets for notebooklm', () => {
    render(<DryRunDialog {...defaultProps} detail={createNotebooklmDetail()} />);
    expect(screen.getByTestId('tool-flow-presets')).toBeInTheDocument();
    expect(screen.getByTestId('tool-flow-select-example-1-notebook-query')).toBeInTheDocument();
    expect(screen.getByTestId('tool-flow-select-example-2-library-discovery-add')).toBeInTheDocument();
  });

  it('prefills first notebooklm preset on first open and executes without reselection', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} detail={createNotebooklmDetail()} onExecute={onExecute} />);

    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"question": "Summarize the architecture decisions in this notebook"');

    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'notebooklm',
      expect.objectContaining({
        question: 'Summarize the architecture decisions in this notebook',
        notebook_url: 'https://notebooklm.google.com/notebook/example',
      }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('updates notebooklm payload when switching between generic examples', () => {
    render(<DryRunDialog {...defaultProps} detail={createNotebooklmDetail()} />);

    fireEvent.click(screen.getByTestId('tool-flow-select-example-2-library-discovery-add'));
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"operation": "add_notebook"');

    fireEvent.click(screen.getByTestId('tool-flow-select-example-1-notebook-query'));
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"question": "Summarize the architecture decisions in this notebook"');
  });

  it('renders workflow candidate audit metadata from skill detail', () => {
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'soffice.py',
          command: 'python scripts/office/soffice.py --headless --convert-to docx document.doc',
          params: ['headless', 'convert_to'],
          depends_on: ['soffice'],
          source: 'instructions',
          evidence: 'python scripts/office/soffice.py --headless --convert-to docx document.doc',
          confidence: 'medium',
          confidence_score: 0.75,
          validation: {
            status: 'warn',
            issues: ['Missing dependency: soffice'],
            missing_dependencies: ['soffice'],
          },
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByTestId('tool-flow-presets')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-workflow-command-preview')).toHaveTextContent('python scripts/office/soffice.py');
    expect(screen.getByTestId('dry-run-workflow-audit')).toHaveTextContent('Source: instructions');
    expect(screen.getByTestId('dry-run-workflow-audit')).toHaveTextContent('Confidence: medium');
    expect(screen.getByTestId('dry-run-workflow-validation')).toHaveTextContent('Validation: WARN');
    expect(screen.getByTestId('dry-run-workflow-evidence')).toHaveTextContent('Evidence: python scripts/office/soffice.py');
  });

  it('shows dependency recovery panel when selected workflow reports missing dependencies', () => {
    setNavigatorUserAgentDataPlatform('macOS');

    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'soffice.py',
          command: 'python scripts/office/soffice.py --headless --convert-to docx document.doc',
          params: ['headless', 'convert_to'],
          depends_on: ['soffice'],
          source: 'instructions',
          validation: {
            status: 'warn',
            issues: ['Missing dependency: soffice'],
            missing_dependencies: ['soffice'],
          },
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    expect(screen.getByTestId('dry-run-missing-deps-panel')).toHaveTextContent('Missing runtime dependencies: soffice');
    expect(screen.getByTestId('dry-run-install-commands')).toHaveTextContent('brew install --cask libreoffice');
    expect(screen.getByTestId('dry-run-install-commands')).toHaveTextContent('which soffice');
  });

  it('copies install commands from dependency recovery panel', async () => {
    setNavigatorUserAgentDataPlatform('macOS');

    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'soffice.py',
          command: 'python scripts/office/soffice.py --headless --convert-to docx document.doc',
          params: ['headless', 'convert_to'],
          depends_on: ['soffice'],
          source: 'instructions',
          validation: {
            status: 'warn',
            issues: ['Missing dependency: soffice'],
            missing_dependencies: ['soffice'],
          },
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    await act(async () => {
      fireEvent.click(screen.getByTestId('dry-run-copy-install-commands'));
    });

    expect(writeText).toHaveBeenCalledWith('brew install --cask libreoffice\nwhich soffice');
    expect(screen.getByTestId('dry-run-copy-install-commands')).toHaveTextContent('Copied');
  });

  it('shows windows install commands when platform is windows', () => {
    setNavigatorUserAgentDataPlatform('Windows');

    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'soffice.py',
          command: 'python scripts/office/soffice.py --headless --convert-to docx document.doc',
          params: ['headless', 'convert_to'],
          depends_on: ['soffice'],
          source: 'instructions',
          validation: {
            status: 'warn',
            issues: ['Missing dependency: soffice'],
            missing_dependencies: ['soffice'],
          },
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    expect(screen.getByTestId('dry-run-install-commands')).toHaveTextContent('winget install TheDocumentFoundation.LibreOffice');
    expect(screen.getByTestId('dry-run-install-commands')).toHaveTextContent('where soffice');
  });

  it('executes selected workflow_id and passes options payload', () => {
    const onExecute = vi.fn();
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'convert',
          command: 'python scripts/office/soffice.py --headless --convert-to docx document.doc',
          params: ['headless'],
          depends_on: ['soffice'],
          source: 'instructions',
        },
        {
          id: 'template_2',
          title: 'unpack',
          command: 'python scripts/office/unpack.py document.docx unpacked/',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-workflow-select-template_2'));
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'docx',
      expect.objectContaining({
        command: 'python scripts/office/unpack.py document.docx unpacked/',
      }),
      false,
      expect.objectContaining({ workflowId: 'template_2' }),
    );
  });

  it('executes selected workflow command even when skill example is selected', () => {
    const onExecute = vi.fn();
    const detail = createDetail({
      name: 'notebooklm',
      display_name: 'notebooklm',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'notebooklm', description: 'NotebookLM workflow helper' }],
      package_examples: [{
        id: 'quick-workflow-1',
        label: 'Quick Workflow 1 · Check library',
        description: 'Check library via command workflow',
        objective: 'Validate quick workflow step: Check library.',
        expectedFocus: ['routing'],
        input: {
          script: 'notebook_manager.py',
          operation: 'list',
          task: 'Check library',
        },
      }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'ask_question.py',
          command: 'python scripts/run.py ask_question.py --question "What is the content of this notebook?"',
          params: ['question'],
          depends_on: ['python'],
          source: 'instructions',
        },
        {
          id: 'template_2',
          title: 'notebook_manager.py · add',
          command: 'python scripts/run.py notebook_manager.py add --url "[URL]" --name "[Based on content]"',
          params: ['url', 'name'],
          depends_on: ['python'],
          source: 'instructions',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-workflow-select-template_2'));
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'notebooklm',
      expect.objectContaining({
        command: 'python scripts/run.py notebook_manager.py add --url [URL] --name [Based on content]',
      }),
      false,
      expect.objectContaining({ workflowId: 'template_2' }),
    );
  });

  it('updates selected execution payload preview when switching workflow candidate', () => {
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      package_examples: [{
        id: 'quick-start',
        label: 'Quick Start',
        description: 'run quick start',
        objective: 'quick-start verification',
        expectedFocus: ['routing'],
        input: { task: 'quick-start' },
      }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'convert',
          command: 'python scripts/office/soffice.py --convert-to docx document.doc',
          params: ['convert_to'],
          depends_on: ['soffice'],
          source: 'instructions',
        },
        {
          id: 'template_2',
          title: 'unpack',
          command: 'python scripts/office/unpack.py document.docx unpacked/',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);
    fireEvent.click(screen.getByTestId('dry-run-workflow-select-template_2'));

    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"workflow_id": "template_2"');
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('unpack.py document.docx unpacked/');
  });

  it('collapses workflow candidates to first 3 by default and expands on More', () => {
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'wf-1',
          command: 'python scripts/run.py a.py one',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
        {
          id: 'template_2',
          title: 'wf-2',
          command: 'python scripts/run.py b.py two',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
        {
          id: 'template_3',
          title: 'wf-3',
          command: 'python scripts/run.py c.py three',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
        {
          id: 'template_4',
          title: 'wf-4',
          command: 'python scripts/run.py d.py four',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
        {
          id: 'template_5',
          title: 'wf-5',
          command: 'python scripts/run.py e.py five',
          params: [],
          depends_on: [],
          source: 'instructions',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    expect(screen.getByTestId('dry-run-workflow-select-template_1')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-workflow-select-template_2')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-workflow-select-template_3')).toBeInTheDocument();
    expect(screen.queryByTestId('dry-run-workflow-select-template_4')).not.toBeInTheDocument();
    expect(screen.queryByTestId('dry-run-workflow-select-template_5')).not.toBeInTheDocument();
    expect(screen.getByTestId('dry-run-workflow-collapse-state')).toHaveTextContent('Showing 3 of 5 examples.');
    expect(screen.getByTestId('dry-run-workflow-toggle')).toHaveTextContent('More (2)');

    fireEvent.click(screen.getByTestId('dry-run-workflow-toggle'));
    expect(screen.getByTestId('dry-run-workflow-select-template_4')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-workflow-select-template_5')).toBeInTheDocument();
    expect(screen.getByTestId('dry-run-workflow-toggle')).toHaveTextContent('Show less');
  });

  it('supports show more/less for long selected execution payload preview', () => {
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      package_examples: [{
        id: 'quick-start',
        label: 'Quick Start',
        description: 'run quick start',
        objective: 'quick-start verification',
        expectedFocus: ['routing'],
        input: {
          task: 'x'.repeat(500),
        },
      }],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    const payloadBlock = screen.getByTestId('dry-run-selected-execution-payload').querySelector('pre');
    expect(payloadBlock).toBeInTheDocument();
    expect(payloadBlock?.className).toContain('max-h-20');

    fireEvent.click(screen.getByTestId('dry-run-selected-payload-toggle'));
    expect(payloadBlock?.className).toContain('max-h-52');

    fireEvent.click(screen.getByTestId('dry-run-selected-payload-toggle'));
    expect(payloadBlock?.className).toContain('max-h-20');
  });

  it('propagates customized workflow command from placeholder inputs to execute payload', () => {
    const onExecute = vi.fn();
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'convert',
          command: 'python scripts/office/soffice.py --convert-to docx document.doc',
          params: ['convert_to'],
          depends_on: ['soffice'],
          source: 'instructions',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} onExecute={onExecute} />);

    fireEvent.change(screen.getByTestId('dry-run-workflow-arg-convert_to'), { target: { value: 'pdf' } });
    fireEvent.change(screen.getByTestId('dry-run-workflow-arg-arg_1'), { target: { value: 'report.docx' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'docx',
      expect.objectContaining({
        command: 'python scripts/office/soffice.py --convert-to pdf report.docx',
      }),
      false,
      expect.objectContaining({ workflowId: 'template_1' }),
    );
  });

  it('keeps quoted notebooklm question as one placeholder value', () => {
    const detail = createDetail({
      name: 'notebooklm',
      display_name: 'notebooklm',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'notebooklm', description: 'NotebookLM workflow helper' }],
      available_workflows: [
        {
          id: 'template_1',
          title: 'ask_question.py',
          command: 'python scripts/run.py ask_question.py --question "What is the content of this notebook?"',
          params: ['question'],
          depends_on: ['python'],
          source: 'instructions',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    const questionInput = screen.getByTestId('dry-run-workflow-arg-question') as HTMLInputElement;
    expect(questionInput.value).toBe('What is the content of this notebook?');
    expect(screen.queryByTestId('dry-run-workflow-arg-arg_1')).not.toBeInTheDocument();
  });

  it('treats prompt-native notebooklm routing as pass when observed args are missing but requested_input matches', () => {
    const result: DryRunResultData = {
      ...createPassResult(),
      llm_verification: {
        success: true,
        message: "LLM correctly called 'notebooklm'",
        tool_call: { name: 'notebooklm', arguments: {} },
        requested_input: {
          question: 'Summarize the architecture decisions in this notebook',
          notebook_url: 'https://notebooklm.google.com/notebook/example',
        },
        phases: [
          { name: 'discovery', label: 'Skill Discovery', timestamp_ms: 0, status: 'pass', data: {} },
          { name: 'activation', label: 'Tool Activation', timestamp_ms: 0, status: 'pass', data: {} },
          { name: 'negotiation', label: 'LLM Negotiation', timestamp_ms: 0, status: 'pass', data: {} },
          { name: 'execution', label: 'LLM Execution', timestamp_ms: 1000, status: 'pass', data: { model: 'gemini-2.5-pro' } },
          { name: 'tool_execution', label: 'Tool Execution', timestamp_ms: 1000, status: 'skip', data: { reason: 'no executor available' } },
        ],
      },
    };

    render(<DryRunDialog {...defaultProps} detail={createNotebooklmDetail()} dryRunResult={result} />);
    revealAllStages();
    const routingStage = screen.getByTestId('dry-run-stage-tool-example-runtime');
    expect(routingStage).toHaveTextContent('PASS');
    expect(routingStage).toHaveTextContent('instruction-driven mode uses requested_input as routing evidence');
  });

  it('accepts skill-creator routing when requested_input matches and observed args are error envelope', () => {
    const result: DryRunResultData = {
      ...createPassResult(),
      llm_verification: {
        success: false,
        message: "LLM returned tool arg formatting error envelope",
        tool_call: {
          name: 'skill-creator',
          arguments: '{"error":"Tool requires valid JSON object input."}',
        },
        requested_input: {
          skill_name: 'pdf-editor',
          goal: 'Create a reusable skill for rotating and extracting PDF content',
        },
        phases: [
          { name: 'discovery', label: 'Skill Discovery', timestamp_ms: 0, status: 'pass', data: {} },
          { name: 'activation', label: 'Tool Activation', timestamp_ms: 0, status: 'pass', data: {} },
          { name: 'negotiation', label: 'LLM Negotiation', timestamp_ms: 0, status: 'pass', data: {} },
          { name: 'execution', label: 'LLM Execution', timestamp_ms: 1000, status: 'warn', data: { error: 'arg format issue' } },
          { name: 'tool_execution', label: 'Tool Execution', timestamp_ms: 1000, status: 'skip', data: { reason: 'no executor available' } },
        ],
      },
    };

    render(<DryRunDialog {...defaultProps} detail={createSkillCreatorDetail()} dryRunResult={result} />);
    revealAllStages();
    const routingStage = screen.getByTestId('dry-run-stage-tool-example-runtime');
    expect(routingStage).toHaveTextContent('PASS');
    expect(routingStage).toHaveTextContent('routing accepted even though llm.success=false');
  });

  it('re-syncs JSON editor to selected skill-creator preset payload when toggling modes', () => {
    const detail = createDetail({
      name: 'skill-creator',
      display_name: 'skill-creator',
      runtime_binding: 'prompt_instructions',
      tools: [{ name: 'skill-creator', description: 'Skill creation guide' }],
      package_examples: [
        {
          id: 'quick-start',
          label: 'Quick Start · SKILL.md',
          description: 'Run package quick start flow',
          input: {
            skill_name: 'example-skill',
            evals: [
              {
                id: 1,
                prompt: "User's task prompt",
                expected_output: 'Description of expected result',
                files: [],
              },
            ],
          },
          source: 'SKILL.md#quick-start',
          expectedFocus: ['quick-start'],
          objective: 'Validate quick-start payload sync',
          confidence: 'medium',
          confidence_breakdown: { score: 0.7 },
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    const textarea = screen.getByTestId('dry-run-json-input').querySelector('textarea')!;
    expect((textarea as HTMLTextAreaElement).value).toContain('example-skill');

    fireEvent.change(textarea, {
      target: {
        value: JSON.stringify({ skill_name: 'pdf-editor', goal: 'Create a reusable skill for rotating and extracting PDF content' }, null, 2),
      },
    });
    expect((textarea as HTMLTextAreaElement).value).toContain('pdf-editor');

    fireEvent.click(screen.getByTestId('dry-run-mode-form'));
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));

    const resetTextarea = screen.getByTestId('dry-run-json-input').querySelector('textarea')!;
    expect((resetTextarea as HTMLTextAreaElement).value).toContain('example-skill');
    expect((resetTextarea as HTMLTextAreaElement).value).not.toContain('pdf-editor');
  });

  it('shows instruction-driven guidance for script_executor_compat skills without schema', () => {
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      instructions_length: 880,
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByText(/Instruction-driven runtime detected/)).toBeInTheDocument();
  });

  it('prefers package_examples presets when provided', () => {
    const onExecute = vi.fn();
    const detail = createDetail({
      name: 'package-driven',
      display_name: 'Package Driven',
      tools: [{ name: 'package-driven', description: 'Package-only tool' }],
      package_examples: [
        {
          id: 'pkg-example-1a',
          label: 'Package Example 1',
          description: 'Package fallback',
          input: { task: 'from package example 1' },
          expectedFocus: ['package_examples'],
          objective: 'package-1',
        },
        {
          id: 'pkg-example-2b',
          label: 'Package Example 2',
          description: 'Package second',
          input: { task: 'from package example 2' },
          expectedFocus: ['package_examples'],
          objective: 'package-2',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} onExecute={onExecute} />);
    expect(screen.getByTestId('tool-flow-select-pkg-example-1a')).toBeInTheDocument();
    expect(screen.getByTestId('tool-flow-select-pkg-example-2b')).toBeInTheDocument();

    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith(
      'package-driven',
      expect.objectContaining({ task: 'from package example 1' }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('falls back to tool presets when package_examples are absent', () => {
    const detail = createDetail({
      name: 'notebooklm',
      display_name: 'NotebookLM',
      package_examples: [],
      tools: [{ name: 'notebooklm', description: 'NotebookLM helper' }],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByTestId('tool-flow-select-example-1-notebook-query')).toBeInTheDocument();
  });

  it('prefers notebooklm package_examples when present', () => {
    const detail = createDetail({
      name: 'notebooklm',
      display_name: 'NotebookLM',
      package_examples: [
        {
          id: 'quick-start-from-skill-md',
          label: 'Quick Start · SKILL.md',
          description: 'Run package quick start',
          input: { action: 'create', task: 'Run the package quick start flow' },
          expectedFocus: ['skill.md #quick-start'],
          objective: 'package quick start',
        },
      ],
      tools: [{ name: 'notebooklm', description: 'NotebookLM helper' }],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByTestId('tool-flow-select-quick-start-from-skill-md')).toBeInTheDocument();
    expect(screen.queryByTestId('tool-flow-select-example-1-notebook-query')).not.toBeInTheDocument();
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"action": "create"');
  });

  it('renders audit metadata for package_examples when source/confidence are provided', () => {
    const detail = createDetail({
      name: 'audit-skill',
      display_name: 'Audit Skill',
      package_examples: [
        {
          id: 'workflow-1-check-auth',
          label: 'Workflow 1 · Check auth',
          description: 'Check auth via command workflow.',
          input: { script: 'auth_manager.py', operation: 'status' },
          expectedFocus: ['skill.md #workflow-steps'],
          objective: 'Validate workflow step: check auth.',
          source: 'SKILL.md#workflow-steps',
          confidence: 'high',
          confidence_reason: 'title matched flow/workflow semantics; command parsed into structured arguments',
          confidence_breakdown: { title_match: 1, step_structure: 1, command_parse: 1, score: 0.93 },
        },
      ],
      tools: [{ name: 'audit-skill', description: 'Audit helper' }],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByTestId('tool-flow-audit-workflow-1-check-auth')).toHaveTextContent('Source: SKILL.md#workflow-steps');
    expect(screen.getByTestId('tool-flow-audit-workflow-1-check-auth')).toHaveTextContent('Confidence: high');
    expect(screen.getByTestId('tool-flow-audit-workflow-1-check-auth')).toHaveTextContent('(93%)');
    expect(screen.getByTestId('tool-flow-confidence-tooltip-workflow-1-check-auth')).toHaveAttribute(
      'title',
      expect.stringContaining('title matched flow/workflow semantics'),
    );
  });

  it('renders generic presets for additional community skills', () => {
    const samples: Array<{ detail: SkillDetail; first: string; second: string }> = [
      {
        detail: createUsingSuperpowersDetail(),
        first: 'tool-flow-select-example-1-workflow-discipline',
        second: 'tool-flow-select-example-2-plan-before-implementation',
      },
      {
        detail: createFrontendDesignDetail(),
        first: 'tool-flow-select-example-1-landing-page',
        second: 'tool-flow-select-example-2-dashboard-redesign',
      },
      {
        detail: createSkillCreatorDetail(),
        first: 'tool-flow-select-example-1-skill-scaffold',
        second: 'tool-flow-select-example-2-skill-iteration',
      },
    ];

    for (const sample of samples) {
      const { unmount } = render(<DryRunDialog {...defaultProps} detail={sample.detail} />);
      expect(screen.getByTestId('tool-flow-presets')).toBeInTheDocument();
      expect(screen.getByTestId(sample.first)).toBeInTheDocument();
      expect(screen.getByTestId(sample.second)).toBeInTheDocument();
      unmount();
    }
  });

  it('shows built-in fallback notice and source label for tool presets', () => {
    render(<DryRunDialog {...defaultProps} detail={createFrontendDesignDetail()} />);

    expect(screen.getByTestId('dry-run-example-source-fallback')).toBeInTheDocument();
    expect(screen.getByTestId('tool-flow-audit-example-1-landing-page')).toHaveTextContent('Source: built-in preset');
  });

  it('does not show built-in fallback notice when package examples are provided', () => {
    const detail = createDetail({
      name: 'docx',
      display_name: 'DOCX',
      runtime_binding: 'script_executor_compat',
      tools: [{ name: 'docx', description: 'DOCX workflow helper' }],
      package_examples: [
        {
          id: 'quick-start',
          label: 'Quick Start',
          description: 'run quick start',
          objective: 'quick-start verification',
          expectedFocus: ['routing'],
          input: { command: 'python scripts/office/unpack.py input.docx unpacked/' },
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    expect(screen.queryByTestId('dry-run-example-source-fallback')).not.toBeInTheDocument();
    expect(screen.getByTestId('tool-flow-audit-quick-start')).toHaveTextContent('Source: package_examples');
  });

  it('executes selected second preset payload for additional community skills', () => {
    const samples: Array<{ detail: SkillDetail; selectId: string; tool: string; expected: Record<string, unknown> }> = [
      {
        detail: createUsingSuperpowersDetail(),
        selectId: 'tool-flow-select-example-2-plan-before-implementation',
        tool: 'using-superpowers',
        expected: { task: 'Build a billing dashboard and choose process skill order first' },
      },
      {
        detail: createFrontendDesignDetail(),
        selectId: 'tool-flow-select-example-2-dashboard-redesign',
        tool: 'frontend-design',
        expected: { aesthetic: 'industrial data-editorial' },
      },
      {
        detail: createSkillCreatorDetail(),
        selectId: 'tool-flow-select-example-2-skill-iteration',
        tool: 'skill-creator',
        expected: { stage: 'iterate' },
      },
    ];

    for (const sample of samples) {
      const onExecute = vi.fn();
      const { unmount } = render(<DryRunDialog {...defaultProps} detail={sample.detail} onExecute={onExecute} />);
      fireEvent.click(screen.getByTestId(sample.selectId));
      fireEvent.click(screen.getByTestId('dry-run-execute'));
      expect(onExecute).toHaveBeenCalledWith(
        sample.tool,
        expect.objectContaining(sample.expected),
        false,
        expect.objectContaining({ workflowId: undefined }),
      );
      unmount();
    }
  });

  it('keeps example 4 status payload stable after switching examples', () => {
    render(<DryRunDialog {...defaultProps} detail={createPlanningToolDetail()} />);

    fireEvent.click(screen.getByTestId('dry-run-workflow-toggle'));
    fireEvent.click(screen.getByTestId('planning-flow-select-example-4-error-recovery'));
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"action": "status"');

    fireEvent.click(screen.getByTestId('planning-flow-select-example-2-bug-fix'));
    fireEvent.click(screen.getByTestId('planning-flow-select-example-4-error-recovery'));

    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"action": "status"');
  });

  it('updates weather city suggestions when country changes', () => {
    render(<DryRunDialog {...defaultProps} detail={createWeatherDetail()} />);

    const country = screen.getByTestId('dry-run-input-country');
    fireEvent.change(country, { target: { value: 'JP' } });

    const cityInput = screen.getByTestId('dry-run-input-city') as HTMLInputElement;
    expect(cityInput.getAttribute('list')).toBe('weather-city-suggestions');
    const datalist = document.getElementById('weather-city-suggestions');
    expect(datalist).toBeTruthy();
    expect(datalist?.innerHTML).toContain('Tokyo');
  });

  it('renders get_location city as dropdown with HouYi style options', () => {
    render(<DryRunDialog {...defaultProps} detail={createLocationDetail()} />);
    const city = screen.getByTestId('dry-run-input-city') as HTMLSelectElement;
    expect(city.tagName).toBe('SELECT');
    expect(Array.from(city.options).map((o) => o.value)).toContain('Beijing');
    expect(Array.from(city.options).map((o) => o.value)).toContain('Hangzhou');
  });

  afterEach(() => {
    setNavigatorUserAgentDataPlatform(undefined);
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

  it('shows no-schema runtime-check message when tool has no schema', () => {
    const detail = createDetail({
      tools: [{ name: 'simple_tool', description: 'No params' }],
    });
    render(<DryRunDialog {...defaultProps} detail={detail} />);
    expect(screen.getByText(/No structured input schema is defined/)).toBeInTheDocument();
  });

  it('renders planning flow presets for planning-with-files tool', () => {
    render(<DryRunDialog {...defaultProps} detail={createPlanningToolDetail()} />);
    expect(screen.getByTestId('planning-flow-presets')).toBeInTheDocument();
    expect(screen.getByTestId('planning-flow-select-example-1-research-task')).toBeInTheDocument();
    expect(screen.getByTestId('planning-flow-select-example-3-feature-development')).toBeInTheDocument();
  });

  it('activates planning create-flow preset and fills create payload', () => {
    render(<DryRunDialog {...defaultProps} detail={createPlanningToolDetail()} />);
    fireEvent.click(screen.getByTestId('planning-flow-select-example-1-research-task'));
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"action": "create"');
    const taskInput = screen.getByTestId('dry-run-input-task') as HTMLInputElement;
    expect(taskInput.value).toContain('Research morning exercise benefits and write summary');
  });

  it('updates selected execution payload when planning action changes in form mode', () => {
    const detail = createDetail({
      name: 'planning-with-files',
      display_name: 'planning-with-files',
      tools: [
        {
          name: 'planning-with-files',
          description: 'Planning workflow skill',
          input_schema: {
            type: 'object',
            properties: {
              action: { type: 'string', enum: ['create', 'update', 'complete', 'status'] },
              task: { type: 'string' },
            },
            required: ['action'],
          },
        },
      ],
      package_examples: [
        {
          id: 'quick-start-from-skill-md',
          label: 'Quick Start · SKILL.md',
          description: 'Run package quick start flow',
          input: { action: 'create', task: 'Run quick start' },
          expectedFocus: ['skill.md #quick-start'],
          objective: 'package quick start',
        },
      ],
    });

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    const actionSelect = screen.getByTestId('dry-run-input-action') as HTMLSelectElement;
    fireEvent.change(actionSelect, { target: { value: 'update' } });

    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"action": "update"');
  });

  it('defaults planning action to create when selected example only provides command/task', () => {
    const detail = createPlanningToolDetail();
    detail.package_examples = [
      {
        id: 'quick-start-from-skill-md',
        label: 'Quick Start · SKILL.md',
        description: 'Run package quick start flow',
        input: { command: '/plan create "Implement user authentication"', task: 'Run quick start flow' },
        expectedFocus: ['skill.md #quick-start'],
        objective: 'package quick start',
      },
    ];

    render(<DryRunDialog {...defaultProps} detail={detail} />);

    expect(screen.queryByTestId('dry-run-input-action')).not.toBeInTheDocument();
    expect(screen.getByTestId('dry-run-selected-execution-payload')).toHaveTextContent('"action": "create"');
  });

  it('hides command placeholders and workflow command preview for planning tool examples', () => {
    render(<DryRunDialog {...defaultProps} detail={createPlanningToolDetail()} />);

    expect(screen.queryByTestId('dry-run-workflow-command-preview')).not.toBeInTheDocument();
    expect(screen.queryByTestId('dry-run-workflow-arg-fields')).not.toBeInTheDocument();
  });

  it('prefills first planning example on first open and executes without reselection', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} detail={createPlanningToolDetail()} onExecute={onExecute} />);

    const taskInput = screen.getByTestId('dry-run-input-task') as HTMLInputElement;
    expect(taskInput.value).toContain('Research morning exercise benefits and write summary');

    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith(
      'ext__planning-with-files',
      expect.objectContaining({
        action: 'create',
        task: expect.stringContaining('Research morning exercise benefits and write summary'),
      }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('enforces single-select example behavior', () => {
    render(<DryRunDialog {...defaultProps} detail={createPlanningToolDetail()} />);

    const ex1 = screen.getByTestId('planning-flow-radio-example-1-research-task') as HTMLInputElement;
    const ex2 = screen.getByTestId('planning-flow-radio-example-2-bug-fix') as HTMLInputElement;

    expect(ex1.checked).toBe(true);
    expect(ex2.checked).toBe(false);

    fireEvent.click(screen.getByTestId('planning-flow-select-example-2-bug-fix'));

    expect(ex1.checked).toBe(false);
    expect(ex2.checked).toBe(true);
  });

  it('does not clear result when re-clicking the already selected example', () => {
    const onClearResult = vi.fn();
    render(
      <DryRunDialog
        {...defaultProps}
        detail={createPlanningToolDetail()}
        onClearResult={onClearResult}
      />,
    );

    onClearResult.mockClear();
    fireEvent.click(screen.getByTestId('planning-flow-select-example-1-research-task'));
    expect(onClearResult).not.toHaveBeenCalled();
  });

  it('does not render planning flow presets for core planning tool', () => {
    render(<DryRunDialog {...defaultProps} detail={createCorePlanningToolDetail()} />);
    expect(screen.queryByTestId('planning-flow-presets')).not.toBeInTheDocument();
  });

  it('renders generic skill example presets for rag-skill', () => {
    render(<DryRunDialog {...defaultProps} detail={createRagSkillDetail()} />);
    expect(screen.getByTestId('tool-flow-presets')).toBeInTheDocument();
    expect(screen.getByTestId('tool-flow-select-example-1-kb-query')).toBeInTheDocument();
  });

  it('uses selected generic preset payload when executing rag-skill with no schema', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} detail={createRagSkillDetail()} onExecute={onExecute} />);

    fireEvent.click(screen.getByTestId('tool-flow-select-example-1-kb-query'));
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'rag-skill',
      expect.objectContaining({
        query: 'What is RAG and when should it be used?',
        knowledge_dir: 'knowledge/',
      }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('uses active planning loop payload when external planning has no schema', () => {
    const onExecute = vi.fn();
    const noSchemaExternalPlanning = createPlanningToolDetail();
    noSchemaExternalPlanning.tools = [{
      name: 'ext__planning-with-files',
      description: 'No schema prompt-native planning tool',
    }];

    render(<DryRunDialog {...defaultProps} detail={noSchemaExternalPlanning} onExecute={onExecute} />);

    fireEvent.click(screen.getByTestId('dry-run-workflow-toggle'));
    fireEvent.click(screen.getByTestId('planning-flow-select-example-4-error-recovery'));
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'ext__planning-with-files',
      expect.objectContaining({ action: 'status' }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('allows live execute for external planning package_examples even when action field is hidden', () => {
    const onExecute = vi.fn();
    const detail = createPlanningToolDetail();
    detail.package_examples = [
      {
        id: 'example-1-research-task',
        label: 'Example 1 · Research Task',
        description: 'Package-derived planning example without explicit action field',
        input: { command: 'Write task_plan.md', task: 'Research morning exercise benefits and write a summary' },
        expectedFocus: ['examples.md#example-1'],
        objective: 'Validate package-native workflow for Example 1',
      },
    ];

    render(<DryRunDialog {...defaultProps} detail={detail} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'ext__planning-with-files',
      expect.objectContaining({
        action: 'create',
        command: 'Write task_plan.md',
        task: 'Research morning exercise benefits and write a summary',
      }),
      true,
      expect.objectContaining({ workflowId: undefined }),
    );
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
    expect(onExecute).toHaveBeenCalledWith(
      'web_search',
      expect.objectContaining({ query: 'test query' }),
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
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
    expect(onExecute).toHaveBeenCalledWith(
      'web_search',
      {
        query: 'test query',
        max_results: 5,
      },
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('executes with JSON input', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} onExecute={onExecute} />);
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    const textarea = screen.getByTestId('dry-run-json-input').querySelector('textarea')!;
    fireEvent.change(textarea, { target: { value: '{"query": "hello"}' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));
    expect(onExecute).toHaveBeenCalledWith(
      'web_search',
      { query: 'hello' },
      false,
      expect.objectContaining({ workflowId: undefined }),
    );
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
    expect(onExecute).toHaveBeenCalledWith(
      'web_search',
      expect.objectContaining({ query: 'test' }),
      true,
      expect.objectContaining({ workflowId: undefined }),
    );
  });

  it('changes button text in live mode', () => {
    render(<DryRunDialog {...defaultProps} />);
    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    expect(screen.getByTestId('dry-run-execute')).toHaveTextContent('Execute Live Dry-run');
  });

  it('shows notebooklm live defaults for provider and model when live is enabled', () => {
    render(<DryRunDialog {...defaultProps} detail={createNotebooklmDetail()} />);

    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));

    const provider = screen.getByTestId('dry-run-live-provider') as HTMLSelectElement;
    const model = screen.getByTestId('dry-run-live-model') as HTMLInputElement;
    expect(provider.value).toBe('vertex');
    expect(model.value).toBe('gemini-2.5-pro');
  });

  it('passes live provider/model overrides when executing notebooklm live dry-run', () => {
    const onExecute = vi.fn();
    render(<DryRunDialog {...defaultProps} detail={createNotebooklmDetail()} onExecute={onExecute} />);

    fireEvent.click(screen.getByTestId('dry-run-live-toggle'));
    fireEvent.change(screen.getByTestId('dry-run-live-provider'), { target: { value: 'google_ai' } });
    fireEvent.change(screen.getByTestId('dry-run-live-model'), { target: { value: 'gemini-2.5-flash' } });
    fireEvent.click(screen.getByTestId('dry-run-execute'));

    expect(onExecute).toHaveBeenCalledWith(
      'notebooklm',
      expect.objectContaining({
        question: 'Summarize the architecture decisions in this notebook',
      }),
      true,
      {
        llmProvider: 'google_ai',
        llmModel: 'gemini-2.5-flash',
      },
    );
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
    // LLM tool-call section
    expect(flow).toHaveTextContent('LLM Tool Call');
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
    expect(screen.getByTestId('dry-run-stage-runtime-readiness').className).toContain('opacity-40');

    // After second tick — runtime-readiness reveals
    act(() => { vi.advanceTimersByTime(200); });
    expect(screen.getByTestId('dry-run-stage-runtime-readiness').className).toContain('animate-stageReveal');
    expect(screen.getByTestId('dry-run-stage-schema').className).toContain('opacity-40');
  });

  it('shows all stages as pending when executing (before result)', () => {
    render(<DryRunDialog {...defaultProps} />);

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
    fireEvent.click(screen.getByTestId('dry-run-input-toggle'));
    fireEvent.click(screen.getByTestId('dry-run-mode-json'));
    // Pipeline should still be visible
    expect(screen.getByTestId('dry-run-result-panel')).toBeInTheDocument();
    expect(screen.getByText('ALL PASSED')).toBeInTheDocument();
  });

  it('preserves pipeline results when switching back to Form mode', () => {
    render(<DryRunDialog {...defaultProps} dryRunResult={createPassResult()} />);
    revealAllStages();
    fireEvent.click(screen.getByTestId('dry-run-input-toggle'));
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
