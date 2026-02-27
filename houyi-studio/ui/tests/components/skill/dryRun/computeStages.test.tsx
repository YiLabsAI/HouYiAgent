import { describe, expect, it } from 'vitest';

import { computeStages } from '@/components/panels/skill/dryRun/computeStages';
import type { DryRunResultData } from '@/components/LeftSidebar/useSkillsLogic';
import type { SkillDetail } from '@/types/websocket';

const createDetail = (overrides: Partial<SkillDetail> = {}): SkillDetail => ({
  name: 'web_search',
  display_name: 'Web Search',
  description: 'Search the web',
  version: '1.0.0',
  tools: [{ name: 'web_search', description: 'Search the web for information' }],
  permissions: [],
  policy: { default_action: 'allow' },
  hooks: [],
  certification: 'gold',
  side_effect: 'network',
  ...overrides,
  is_core: overrides.is_core ?? false,
});

const createPassResult = (overrides: Partial<DryRunResultData> = {}): DryRunResultData => ({
  valid: true,
  schema_errors: [],
  policy_result: 'allow',
  capability_gaps: [],
  estimated_side_effects: [],
  ...overrides,
});

describe('computeStages', () => {
  it('builds baseline stages in pending/running mode when result is null', () => {
    const stages = computeStages(null, createDetail(), false);

    expect(stages.map((s) => s.id)).toEqual([
      'registration',
      'runtime-readiness',
      'schema',
      'policy',
      'side-effects',
      'hooks',
    ]);
    expect(stages[0]?.status).toBe('running');
    expect(stages.slice(1).every((s) => s.status === 'pending')).toBe(true);
  });

  it('runtime readiness passes when executable and ready', () => {
    const result = createPassResult();
    const detail = createDetail({ capability_tier: 'executable', runtime_status: 'ready' });
    const stages = computeStages(result, detail, false);
    const rt = stages.find((s) => s.id === 'runtime-readiness');
    expect(rt?.status).toBe('pass');
    expect(rt?.summary).toContain('Fully operational');
  });

  it('runtime readiness warns when executable but degraded', () => {
    const result = createPassResult();
    const detail = createDetail({ capability_tier: 'executable', runtime_status: 'degraded' });
    const stages = computeStages(result, detail, false);
    const rt = stages.find((s) => s.id === 'runtime-readiness');
    expect(rt?.status).toBe('warn');
    expect(rt?.summary).toContain('Partially operational');
  });

  it('runtime readiness fails when not executable', () => {
    const result = createPassResult();
    const detail = createDetail({ capability_tier: 'metadata', runtime_status: 'unavailable' });
    const stages = computeStages(result, detail, false);
    const rt = stages.find((s) => s.id === 'runtime-readiness');
    expect(rt?.status).toBe('fail');
    expect(rt?.summary).toContain('Not executable');
  });

  it('runtime readiness defaults to metadata/unavailable when fields absent', () => {
    const result = createPassResult();
    const stages = computeStages(result, createDetail(), false);
    const rt = stages.find((s) => s.id === 'runtime-readiness');
    expect(rt?.status).toBe('fail');
    expect(rt?.summary).toContain('metadata');
  });

  it('adds capability gaps stage when gaps exist', () => {
    const result = createPassResult({
      capability_gaps: ['network access unavailable'],
      policy_result: 'deny',
    });

    const stages = computeStages(result, createDetail(), false);

    const gaps = stages.find((s) => s.id === 'gaps');
    expect(gaps).toBeDefined();
    expect(gaps?.status).toBe('warn');
    expect(gaps?.summary).toContain('1 gap(s) detected');
  });

  it('adds llm verification and tool execution stages when llm phase is present', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: { name: 'web_search', arguments: { query: 'test' } },
        phases: [
          {
            name: 'tool_execution',
            label: 'Tool Execution',
            timestamp_ms: 30,
            status: 'warn',
            data: { reason: 'timeout' },
          },
        ],
      },
    });

    const stages = computeStages(result, createDetail(), false);

    const llm = stages.find((s) => s.id === 'llm-verify');
    const toolExec = stages.find((s) => s.id === 'tool-execution');

    expect(llm?.status).toBe('pass');
    expect(toolExec?.status).toBe('warn');
    expect(toolExec?.summary).toContain('timeout');
  });

  it('shows llm verification stage in live mode even if llm result is absent', () => {
    const stages = computeStages(createPassResult(), createDetail(), true);

    const llm = stages.find((s) => s.id === 'llm-verify');
    expect(llm).toBeDefined();
    expect(llm?.status).toBe('skip');
    expect(llm?.summary).toContain('not available');
  });

  it('marks llm verification as fail when llm verification fails', () => {
    const stages = computeStages(
      createPassResult({
        llm_verification: {
          success: false,
          message: 'LLM did not produce a tool call',
        },
      }),
      createDetail(),
      true,
    );
    const llm = stages.find((s) => s.id === 'llm-verify');
    expect(llm?.status).toBe('fail');
    expect(llm?.summary).toContain('did not produce a tool call');
  });

  it('adds planning workflow coverage stage for planning-with-files', () => {
    const result = createPassResult();
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, false);
    const planning = stages.find((s) => s.id === 'planning-flow-static');
    expect(planning).toBeDefined();
    expect(planning?.status).toBe('pass');
    expect(planning?.label).toBe('Planning Example Spec Alignment');
    expect(planning?.summary).toContain('7/7');
  });

  it('does not add planning examples stage for core planning-with-files', () => {
    const stages = computeStages(
      createPassResult(),
      createDetail({
        name: 'planning-with-files',
        display_name: 'planning-with-files',
        runtime_binding: 'python_executor',
      }),
      false,
    );
    expect(stages.find((s) => s.id === 'planning-flow-static')).toBeUndefined();
  });

  it('adds runtime evidence stage and passes when selected example action matches llm call', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: { name: 'ext__planning-with-files', arguments: { action: 'create' } },
        requested_input: { action: 'create' },
        phases: [
          {
            name: 'tool_execution',
            label: 'Tool Execution',
            timestamp_ms: 42,
            status: 'pass',
            data: { result_preview: 'done' },
          },
        ],
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
    expect(runtime?.label).toBe('Planning Example LLM Routing');
    expect(runtime?.summary).toContain('expected action "create"');
  });

  it('passes runtime routing when observed tool-call arguments are JSON string', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: { name: 'ext__planning-with-files', arguments: '{"action":"create"}' },
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
  });

  it('extracts planning runtime action from backend-normalized tool_call.action when args are empty', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: {
          name: 'ext__planning-with-files',
          arguments: {},
          action: 'create',
        },
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Always invoke the planning tool with action.
`,
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
  });

  it('passes runtime routing when observed tool_call uses nested function.arguments shape', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: {
          function: {
            name: 'ext__planning-with-files',
            arguments: { kwargs: { Action: 'create' } },
          },
        } as unknown as Record<string, unknown>,
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
  });

  it('passes runtime routing when action is only present in fenced raw_content JSON', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: { name: 'ext__planning-with-files', arguments: {} },
        raw_content: 'tool args:\n```json\n{"action":"create"}\n```',
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
  });

  it('passes runtime routing when action is nested in observed arguments payload', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: {
          name: 'ext__planning-with-files',
          arguments: { input: { arguments: { action: 'create' } } },
        },
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
  });

  it('passes runtime routing when observed arguments use single-quoted pseudo-json', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: {
          name: 'ext__planning-with-files',
          arguments: "{'action':'create'}",
        },
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-1-research-task',
      planningFlowLabel: 'Example 1 · Research Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('pass');
  });

  it('fails runtime routing stage when requested snapshot action mismatches selected example', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: { name: 'ext__planning-with-files', arguments: { action: 'update' } },
        requested_input: { action: 'create' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-2-bug-fix',
      planningFlowLabel: 'Example 2 · Bug Fix Task',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('fail');
    expect(runtime?.summary).toContain('LLM routing mismatch');
  });

  it('warns runtime routing stage when observed tool-call action evidence is missing', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: { name: 'ext__planning-with-files', arguments: {} },
        requested_input: { action: 'status' },
      },
    });
    const detail = createDetail({
      name: 'ext__planning-with-files',
      display_name: 'ext__planning-with-files',
      runtime_binding: 'prompt_instructions',
      instructions_length: 5000,
      instructions: `
Create task_plan.md findings.md progress.md
Create Plan First
2-Action Rule
Log ALL Errors
Never Repeat Failures
Use for:
Skip for:
examples.md
reference.md
`,
      hook_specs: [
        { event: 'PreToolUse', type: 'command', matcher: 'Write|Edit', command: 'cat task_plan.md', handler: null },
        { event: 'PostToolUse', type: 'command', matcher: 'Write|Edit', command: 'update task_plan.md status', handler: null },
        { event: 'Stop', type: 'command', matcher: '*', command: 'sh scripts/check-complete.sh', handler: null },
      ],
    });

    const stages = computeStages(result, detail, true, {
      planningFlowId: 'example-3-feature-development',
      planningFlowLabel: 'Example 3 · Feature Development',
    });

    const runtime = stages.find((s) => s.id === 'planning-flow-runtime');
    expect(runtime?.status).toBe('warn');
    expect(runtime?.summary).toContain('observed action evidence is missing');
  });

  it('adds generic external example stages and passes strict subset routing for notebooklm', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: {
          name: 'notebooklm',
          arguments: {
            question: 'Summarize the architecture decisions in this notebook',
            notebook_url: 'https://notebooklm.google.com/notebook/example',
            extra: 'ignored',
          },
        },
        requested_input: {
          question: 'Summarize the architecture decisions in this notebook',
          notebook_url: 'https://notebooklm.google.com/notebook/example',
          extra: 'ignored',
        },
      },
    });

    const detail = createDetail({
      name: 'notebooklm',
      display_name: 'notebooklm',
      runtime_binding: 'prompt_instructions',
      instructions: `Always use run.py\npython scripts/run.py auth_manager.py status\nauthenticate`,
      instructions_length: 1200,
      tools: [{ name: 'notebooklm', description: 'Notebook assistant' }],
    });

    const stages = computeStages(result, detail, true, {
      selectedExampleId: 'example-1-notebook-query',
      selectedExampleLabel: 'Example 1 · Notebook Query',
      selectedToolName: 'notebooklm',
    });

    const staticStage = stages.find((s) => s.id === 'tool-example-static');
    const runtimeStage = stages.find((s) => s.id === 'tool-example-runtime');
    expect(staticStage?.status).toBe('pass');
    expect(runtimeStage?.status).toBe('pass');
  });

  it('fails generic external example runtime stage when requested/observed subset mismatches', () => {
    const result = createPassResult({
      llm_verification: {
        success: true,
        message: 'ok',
        tool_call: {
          name: 'notebooklm',
          arguments: {
            question: 'wrong question',
          },
        },
        requested_input: {
          question: 'wrong question',
        },
      },
    });

    const detail = createDetail({
      name: 'notebooklm',
      display_name: 'notebooklm',
      runtime_binding: 'prompt_instructions',
      instructions: `Always use run.py\npython scripts/run.py auth_manager.py status\nauthenticate`,
      instructions_length: 1200,
      tools: [{ name: 'notebooklm', description: 'Notebook assistant' }],
    });

    const stages = computeStages(result, detail, true, {
      selectedExampleId: 'example-1-notebook-query',
      selectedExampleLabel: 'Example 1 · Notebook Query',
      selectedToolName: 'notebooklm',
    });

    const runtimeStage = stages.find((s) => s.id === 'tool-example-runtime');
    expect(runtimeStage?.status).toBe('fail');
    expect(runtimeStage?.summary).toContain('LLM routing mismatch');
  });
});
