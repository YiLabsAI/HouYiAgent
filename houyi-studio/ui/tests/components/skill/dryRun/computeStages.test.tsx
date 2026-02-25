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
      'schema',
      'policy',
      'side-effects',
      'hooks',
    ]);
    expect(stages[0]?.status).toBe('running');
    expect(stages.slice(1).every((s) => s.status === 'pending')).toBe(true);
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
});
