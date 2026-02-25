import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { LlmVerificationDetails } from '@/components/panels/skill/dryRun/LlmVerificationDetails';
import type { DryRunResultData } from '@/components/LeftSidebar/useSkillsLogic';

const createLlmVerification = (): NonNullable<DryRunResultData['llm_verification']> => ({
  success: true,
  message: "LLM correctly called 'web_search'",
  tool_call: { name: 'web_search', arguments: { query: 'test' } },
  system_prompt: 'You are a helpful assistant.',
  raw_content: 'I will search for that.',
  execution_result: 'search_result: ok',
  tool_definitions: [
    {
      type: 'function',
      function: {
        name: 'web_search',
        description: 'Search the web',
        parameters: { properties: { query: { type: 'string' } }, required: ['query'] },
      },
    },
  ],
  model_name: 'gemini-3-pro-preview',
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  phases: [
    {
      name: 'discovery',
      label: 'Skill Discovery',
      timestamp_ms: 0,
      status: 'pass',
      data: { skill_name: 'web_search', description: 'Search the web' },
    },
    {
      name: 'execution',
      label: 'LLM Execution',
      timestamp_ms: 1200,
      status: 'pass',
      data: { model: 'gemini-3-pro-preview', latency_ms: 1180 },
    },
  ],
});

describe('LlmVerificationDetails', () => {
  it('renders timeline, tool call, usage and model badge', () => {
    render(<LlmVerificationDetails llm={createLlmVerification()} />);

    const flow = screen.getByTestId('llm-verify-flow');
    expect(flow).toHaveTextContent('Skill Discovery');
    expect(flow).toHaveTextContent('LLM Execution');
    expect(flow).toHaveTextContent('t=0ms');
    expect(flow).toHaveTextContent('LLM Tool Call');
    expect(flow).toHaveTextContent('prompt_tokens: 10');
    expect(flow).toHaveTextContent('gemini-3-pro-preview');
    expect(flow).toHaveTextContent('I will search for that.');
  });

  it('toggles tool definitions json block', () => {
    render(<LlmVerificationDetails llm={createLlmVerification()} />);

    expect(screen.queryByText(/"description": "Search the web"/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('▸ Show tool definitions (JSON)'));
    expect(screen.getByText(/"description": "Search the web"/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('▾ Hide tool definitions'));
    expect(screen.queryByText(/"description": "Search the web"/)).not.toBeInTheDocument();
  });

  it('collapses and expands execution phase details', () => {
    render(<LlmVerificationDetails llm={createLlmVerification()} />);

    expect(screen.getByText('latency:')).toBeInTheDocument();
    fireEvent.click(screen.getByText('LLM Execution'));
    expect(screen.queryByText('latency:')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('LLM Execution'));
    expect(screen.getByText('latency:')).toBeInTheDocument();
  });

  it('shows tool execution result block when present', () => {
    render(<LlmVerificationDetails llm={createLlmVerification()} />);
    expect(screen.getByText('Tool Execution Result')).toBeInTheDocument();
    expect(screen.getByText('search_result: ok')).toBeInTheDocument();
  });
});
