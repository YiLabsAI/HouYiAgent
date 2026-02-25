import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { SkillFieldInput, type DryRunSchemaField } from '@/components/panels/skill/dryRun/SkillFieldInput';

const baseField: DryRunSchemaField = {
  name: 'query',
  type: 'string',
  title: 'Query',
  description: 'Search query',
  required: true,
  nullable: false,
};

describe('SkillFieldInput', () => {
  it('renders weather country as select with country options', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'country' }}
        value=""
        isWeatherTool
        isLocationTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const el = screen.getByTestId('dry-run-input-country') as HTMLSelectElement;
    expect(el.tagName).toBe('SELECT');
    expect(Array.from(el.options).map((o) => o.value)).toContain('CN');
  });

  it('renders weather city with datalist suggestions', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'city' }}
        value=""
        isWeatherTool
        isLocationTool={false}
        weatherCityOptions={['Tokyo', 'Osaka']}
        onChange={vi.fn()}
        placeholder="Required"
      />,
    );

    const input = screen.getByTestId('dry-run-input-city') as HTMLInputElement;
    expect(input.tagName).toBe('INPUT');
    expect(input.getAttribute('list')).toBe('weather-city-suggestions');
    const datalist = document.getElementById('weather-city-suggestions');
    expect(datalist?.innerHTML).toContain('Tokyo');
  });

  it('renders get_location city as select with preset options', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'city' }}
        value=""
        isWeatherTool={false}
        isLocationTool
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-city') as HTMLSelectElement;
    expect(select.tagName).toBe('SELECT');
    expect(Array.from(select.options).map((o) => o.value)).toContain('Hangzhou');
  });

  it('renders boolean as select and applies defaultRaw fallback', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'use_cache', type: 'boolean', defaultRaw: true }}
        value=""
        isWeatherTool={false}
        isLocationTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-use_cache') as HTMLSelectElement;
    expect(select.tagName).toBe('SELECT');
    expect(select.value).toBe('true');
  });

  it('calls onChange with selected enum value', () => {
    const onChange = vi.fn();
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'mode', required: false, enum: ['fast', 'balanced'] }}
        value=""
        isWeatherTool={false}
        isLocationTool={false}
        weatherCityOptions={[]}
        onChange={onChange}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-mode');
    fireEvent.change(select, { target: { value: 'balanced' } });
    expect(onChange).toHaveBeenCalledWith('balanced');
  });
});
