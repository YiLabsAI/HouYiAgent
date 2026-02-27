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
        isWebSearchTool={false}
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
        isWebSearchTool={false}
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

  it('renders weather date as preset dropdown', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'date', required: false }}
        value=""
        isWeatherTool
        isLocationTool={false}
        isWebSearchTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-date') as HTMLSelectElement;
    expect(select.tagName).toBe('SELECT');
    expect(Array.from(select.options).map((o) => o.value)).toEqual([
      '',
      'today',
      'tomorrow',
      'day_after_tomorrow',
    ]);
  });

  it('renders get_location city as select with preset options', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'city' }}
        value=""
        isWeatherTool={false}
        isLocationTool
        isWebSearchTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-city') as HTMLSelectElement;
    expect(select.tagName).toBe('SELECT');
    expect(Array.from(select.options).map((o) => o.value)).toContain('Hangzhou');
  });

  it('applies weather lat/lon range defaults when schema has no bounds', () => {
    const { rerender } = render(
      <SkillFieldInput
        field={{ ...baseField, name: 'lat', type: 'number', required: false }}
        value=""
        isWeatherTool
        isLocationTool={false}
        isWebSearchTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const lat = screen.getByTestId('dry-run-input-lat') as HTMLInputElement;
    expect(lat.min).toBe('-90');
    expect(lat.max).toBe('90');

    rerender(
      <SkillFieldInput
        field={{ ...baseField, name: 'lon', type: 'number', required: false }}
        value=""
        isWeatherTool
        isLocationTool={false}
        isWebSearchTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const lon = screen.getByTestId('dry-run-input-lon') as HTMLInputElement;
    expect(lon.min).toBe('-180');
    expect(lon.max).toBe('180');
  });

  it('renders boolean as select and applies defaultRaw fallback', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'use_cache', type: 'boolean', defaultRaw: true }}
        value=""
        isWeatherTool={false}
        isLocationTool={false}
        isWebSearchTool={false}
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
        isWebSearchTool={false}
        weatherCityOptions={[]}
        onChange={onChange}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-mode');
    fireEvent.change(select, { target: { value: 'balanced' } });
    expect(onChange).toHaveBeenCalledWith('balanced');
  });

  it('renders required enum with explicit select placeholder', () => {
    render(
      <SkillFieldInput
        field={{ ...baseField, name: 'action', required: true, enum: ['create', 'update'] }}
        value=""
        isWeatherTool={false}
        isLocationTool={false}
        isWebSearchTool={false}
        weatherCityOptions={[]}
        onChange={vi.fn()}
        placeholder=""
      />,
    );

    const select = screen.getByTestId('dry-run-input-action') as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual(['', 'create', 'update']);
    expect(select.options[0].textContent).toBe('— select —');
    expect(select.value).toBe('');
  });

  it('renders web_search provider/mode as preset selects', () => {
    render(
      <>
        <SkillFieldInput
          field={{ ...baseField, name: 'provider', required: false }}
          value=""
          isWeatherTool={false}
          isLocationTool={false}
          isWebSearchTool
          weatherCityOptions={[]}
          onChange={vi.fn()}
          placeholder=""
        />
        <SkillFieldInput
          field={{ ...baseField, name: 'mode', required: false }}
          value=""
          isWeatherTool={false}
          isLocationTool={false}
          isWebSearchTool
          weatherCityOptions={[]}
          onChange={vi.fn()}
          placeholder=""
        />
      </>,
    );

    const provider = screen.getByTestId('dry-run-input-provider') as HTMLSelectElement;
    const mode = screen.getByTestId('dry-run-input-mode') as HTMLSelectElement;
    expect(Array.from(provider.options).map((o) => o.value)).toEqual([
      '',
      'ddg',
      'serper',
      'tavily',
      'bocha',
      'searxng',
    ]);
    expect(Array.from(mode.options).map((o) => o.value)).toEqual(['', 'search', 'browse']);
  });
});
