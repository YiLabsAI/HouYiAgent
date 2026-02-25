import React from 'react';

import {
  LOCATION_CITY_OPTIONS,
  WEATHER_COUNTRIES,
} from './inputPresets';

export interface DryRunSchemaField {
  name: string;
  type: string;
  title: string;
  description: string;
  required: boolean;
  default?: unknown;
  defaultRaw?: unknown;
  nullable: boolean;
  enum?: string[];
  minimum?: number;
  maximum?: number;
}

interface SkillFieldInputProps {
  field: DryRunSchemaField;
  value: string;
  error?: string;
  isWeatherTool: boolean;
  isLocationTool: boolean;
  weatherCityOptions: string[];
  onChange: (value: string) => void;
  placeholder: string;
}

export const SkillFieldInput: React.FC<SkillFieldInputProps> = ({
  field,
  value,
  error,
  isWeatherTool,
  isLocationTool,
  weatherCityOptions,
  onChange,
  placeholder,
}) => {
  const baseClass = `w-full bg-gray-900 border rounded px-2 py-1.5 text-xs text-gray-200 focus:border-blue-500 focus:outline-none ${
    error ? 'border-red-500' : 'border-gray-700'
  }`;

  if (isWeatherTool && field.name === 'country') {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="">— select country —</option>
        {WEATHER_COUNTRIES.map((c) => (
          <option key={c.code} value={c.code}>{c.label}</option>
        ))}
      </select>
    );
  }

  if (isWeatherTool && field.name === 'provider') {
    return (
      <select
        value={value || (field.default !== undefined ? String(field.default) : 'auto')}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="auto">auto</option>
        <option value="openmeteo">openmeteo</option>
        <option value="wttr">wttr</option>
      </select>
    );
  }

  if (isWeatherTool && field.name === 'city') {
    return (
      <>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          list="weather-city-suggestions"
          className={`${baseClass} placeholder:text-gray-600`}
          data-testid={`dry-run-input-${field.name}`}
        />
        <datalist id="weather-city-suggestions">
          {weatherCityOptions.map((city) => (
            <option key={city} value={city} />
          ))}
        </datalist>
      </>
    );
  }

  if (isLocationTool && field.name === 'city') {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="">— default (Hangzhou) —</option>
        {LOCATION_CITY_OPTIONS.map((city) => (
          <option key={city} value={city}>{city}</option>
        ))}
      </select>
    );
  }

  if (field.type === 'boolean') {
    return (
      <select
        value={value || (field.defaultRaw !== undefined ? String(field.defaultRaw) : 'false')}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  }

  if (field.enum) {
    return (
      <select
        value={value || (field.default !== undefined ? String(field.default) : '')}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        {!field.required && field.default === undefined && (
          <option value="">— not set —</option>
        )}
        {field.enum.map((v) => (
          <option key={v} value={v}>{v}</option>
        ))}
      </select>
    );
  }

  return (
    <input
      type={field.type === 'number' || field.type === 'integer' ? 'number' : 'text'}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      min={field.minimum}
      max={field.maximum}
      className={`${baseClass} placeholder:text-gray-600`}
      data-testid={`dry-run-input-${field.name}`}
    />
  );
};
