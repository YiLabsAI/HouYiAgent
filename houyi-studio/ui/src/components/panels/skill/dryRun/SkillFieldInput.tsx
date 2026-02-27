import React from 'react';

import {
  LOCATION_CITY_OPTIONS,
  WEATHER_DATE_OPTIONS,
  WEB_SEARCH_MODE_OPTIONS,
  WEB_SEARCH_PROVIDER_OPTIONS,
  WEATHER_COUNTRIES,
} from './inputPresets';

export interface DryRunSchemaField {
  name: string;
  type: string;
  format?: string;
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
  isWebSearchTool: boolean;
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
  isWebSearchTool,
  weatherCityOptions,
  onChange,
  placeholder,
}) => {
  const resolvedMin =
    field.minimum ?? (isWeatherTool && field.name === 'lat' ? -90 : isWeatherTool && field.name === 'lon' ? -180 : undefined);
  const resolvedMax =
    field.maximum ?? (isWeatherTool && field.name === 'lat' ? 90 : isWeatherTool && field.name === 'lon' ? 180 : undefined);

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

  if (isWeatherTool && field.name === 'date') {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="">— not set (provider default) —</option>
        {WEATHER_DATE_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    );
  }

  if (isWebSearchTool && field.name === 'provider') {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="">— auto by env/runtime —</option>
        {WEB_SEARCH_PROVIDER_OPTIONS.map((provider) => (
          <option key={provider} value={provider}>{provider}</option>
        ))}
      </select>
    );
  }

  if (isWebSearchTool && field.name === 'mode') {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        <option value="">— default (search) —</option>
        {WEB_SEARCH_MODE_OPTIONS.map((mode) => (
          <option key={mode} value={mode}>{mode}</option>
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
    const hasDefault = field.default !== undefined;
    const placeholderLabel = field.required ? '— select —' : '— not set —';
    return (
      <select
        value={value || (hasDefault ? String(field.default) : '')}
        onChange={(e) => onChange(e.target.value)}
        className={baseClass}
        data-testid={`dry-run-input-${field.name}`}
      >
        {!hasDefault && (
          <option value="">{placeholderLabel}</option>
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
      min={resolvedMin}
      max={resolvedMax}
      className={`${baseClass} placeholder:text-gray-600`}
      data-testid={`dry-run-input-${field.name}`}
    />
  );
};
