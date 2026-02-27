export const WEATHER_COUNTRIES = [
  { code: 'CN', label: 'China (CN)' },
  { code: 'US', label: 'United States (US)' },
  { code: 'JP', label: 'Japan (JP)' },
  { code: 'KR', label: 'South Korea (KR)' },
  { code: 'SG', label: 'Singapore (SG)' },
  { code: 'GB', label: 'United Kingdom (GB)' },
  { code: 'DE', label: 'Germany (DE)' },
  { code: 'FR', label: 'France (FR)' },
  { code: 'CA', label: 'Canada (CA)' },
  { code: 'AU', label: 'Australia (AU)' },
] as const;

export const WEATHER_CITY_SUGGESTIONS: Record<string, string[]> = {
  CN: ['Beijing', 'Shanghai', 'Shenzhen', 'Guangzhou', 'Hangzhou', 'Chengdu'],
  US: ['New York', 'San Francisco', 'Los Angeles', 'Seattle', 'Chicago', 'Austin'],
  JP: ['Tokyo', 'Osaka', 'Kyoto', 'Yokohama', 'Sapporo'],
  KR: ['Seoul', 'Busan', 'Incheon', 'Daegu'],
  SG: ['Singapore'],
  GB: ['London', 'Manchester', 'Birmingham', 'Edinburgh'],
  DE: ['Berlin', 'Munich', 'Hamburg', 'Frankfurt'],
  FR: ['Paris', 'Lyon', 'Marseille', 'Toulouse'],
  CA: ['Toronto', 'Vancouver', 'Montreal', 'Calgary'],
  AU: ['Sydney', 'Melbourne', 'Brisbane', 'Perth'],
};

export const DEFAULT_WEATHER_CITY_OPTIONS = [
  'Beijing',
  'Shanghai',
  'London',
  'San Francisco',
  'Tokyo',
  'Singapore',
];

export const LOCATION_CITY_OPTIONS = [
  'Hangzhou',
  'Beijing',
  'Shanghai',
  'Shenzhen',
  'Guangzhou',
  'Chengdu',
  'Tokyo',
  'Singapore',
  'London',
  'New York',
  'San Francisco',
];

export const WEB_SEARCH_PROVIDER_OPTIONS = ['ddg', 'serper', 'tavily', 'bocha', 'searxng'] as const;

export const WEB_SEARCH_MODE_OPTIONS = ['search', 'browse'] as const;

export const WEATHER_DATE_OPTIONS = [
  { value: 'today', label: 'today' },
  { value: 'tomorrow', label: 'tomorrow' },
  { value: 'day_after_tomorrow', label: 'day_after_tomorrow' },
] as const;
