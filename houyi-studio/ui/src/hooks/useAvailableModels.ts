/**
 * Hook to fetch available models from configured providers.
 *
 * Models are fetched from /api/chat/models (which reads enabled providers)
 * and cached in memory. Falls back to MODEL_OPTIONS when no providers
 * are configured.
 *
 * models come from provider API, not hardcoded.
 */
import { useState, useEffect, useRef } from 'react';
import { MODEL_OPTIONS } from '@/constants/models';

export interface AvailableModel {
  model: string;
  provider: string;
}

const API_BASE = '/api/chat';

// Module-level cache to avoid refetching on every mount
let cachedModels: AvailableModel[] | null = null;
let cacheTimestamp = 0;
const CACHE_TTL_MS = 60_000; // 1 minute
let inFlightModels: Promise<AvailableModel[]> | null = null;

export function invalidateModelCache() {
  cachedModels = null;
  cacheTimestamp = 0;
}

export function useAvailableModels() {
  const [models, setModels] = useState<AvailableModel[]>(cachedModels ?? []);
  const [isLoading, setIsLoading] = useState(!cachedModels);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    const now = Date.now();
    if (cachedModels && now - cacheTimestamp < CACHE_TTL_MS) {
      setModels(cachedModels);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);

    const load = async (): Promise<AvailableModel[]> => {
      const now2 = Date.now();
      if (cachedModels && now2 - cacheTimestamp < CACHE_TTL_MS) {
        return cachedModels;
      }

      if (inFlightModels) return inFlightModels;

      inFlightModels = fetch(`${API_BASE}/models`)
        .then((r) => r.json())
        .then((data: { models: AvailableModel[] }) => {
          cachedModels = data.models ?? [];
          cacheTimestamp = Date.now();
          return cachedModels;
        })
        .catch(() => {
          const fallback = MODEL_OPTIONS.map((o) => ({
            model: o.value,
            provider: 'fallback',
          }));
          cachedModels = fallback;
          cacheTimestamp = Date.now();
          return fallback;
        })
        .finally(() => {
          inFlightModels = null;
        });

      return inFlightModels;
    };

    load()
      .then((m) => setModels(m))
      .finally(() => setIsLoading(false));
  }, []);

  // Convenience: flat list of model IDs for simple dropdowns
  const modelIds = models.map((m) => m.model);

  return { models, modelIds, isLoading };
}
