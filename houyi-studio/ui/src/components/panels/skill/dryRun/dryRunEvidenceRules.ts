export const PLANNING_EXPECTED_ACTION_BY_EXAMPLE_ID: Record<string, string> = {
  'example-1-research-task': 'create',
  'example-2-bug-fix': 'update',
  'example-3-feature-development': 'status',
  'example-4-error-recovery': 'status',
};

export const EXTERNAL_EXAMPLE_STATIC_CHECKS: Record<string, Array<{ key: string; label: string; needles: string[] }>> = {
  notebooklm: [
    { key: 'run-wrapper', label: 'Uses scripts/run.py wrapper discipline', needles: ['scripts/run.py', 'always use run.py'] },
    { key: 'auth-first', label: 'Includes auth-first workflow guidance', needles: ['auth_manager.py status', 'authenticate'] },
  ],
  'skill-creator': [
    { key: 'init-skill', label: 'Mentions init_skill.py based scaffolding', needles: ['init_skill.py', 'initializing the skill'] },
    { key: 'package-skill', label: 'Mentions package_skill.py packaging workflow', needles: ['package_skill.py', 'packaging a skill'] },
  ],
  'using-superpowers': [
    { key: 'skill-first', label: 'Enforces skill invocation before response', needles: ['before any response', 'must invoke the skill'] },
    { key: 'priority-order', label: 'Defines skill priority/process order', needles: ['skill priority', 'process skills first'] },
  ],
  'frontend-design': [
    { key: 'bold-direction', label: 'Requires explicit visual direction before coding', needles: ['bold aesthetic direction', 'before coding'] },
    { key: 'anti-generic', label: 'Disallows generic AI aesthetics', needles: ['never use generic ai-generated aesthetics'] },
  ],
  'rag-skill': [
    { key: 'kb-root', label: 'Defines knowledge root/data_structure discovery', needles: ['data_structure.md', 'knowledge/'] },
    { key: 'progressive-retrieval', label: 'Defines progressive retrieval behavior', needles: ['progressive retrieval', 'do not read entire files'] },
  ],
};

export const isInstructionDrivenRuntime = (runtimeBinding: string | undefined | null): boolean => (
  runtimeBinding === 'prompt_instructions' || runtimeBinding === 'script_executor_compat'
);

export function parseLooseJson(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  const unfenced = trimmed
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/, '')
    .trim();

  const extractBracketed = (text: string): string => {
    const pairs: Array<[string, string]> = [['{', '}'], ['[', ']']];
    for (const [startCh, endCh] of pairs) {
      const start = text.indexOf(startCh);
      if (start < 0) continue;
      let depth = 0;
      for (let i = start; i < text.length; i += 1) {
        if (text[i] === startCh) depth += 1;
        if (text[i] === endCh) depth -= 1;
        if (depth === 0) return text.slice(start, i + 1);
      }
    }
    return text;
  };

  const candidate = extractBracketed(unfenced);

  try {
    return JSON.parse(candidate);
  } catch {
    const normalized = candidate
      .replace(/([{,]\s*)'([^']+?)'\s*:/g, '$1"$2":')
      .replace(/:\s*'([^']*?)'/g, ':"$1"');
    try {
      return JSON.parse(normalized);
    } catch {
      return null;
    }
  }
}

export function extractAction(payload: unknown, depth = 0): string {
  if (depth > 6 || payload == null) return '';

  if (typeof payload === 'string') {
    const parsed = parseLooseJson(payload);
    return parsed == null ? '' : extractAction(parsed, depth + 1);
  }

  if (Array.isArray(payload)) {
    for (const item of payload) {
      const nested = extractAction(item, depth + 1);
      if (nested) return nested;
    }
    return '';
  }

  if (typeof payload !== 'object') return '';
  const record = payload as Record<string, unknown>;

  for (const key of Object.keys(record)) {
    if (key.toLowerCase() === 'action' && typeof record[key] === 'string') {
      return String(record[key]);
    }
  }

  for (const key of ['arguments', 'input', 'params', 'payload', 'data', 'args', 'kwargs', 'tool_input']) {
    if (!(key in record)) continue;
    const nested = extractAction(record[key], depth + 1);
    if (nested) return nested;
  }

  for (const value of Object.values(record)) {
    const nested = extractAction(value, depth + 1);
    if (nested) return nested;
  }

  return '';
}

export function subsetMatch(actual: unknown, expected: unknown, depth = 0): boolean {
  if (depth > 6) return false;
  if (expected === undefined) return true;
  if (expected === null || typeof expected !== 'object') {
    return actual === expected;
  }

  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length < expected.length) return false;
    return expected.every((item, idx) => subsetMatch(actual[idx], item, depth + 1));
  }

  if (!actual || typeof actual !== 'object' || Array.isArray(actual)) return false;
  const a = actual as Record<string, unknown>;
  return Object.entries(expected as Record<string, unknown>).every(
    ([key, value]) => subsetMatch(a[key], value, depth + 1),
  );
}
