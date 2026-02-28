import type { DryRunSchemaField } from './SkillFieldInput';
import type { ToolDryRunPreset } from './dryRunToolRules';

export const presetToFormValues = (preset: ToolDryRunPreset | null): Record<string, string> => {
  if (!preset) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(preset.input ?? {})) {
    if (Array.isArray(v)) {
      out[k] = v.join(', ');
    } else if (typeof v === 'boolean') {
      out[k] = v ? 'true' : 'false';
    } else if (v !== undefined && v !== null) {
      out[k] = String(v);
    }
  }
  return out;
};

const resolveRefNode = (schema: Record<string, unknown>, ref: string): Record<string, unknown> => {
  if (!ref.startsWith('#/')) return {};
  const path = ref.slice(2).split('/');
  let cursor: unknown = schema;
  for (const segment of path) {
    if (!cursor || typeof cursor !== 'object' || !(segment in cursor)) {
      return {};
    }
    cursor = (cursor as Record<string, unknown>)[segment];
  }
  return (cursor && typeof cursor === 'object') ? (cursor as Record<string, unknown>) : {};
};

const resolveNode = (schema: Record<string, unknown>, node: unknown): Record<string, unknown> => {
  if (!node || typeof node !== 'object') return {};
  const raw = node as Record<string, unknown>;
  const ref = typeof raw.$ref === 'string' ? resolveRefNode(schema, raw.$ref) : {};
  const rest = { ...raw };
  delete rest.$ref;
  return { ...ref, ...rest };
};

export const parseSchemaFields = (inputSchema?: Record<string, unknown>): DryRunSchemaField[] => {
  if (!inputSchema) return [];

  const schema = inputSchema;
  const props = (schema.properties as Record<string, unknown>) || {};
  const required = new Set((schema.required as string[]) || []);

  return Object.entries(props).map(([name, def]) => {
    const d = resolveNode(schema, def);
    let fieldType = d.type as string | undefined;
    let fieldFormat = d.format as string | undefined;
    let nullable = false;
    let enumValues = d.enum as string[] | undefined;

    if (!fieldType && Array.isArray(d.anyOf)) {
      const variants = (d.anyOf as Array<unknown>).map((variant) => resolveNode(schema, variant));
      const types = variants.map((t) => t.type as string).filter(Boolean);
      const nonNullTypes = types.filter((t) => t !== 'null');
      nullable = types.includes('null');
      fieldType = nonNullTypes[0] || 'string';
      if (!fieldFormat) {
        const withFormat = variants.find((v) => typeof v.format === 'string');
        fieldFormat = withFormat?.format as string | undefined;
      }
      if (!enumValues) {
        for (const v of variants) {
          if (Array.isArray(v.enum)) {
            enumValues = v.enum as string[];
            break;
          }
        }
      }
    }

    const defaultVal = d.default;
    const isRequired = required.has(name) && !nullable && defaultVal === undefined;
    const hasDefault = defaultVal !== undefined && defaultVal !== null;

    return {
      name,
      type: fieldType || 'string',
      format: fieldFormat,
      title: (d.title as string) || '',
      description: (d.description as string) || '',
      required: isRequired,
      default: hasDefault ? defaultVal : undefined,
      defaultRaw: defaultVal,
      nullable,
      enum: enumValues,
      minimum: d.minimum as number | undefined,
      maximum: d.maximum as number | undefined,
    };
  });
};

export const buildExecutionInputFromForm = (
  visibleSchemaFields: Array<DryRunSchemaField & { visible?: boolean }>,
  formValues: Record<string, string>,
  errors: Record<string, string>,
): Record<string, unknown> => {
  const input: Record<string, unknown> = {};

  for (const field of visibleSchemaFields) {
    const val = formValues[field.name];
    if (field.required && (val === undefined || val === '')) {
      errors[field.name] = 'This field is required';
    }
    if (val !== undefined && val !== '') {
      if (field.type === 'number' || field.type === 'integer') {
        input[field.name] = Number(val);
      } else if (field.type === 'boolean') {
        input[field.name] = val === 'true';
      } else {
        input[field.name] = val;
      }
    }
  }

  return input;
};

export const fieldPlaceholder = (field: DryRunSchemaField): string => {
  if (field.required) return 'Required';
  if (field.default !== undefined) return `Default: ${field.default}`;
  return 'Optional';
};
