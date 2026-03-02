import type { DryRunSchemaField } from './SkillFieldInput';

export interface DryRunFieldRules {
  visible: boolean;
  required: boolean;
}

export interface ToolDryRunPreset {
  id: string;
  label: string;
  description: string;
  input: Record<string, unknown>;
  /** Optional command template rendered with placeholders before execution. */
  command_template?: string;
  /** Optional workflow identifier when the example comes from workflow extraction. */
  workflow_id?: string;
  /** Optional workflow evidence text used in dry-run audit panel. */
  evidence?: string;
  /** Optional workflow validation state copied from extraction output. */
  validation_status?: 'pass' | 'warn' | 'fail';
  validation_issues?: string[];
  /** Tracks where this example originated for merge/trace UI. */
  origin?: 'package_examples' | 'tool_preset' | 'frontmatter_workflow' | 'instruction_workflow';
  expectedFocus: string[];
  objective?: string;
  source?: string;
  confidence?: 'high' | 'medium' | 'low' | string;
  confidence_reason?: string;
  confidence_breakdown?: Record<string, number>;
}

export type PlanningFlowPreset = ToolDryRunPreset;

const PLANNING_ACTIONS = new Set(['create', 'update', 'complete', 'status']);

const getAction = (value: unknown): string => (typeof value === 'string' ? value : '');

interface PlanningDetectionOptions {
  inputSchema?: Record<string, unknown>;
}

const planningActionFromSchema = (inputSchema?: Record<string, unknown>): string[] => {
  if (!inputSchema) return [];
  const properties = inputSchema.properties;
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) return [];
  const actionSchema = (properties as Record<string, unknown>).action;
  if (!actionSchema || typeof actionSchema !== 'object' || Array.isArray(actionSchema)) return [];
  const enumValues = (actionSchema as Record<string, unknown>).enum;
  if (!Array.isArray(enumValues)) return [];
  return enumValues.filter((value): value is string => typeof value === 'string');
};

const hasPlanningPresetActions = (toolName: string): boolean => {
  const presets = getToolDryRunPresets(toolName);
  return presets.some((preset) => PLANNING_ACTIONS.has(getAction(preset.input.action)));
};

const inferPlanningBySchema = (inputSchema?: Record<string, unknown>): boolean => {
  const actions = planningActionFromSchema(inputSchema);
  if (actions.length === 0) return false;
  return actions.some((action) => PLANNING_ACTIONS.has(action));
};

// Data-driven planning detection:
// - Prefer schema capability (action enum contains planning actions).
// - Fallback to tool preset examples when schema is absent.
export const isPlanningWithFilesTool = (
  toolName: string,
  options?: PlanningDetectionOptions,
): boolean => inferPlanningBySchema(options?.inputSchema) || hasPlanningPresetActions(toolName);

// External planning mode now follows namespace convention instead of one fixed tool id.
export const isExternalPlanningWithFilesTool = (
  toolName: string,
  options?: PlanningDetectionOptions,
): boolean => isPlanningWithFilesTool(toolName, options) && toolName.startsWith('ext__');

export const getToolFieldRules = (
  toolName: string,
  field: DryRunSchemaField,
  formValues: Record<string, string>,
  options?: PlanningDetectionOptions,
): DryRunFieldRules => {
  if (!isPlanningWithFilesTool(toolName, options)) {
    return { visible: true, required: field.required };
  }

  if (field.name === 'action') {
    return { visible: true, required: true };
  }

  const action = getAction(formValues.action);
  if (!action) {
    return { visible: false, required: false };
  }

  if (action === 'create') {
    if (field.name === 'task') {
      return { visible: true, required: true };
    }
    return { visible: field.name === 'subtasks', required: false };
  }

  if (action === 'update') {
    if (field.name === 'subtask_index') {
      return { visible: true, required: true };
    }
    return { visible: field.name === 'completed', required: false };
  }

  return { visible: false, required: false };
};

export const applyToolSpecificValidation = (
  toolName: string,
  input: Record<string, unknown>,
  errors: Record<string, string>,
  options?: PlanningDetectionOptions,
): void => {
  if (!isPlanningWithFilesTool(toolName, options)) return;

  const action = getAction(input.action);

  if (action === 'create') {
    const task = typeof input.task === 'string' ? input.task.trim() : '';
    if (!task) {
      errors.task = 'Task is required when action is create';
    }
  }

  if (action === 'update') {
    const idx = input.subtask_index;
    if (idx === undefined || idx === null || idx === '') {
      errors.subtask_index = 'subtask_index is required when action is update';
    }
  }
};

export const normalizeToolInput = (
  toolName: string,
  input: Record<string, unknown>,
  options?: PlanningDetectionOptions,
): Record<string, unknown> => {
  if (!isPlanningWithFilesTool(toolName, options)) return input;

  const normalized = { ...input };
  const action = getAction(normalized.action);
  if (action === 'create' && typeof normalized.task === 'string') {
    normalized.task = normalized.task.trim();
    if (typeof normalized.subtasks === 'string') {
      normalized.subtasks = normalized.subtasks
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item.length > 0);
    }
    delete normalized.subtask_index;
    delete normalized.completed;
    return normalized;
  }

  if (action === 'update') {
    delete normalized.task;
    delete normalized.subtasks;
    delete normalized.command;
    return normalized;
  }

  if (action === 'complete' || action === 'status') {
    delete normalized.task;
    delete normalized.subtasks;
    delete normalized.subtask_index;
    delete normalized.completed;
    delete normalized.command;
    return normalized;
  }

  return normalized;
};

export const shouldShowActionHint = (
  toolName: string,
  formValues: Record<string, string>,
  options?: PlanningDetectionOptions,
): boolean => isPlanningWithFilesTool(toolName, options) && !getAction(formValues.action);

const toFormStringValues = (input: Record<string, unknown>): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(input)) {
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

const TOOL_DRY_RUN_PRESETS: Record<string, ToolDryRunPreset[]> = {
  'ext__planning-with-files': [
    {
      id: 'example-1-research-task',
      label: 'Example 1 · Research Task',
      description: 'Covers create-plan + research/synthesize/deliver workflow for morning exercise summary.',
      input: {
        action: 'create',
        task: 'Research morning exercise benefits and write summary',
        subtasks: [
          'Phase 1: Create this plan',
          'Phase 2: Search and gather sources',
          'Phase 3: Synthesize findings',
          'Phase 4: Deliver summary',
        ],
      },
      expectedFocus: ['examples.md #example-1', 'plan bootstrap', 'loop progression'],
      objective: 'Bootstrap research loop with a concrete plan creation payload.',
    },
    {
      id: 'example-2-bug-fix',
      label: 'Example 2 · Bug Fix Task',
      description: 'Covers root-cause analysis style plan updates for authentication login bug scenario.',
      input: {
        action: 'update',
        subtask_index: 2,
        completed: true,
      },
      expectedFocus: ['examples.md #example-2', 'phase transition update', 'error logging discipline'],
      objective: 'Validate root-cause style progress updates and rule discipline.',
    },
    {
      id: 'example-3-feature-development',
      label: 'Example 3 · Feature Development',
      description: 'Covers 3-file pattern for dark-mode feature planning and execution progress.',
      input: {
        action: 'status',
      },
      expectedFocus: ['examples.md #example-3', 'three-file pattern', 'delivery artifact tracking'],
      objective: 'Validate 3-file planning structure for feature delivery tracking.',
    },
    {
      id: 'example-4-error-recovery',
      label: 'Example 4 · Error Recovery Pattern',
      description: 'Covers failure logging and non-silent retry guard before completion.',
      input: { action: 'status' },
      expectedFocus: ['examples.md #example-4', 'log-all-errors rule', 'never-repeat-failures rule'],
      objective: 'Validate error logging and recovery guards before completion.',
    },
  ],
  notebooklm: [
    {
      id: 'example-1-notebook-query',
      label: 'Example 1 · Notebook Query',
      description: 'Ask NotebookLM for a concise source-grounded answer with explicit question payload.',
      input: {
        question: 'Summarize the architecture decisions in this notebook',
        notebook_url: 'https://notebooklm.google.com/notebook/example',
      },
      expectedFocus: ['run.py wrapper', 'ask_question.py', 'source-grounded response'],
      objective: 'Verify question routing and notebook-targeted query payload.',
    },
    {
      id: 'example-2-library-discovery-add',
      label: 'Example 2 · Library Discovery + Add',
      description: 'Discover notebook content first, then add with complete metadata fields.',
      input: {
        operation: 'add_notebook',
        notebook_url: 'https://notebooklm.google.com/notebook/example',
        name: 'Architecture Playbook',
        description: 'Architecture decisions and migration records',
        topics: 'architecture,migration,platform',
      },
      expectedFocus: ['smart add workflow', 'notebook_manager.py add', 'metadata completeness'],
      objective: 'Verify add-flow discipline with complete notebook metadata.',
    },
  ],
  'skill-creator': [
    {
      id: 'example-1-skill-scaffold',
      label: 'Example 1 · Skill Scaffold Plan',
      description: 'Plan a new skill package with reusable scripts/references/assets.',
      input: {
        skill_name: 'pdf-editor',
        goal: 'Create a reusable skill for rotating and extracting PDF content',
      },
      expectedFocus: ['init_skill.py', 'package_skill.py', 'resource planning'],
      objective: 'Verify tool routing for structured skill creation requests.',
    },
    {
      id: 'example-2-skill-iteration',
      label: 'Example 2 · Improve Existing Skill',
      description: 'Iterate an existing skill based on usage feedback and package again.',
      input: {
        skill_name: 'pdf-editor',
        goal: 'Improve extraction reliability and add troubleshooting references',
        stage: 'iterate',
      },
      expectedFocus: ['6-step process', 'resource update', 're-package workflow'],
      objective: 'Verify iterative skill refinement flow after real task feedback.',
    },
  ],
  'using-superpowers': [
    {
      id: 'example-1-workflow-discipline',
      label: 'Example 1 · Workflow Discipline',
      description: 'Enforce skill-first workflow before implementation response.',
      input: {
        task: 'Fix login bug with strict skill invocation discipline',
      },
      expectedFocus: ['invoke skill first', 'process before implementation'],
      objective: 'Verify meta-process skill routing under execution tasks.',
    },
    {
      id: 'example-2-plan-before-implementation',
      label: 'Example 2 · Plan Before Coding',
      description: 'Ensure process skills are invoked before implementation skills on build requests.',
      input: {
        task: 'Build a billing dashboard and choose process skill order first',
      },
      expectedFocus: ['skill priority', 'process skills first', 'announce skill usage'],
      objective: 'Verify sequencing discipline across process and implementation skills.',
    },
  ],
  'frontend-design': [
    {
      id: 'example-1-landing-page',
      label: 'Example 1 · Bold Landing Page',
      description: 'Build a production-ready landing page with clear visual direction.',
      input: {
        brief: 'Create a bold AI infra landing page with mobile and desktop support',
        aesthetic: 'brutalist editorial',
      },
      expectedFocus: ['distinct typography', 'non-generic layout', 'production-grade output'],
      objective: 'Verify frontend skill routing with explicit aesthetic intent.',
    },
    {
      id: 'example-2-dashboard-redesign',
      label: 'Example 2 · Product Dashboard Redesign',
      description: 'Redesign a dense dashboard with clear visual hierarchy and responsive behavior.',
      input: {
        brief: 'Redesign analytics dashboard for product team with responsive desktop/mobile layouts',
        aesthetic: 'industrial data-editorial',
      },
      expectedFocus: ['visual hierarchy', 'responsive implementation', 'intentional motion'],
      objective: 'Verify complex UI redesign flow with explicit style direction.',
    },
  ],
  'rag-skill': [
    {
      id: 'example-1-kb-query',
      label: 'Example 1 · Knowledge Query',
      description: 'Query local knowledge base with explicit root and question.',
      input: {
        query: 'What is RAG and when should it be used?',
        knowledge_dir: 'knowledge/',
      },
      expectedFocus: ['data_structure discovery', 'progressive retrieval', 'source citation'],
      objective: 'Verify rag-skill routing for concrete knowledge retrieval payload.',
    },
    {
      id: 'example-2-pdf-excel-analysis',
      label: 'Example 2 · PDF + Excel Analysis',
      description: 'Run mixed-file retrieval with mandatory reference-first workflow before extraction.',
      input: {
        query: 'Compare KPI trends from the quarterly PDF report and Excel workbook',
        knowledge_dir: 'knowledge/',
        file_types: ['pdf', 'xlsx'],
      },
      expectedFocus: ['reference-first processing', 'iterative retrieval', 'evidence synthesis'],
      objective: 'Verify multi-file retrieval flow with PDF/Excel processing discipline.',
    },
  ],
};

export const getToolDryRunPresets = (toolName: string): ToolDryRunPreset[] => (
  TOOL_DRY_RUN_PRESETS[toolName] ?? []
);

export const getToolDryRunPreset = (toolName: string, presetId: string): ToolDryRunPreset | null => (
  getToolDryRunPresets(toolName).find((p) => p.id === presetId) ?? null
);

export const hasToolDryRunPresets = (toolName: string): boolean => getToolDryRunPresets(toolName).length > 0;

export const toolPresetFormValues = (toolName: string, presetId: string): Record<string, string> => {
  const preset = getToolDryRunPreset(toolName, presetId);
  if (!preset) return {};
  return toFormStringValues(preset.input);
};

const resolvePlanningPresetSourceTool = (): string | null => {
  const entries = Object.entries(TOOL_DRY_RUN_PRESETS);
  for (const [toolName, presets] of entries) {
    const actions = new Set(
      presets
        .map((preset) => getAction(preset.input.action))
        .filter((action) => PLANNING_ACTIONS.has(action)),
    );
    if (actions.has('create') && actions.has('update') && actions.has('status')) {
      return toolName;
    }
  }
  return null;
};

const PLANNING_PRESET_SOURCE_TOOL = resolvePlanningPresetSourceTool();

export const PLANNING_FLOW_PRESETS: PlanningFlowPreset[] = PLANNING_PRESET_SOURCE_TOOL
  ? getToolDryRunPresets(PLANNING_PRESET_SOURCE_TOOL)
  : [];

/* Backward-compatible wrappers for planning-specific call sites/tests */
export const getPlanningFlowPreset = (presetId: string): PlanningFlowPreset | null => (
  PLANNING_PRESET_SOURCE_TOOL
    ? getToolDryRunPreset(PLANNING_PRESET_SOURCE_TOOL, presetId)
    : null
);

export const planningPresetFormValues = (presetId: string): Record<string, string> => (
  PLANNING_PRESET_SOURCE_TOOL
    ? toolPresetFormValues(PLANNING_PRESET_SOURCE_TOOL, presetId)
    : {}
);
