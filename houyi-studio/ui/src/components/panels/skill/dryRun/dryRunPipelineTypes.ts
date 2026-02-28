import type React from 'react';

export type StageStatus = 'pending' | 'running' | 'pass' | 'fail' | 'warn' | 'skip';

export interface PipelineStage {
  id: string;
  number: number;
  label: string;
  status: StageStatus;
  summary: string;
  details?: React.ReactNode;
}

export interface DryRunPipelineContext {
  planningFlowId?: string | null;
  planningFlowLabel?: string | null;
  selectedExampleId?: string | null;
  selectedExampleLabel?: string | null;
  selectedToolName?: string | null;
  selectedExampleInput?: Record<string, unknown> | null;
  selectedExampleObjective?: string | null;
}
