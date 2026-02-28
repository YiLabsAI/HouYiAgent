import type { SkillDetail } from '../../../../types/websocket';
import type { DryRunResultData } from '../../../LeftSidebar/useSkillsLogic';
import {
  buildCoreVerificationStages,
  buildExternalPlanningStages,
  buildGenericExampleStages,
  buildLlmVerificationStages,
  buildRegistrationStages,
} from './dryRunStageBuilders';
import type { DryRunPipelineContext, PipelineStage } from './dryRunPipelineTypes';

export type { DryRunPipelineContext, PipelineStage, StageStatus } from './dryRunPipelineTypes';

export function computeStages(
  result: DryRunResultData | null,
  detail: SkillDetail,
  liveMode: boolean,
  context?: DryRunPipelineContext,
): PipelineStage[] {
  const drafts = [
    ...buildRegistrationStages(result, detail),
    ...buildExternalPlanningStages(result, detail, liveMode, context),
    ...buildGenericExampleStages(result, detail, liveMode, context),
    ...buildCoreVerificationStages(result, detail),
    ...buildLlmVerificationStages(result, detail, liveMode),
  ];

  return drafts.map((stage, idx) => ({
    ...stage,
    number: idx + 1,
  }));
}
