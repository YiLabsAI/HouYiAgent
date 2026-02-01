import { buildToolStatistics, type ToolStatistics } from '../utils/toolStats';

type StoreSet = (partial: any | ((state: any) => any)) => void;
type StoreGet = () => any;

export const createToolStatsActions = (_set: StoreSet, get: StoreGet) => ({
  getToolStatistics: (): ToolStatistics => {
    const state = get();
    const execution = state.getViewExecution();
    const plan = state.currentPlan;
    return buildToolStatistics(execution, plan);
  },
});
