import type { ExecutionIR, PlanIR } from '@/types/ir';

export interface ToolStatistics {
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
  toolsByName: Record<string, { count: number; successful: number; failed: number }>;
  toolNodes: number;
  totalNodes: number;
}

export const buildToolStatistics = (
  execution: ExecutionIR | null,
  plan: PlanIR | null,
): ToolStatistics => {
  const stats: ToolStatistics = {
    totalCalls: 0,
    successfulCalls: 0,
    failedCalls: 0,
    toolsByName: {},
    toolNodes: 0,
    totalNodes: 0,
  };

  if (!execution || !plan) {
    return stats;
  }

  const toolNodes = plan.nodes.filter((n) => {
    const rawType = (n as any).node_type?.value ?? (n as any).node_type;
    return typeof rawType === 'string' && rawType.toLowerCase() === 'tool';
  });
  stats.toolNodes = toolNodes.length;
  stats.totalNodes = plan.nodes.length;

  const toolCallsByName = new Set<string>();

  for (const [nodeId, nodeExec] of Object.entries(execution.node_executions || {})) {
    const node = plan.nodes.find((n) => n.node_id === nodeId);
    if (!node) continue;

    const nodeType = (node as any).node_type?.value ?? (node as any).node_type;
    const normalizedType = typeof nodeType === 'string' ? nodeType.toLowerCase() : nodeType;

    if (normalizedType === 'llm' && nodeExec.outputs) {
      const outputs = nodeExec.outputs as any;
      if (outputs.type === 'llm_response' && outputs.tool_calls) {
        const toolCalls = outputs.tool_calls || [];
        for (const toolCall of toolCalls) {
          const toolName =
            toolCall.tool_name
            || toolCall.requested_tool_name
            || toolCall.function?.name
            || 'unknown';
          toolCallsByName.add(toolName);
          stats.totalCalls += 1;

          if (!stats.toolsByName[toolName]) {
            stats.toolsByName[toolName] = { count: 0, successful: 0, failed: 0 };
          }
          stats.toolsByName[toolName].count += 1;

          if (toolCall.result?.is_error || toolCall.result?.raw?.error) {
            stats.failedCalls += 1;
            stats.toolsByName[toolName].failed += 1;
          } else {
            stats.successfulCalls += 1;
            stats.toolsByName[toolName].successful += 1;
          }
        }
      }
    }

    if (normalizedType === 'tool') {
      const toolName = node.config?.tool_name || node.metadata?.tool_name || node.config?.skill_name || 'unknown';
      if (toolCallsByName.has(toolName)) {
        continue;
      }
      stats.totalCalls += 1;
      if (!stats.toolsByName[toolName]) {
        stats.toolsByName[toolName] = { count: 0, successful: 0, failed: 0 };
      }
      if (nodeExec.status === 'completed' && !nodeExec.error) {
        stats.successfulCalls += 1;
        stats.toolsByName[toolName].count += 1;
        stats.toolsByName[toolName].successful += 1;
      } else if (nodeExec.status === 'failed' || nodeExec.error) {
        stats.failedCalls += 1;
        stats.toolsByName[toolName].count += 1;
        stats.toolsByName[toolName].failed += 1;
      }
    }
  }

  return stats;
};
