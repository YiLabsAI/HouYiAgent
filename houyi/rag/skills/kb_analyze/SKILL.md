---
name: kb-analyze
version: "1.0.0"
description: |
  知识库分析工具。提供知识库统计、健康检查和优化建议。
  分析文档分布、索引状态、查询性能等。

user-invocable: true

allowed-tools:
  - Read
  - Glob

hooks:
  PreToolUse:
    - matcher: "Read|Glob"
      type: handler
      handler: houyi.rag.skills.kb_analyze.hooks:pre_analyze_hook
  PostToolUse:
    - matcher: ".*"
      type: handler
      handler: houyi.rag.skills.kb_analyze.hooks:post_analyze_hook
  Stop:
    - type: handler
      handler: houyi.rag.skills.kb_analyze.hooks:stop_hook

invocationPolicy:
  modelAutoInvoke: allow
  userInvocable: true
  sideEffect: none

permissions:
  filesystem:
    read: true
    write: false
    paths:
      - "${KNOWLEDGE_DIR}/**/*"
      - "${WORKSPACE}/.rag_index/**/*"
---

# 知识库分析 Skill（kb-analyze）

知识库分析工具，提供统计信息、健康检查和优化建议。

## 触发条件

用户请求以下操作时激活：
- "分析知识库"
- "知识库统计信息"
- "检查知识库健康状态"
- "知识库优化建议"

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| knowledge_dir | str | "knowledge/" | 知识库根目录 |
| analysis_type | str | "full" | 分析类型：full/stats/health/optimize |
| include_content | bool | false | 是否包含内容分析 |

## 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| stats | Stats | 统计信息 |
| health | Health | 健康状态 |
| recommendations | list[str] | 优化建议 |
| issues | list[Issue] | 发现的问题 |

## 工作流程

1. **收集统计**
   - 文档数量和类型分布
   - 分块数量和大小分布
   - 索引大小和状态

2. **健康检查**
   - 索引完整性
   - 文档覆盖率
   - 嵌入质量

3. **性能分析**
   - 查询延迟分布
   - 召回率估计

4. **生成建议**
   - 索引优化建议
   - 分块策略调整
   - 资源使用优化

## 分析类型

- **full**: 完整分析（统计 + 健康 + 优化）
- **stats**: 仅统计信息
- **health**: 仅健康检查
- **optimize**: 仅优化建议

## 示例

```
用户: 分析知识库状态
Skill: 执行完整分析
输出:
  - 文档数: 150，分块数: 2340
  - 索引健康: 良好
  - 建议: 考虑对大文档启用分层索引
```

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "knowledge_dir": {
      "type": "string",
      "default": "knowledge/",
      "description": "Knowledge base directory"
    },
    "analysis_type": {
      "type": "string",
      "enum": ["full", "stats", "health", "optimize"],
      "default": "full"
    },
    "include_content": {
      "type": "boolean",
      "default": false
    }
  }
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "stats": {
      "type": "object",
      "properties": {
        "total_documents": { "type": "integer" },
        "total_chunks": { "type": "integer" },
        "total_entities": { "type": "integer" },
        "index_size_mb": { "type": "number" },
        "file_types": {
          "type": "object",
          "additionalProperties": { "type": "integer" }
        }
      }
    },
    "health": {
      "type": "object",
      "properties": {
        "status": {
          "type": "string",
          "enum": ["healthy", "degraded", "unhealthy"]
        },
        "index_integrity": { "type": "boolean" },
        "coverage_percent": { "type": "number" }
      }
    },
    "recommendations": {
      "type": "array",
      "items": { "type": "string" }
    },
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "severity": { "type": "string" },
          "message": { "type": "string" }
        }
      }
    }
  }
}
```
