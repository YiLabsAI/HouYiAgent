---
name: kb-graph
version: "1.0.0"
description: |
  知识图谱查询工具。支持实体检索、关系遍历和多跳推理。
  基于 PPR（Personalized PageRank）算法进行相关性排序。

user-invocable: true

allowed-tools:
  - Read
  - Grep

hooks:
  PreToolUse:
    - matcher: "Read|Grep"
      type: handler
      handler: houyi.rag.skills.kb_graph.hooks:pre_graph_hook
  PostToolUse:
    - matcher: ".*"
      type: handler
      handler: houyi.rag.skills.kb_graph.hooks:post_graph_hook
  Stop:
    - type: handler
      handler: houyi.rag.skills.kb_graph.hooks:stop_hook

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

# 知识图谱查询 Skill（kb-graph）

知识图谱查询工具，支持实体检索、关系遍历和多跳推理。

## 触发条件

用户请求以下操作时激活：
- "查找与 X 相关的实体"
- "X 和 Y 之间有什么关系"
- "遍历知识图谱找到..."
- "多跳推理..."

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | str | 必填 | 查询内容 |
| start_entities | list[str] | [] | 起始实体列表 |
| max_hops | int | 3 | 最大遍历跳数 |
| top_k | int | 10 | 返回结果数量 |
| use_ppr | bool | true | 是否使用 PPR 排序 |

## 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| entities | list[Entity] | 找到的实体列表 |
| relations | list[Relation] | 找到的关系列表 |
| paths | list[Path] | 实体间的路径 |
| confidence | float | 置信度 (0-1) |

## 工作流程

1. **实体识别**
   - 从查询中提取实体
   - 在图谱中匹配实体节点

2. **图遍历**
   - 从起始实体开始 BFS/DFS
   - 限制最大跳数

3. **PPR 排序**
   - 计算 Personalized PageRank
   - 按相关性排序结果

4. **结果组织**
   - 提取相关实体和关系
   - 构建实体间路径

## 示例

```
用户: 找出与 "机器学习" 相关的所有技术概念
Skill: 从 "机器学习" 节点开始，遍历知识图谱
输出: 找到 15 个相关实体，包括 "深度学习"、"神经网络"...
```

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Query for graph traversal"
    },
    "start_entities": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    },
    "max_hops": {
      "type": "integer",
      "default": 3
    },
    "top_k": {
      "type": "integer",
      "default": 10
    },
    "use_ppr": {
      "type": "boolean",
      "default": true
    }
  },
  "required": ["query"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "name": { "type": "string" },
          "type": { "type": "string" },
          "score": { "type": "number" }
        }
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source": { "type": "string" },
          "target": { "type": "string" },
          "type": { "type": "string" }
        }
      }
    },
    "paths": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "confidence": { "type": "number" }
  }
}
```
