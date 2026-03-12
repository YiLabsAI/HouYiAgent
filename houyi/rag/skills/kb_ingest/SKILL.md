---
name: kb-ingest
version: "1.0.0"
description: |
  知识库文档摄入工具。将文档加载、分块、嵌入并构建索引。
  支持多种文件格式：Markdown、TXT、PDF、HTML。
  支持增量更新和完整重建。

user-invocable: true

allowed-tools:
  - Read
  - Write
  - Glob

hooks:
  PreToolUse:
    - matcher: "Write"
      type: handler
      handler: houyi.rag.skills.kb_ingest.hooks:pre_ingest_hook
  PostToolUse:
    - matcher: ".*"
      type: handler
      handler: houyi.rag.skills.kb_ingest.hooks:post_ingest_hook
  Stop:
    - type: handler
      handler: houyi.rag.skills.kb_ingest.hooks:stop_hook

invocationPolicy:
  modelAutoInvoke: allow_with_consent
  userInvocable: true
  sideEffect: filesystem

permissions:
  filesystem:
    read: true
    write: true
    paths:
      - "${KNOWLEDGE_DIR}/**/*"
      - "${WORKSPACE}/knowledge/**/*"
      - "${WORKSPACE}/.rag_index/**/*"
---

# 知识库摄入 Skill（kb-ingest）

文档摄入工具，将文档加载、分块、嵌入并构建检索索引。

## 触发条件

用户请求以下操作时激活：
- "添加文档到知识库"
- "索引这些文件"
- "更新知识库索引"
- "重建知识库"

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| paths | list[str] | 必填 | 要摄入的文件/目录路径 |
| knowledge_dir | str | "knowledge/" | 知识库根目录 |
| mode | str | "incremental" | 摄入模式：incremental/full |
| build_graph | bool | false | 是否构建知识图谱 |
| chunk_size | int | 512 | 文档分块大小 |
| chunk_overlap | int | 64 | 分块重叠大小 |

## 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| success | bool | 是否成功 |
| documents_processed | int | 处理的文档数 |
| chunks_created | int | 创建的分块数 |
| index_path | str | 索引存储路径 |
| message | str | 状态消息 |

## 工作流程

1. **扫描文件**
   - 递归扫描指定路径
   - 过滤支持的文件格式

2. **加载文档**
   - 根据文件类型使用对应 loader
   - 提取文本内容和元数据

3. **分块处理**
   - 按 chunk_size 分割文档
   - 保留 chunk_overlap 重叠

4. **嵌入生成**
   - 调用嵌入模型生成向量
   - 支持批量处理

5. **索引构建**
   - 构建向量索引 (HNSW)
   - 构建稀疏索引 (BM25)
   - 可选构建知识图谱

6. **持久化**
   - 保存索引到磁盘
   - 更新索引元数据

## 支持的文件格式

- Markdown (.md)
- 纯文本 (.txt)
- PDF (.pdf) - 需要 pypdf
- HTML (.html, .htm)

## 示例

```
用户: 将 docs/ 目录添加到知识库
Skill: 扫描 docs/ 目录，处理所有支持的文件
输出: 成功摄入 25 个文档，创建 342 个分块
```

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "paths": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Files or directories to ingest"
    },
    "knowledge_dir": {
      "type": "string",
      "default": "knowledge/",
      "description": "Knowledge base directory"
    },
    "mode": {
      "type": "string",
      "enum": ["incremental", "full"],
      "default": "incremental"
    },
    "build_graph": {
      "type": "boolean",
      "default": false
    },
    "chunk_size": {
      "type": "integer",
      "default": 512
    },
    "chunk_overlap": {
      "type": "integer",
      "default": 64
    }
  },
  "required": ["paths"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": { "type": "boolean" },
    "documents_processed": { "type": "integer" },
    "chunks_created": { "type": "integer" },
    "index_path": { "type": "string" },
    "message": { "type": "string" }
  }
}
```
