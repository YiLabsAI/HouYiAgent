---
name: kb-search
description: |
  知识库智能检索助手。根据查询复杂度自动选择检索模式：
  - 小规模/简单查询：Agentic 模式（grep + read_file）
  - 大规模/复杂查询：Indexed 模式（Vector + Graph + BM25）
  支持渐进式检索，最多 5 轮迭代确保找到最相关信息。
---

# 本地知识库检索 Skill（kb-search）

## 触发条件

用户问题涉及以下场景时激活本 Skill：
- "从知识库查找/检索信息"
- "在文档中搜索..."
- "查找资料/文档..."
- "帮我找一下关于..."

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| query | str | 必填 | 用户查询 |
| knowledge_dir | str | "knowledge/" | 知识库根目录 |
| mode | str | "auto" | 检索模式：auto/agentic/indexed |
| max_rounds | int | 5 | 最大检索轮数（Agentic 模式） |

## 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| answer | str | 回答内容 |
| sources | list[Source] | 引用来源（文件路径 + 位置） |
| confidence | float | 置信度 (0-1) |

## 工作流程

### Agentic 模式（小规模知识库）

1. **读取目录索引**
   - 读取 `knowledge_dir/data_structure.md` 了解知识库结构
   - 识别子目录和文件用途

2. **选择候选文件**
   - 根据查询内容选择最相关的目录和文件
   - 递归进入子目录获取更多候选

3. **渐进式检索**（最多 max_rounds 轮）
   - Round 1: 使用初始关键词 grep 搜索
   - Round 2: 扩展同义词和相关词
   - Round 3: 扩展上下文范围
   - Round 4: 跨文件关联检索
   - Round 5: 最终确认

4. **组织答案**
   - 汇总检索结果
   - 标注引用来源

### Indexed 模式（大规模知识库）

1. **查询分析**
   - 识别查询类型（精确匹配/语义搜索/因果推理）
   - 选择检索策略组合

2. **并行检索**
   - BM25 稀疏检索
   - Vector 向量检索（HNSW）
   - Graph 图检索（PPR，可选）

3. **结果融合**
   - RRF 融合多路检索结果
   - LLM 重排序

4. **生成答案**
   - CRAG 验证相关性
   - 生成最终答案

## 协同工具

在 Agentic 模式下，本 Skill 会使用以下工具：

- **grep**: 关键词搜索
- **read_file**: 局部文件读取
- **pdftotext/pdfplumber**: PDF 处理（需先读取 references/pdf_reading.md）
- **pandas**: Excel 分析（需先读取 references/excel_reading.md）

## 示例

### 示例 1: 简单查询

```
用户: 什么是 RAG?
Skill: 使用 Agentic 模式，在知识库中搜索 "RAG" 相关内容
输出: RAG (Retrieval-Augmented Generation) 是一种...
来源: knowledge/concepts/rag.md
```

### 示例 2: 复杂分析

```
用户: 分析 2023 年销售趋势
Skill: 使用 Indexed 模式，结合向量搜索和图检索
输出: 根据财务报告分析，2023 年销售呈现...
来源: knowledge/reports/2023_annual.pdf, knowledge/data/sales.xlsx
```

## 注意事项

1. **文件格式处理**
   - 遇到 PDF/Excel 文件时，先读取对应的 references 文档学习处理方法
   - 避免直接读取整个大文件

2. **Token 优化**
   - 使用 grep 定位后只读取匹配附近的上下文
   - 设置合理的 chunk_limit 避免过多 token 消耗

3. **来源引用**
   - 始终标注信息来源
   - 包含文件路径和大致位置
