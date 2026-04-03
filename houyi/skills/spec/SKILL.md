---
name: houyi_spec
version: 0.3.0
description: |
  HouYi 项目设计文档与交付验收文档生成规范。
  当需要为新特性创建设计文档（design doc）、验收跟踪文档（acceptance doc），
  或维护交付日志时，使用本 skill 提供的标准模板。
  中文模板为基准版本（.CN.md），英文模板无后缀（默认）。
author: Houyi Team
---

# HouYi Spec — 设计与交付文档规范

## 概述

每个 HouYi 特性由**两份配套文档**驱动：

| 文档 | 用途 | 文件命名 |
|------|------|---------|
| **设计文档** | 架构、API、行为规范 — 唯一基准定义 | `<feature>-design.md` |
| **验收文档** | 交付跟踪、Gate 验证、测试证据 | `<feature>-acceptance.md` |

存放于 `docs/design/`。验收文档**引用**设计文档，不复制接口定义。

## 模板文件

| 模板 | 路径 | 语言 |
|------|------|------|
| 设计文档模板（基准） | [design-template.CN.md](templates/design-template.CN.md) | 中文 |
| 设计文档模板 | [design-template.md](templates/design-template.md) | English |
| 验收文档模板（基准） | [acceptance-template.CN.md](templates/acceptance-template.CN.md) | 中文 |
| 验收文档模板 | [acceptance-template.md](templates/acceptance-template.md) | English |

**使用方式**：读取对应模板，替换 `<Feature>` / `<feature>` 占位符，按模板结构填写内容。

## 核心规则速查

1. **设计文档先行** — 先确立 API 契约，再派生 Sprint 任务
2. **评审记录前置** — §0 设计评审记录 + 变更历史放在文档最前面
3. **以终为始** — 对标业界 SOTA，设定有挑战性的终态目标，再拆解为可落地的阶段里程碑
4. **Benchmark 闭环** — 每个目标对应可量化的 benchmark（业界公认 / 自研），脚手架入库，Gate 前必跑
5. **前沿洞察** — 业界调研覆盖最新理念与实践，确保设计方向不滞后
6. **完成链传导** — Task ✅ → Sprint ✅ → Gate ✅ → Phase ✅，不可跳层
7. **禁止虚标** — 任何 ✅ 必须有可运行的验证证据
8. **三区分离** — Sprint 交付 / Bug 修复 / 架构修正 不得混排
9. **默认中文** — 沟通语言为中文时，输出文档使用中文（CN 模板）；仅在用户明确要求时生成英文或双语版本
10. **单份输出** — 每次生成仅产出一份文档，不自动生成双语副本
11. **四图必备** — 架构图、时序图、流程图、类图

详细说明见各模板文件。
