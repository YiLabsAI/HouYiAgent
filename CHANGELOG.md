# Changelog

All notable changes to HouYi will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed datetime API deprecation warnings in `runtime/executor.py` and `orchestration/state.py`
- Fixed 8 failing executor tests related to datetime usage
- Fixed pytest collection warnings for `TestCase` Pydantic model

### Planned for v0.2.0
- NeuroSymbolic Engine with verification layer
- Code verification before execution (Python, SQL)
- Structured output enforcement with JSON Schema
- Auto-retry mechanism with error feedback

---

## [0.1.0] - 2025-01-04

### Added

#### Core Framework
- Multi-agent orchestration system with DAG-based execution engine
- Schema-first design using Pydantic v2 for type safety
- `AgentSpec`, `TaskSpec`, and `SkillSpec` for defining agent behaviors
- `LocalExecutor` with topological scheduling for concurrent node execution
- Immutable `SessionState` for state management and context propagation
- IR-based `ExecutionPlan` with 5 node types: LLM, TOOL, VERIFY, LOGIC, ROUTE

#### LLM Integration
- OpenAI adapter with GPT-3.5/GPT-4 support
- Anthropic adapter with Claude support
- Unified `LLMAdapter` interface with streaming capabilities
- Function calling and tool use support
- `SkillExecutor` with input/output validation, retry logic, and timeout control

#### Evaluation System
- 19 built-in evaluators across multiple dimensions:
  - **Quality**: Accuracy, Completeness, Relevance, SemanticSimilarity
  - **Performance**: Cost, Latency
  - **Safety**: Toxicity, Hallucination, Safety
  - **Bias & Facts**: Bias, Factuality
  - **RAG**: Groundedness, ContextPrecision, ContextRecall, Faithfulness
  - **Structure**: Coherence, SkillUsage
  - **Advanced**: LLMJudge, Custom
- Batch evaluation with `Dataset` support (JSON/CSV/YAML)
- Report generation in HTML, JSON, and Markdown formats

#### Observability
- Lightweight `TraceManager` with zero-config setup
- 4 trace exporters:
  - Console (simple/verbose modes)
  - JSON (file-based)
  - Jaeger (OTLP/HTTP)
  - Datadog (Agent API v0.4)
- OpenTelemetry-compatible tracing

#### AgentSkills.io Integration
- Load skills from local files: `Skill.from_file()`
- Load skills from URLs with caching: `Skill.from_url()`
- Load skills from AgentSkills.io registry: `Skill.from_registry()`
- Export skills to skill.md format: `export_skill_md()`

#### Documentation
- Comprehensive README with quickstart guide
- Development guidelines and architecture documentation
- API reference and usage examples

### Fixed
- Migrated from deprecated Pydantic `Config` to `ConfigDict`
- Resolved all Pydantic v2 compatibility warnings

### Technical Details
- Full asyncio support for concurrent operations
- Complete type hints with mypy strict mode
- Zero-configuration setup - works out of the box
- Python 3.10+ support

---

