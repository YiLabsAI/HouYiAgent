<div align="center">
  <img src=".github/images/logo-text.svg" alt="HouYi Logo" width="400">

  <h3>Lightweight · Extensible · Production-Grade · Context-Engineered Multi-Agent Framework</h3>
  <p><em>with neuro-symbolic verification</em></p>

  <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://github.com/YiLabsAI/HouYiAgent/actions"><img src="https://img.shields.io/github/actions/workflow/status/YiLabsAI/HouYiAgent/tests.yml?branch=main&label=tests" alt="Tests"></a>
    <a href="https://codecov.io/gh/YiLabsAI/HouYiAgent"><img src="https://codecov.io/gh/YiLabsAI/HouYiAgent/branch/main/graph/badge.svg" alt="Coverage"></a>
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue" alt="Python Versions">
    <br>
    <a href="https://twitter.com/YiLabsAI"><img src="https://img.shields.io/twitter/follow/YiLabsAI?style=social" alt="Twitter Follow"></a>
  </p>
</div>

---

## Overview

**HouYi** is a lightweight, extensible, production-grade multi-agent **framework** that ships with SOTA built-in agents (Deep Research, Chatbox, Memory Inbox). One `Agent` class, one SDK — define, orchestrate, evaluate, and ship agents from prototype to production without changing your API surface.

**Why HouYi**

- **Full-lifecycle harness** — Not just execution: definition → orchestration → context engineering → evaluation → observability → governance. Every layer is pluggable, every extension point is documented for community and enterprise customization.
- **Context engineering as first-class** — Token budgeting, persistent memory with emphasis-aware recall, RAG, context compression, and Reminders injection at the Transformer attention sweet spot — built into the SDK, not afterthoughts.
- **Neuro-symbolic verification** — Z3 SMT solver validates LLM outputs against business constraints, separating probabilistic reasoning from deterministic correctness for production reliability.
- **Ships with SOTA agents** — Deep Research (plan → multi-round search → conflict resolution → citation-verified report with RACE/FACT scoring), Chatbox (multi-turn with tool calling and memory), Memory Inbox (LLM-powered extraction with review workflow). Use them directly or study their source as reference implementations.

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HouYi Studio (Ideas Foundry)                        │
│   Graph Orchestration  · Chatbox · Agent Hub · Deep Research            │
├─────────────────────────────────────────────────────────────────────────┤
│                   Studio Server (FastAPI + SSE)                         │
│   Chat API · Research API · Memory API · Knowledge API                  │
├─────────────────────────────────────────────────────────────────────────┤
│                         HouYi SDK (Core)                                │
│                                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │  Agent   │  │  AgentTeam   │  │  Team    │  │  DAG               │   │
│  │  Runner  │  │  Manager     │  │  Task    │  │  Engine            │   │
│  └─────┬────┘  └──────┬───────┘  └────┬─────┘  └────────┬───────────┘   │
│        └──────────┬───┴───────────────┴─────────────────┘               │
│            ┌──────┴───────┐                                             │
│            │ Orchestrator │  Delegate · Autonomous · DAG                │
│            └──────┬───────┘                                             │
│  ┌────────────────┼────────────────────────────────────────────────┐    │
│  │         Context Engineering Layer                 ★ Pluggable   │    │
│  │  Token Budget · Tools · Memory · RAG · State Checkpoints        │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │         Capabilities Layer                        ★ Pluggable   │    │
│  │  SimpleSkill · Web Search · Shell Exec · A2A · Self-Evolver     │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │         Quality & Governance Layer                ★ Pluggable   │    │
│  │  Evaluators · Z3 Verification · Sandbox · Cost Control          │    │
│  │  OTEL Tracing · Error Policy · Conflict Resolution              │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │         Adapters Layer                            ★ Pluggable   │    │
│  │  OpenAI · Anthropic · Gemini · more...                          │    │
│  │  Memory Store · Embedding Provider · Persistence Backend        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Extension Points

HouYi is designed for community contribution and enterprise customization. Every `★ Pluggable` layer exposes well-defined extension points:

```
Extension Point                  Protocol / Base Class          Implementations
─────────────────────────────────────────────────────────────────────────────
LLM Adapter                      LLMAdapter                     OpenAI, Anthropic, Gemini, Ollama, vLLM
Memory Backend                   MemoryStore                    SQLite, Redis, QMD
Embedding Provider               EmbeddingProvider              FastEmbed, OpenAI, HuggingFace
Search Provider                  WebSearchService               Bocha, DuckDuckGo, Tavily, Serper
Skill / Tool                     @tool / SkillSpec              Any Python function → auto-schema
Context Source                   ContextSource                  RAG, Memory, MCP server, custom retriever
Evaluator                        Evaluator                      19+ built-in evaluators, extensible strategy Pattern 
Observability Exporter           OTEL SpanExporter              Jaeger, Zipkin, Datadog, Prometheus
Message Bus Backend              AgentMessageBus                In-process queue, NATS, Kafka, RocketMQ
Orchestration Mode               AgentOrchestrator              Delegate, Autonomous, DAG, custom
Error / Conflict Policy          ErrorPolicy / ConflictResolver Retry, fallback, source voting, LLM arbiter
Verification Backend             Z3 Solver                      SMT constraints, custom verifier
State / Persistence              StateStore                     SQLite, filesystem, Redis
```

## ✨ Key Features

| Category | Feature | Highlight |
|----------|---------|-----------|
| **Orchestration** | Lightweight Pydantic Core | Declarative agents, tasks, and workflows as Python classes with automatic validation and serialization — "code as configuration" |
| | Unified Multi-Agent Engine | Same `Agent` class, same SDK: tool-loop, `mode="delegate"` (supervisor dispatches sub-agents), `mode="autonomous"` (shared state + message bus). Scale from a single chatbot to a multi-agent research team without API fragmentation |
| | Async DAG Execution | Built on `asyncio` with DAG-based task orchestration — parallel execution, dynamic graph evolution, and non-blocking I/O for high-concurrency scenarios |
| **Context** | Context Engineering Pipeline | Dynamic token budgeting, RAG integration, persistent Memory with hybrid retrieval (full-text + embedding), emphasis-aware recall that prioritizes user-stressed instructions, and context compression with Reminders injection at the Transformer attention sweet spot |
| | SimpleSkill Specification | Cross-platform skill model with built-in governance, evaluation hooks, and host-portable capability negotiation. Any Python function becomes a governed, evaluable capability unit |
| **Quality** | Neuro-Symbolic Verification | Z3 SMT solver formally verifies LLM outputs against business constraints, separating probabilistic reasoning from deterministic correctness for production reliability |
| | Extensible Evaluator Framework | 19+ evaluators across 4 categories — **Quality** (accuracy, completeness, relevance, coherence, factuality), **Safety** (toxicity, bias, hallucination), **RAG** (groundedness, faithfulness, context precision/recall), **Performance** (cost, latency). Add custom evaluators via `Evaluator` base class |
| | Cost-Aware Governance | Token budget control with dynamic model routing enables automatic cost optimization while maintaining quality through intelligent provider fallback |
| **Infrastructure** | A2A Pub/Sub Protocol | Native Agent-to-Agent messaging (P2P, Pub/Sub, Broadcast) aligned with the [A2A Pub/Sub draft](https://github.com/a2aproject/A2A/issues/1029). Pluggable transport: in-process queues for dev, NATS/Kafka/RocketMQ for distributed production |
| | Zero-Config Observability | OpenTelemetry auto-instruments every agent execution with distributed tracing across LLM calls, tool invocations, and state transitions — <3% overhead, no manual setup |
| | Persistent State & Workflows | Automatic execution snapshots support pause/resume, external event handling, and human-in-the-loop workflows — agents wait for async callbacks and resume exactly where they left off |
| | Secure Sandbox Execution | Isolated execution environment with permission controls prevents LLM-generated code from accessing unauthorized resources, ensuring enterprise-grade security |

## 📦 Installation

```bash
git clone https://github.com/YiLabsAI/HouYiAgent.git
cd HouYiAgent
uv sync --extra dev
```

### Launch HouYi Studio

HouYi Studio is a full-featured web IDE with Chatbox, Agent Hub, Deep Research, and Memory Inbox. Start it locally with one command:

```bash
cp .env.example .env   # configure your LLM and search API keys
./scripts/dev.sh        # launches backend (FastAPI) + frontend (Vite) via tmux
```

Open `http://localhost:3000` to access the Studio.

## 🚀 Quick Start

### Simple Agent

```python
from houyi import Agent, tool
from houyi.llm import OpenAIAdapter

@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for {query}"]

agent = Agent(
    role="Researcher",
    skills=[search],
    llm=OpenAIAdapter(model="gpt-4o-mini"),
)

result = agent.run("What is HouYi?")
```

### Multi-Agent Team

```python
from houyi import Agent, Task, Team

researcher = Agent(role="Researcher", skills=[search], llm=llm)
analyst = Agent(role="Analyst", skills=[analyze], llm=llm)

team = Team(
    agents=[researcher, analyst],
    tasks=[
        Task("Research AI trends", agent=researcher),
        Task("Analyze findings", agent=analyst, context=[0]),
    ],
)
result = team.run()
```

### Sub-Agent Delegation (Supervisor Pattern)

```python
from houyi import Agent, AgentTeamConfig

supervisor = Agent(
    role="Research Supervisor",
    llm=llm,
    tools=[web_search],
    sub_agents=[
        AgentTeamConfig(role="Searcher", skills=["web_search"]),
        AgentTeamConfig(role="Analyst", skills=["code_execute"]),
    ],
    mode="delegate",
)

result = supervisor.run("Deep research on AI agent architectures")
```

### Memory — Persistent Context Across Sessions

```python
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore

store = MemoryStore(data_dir="./memory_data")
engine = MemoryEngine(store)

await engine.add("User prefers Python over JavaScript", tags=["preference"])
memories = await engine.recall("programming language preference?", top_k=5)
context = await engine.build_context("coding question", max_tokens=500)
```

### Context Engineering — Reminders Injection

```python
from houyi.application.context.reminders import ReminderInjector, CITATION_REMINDER

injector = ReminderInjector([CITATION_REMINDER])
messages = injector.inject(conversation_messages)
# Critical instructions injected at context tail — Transformer attention sweet spot
```

### Evaluation

```python
from houyi import evaluate

results = evaluate(
    agent=agent,
    test_cases=[{"input": "What is AI?", "expected_output": "..."}],
    evaluators=["accuracy", "completeness", "relevance"],
)
print(results.summary())
```

## 🤖 Built-in Agents

HouYi ships with production-ready agent applications built on top of the SDK:

| Agent | Description |
|-------|-------------|
| **Deep Research** | Automated research: plan decomposition → multi-round web search → source aggregation → intermediate analysis → conflict resolution → citation-verified report with RACE/FACT quality scoring |
| **Chatbox** | Multi-turn conversational AI with streaming, tool calling, memory integration, and full context engineering pipeline |
| **Memory Inbox** | LLM-powered memory extraction from conversations with human-in-the-loop review/approve/reject workflow |

Each is a production-grade application that exercises every layer of the SDK. Study their source as reference implementations for building your own agents.

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](./docs/getting-started.md) | Installation, quick start, core concepts |
| [API Reference](./docs/api-reference.md) | Complete API documentation |
| [Advanced Features](./docs/advanced-features.md) | Observability, multi-LLM, DAG execution, context engineering |
| [Evaluation](./docs/evaluation.md) | Evaluator framework and all built-in evaluators |
| [Development Guide](./agent.md) | Coding standards and engineering practices |
| [Examples](./examples/) | Runnable code examples |

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](./CONTRIBUTING.md).

```bash
make check          # lint + type check + unit tests
make test-e2e       # integration tests with real LLM
```

## 🏛 Standards & Acknowledgments

HouYi is built on and contributes to open standards:

| Standard | Role in HouYi |
|----------|---------------|
| **[OpenTelemetry](https://opentelemetry.io/)** | Zero-config distributed tracing across LLM calls, tools, and agent state transitions |
| **[SimpleSkill](./docs/simpleskill-spec.md)** | HouYi's native skill specification — cross-platform, governable, evaluable capability units (originated from this project) |
| **[MCP](https://modelcontextprotocol.io/)** | Model Context Protocol integration for external context sources |
| **[A2A](https://github.com/google/A2A)** | Agent-to-Agent protocol with native Pub/Sub messaging for distributed multi-agent communication |
