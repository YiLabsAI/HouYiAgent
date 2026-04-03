# Getting Started with HouYi

This guide will help you get started with HouYi, a lightweight multi-agent framework.

## Installation

### Prerequisites

- Python 3.11, 3.12, or 3.13
- uv (recommended for development)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/YiLabsAI/HouYiAgent.git
cd HouYiAgent

# Install uv (see https://docs.astral.sh/uv/)

# Use Python 3.11 by default
uv python install 3.11

# Create/sync the virtualenv in .venv and install dev dependencies
uv sync --extra dev

# If you plan to run real OpenAI/Anthropic adapters
uv sync --extra dev --extra model-adapters

# If you plan to run the full local dev/test stack used by Studio chat + tool loop + RAG + Vertex tests
uv sync --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability
```

## Quick Start

### 1. Basic Agent (Without LLM)

For simple demos and testing, you can create agents without LLM:

```python
from houyi import Agent, tool

@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for {query}"]

# Create an agent (uses fallback mode without LLM)
agent = Agent(role="Researcher", skills=[search])

# Run it
result = agent.run("What is AI?")
print(result)
```

> **Note**: Without LLM, the agent uses simple heuristics to extract parameters from the task string. This works for basic scenarios but is limited for complex tasks.

### Agent with LLM (Recommended)

For production use, configure an LLM adapter:

```python
from houyi import Agent, tool
from houyi.llm import OpenAIAdapter

@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for {query}"]

# Create agent with OpenAI
agent = Agent(
    role="Researcher",
    skills=[search],
    llm=OpenAIAdapter(
        model="gpt-4",
        api_key="your-api-key"  # or set OPENAI_API_KEY env var
    )
)

result = agent.run("What is AI?")
```

**Supported LLM Adapters**:
- `OpenAIAdapter` - OpenAI models (GPT-4, GPT-3.5, etc.)
- `AnthropicAdapter` - Anthropic models (Claude 3.5 Sonnet, etc.)

### 2. Agent with LLM

Integrate with OpenAI or Anthropic:

```python
from houyi import Agent, tool
from houyi.llm import OpenAIAdapter

@tool
def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)

# Create agent with LLM
agent = Agent(
    role="Math Assistant",
    skills=[calculate],
    llm=OpenAIAdapter(model="gpt-4", api_key="your-api-key")
)

result = agent.run("What is 25 * 4 + 10?")
```

### 3. Multi-Agent Team

Coordinate multiple agents:

```python
from houyi import Agent, Task, Team, tool

# Define agents
researcher = Agent(role="Researcher", skills=[search])
analyst = Agent(role="Analyst", skills=[analyze])

# Create team with tasks
team = Team(
    agents=[researcher, analyst],
    tasks=[
        Task("Research AI trends", agent=researcher),
        Task("Analyze findings", agent=analyst, context=[0])  # Depends on task 0
    ]
)

result = team.run()
```

### 4. Evaluation

Evaluate your agent's performance:

```python
from houyi import Agent, evaluate

# Run evaluation
results = evaluate(
    agent=agent,
    test_cases=[
        {
            "input": "What is AI?",
            "expected_output": "AI is artificial intelligence."
        }
    ],
    evaluators=["accuracy", "completeness", "relevance"]
)

print(results.summary())
```

### 5. Studio Chat Tool Calling (API)

When using Studio server chat endpoints, enable tools per request:

```bash
curl -N -X POST "http://127.0.0.1:8000/api/chat/conversations/<conversation_id>/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Find TODO comments in this repo and summarize them",
    "enable_tool_calls": true,
    "tool_call_strategy": "balanced",
    "enable_skills": ["houyi_find_files", "houyi_grep", "houyi_read_file"],
    "enable_web_search": false,
    "max_tool_iterations": 6
  }'
```

`tool_call_strategy` behavior:

- `conservative`: requires explicit `enable_skills` or `enable_web_search`.
- `balanced` (default): explicit requests run, otherwise heuristics decide.
- `aggressive`: default-on unless `enable_tool_calls=false`.

If you hit OpenAI-compatible provider context-limit errors during tool loops,
adjust these env vars:

- `HOUYI_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS`
- `HOUYI_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS`
- `HOUYI_TOOLCALL_LOOP_MAX_MESSAGE_CHARS`
- `HOUYI_TOOLCALL_LOOP_MAX_TOTAL_CHARS`

### 6. Sub-Agent Delegation (Supervisor Pattern)

Coordinate sub-agents through a supervisor that decomposes tasks and merges results:

```python
from houyi import Agent
from houyi.domain.agent import AgentTeamConfig

supervisor = Agent(
    role="Research Supervisor",
    llm=OpenAIAdapter(model="gpt-4o-mini"),
    tools=[web_search_tool],
    sub_agents=[
        AgentTeamConfig(role="Searcher", skills=["web_search"]),
        AgentTeamConfig(role="Analyst", skills=["code_execute"]),
    ],
    mode="delegate",
)

result = await supervisor.arun("Deep research on AI agent architectures")
```

The supervisor LLM autonomously decides how to decompose the task, dispatches sub-questions to the appropriate sub-agents, and merges their results.

### 7. Memory — Persistent Context Across Sessions

```python
from houyi.adapters.memory.engine import MemoryEngine
from houyi.adapters.memory.store import MemoryStore

store = MemoryStore(data_dir="./memory_data")
engine = MemoryEngine(store)

await engine.add("User prefers Python over JavaScript", tags=["preference"])
memories = await engine.recall("What language does the user prefer?", top_k=5)
context = await engine.build_context("programming question", max_tokens=500)
```

## Core Concepts

### Agent

The `Agent` class is the universal entry point with three execution paths:

- **Tool-loop** — agent has tools, no sub_agents: iterative LLM → tool-call → result loop
- **Orchestrated** — agent has `sub_agents` or `mode` set: `AgentOrchestrator` handles delegate / autonomous collaboration
- **DAG** — fallback graph-based execution via planner + executor

### Skill

A skill is a function wrapped with Pydantic validation:

- Use `@tool` decorator for automatic schema inference
- Input/output validation and type safety

### AgentTeamConfig

Declarative sub-agent definition for supervisor patterns:

- **role**: Sub-agent's specialization
- **skills**: Capabilities the sub-agent can use
- Supervisor LLM autonomously decides dispatch and merge

### Team & Task

Multi-agent DAG orchestration:

- **DAG Execution**: Parallel execution with dependency management
- **Context Passing**: Share results between tasks
- **Cycle Detection**: Prevents circular dependencies

### Memory

Persistent, queryable context store:

- **SQLite + FTS5**: Full-text search with embedding-based hybrid retrieval
- **LLM-powered extraction**: Automatic memory extraction from conversations
- **Emphasis-aware recall**: Prioritizes memories the user has emphasized

## Next Steps

- [API Reference](./api-reference.md) - Complete API documentation
- [Advanced Features](./advanced-features.md) - Observability, multi-LLM, DAG execution
- [Evaluation Guide](./evaluation.md) - All 19 evaluators explained
- [Examples](../examples/) - More code examples

## Need Help?

- [GitHub Issues](https://github.com/YiLabsAI/HouYiAgent/issues)
- [Development Guide](../agent.md)
- [CHANGELOG](../CHANGELOG.md)
