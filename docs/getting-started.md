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

## Core Concepts

### Agent

An agent is a runtime instance with execution capabilities:

- **Role**: Defines the agent's purpose
- **Skills**: Functions the agent can execute
- **LLM**: Optional language model integration
- **Observability**: Built-in tracing (enabled by default)

### Skill

A skill is a function wrapped with Pydantic validation:

- Use `@tool` decorator for automatic schema inference
- Input/output validation
- Type safety with Pydantic

### Task

A task represents work to be done:

- **Description**: What needs to be done
- **Agent**: Which agent should do it
- **Context**: Dependencies on other tasks (for DAG execution)

### Team

A team coordinates multiple agents:

- **DAG Execution**: Parallel execution with dependency management
- **Context Passing**: Share results between tasks
- **Cycle Detection**: Prevents circular dependencies

## Next Steps

- [API Reference](./api-reference.md) - Complete API documentation
- [Advanced Features](./advanced-features.md) - Observability, multi-LLM, DAG execution
- [Evaluation Guide](./evaluation.md) - All 19 evaluators explained
- [Examples](../examples/) - More code examples

## Need Help?

- [GitHub Issues](https://github.com/YiLabsAI/HouYiAgent/issues)
- [Development Guide](../agent.md)
- [CHANGELOG](../CHANGELOG.md)
