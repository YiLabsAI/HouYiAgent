# Getting Started with HouYi

This guide will help you get started with HouYi, a lightweight multi-agent framework.

## Installation

### Prerequisites

- Python 3.11, 3.12, or 3.13
- pip or conda

### Install from Source

```bash
# Clone the repository
git clone https://github.com/YiLabsAI/HouYiAgent.git
cd HouYiAgent

# Install in editable mode
pip install -e .
```

### Using Conda (Recommended for Development)

```bash
# Clone the repository first
git clone https://github.com/YiLabsAI/HouYiAgent.git
cd HouYiAgent

# Create and activate conda environment
conda create -n houyi python=3.11 -y
conda activate houyi

# Install in editable mode
pip install -e .
```

## Quick Start

### 1. Basic Agent

Create your first agent in just 2 lines:

```python
from houyi import Agent, tool

# Define a skill
@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for {query}"]

# Create an agent
agent = Agent(role="Researcher", skills=[search])

# Run it
result = agent.run("What is HouYi?")
print(result)
```

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
