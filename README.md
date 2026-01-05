<div align="center">
  <img src=".github/images/logo-text.svg" alt="HouYi Logo" width="400">

  <h3>Next-generation lightweight multi-agent framework</h3>

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

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

## Overview

HouYi is a next-generation, lightweight multi-agent programming framework designed for production use. Built on Pydantic v2 with zero-config observability and comprehensive evaluation capabilities.

**Why HouYi?**
- ✅ **2-line agent creation** - Minimal boilerplate
- ✅ **Built-in tracing** - Zero-config observability with <3% overhead
- ✅ **19 evaluators** - Quality, safety, performance metrics out of the box
- ✅ **Type-safe** - Full Pydantic v2 validation
- ✅ **Production-ready** - DAG execution engine with parallel task support

## ✨ Key Features

- 🪶 **Lightweight Core** - Pydantic-based declarative definitions
- 🔍 **Zero-Config Observability** - Built-in OpenTelemetry tracing (<3% overhead)
- 📊 **Comprehensive Evaluation** - 19 built-in evaluators (quality, safety, performance)
- ⚡ **DAG Execution** - Parallel task execution with dependency management
- 🤖 **Multi-LLM Support** - Unified interface for OpenAI, Anthropic, and more
- 🔒 **Type Safety** - Full Pydantic v2 validation and serialization
- 🎯 **Production-Ready** - Battle-tested execution engine

> **Note**: Advanced features like bidirectional visual sync, distributed execution, and neuro-symbolic verification are planned for future releases. See [roadmap](./docs/roadmap.md) for details.

## 📦 Installation

### Prerequisites

- Python 3.11, 3.12, or 3.13
- pip or conda

### Quick Install

```bash
# From PyPI (coming soon)
pip install houyi
```

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

# Install dev dependencies for testing
pip install pytest pytest-cov ruff mypy
```

## 🚀 Quick Start

```python
from houyi import Agent, tool

# 1. Define a skill
@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for {query}"]

# 2. Create an agent
agent = Agent(role="Researcher", skills=[search])

# 3. Run it
result = agent.run("What is HouYi?")
# Output: ✅ agent.run (13.16ms)
```

**See [Getting Started Guide](./docs/getting-started.md) for complete tutorial.**

## 💡 Examples

Explore the [`examples/`](./examples/) directory:

```bash
python examples/quickstart.py          # Basic agent
python examples/llm_example.py         # LLM integration
python examples/team_example.py        # Multi-agent team
python examples/evaluation_example.py  # All 19 evaluators
```

## 📚 Documentation

**User Guides:**
- [Getting Started](./docs/getting-started.md) - Installation, quick start, core concepts
- [API Reference](./docs/api-reference.md) - Complete API documentation
- [Advanced Features](./docs/advanced-features.md) - Observability, multi-LLM, DAG execution
- [Evaluation](./docs/evaluation.md) - All 19 evaluators explained

**Development:**
- [CONTRIBUTING.md](./CONTRIBUTING.md) - How to contribute
- [agent.md](./agent.md) - Development guide and coding standards
- [CHANGELOG.md](./CHANGELOG.md) - Version history

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## 🙏 Acknowledgments

- Built on industry standards:
  - **OpenTelemetry (OTEL)** for observability and distributed tracing
  - **AgentSkills.io** for skill interoperability and sharing
  - **MCP (Model Context Protocol)** for context management
  - **A2A (Agent-to-Agent)** protocol for multi-agent communication

---

**Built with ❤️ for the AI agent community**
