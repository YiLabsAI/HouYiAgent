# Advanced Features

Explore HouYi's advanced capabilities for production use.

## Observability

HouYi includes built-in tracing with **< 3% performance overhead**.

### Zero-Config Tracing

By default, agents have observability enabled:

```python
from houyi import Agent, tool

@tool
def search(query: str) -> list[str]:
    return ["result"]

agent = Agent(role="Researcher", skills=[search])
result = agent.run("Query")
# Output: ✅ agent.run (13.16ms)
```

### Verbose Mode

Enable detailed traces:

```python
agent = Agent(
    role="Researcher",
    skills=[search],
    observability={
        "enabled": True,
        "exporters": [
            {"type": "console", "verbose": True}
        ]
    }
)
```

### Multiple Exporters

Export traces to multiple destinations:

```python
agent = Agent(
    role="Researcher",
    skills=[search],
    observability={
        "enabled": True,
        "exporters": [
            {"type": "console", "verbose": False},
            {"type": "json", "filepath": "traces.json"},
            {"type": "jaeger", "endpoint": "http://localhost:4318"}
        ]
    }
)
```

**Supported Exporters:**
- ✅ **ConsoleExporter**: Simple/verbose console output
- ✅ **JSONExporter**: Export to JSON file
- ✅ **JaegerExporter**: Export to Jaeger (OpenTelemetry)
- ⚠️ **DatadogExporter**: Placeholder (coming soon)

### Custom Spans

Create custom spans for fine-grained tracing:

```python
from houyi.observability import TraceManager

tm = TraceManager()

with tm.start_span("custom_operation") as span:
    span.set_attribute("user_id", "123")
    span.add_event("checkpoint", {"step": 1})
    # Your code here
```

## Multi-LLM Support

Unified interface for multiple LLM providers.

### OpenAI

```python
from houyi import Agent
from houyi.llm import OpenAIAdapter

agent = Agent(
    role="Assistant",
    skills=[...],
    llm=OpenAIAdapter(
        model="gpt-4",
        api_key="sk-...",
        temperature=0.7,
        max_tokens=1000
    )
)
```

### Anthropic

```python
from houyi.llm import AnthropicAdapter

agent = Agent(
    role="Assistant",
    skills=[...],
    llm=AnthropicAdapter(
        model="claude-3-5-sonnet",
        api_key="sk-ant-...",
        temperature=0.7
    )
)
```

### Features

- **Streaming Support**: Real-time response streaming
- **Function Calling**: Tool use with LLMs
- **Automatic Retry**: Exponential backoff on failures
- **Pydantic Validation**: Type-safe inputs/outputs

## Connection Resilience and Fault Tolerance

HouYi uses a unified, provider-agnostic retry model designed for production stability.
The same retry semantics are applied across HTTP-based LLM adapters, including Vertex AI,
SiliconFlow, and OpenAI integration paths.

### Why this is production-grade

- **Unified policy model**: One retry policy abstraction for all adapters (total budget + error class budgets).
- **Failure-class aware retries**: Distinguishes connection, read, status, and other failures.
- **Server-directed pacing**: Honors `Retry-After` for rate-limit and service-unavailable responses.
- **Jittered exponential backoff**: Uses full-jitter to reduce synchronized retry storms.
- **Streaming-safe retry semantics**: Retries only before the first streamed token/chunk is emitted.

### Retry decision model

The retry controller evaluates both global and per-class budgets:

- `total` retry budget
- `connect` retry budget
- `read` retry budget
- `status` retry budget
- `other` retry budget

This allows strict control over reliability behavior under different failure modes.

### `Retry-After` precedence

For retryable HTTP statuses such as `429` and `503`:

1. If `Retry-After` is present and valid, HouYi uses that delay.
2. Otherwise, HouYi falls back to local exponential backoff with jitter.

### Streaming behavior: pre-first-chunk retry only

For streaming responses, HouYi retries only if the request fails **before** any output
chunk is emitted. Once at least one chunk has been delivered to the caller, automatic
retry is disabled for that request to avoid duplicate output or broken stream ordering.

This provides a practical reliability/consistency trade-off for real-world streaming APIs.

## DAG Execution Engine

Parallel execution with dependency management.

### Basic DAG

```python
from houyi import Agent, Task, Team

team = Team(
    agents=[agent1, agent2, agent3],
    tasks=[
        Task("Task 1", agent=agent1),           # Runs first
        Task("Task 2", agent=agent2),           # Runs in parallel with Task 1
        Task("Task 3", agent=agent3, context=[0, 1])  # Runs after Task 1 & 2
    ]
)

result = team.run()
```

### Features

- **Topological Sorting**: Automatic task ordering
- **Concurrent Execution**: Parallel execution with asyncio
- **Dependency Tracking**: Context passing between tasks
- **Cycle Detection**: Prevents circular dependencies

### Complex DAG

```python
team = Team(
    agents=[researcher, analyst, writer],
    tasks=[
        Task("Research topic A", agent=researcher),      # Task 0
        Task("Research topic B", agent=researcher),      # Task 1
        Task("Analyze A", agent=analyst, context=[0]),   # Task 2 (depends on 0)
        Task("Analyze B", agent=analyst, context=[1]),   # Task 3 (depends on 1)
        Task("Write report", agent=writer, context=[2, 3])  # Task 4 (depends on 2, 3)
    ]
)
```

**Execution Flow:**
1. Tasks 0 and 1 run in parallel
2. Tasks 2 and 3 run in parallel (after 0 and 1 complete)
3. Task 4 runs after 2 and 3 complete

## Comprehensive Evaluation

19 built-in evaluators across multiple dimensions.

### Quality Evaluators

```python
from houyi import evaluate

results = evaluate(
    agent, test_cases,
    evaluators=["accuracy", "completeness", "relevance", "semantic_similarity"]
)
```

### Safety Evaluators

```python
results = evaluate(
    agent, test_cases,
    evaluators=["toxicity", "hallucination", "safety"]
)
```

### Performance Evaluators

```python
results = evaluate(
    agent, test_cases,
    evaluators=["cost", "latency"]
)
```

### RAG Evaluators

```python
results = evaluate(
    agent, test_cases,
    evaluators=["groundedness", "context_precision", "context_recall", "faithfulness"]
)
```

### All 19 Evaluators

- **Quality**: Accuracy, Completeness, Relevance, SemanticSimilarity
- **Performance**: Cost, Latency
- **Safety**: Toxicity, Hallucination, Safety
- **Bias & Facts**: Bias, Factuality
- **RAG**: Groundedness, ContextPrecision, ContextRecall, Faithfulness
- **Structure**: Coherence, SkillUsage
- **Advanced**: LLMJudge, Custom

## Custom Evaluators

Extend with your own evaluation logic:

```python
from houyi import Evaluator, EvaluationResult

class CustomEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "custom"

    def evaluate(self, input, output, expected, metadata):
        score = self.calculate_score(output, expected)
        return EvaluationResult(
            evaluator=self.name,
            input=input,
            output=output,
            expected_output=expected,
            score=score,
            passed=score > 0.8,
            feedback=f"Custom score: {score:.2%}"
        )

# Use it
results = evaluate(agent, test_cases, evaluators=[CustomEvaluator()])
```

## Best Practices

### 1. Use Type Hints

Always use type hints for automatic schema inference:

```python
@tool
def search(query: str, max_results: int = 10) -> list[dict]:
    """Search with type hints."""
    return [...]
```

### 2. Enable Observability

Keep observability enabled in production:

```python
agent = Agent(
    role="Production Agent",
    skills=[...],
    observability={"enabled": True}  # Default
)
```

### 3. Evaluate Regularly

Run evaluations on representative test cases:

```python
results = evaluate(
    agent,
    test_cases=load_test_cases(),
    evaluators=["accuracy", "latency", "cost"]
)
```

### 4. Handle Errors Gracefully

Use try-except in skills:

```python
@tool
def api_call(endpoint: str) -> dict:
    """Call external API."""
    try:
        response = requests.get(endpoint)
        return response.json()
    except Exception as e:
        return {"error": str(e)}
```

## See Also

- [Getting Started Guide](./getting-started.md)
- [API Reference](./api-reference.md)
- [Evaluation Guide](./evaluation.md)
- [Examples](../examples/)
