# API Reference

Complete API documentation for HouYi framework.

## Core Classes

### Agent

```python
from houyi import Agent

agent = Agent(
    role: str,                    # Agent's role/purpose
    skills: list[SkillSpec] = [], # List of skills
    llm: LLMAdapter = None,       # Optional LLM adapter
    system_prompt: str = None,    # Custom system prompt
    observability: dict = None,   # Observability config
    memory: bool = False,         # Enable memory
    policies: dict = {}           # Custom policies
)
```

**Methods:**
- `run(task: str | Task) -> Any` - Execute a task
- `to_system_prompt() -> str` - Generate system prompt

### Task

```python
from houyi import Task

task = Task(
    description: str,              # Task description
    agent: Agent = None,           # Assigned agent
    expected_output: str = None,   # Expected output
    context: list[int] = None      # Task dependencies (indices)
)
```

### Team

```python
from houyi import Team

team = Team(
    agents: list[Agent],           # List of agents
    tasks: list[Task],             # List of tasks
    observability: dict = None     # Observability config
)
```

**Methods:**
- `run() -> dict` - Execute all tasks
- `execute() -> dict` - Alias for run()

### SkillSpec

```python
from houyi import SkillSpec

skill = SkillSpec(
    name: str,                     # Skill name
    description: str,              # Skill description
    input_schema: Type[BaseModel], # Input Pydantic model
    output_schema: Type[BaseModel],# Output Pydantic model
    executor: Callable,            # Execution function
    constraints: dict = {}         # Optional constraints
)
```

## Decorators

### @tool

Decorator for creating skills from functions:

- Public entrypoint: `from houyi import tool`
- Implementation location: `houyi/decorators.py`
- Responsibility: convert a typed Python function into a `SkillSpec` by deriving input/output schemas from type hints and binding the original function as the executor

```python
from houyi import tool

@tool
def search(query: str, max_results: int = 10) -> list[dict]:
    """Search the web for information."""
    return [...]
```

Automatically infers input/output schemas from type hints.

## LLM Adapters

### OpenAIAdapter

```python
from houyi.llm import OpenAIAdapter

llm = OpenAIAdapter(
    model: str = "gpt-4",          # Model name
    api_key: str = None,           # API key (or from env)
    temperature: float = 0.7,      # Temperature
    max_tokens: int = None,        # Max tokens
    timeout: int = 30              # Request timeout
)
```

### AnthropicAdapter

```python
from houyi.llm import AnthropicAdapter

llm = AnthropicAdapter(
    model: str = "claude-3-5-sonnet",
    api_key: str = None,
    temperature: float = 0.7,
    max_tokens: int = None,
    timeout: int = 30
)
```

### Retry and Streaming Semantics (HTTP-based adapters)

HouYi applies a unified retry controller to HTTP-based LLM adapter paths.

**Retry classes:**
- `connect` (connection establishment failures)
- `read` (read/timeout/protocol interruptions)
- `status` (retryable HTTP statuses)
- `other` (other transient errors)

**Budget model:**
- Global `total` retry budget
- Per-class budgets (`connect`, `read`, `status`, `other`)

**Retryable status behavior:**
- For `429` / `503` (and other configured retryable statuses), HouYi checks `Retry-After` first.
- If `Retry-After` is missing or invalid, HouYi uses local exponential backoff with jitter.

**Streaming safety rule:**
- Retry is allowed only before the first output chunk is emitted.
- After any chunk has been emitted, automatic retry is disabled for that request.

### Chat Tool-Calling (Studio Server)

`POST /api/chat/conversations/{conversation_id}/messages` supports tool-calling controls:

- `enable_tool_calls: bool | null` — when `false`, hard-disables the tool loop.
- `tool_call_strategy: "conservative" | "balanced" | "aggressive" | null` — chat gating policy.
- `enable_skills: list[str] | null` — allow additional skill names for this request.
- `enable_web_search: bool | null` — include `web_search` when `true`.
- `max_tool_iterations: int | null` — cap tool-loop rounds (`1..50`).

Gating semantics:

- `conservative`: tools run only for explicit `enable_skills` / `enable_web_search` requests.
- `balanced` (default): explicit requests always run; otherwise tool-intent heuristics decide.
- `aggressive`: tools are default-on unless `enable_tool_calls=false`.

Runtime notes:

- Chat mode includes a built-in tool allowlist; `enable_skills` appends extra skills.
- Tool-loop execution uses parallel tool calls internally.

Tool loop payload protection (for OpenAI-compatible providers with strict context windows):

- `HOUYI_CHAT_TOOL_LOOP_MAX_MESSAGE_CHARS` (default `12000`)
- `HOUYI_CHAT_TOOL_LOOP_MAX_TOTAL_CHARS` (default `160000`)
- `HOUYI_TOOLCALL_LOOP_MAX_MESSAGE_CHARS` (default `12000`)
- `HOUYI_TOOLCALL_LOOP_MAX_TOTAL_CHARS` (default `160000`)

The server sanitizes message content and tool-call arguments to strings, truncates oversized items, and drops oldest non-system turns when budget is exceeded.

## Evaluation

### evaluate()

```python
from houyi import evaluate

results = evaluate(
    agent: Agent,                  # Agent to evaluate
    test_cases: list[dict],        # Test cases
    evaluators: list[str | Evaluator] = None,  # Evaluators
    dataset: Dataset = None        # Optional dataset
)
```

**Test Case Format:**
```python
{
    "input": "Question or task",
    "expected_output": "Expected answer",
    "expected_skills": ["skill1", "skill2"]  # Optional
}
```

### Built-in Evaluators

**Quality:**
- `accuracy` - Exact match accuracy
- `completeness` - Output completeness
- `relevance` - Output relevance
- `semantic_similarity` - Semantic similarity

**Performance:**
- `cost` - Execution cost
- `latency` - Execution latency

**Safety:**
- `toxicity` - Toxicity detection
- `hallucination` - Hallucination detection
- `safety` - Safety check

**RAG:**
- `groundedness` - Grounding in context
- `context_precision` - Context precision
- `context_recall` - Context recall
- `faithfulness` - Faithfulness to source

**Structure:**
- `coherence` - Output coherence
- `skill_usage` - Skill usage correctness
- `bias` - Bias detection
- `factuality` - Factual accuracy

**Advanced:**
- `llm_judge` - LLM-based evaluation
- Custom evaluators

### Custom Evaluator

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
            feedback=f"Score: {score:.2%}"
        )
```

## Observability

### TraceManager

```python
from houyi.observability import TraceManager

tm = TraceManager(
    enabled: bool = True,          # Enable tracing
    exporters: list[Exporter] = [] # List of exporters
)
```

**Methods:**
- `start_span(name: str) -> Span` - Start a span

### Exporters

**ConsoleExporter:**
```python
from houyi.observability import ConsoleExporter

exporter = ConsoleExporter(verbose: bool = False)
```

**JSONExporter:**
```python
from houyi.observability import JSONExporter

exporter = JSONExporter(filepath: str)
```

**JaegerExporter:**
```python
from houyi.observability import JaegerExporter

exporter = JaegerExporter(
    endpoint: str = "http://localhost:4318",
    service_name: str = "houyi-agent"
)
```

## Data Models

### LLMMessage

```python
from houyi.llm import LLMMessage, MessageRole

message = LLMMessage(
    role: MessageRole,             # USER, ASSISTANT, SYSTEM
    content: str,                  # Message content
    name: str = None,              # Optional name
    tool_calls: list = None        # Optional tool calls
)
```

### EvaluationResult

```python
from houyi import EvaluationResult

result = EvaluationResult(
    evaluator: str,                # Evaluator name
    input: str,                    # Input text
    output: str,                   # Output text
    expected_output: str,          # Expected output
    score: float,                  # Score (0-1)
    passed: bool,                  # Pass/fail
    metrics: dict = {},            # Additional metrics
    feedback: str = "",            # Feedback text
    duration_ms: float = 0.0,      # Execution duration
    cost: float = 0.0              # Execution cost
)
```

## See Also

- [Getting Started Guide](./getting-started.md)
- [Advanced Features](./advanced-features.md)
- [Evaluation Guide](./evaluation.md)
