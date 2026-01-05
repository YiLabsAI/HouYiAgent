# Evaluation Guide

Complete guide to HouYi's 19 built-in evaluators.

## Overview

HouYi provides comprehensive evaluation capabilities across multiple dimensions:

- **Quality**: Accuracy, completeness, relevance
- **Performance**: Cost, latency
- **Safety**: Toxicity, hallucination detection
- **RAG**: Grounding, context precision/recall
- **Structure**: Coherence, skill usage
- **Advanced**: LLM-based judging, custom evaluators

## Quick Start

```python
from houyi import Agent, evaluate

# Create agent
agent = Agent(role="Assistant", skills=[...])

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

## Quality Evaluators

### Accuracy

Measures exact match between output and expected output.

```python
results = evaluate(agent, test_cases, evaluators=["accuracy"])
```

**Score**: 1.0 if exact match, 0.0 otherwise

### Completeness

Evaluates if output covers all required information.

```python
results = evaluate(agent, test_cases, evaluators=["completeness"])
```

**Score**: 0.0 to 1.0 based on coverage

### Relevance

Measures how relevant the output is to the input.

```python
results = evaluate(agent, test_cases, evaluators=["relevance"])
```

**Score**: 0.0 to 1.0 based on relevance

### Semantic Similarity

Compares semantic similarity using embeddings.

```python
results = evaluate(agent, test_cases, evaluators=["semantic_similarity"])
```

**Score**: Cosine similarity (0.0 to 1.0)

## Performance Evaluators

### Cost

Tracks execution cost (LLM API calls, etc.).

```python
results = evaluate(agent, test_cases, evaluators=["cost"])
```

**Metrics**:
- Total cost in USD
- Cost per test case
- Cost breakdown by operation

### Latency

Measures execution time.

```python
results = evaluate(agent, test_cases, evaluators=["latency"])
```

**Metrics**:
- Duration in milliseconds
- Average latency
- P95, P99 latency

## Safety Evaluators

### Toxicity

Detects toxic or harmful content.

```python
results = evaluate(agent, test_cases, evaluators=["toxicity"])
```

**Score**: 0.0 (safe) to 1.0 (toxic)

### Hallucination

Detects hallucinated or fabricated information.

```python
results = evaluate(agent, test_cases, evaluators=["hallucination"])
```

**Score**: 0.0 (no hallucination) to 1.0 (hallucinated)

### Safety

General safety check for harmful outputs.

```python
results = evaluate(agent, test_cases, evaluators=["safety"])
```

**Score**: 1.0 (safe) to 0.0 (unsafe)

## RAG Evaluators

### Groundedness

Checks if output is grounded in provided context.

```python
results = evaluate(agent, test_cases, evaluators=["groundedness"])
```

**Score**: 0.0 to 1.0 based on grounding

### Context Precision

Measures precision of retrieved context.

```python
results = evaluate(agent, test_cases, evaluators=["context_precision"])
```

**Score**: Precision score (0.0 to 1.0)

### Context Recall

Measures recall of retrieved context.

```python
results = evaluate(agent, test_cases, evaluators=["context_recall"])
```

**Score**: Recall score (0.0 to 1.0)

### Faithfulness

Checks if output is faithful to source material.

```python
results = evaluate(agent, test_cases, evaluators=["faithfulness"])
```

**Score**: 0.0 to 1.0 based on faithfulness

## Structure Evaluators

### Coherence

Evaluates logical flow and coherence.

```python
results = evaluate(agent, test_cases, evaluators=["coherence"])
```

**Score**: 0.0 to 1.0 based on coherence

### Skill Usage

Checks if correct skills were used.

```python
test_cases = [
    {
        "input": "Search for AI",
        "expected_output": "...",
        "expected_skills": ["search"]
    }
]

results = evaluate(agent, test_cases, evaluators=["skill_usage"])
```

**Score**: 1.0 if correct skills used, 0.0 otherwise

### Bias

Detects biased content.

```python
results = evaluate(agent, test_cases, evaluators=["bias"])
```

**Score**: 0.0 (unbiased) to 1.0 (biased)

### Factuality

Checks factual accuracy.

```python
results = evaluate(agent, test_cases, evaluators=["factuality"])
```

**Score**: 0.0 to 1.0 based on factual accuracy

## Advanced Evaluators

### LLM Judge

Uses an LLM to evaluate outputs.

```python
from houyi.evaluation import LLMJudgeEvaluator

evaluator = LLMJudgeEvaluator(
    criteria="Is the response helpful and accurate?",
    llm=OpenAIAdapter(model="gpt-4")
)

results = evaluate(agent, test_cases, evaluators=[evaluator])
```

### Custom Evaluator

Create your own evaluator:

```python
from houyi import Evaluator, EvaluationResult

class LengthEvaluator(Evaluator):
    def __init__(self, min_length: int = 10, max_length: int = 1000):
        self.min_length = min_length
        self.max_length = max_length

    @property
    def name(self) -> str:
        return "length"

    def evaluate(self, input, output, expected, metadata):
        length = len(output)
        passed = self.min_length <= length <= self.max_length
        score = 1.0 if passed else 0.0

        return EvaluationResult(
            evaluator=self.name,
            input=input,
            output=output,
            expected_output=expected,
            score=score,
            passed=passed,
            metrics={"length": length},
            feedback=f"Length: {length} (min: {self.min_length}, max: {self.max_length})"
        )

# Use it
results = evaluate(agent, test_cases, evaluators=[LengthEvaluator()])
```

## Batch Evaluation

Evaluate multiple test cases efficiently:

```python
from houyi.evaluation import Dataset

# Load dataset
dataset = Dataset.from_file("tests/dataset.json")

# Run evaluation
results = evaluate(
    agent=agent,
    dataset=dataset,
    evaluators=["accuracy", "latency", "cost"]
)

# Save report
results.save_report("evaluation_report.html")
```

## Best Practices

### 1. Choose Relevant Evaluators

Select evaluators based on your use case:

```python
# For chatbots
evaluators = ["accuracy", "relevance", "toxicity", "coherence"]

# For RAG systems
evaluators = ["groundedness", "faithfulness", "context_precision"]

# For production monitoring
evaluators = ["latency", "cost", "safety"]
```

### 2. Create Representative Test Cases

```python
test_cases = [
    # Happy path
    {"input": "Normal query", "expected_output": "..."},

    # Edge cases
    {"input": "Empty query", "expected_output": "..."},
    {"input": "Very long query...", "expected_output": "..."},

    # Error cases
    {"input": "Invalid input", "expected_output": "error message"}
]
```

### 3. Set Thresholds

```python
results = evaluate(agent, test_cases, evaluators=["accuracy"])

# Check if meets threshold
if results.avg_score < 0.8:
    print("⚠️  Agent accuracy below threshold!")
```

### 4. Track Over Time

```python
import json
from datetime import datetime

# Save results
results_data = {
    "timestamp": datetime.now().isoformat(),
    "avg_score": results.avg_score,
    "passed_count": results.passed_count,
    "total_cases": results.total_cases
}

with open("evaluation_history.jsonl", "a") as f:
    f.write(json.dumps(results_data) + "\n")
```

## See Also

- [Getting Started Guide](./getting-started.md)
- [API Reference](./api-reference.md)
- [Advanced Features](./advanced-features.md)
- [Examples](../examples/evaluation_example.py)
