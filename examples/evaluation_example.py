"""Example: Using HouYi's 19 built-in evaluators.

This example demonstrates all available evaluators:
- Quality: accuracy, completeness, relevance, semantic_similarity
- Performance: cost, latency
- Safety: toxicity, hallucination, safety
- Bias & Facts: bias, factuality
- RAG: groundedness, context_precision, context_recall, faithfulness
- Structure: coherence, skill_usage
- Advanced: llm_judge, custom
"""

from houyi import Agent, evaluate, tool


# Define a skill
@tool
def search(query: str) -> list[str]:
    """Search for information."""
    return [f"Information about {query}", "Additional context"]


# Create agent
agent = Agent(role="Research Assistant", skills=[search])

# Test cases
test_cases = [
    {
        "input": "What is artificial intelligence? How does it work?",
        "expected_output": "Artificial intelligence is a field of computer science that focuses on creating intelligent machines. It works through machine learning algorithms.",
    },
    {
        "input": "Explain machine learning in simple terms.",
        "expected_output": "Machine learning is a way for computers to learn from data without being explicitly programmed.",
    },
    {
        "input": "What are the benefits of AI?",
        "expected_output": "AI can automate tasks, improve decision-making, and solve complex problems.",
    },
]

print("=" * 70)
print("HouYi Evaluation System - 19 Built-in Evaluators")
print("=" * 70)
print()

# Example 1: Core evaluators (Phase 1)
print("=== Example 1: Core Evaluators (Phase 1) ===\n")

results = evaluate(
    agent=agent, test_cases=test_cases, evaluators=["accuracy", "cost", "latency", "skill_usage"]
)

print(results.summary())
print()

# Example 2: Quality evaluators (Phase 2)
print("=== Example 2: Quality Evaluators (Phase 2) ===\n")

results = evaluate(
    agent=agent,
    test_cases=test_cases,
    evaluators=["completeness", "relevance", "semantic_similarity"],
)

print(results.summary())
print()

# Example 3: Safety evaluators (Phase 2)
print("=== Example 3: Safety Evaluators (Phase 2) ===\n")

results = evaluate(agent=agent, test_cases=test_cases, evaluators=["toxicity", "hallucination"])

print(results.summary())
print()

# Example 4: LLM Judge (Phase 2)
print("=== Example 4: LLM Judge Evaluator ===\n")

results = evaluate(agent=agent, test_cases=test_cases, evaluators=["llm_judge"])

print(results.summary())
print()

# Example 5: Multiple evaluators combined
print("=== Example 5: Multiple Evaluators Combined ===\n")

results = evaluate(
    agent=agent,
    test_cases=test_cases,
    evaluators=[
        # Quality
        "accuracy",
        "completeness",
        "relevance",
        "semantic_similarity",
        # Performance
        "cost",
        "latency",
        # Safety
        "toxicity",
        "hallucination",
        # Structure
        "skill_usage",
        # Advanced
        "llm_judge",
    ],
)

print(results.summary())
print()

# Example 6: Custom evaluator selection
print("=== Example 6: Custom Evaluator Selection ===\n")

# Select evaluators based on use case
quality_evaluators = ["accuracy", "completeness", "relevance", "semantic_similarity"]
safety_evaluators = ["toxicity", "hallucination"]

print("Quality Evaluation:")
results = evaluate(agent, test_cases, quality_evaluators)
print(f"  Avg Score: {results.avg_score:.2%}")
print(f"  Pass Rate: {results.passed_cases}/{results.total_cases} ({results.pass_rate:.1%})")
print()

print("Safety Evaluation:")
results = evaluate(agent, test_cases, safety_evaluators)
print(f"  Avg Score: {results.avg_score:.2%}")
print(f"  Pass Rate: {results.passed_cases}/{results.total_cases} ({results.pass_rate:.1%})")
print()

# Example 7: Detailed results
print("=== Example 7: Detailed Results ===\n")

results = evaluate(
    agent=agent,
    test_cases=[test_cases[0]],  # Single test case
    evaluators=["completeness", "relevance", "hallucination"],
)

for result in results.results:
    print(f"Evaluator: {result.evaluator}")
    print(f"  Score: {result.score:.2%}")
    print(f"  Passed: {result.passed}")
    print(f"  Feedback: {result.feedback}")
    if result.metrics:
        print(f"  Metrics: {result.metrics}")
    print()

print("=" * 70)
print("✅ All evaluation examples completed!")
print("=" * 70)
