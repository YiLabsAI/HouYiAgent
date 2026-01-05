"""Example demonstrating Dataset and Report generation."""

from houyi import Agent, tool
from houyi.evaluation import Dataset, evaluate, ReportGenerator

print("=" * 70)
print("Phase 3.3: Dataset & Report Generation Example")
print("=" * 70)
print()

# Create a simple agent
@tool
def answer_question(question: str) -> str:
    """Answer a question."""
    # Simple mock responses
    responses = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "machine learning": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "houyi": "HouYi is a lightweight multi-agent framework with 19 built-in evaluators and AgentSkills.io support.",
    }
    
    question_lower = question.lower()
    for key, response in responses.items():
        if key in question_lower:
            return response
    
    return "I don't have information about that topic."

agent = Agent(role="Q&A Assistant", skills=[answer_question])

# Example 1: Load dataset from JSON
print("Example 1: Load dataset from JSON")
print("-" * 70)

dataset = Dataset.from_file("datasets/example_dataset.json")
print(f"Loaded dataset: {dataset.name}")
print(f"Description: {dataset.description}")
print(f"Test cases: {len(dataset)}")
print()

# Example 2: Run evaluation with dataset
print("Example 2: Run evaluation with dataset")
print("-" * 70)

results = evaluate(
    agent=agent,
    dataset=dataset,
    evaluators=["accuracy", "completeness", "coherence", "relevance"]
)

print(f"Evaluation completed!")
print(f"Total cases: {results.total_cases}")
print(f"Passed: {results.passed_cases} ({results.pass_rate:.1%})")
print(f"Avg score: {results.avg_score:.2f}")
print()

# Example 3: Generate HTML report
print("Example 3: Generate HTML report")
print("-" * 70)

ReportGenerator.generate_html(
    results,
    "reports/evaluation_report.html",
    title="HouYi Agent Evaluation Report"
)
print("✅ HTML report saved to: reports/evaluation_report.html")
print()

# Example 4: Generate JSON report
print("Example 4: Generate JSON report")
print("-" * 70)

ReportGenerator.generate_json(
    results,
    "reports/evaluation_report.json"
)
print("✅ JSON report saved to: reports/evaluation_report.json")
print()

# Example 5: Generate Markdown report
print("Example 5: Generate Markdown report")
print("-" * 70)

ReportGenerator.generate_markdown(
    results,
    "reports/evaluation_report.md",
    title="HouYi Agent Evaluation Report"
)
print("✅ Markdown report saved to: reports/evaluation_report.md")
print()

# Example 6: Load dataset from CSV
print("Example 6: Load dataset from CSV")
print("-" * 70)

csv_dataset = Dataset.from_file("datasets/example_dataset.csv")
print(f"Loaded CSV dataset: {len(csv_dataset)} test cases")
print()

# Example 7: Create and save custom dataset
print("Example 7: Create and save custom dataset")
print("-" * 70)

from houyi.evaluation.dataset import TestCase

custom_dataset = Dataset(
    name="Custom Test Dataset",
    description="Manually created test dataset",
    test_cases=[
        TestCase(
            input="What is AI?",
            expected_output="AI is artificial intelligence.",
            metadata={"category": "ai", "difficulty": "easy"}
        ),
        TestCase(
            input="Explain neural networks",
            expected_output="Neural networks are computing systems inspired by biological neural networks.",
            metadata={"category": "ai", "difficulty": "hard"}
        ),
    ]
)

custom_dataset.to_file("datasets/custom_dataset.json")
print(f"✅ Custom dataset saved to: datasets/custom_dataset.json")
print()

# Summary
print("=" * 70)
print("Summary: All Dataset & Report features demonstrated!")
print("=" * 70)
print()
print("✅ Feature 1: Load datasets from JSON/CSV/YAML")
print("✅ Feature 2: Run evaluation with Dataset")
print("✅ Feature 3: Generate HTML reports (beautiful UI)")
print("✅ Feature 4: Generate JSON reports (machine-readable)")
print("✅ Feature 5: Generate Markdown reports (documentation)")
print("✅ Feature 6: Create and save custom datasets")
print()
print("🎉 Phase 3.3 Complete: Dataset & Report Generation!")
print("=" * 70)
