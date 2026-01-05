"""Quickstart example for HouYi framework."""

from houyi import Agent, Task, Team, evaluate, tool


# 1. Define skills using @tool decorator
@tool
def search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for {query}", "Additional info"]

@tool
def analyze(data: list[str]) -> str:
    """Analyze search results."""
    return f"Analysis of {len(data)} results: {data[0]}"

# 2. Create agents
researcher = Agent(
    role="Researcher",
    skills=[search]
)

analyst = Agent(
    role="Analyst",
    skills=[analyze]
)

print("=== Example 1: Simple Agent Execution ===")
result = researcher.run("What is HouYi?")
print(f"Result: {result}\n")

print("=== Example 2: Task-Driven Execution ===")
task = Task(
    description="Research HouYi framework",
    expected_output="A detailed report"
)
result = researcher.run(task)
print(f"Result: {result}\n")

print("=== Example 3: Multi-Agent Team ===")
team = Team(
    agents=[researcher, analyst],
    tasks=[
        Task("Research HouYi", agent=researcher),
        Task("Analyze findings", agent=analyst, context=[0])
    ]
)
result = team.run()
print(f"Result: {result}\n")

print("=== Example 4: Evaluation ===")
test_cases = [
    {"input": "What is HouYi?", "expected_output": "A multi-agent framework"},
    {"input": "How does it work?", "expected_output": "Uses agents and tasks"},
]

results = evaluate(
    agent=researcher,
    test_cases=test_cases,
    evaluators=["accuracy", "cost", "latency"]
)
print(f"Evaluation: {results.summary()}")
