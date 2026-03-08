"""Example demonstrating skill.md file support (AgentSkills.io standard)."""

from houyi import Agent, tool
from houyi.domain.skill.spec import SkillSpec

print("=" * 70)
print("Phase 3.1: Skill.md Support Examples")
print("=" * 70)
print()

# Example 1: Create a skill using @tool decorator
print("Example 1: Create skill with @tool decorator")
print("-" * 70)


@tool
def search(query: str, max_results: int = 10) -> list[dict]:
    """Search the web for information."""
    return [
        {
            "title": f"Result for {query}",
            "url": "https://example.com",
            "snippet": "Sample search result...",
        }
    ]


print(f"Created skill: {search.name}")
print(f"Description: {search.description}")
print()

# Example 2: Load skill from local skill.md file
print("Example 2: Load skill from local file")
print("-" * 70)

web_search_skill = SkillSpec.from_file("skills/web_search.md")
print(f"Loaded skill: {web_search_skill.name}")
print(f"Description: {web_search_skill.description}")
print()


# Bind executor to loaded skill
def web_search_executor(query: str, max_results: int = 10):
    """Execute web search."""
    return {
        "results": [
            {
                "title": f"Result for '{query}'",
                "url": "https://example.com",
                "snippet": "This is a sample search result...",
            }
        ]
    }


web_search_skill.bind_executor(web_search_executor)
print(f"Executor bound to {web_search_skill.name}")
print()

# Example 3: Load calculator skill and test execution
print("Example 3: Load calculator skill and test execution")
print("-" * 70)

calc_skill = SkillSpec.from_file("skills/calculator.md")
print(f"Loaded skill: {calc_skill.name}")


# Bind calculator executor
def calculator_executor(expression: str):
    """Execute calculation."""
    try:
        result = eval(expression)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"result": 0, "expression": expression, "error": str(e)}


calc_skill.bind_executor(calculator_executor)

# Test execution
test_expr = "2 + 2 * 3"
test_input = calc_skill.input_schema(expression=test_expr)
result = calc_skill.executor(**test_input.model_dump())
print(f"Test calculation: {test_expr} = {result['result']}")
print()

# Example 4: Export skill to skill.md
print("Example 4: Export @tool skill to skill.md")
print("-" * 70)


@tool
def text_analyzer(text: str, language: str = "en") -> dict:
    """Analyze text and return statistics."""
    return {"word_count": len(text.split()), "char_count": len(text), "language": language}


# Export with metadata and examples
text_analyzer.export_skill_md(
    "skills/text_analyzer.md",
    metadata={"language": "Python", "runtime": "sync", "timeout": "5s", "cost": "$0.0001 per call"},
    examples=[
        {
            "input": {"text": "Hello world", "language": "en"},
            "output": {"word_count": 2, "char_count": 11, "language": "en"},
        }
    ],
)
print("Exported skill to: skills/text_analyzer.md")
print()

# Example 5: Create Agent with loaded skills
print("Example 5: Create Agent with loaded skills")
print("-" * 70)

agent = Agent(role="Research Assistant", skills=[web_search_skill, calc_skill, text_analyzer])
print(f"Created agent with {len(agent.skills)} skills:")
for skill in agent.skills:
    print(f"   - {skill.name}: {skill.description[:50]}...")
print()

# Example 6: Load from URL (demonstration)
print("Example 6: Load from URL (demonstration)")
print("-" * 70)
print("# To load from URL:")
print("# skill = SkillSpec.from_url('https://example.com/skill.md')")
print("# ")
print("# To load from AgentSkills.io registry:")
print("# skill = SkillSpec.from_registry('web_search')")
print()

# Summary
print("=" * 70)
print("Summary: All Skill.md features demonstrated!")
print("=" * 70)
print()
print("Feature 1: @tool decorator - Create skills from functions")
print("Feature 2: from_file() - Load skills from local skill.md files")
print("Feature 3: bind_executor() - Bind execution functions to loaded skills")
print("Feature 4: export_skill_md() - Export skills to skill.md format")
print("Feature 5: Agent integration - Use loaded skills in agents")
print("Feature 6: from_url() - Load skills from URLs (network required)")
print("Feature 7: from_registry() - Load from AgentSkills.io (network required)")
print()
print("Phase 3.1 Complete: Skill.md Support Fully Implemented!")
print("=" * 70)
