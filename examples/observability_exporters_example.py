"""Example demonstrating Jaeger and Datadog exporters."""

from houyi import Agent, tool
from houyi.observability import ObservabilityConfig
from houyi.observability.exporters import ConsoleExporter, DatadogExporter, JaegerExporter

print("=" * 70)
print("Phase 3.4: Jaeger & Datadog Exporters Example")
print("=" * 70)
print()

# Create a simple agent with a tool
@tool
def calculate(a: int, b: int, operation: str = "add") -> int:
    """Perform a calculation."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    else:
        return 0

# Example 1: Console Exporter (default)
print("Example 1: Console Exporter")
print("-" * 70)

agent1 = Agent(
    role="Calculator",
    skills=[calculate],
    observability=ObservabilityConfig(
        enabled=True,
        exporters=[ConsoleExporter(verbose=False)]
    )
)

result = agent1.run("Calculate 5 + 3")
print(f"Result: {result}")
print()

# Example 2: Jaeger Exporter
print("Example 2: Jaeger Exporter")
print("-" * 70)
print("Note: Requires Jaeger running on localhost:4318")
print("Start Jaeger with: docker run -d -p 4318:4318 -p 16686:16686 jaegertracing/all-in-one:latest")
print()

agent2 = Agent(
    role="Calculator",
    skills=[calculate],
    observability=ObservabilityConfig(
        enabled=True,
        exporters=[
            JaegerExporter(
                endpoint="http://localhost:4318",
                service_name="houyi-calculator",
                batch_size=5
            )
        ]
    )
)

# Run multiple operations to trigger batch export
for i in range(3):
    result = agent2.run(f"Calculate {i} + {i+1}")
    print(f"  Operation {i+1}: {result}")

# Flush remaining spans
for exporter in agent2.observability.exporters:
    exporter.flush()

print()

# Example 3: Datadog Exporter
print("Example 3: Datadog Exporter")
print("-" * 70)
print("Note: Requires Datadog Agent running on localhost:8126")
print("Install: https://docs.datadoghq.com/agent/")
print()

agent3 = Agent(
    role="Calculator",
    skills=[calculate],
    observability=ObservabilityConfig(
        enabled=True,
        exporters=[
            DatadogExporter(
                agent_url="http://localhost:8126",
                service_name="houyi-calculator",
                env="development",
                batch_size=5
            )
        ]
    )
)

# Run operations
for i in range(3):
    result = agent3.run(f"Calculate {i} * 2")
    print(f"  Operation {i+1}: {result}")

# Flush remaining traces
for exporter in agent3.observability.exporters:
    exporter.flush()

print()

# Example 4: Multiple Exporters
print("Example 4: Multiple Exporters (Console + JSON)")
print("-" * 70)

from houyi.observability.exporters import JSONExporter

agent4 = Agent(
    role="Calculator",
    skills=[calculate],
    observability=ObservabilityConfig(
        enabled=True,
        exporters=[
            ConsoleExporter(verbose=False),
            JSONExporter(filepath="traces/calculator_traces.json")
        ]
    )
)

result = agent4.run("Calculate 10 + 20")
print(f"Result: {result}")

# Flush JSON exporter
for exporter in agent4.observability.exporters:
    exporter.flush()

print("✅ Traces also saved to: traces/calculator_traces.json")
print()

# Summary
print("=" * 70)
print("Summary: Observability Exporters")
print("=" * 70)
print()
print("✅ ConsoleExporter: Simple console output")
print("✅ JSONExporter: Save traces to JSON file")
print("✅ JaegerExporter: Export to Jaeger (OTLP protocol)")
print("✅ DatadogExporter: Export to Datadog APM")
print()
print("Features:")
print("- Automatic batching for performance")
print("- OTLP format for Jaeger v1.35+")
print("- Datadog Agent API v0.4")
print("- Multiple exporters support")
print("- Zero external dependencies (uses urllib)")
print()
print("🎉 Phase 3.4: Real Exporters Implemented!")
print("=" * 70)
