"""Example of using verification in HouYi.

This example demonstrates:
1. SQL verification with auto-fix
2. Python code verification
3. Neuro-symbolic engine with retry logic
4. Different verification modes
"""

import asyncio

from houyi.assurance.verification import (
    NeuroSymbolicEngine,
    PythonVerifier,
    SQLVerifier,
    VerificationConfig,
    VerificationRule,
)


async def example_sql_verification():
    """Example: SQL verification with auto-fix."""
    print("\n=== SQL Verification Example ===")

    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="sql_check",
        verifier_type="sql",
        rule_spec={
            "check_syntax": True,
            "check_injection": True,
            "allowed_operations": ["SELECT"],
        },
    )

    # Valid SQL
    sql1 = "SELECT * FROM users WHERE id = 1;"
    result1 = await verifier.verify(sql1, rule)
    print(f"✓ Valid SQL: {result1.passed}")

    # SQL with missing semicolon (auto-fixable)
    sql2 = "SELECT * FROM users"
    result2 = await verifier.verify(sql2, rule)
    print(f"  Query: {sql2}")
    print(f"  Passed: {result2.passed}")
    print(f"  Error: {result2.error_message}")
    if result2.auto_fixable:
        fixed, _success = await verifier.auto_fix(sql2, result2)
        print(f"  → Auto-fixed: {fixed}")

    # SQL injection attempt
    sql3 = "SELECT * FROM users WHERE id = 1 OR 1=1;"
    result3 = await verifier.verify(sql3, rule)
    print(f"✗ SQL injection: {result3.passed} - {result3.error_message}")


async def example_python_verification():
    """Example: Python code verification."""
    print("\n=== Python Verification Example ===")

    verifier = PythonVerifier()
    rule = VerificationRule(
        rule_id="python_check",
        verifier_type="python",
        rule_spec={
            "check_syntax": True,
            "check_imports": True,
        },
    )

    # Valid Python code
    code1 = "x = 1 + 2\nprint(x)"
    result1 = await verifier.verify(code1, rule)
    print(f"✓ Valid Python: {result1.passed}")

    # Unsafe import
    code2 = "import os\nos.system('rm -rf /')"
    result2 = await verifier.verify(code2, rule)
    print(f"✗ Unsafe import: {result2.passed} - {result2.error_message}")


async def example_neuro_symbolic_engine():
    """Example: Neuro-symbolic engine with retry logic."""
    print("\n=== Neuro-Symbolic Engine Example ===")

    # Lenient mode: auto-fix enabled
    config = VerificationConfig.lenient()
    engine = NeuroSymbolicEngine(config=config)

    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="sql_gen",
        verifier_type="sql",
        rule_spec={"check_syntax": True, "check_injection": True},
    )

    # Simulate LLM generator that returns SQL without semicolon
    attempt = 0

    async def generator():
        nonlocal attempt
        attempt += 1
        print(f"  Generator attempt {attempt}")
        return "SELECT * FROM users"

    output, success = await engine.generate_and_verify(
        generator, verifier, rule, task_id="example_task"
    )

    print(f"✓ Final output: {output}")
    print(f"✓ Success: {success}")
    print(f"✓ Metrics: {engine.get_metrics()}")


async def example_verification_modes():
    """Example: Different verification modes."""
    print("\n=== Verification Modes Example ===")

    verifier = SQLVerifier()
    rule = VerificationRule(
        rule_id="sql_mode_test",
        verifier_type="sql",
        rule_spec={"check_syntax": True},
    )

    invalid_sql = "SELECT * FROM users"  # Missing semicolon

    # Strict mode: fail immediately
    print("\n1. Strict Mode:")
    strict_config = VerificationConfig.strict()
    strict_engine = NeuroSymbolicEngine(config=strict_config)

    async def gen_invalid():
        return invalid_sql

    output, success = await strict_engine.generate_and_verify(gen_invalid, verifier, rule)
    print(f"   Result: {success} (expected: False)")

    # Lenient mode: auto-fix
    print("\n2. Lenient Mode:")
    lenient_config = VerificationConfig.lenient()
    lenient_engine = NeuroSymbolicEngine(config=lenient_config)

    output, success = await lenient_engine.generate_and_verify(gen_invalid, verifier, rule)
    print(f"   Result: {success} (expected: True)")
    print(f"   Fixed output: {output}")

    # Audit mode: log but don't block
    print("\n3. Audit Mode:")
    audit_config = VerificationConfig.audit()
    audit_engine = NeuroSymbolicEngine(config=audit_config)

    output, success = await audit_engine.generate_and_verify(gen_invalid, verifier, rule)
    print(f"   Result: {success} (expected: True, logs warning)")

    # Disabled mode: no verification
    print("\n4. Disabled Mode:")
    disabled_config = VerificationConfig.disabled()
    disabled_engine = NeuroSymbolicEngine(config=disabled_config)

    output, success = await disabled_engine.generate_and_verify(gen_invalid, verifier, rule)
    print(f"   Result: {success} (expected: True)")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("HouYi Verification Examples")
    print("=" * 60)

    await example_sql_verification()
    await example_python_verification()
    await example_neuro_symbolic_engine()
    await example_verification_modes()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
