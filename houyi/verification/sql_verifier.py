"""SQL verification implementation."""

import re
from typing import Any

from houyi.verification.constraint_solver import Constraint, SQLConstraintSolver
from houyi.verification.verifier import VerificationResult, VerificationRule, Verifier


class SQLVerifier(Verifier):
    """Verifies SQL queries for syntax, security, and permissions."""

    UNSAFE_PATTERNS = [
        r";\s*DROP\s+",
        r";\s*DELETE\s+FROM\s+",
        r";\s*UPDATE\s+.*\s+SET\s+",
        r"--",
        r"/\*.*\*/",
        r"UNION\s+SELECT",
        r"OR\s+1\s*=\s*1",
        r"OR\s+'1'\s*=\s*'1'",
    ]

    ALLOWED_OPERATIONS = {"SELECT", "INSERT", "UPDATE", "DELETE"}

    def __init__(self, use_constraint_solver: bool = True, use_cache: bool = True):
        """Initialize SQL verifier.

        Args:
            use_constraint_solver: Whether to use Z3 constraint solver for formal verification
            use_cache: Enable caching of verification results
        """
        super().__init__(use_cache=use_cache)

        try:
            import sqlparse

            self.sqlparse = sqlparse
        except ImportError as err:
            raise ImportError(
                "sqlparse is required for SQL verification. Install it with: pip install sqlparse"
            ) from err

        self.use_constraint_solver = use_constraint_solver
        if use_constraint_solver:
            self.constraint_solver = SQLConstraintSolver(use_cache=use_cache)
        else:
            self.constraint_solver = None

    async def _verify_impl(
        self,
        output: Any,
        rule: VerificationRule,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Internal SQL verification implementation.

        Args:
            output: SQL query string
            rule: Verification rule
            context: Additional context

        Returns:
            VerificationResult
        """
        if not isinstance(output, str):
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message="Output is not a string",
                error_type="invalid_type",
                severity=rule.severity,
            )

        sql = output.strip()
        if not sql:
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message="SQL query is empty",
                error_type="empty_query",
                severity=rule.severity,
            )

        rule_spec = rule.rule_spec

        # 1. Check syntax
        if rule_spec.get("check_syntax", True):
            try:
                parsed = self.sqlparse.parse(sql)
                if not parsed:
                    return VerificationResult(
                        rule_id=rule.rule_id,
                        passed=False,
                        error_message="Failed to parse SQL query",
                        error_type="sql_syntax",
                        auto_fixable=False,
                        severity=rule.severity,
                    )
            except Exception as e:
                return VerificationResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    error_message=f"SQL syntax error: {str(e)}",
                    error_type="sql_syntax",
                    auto_fixable=False,
                    severity=rule.severity,
                )

        # 2. Check for SQL injection patterns
        if rule_spec.get("check_injection", True):
            sql_upper = sql.upper()
            for pattern in self.UNSAFE_PATTERNS:
                if re.search(pattern, sql_upper, re.IGNORECASE):
                    return VerificationResult(
                        rule_id=rule.rule_id,
                        passed=False,
                        error_message=f"Potential SQL injection detected: pattern '{pattern}'",
                        error_type="sql_injection",
                        auto_fixable=False,
                        severity="error",
                        fix_suggestion="Use parameterized queries instead of string concatenation",
                    )

        # 3. Check allowed operations
        allowed_ops = rule_spec.get("allowed_operations")
        if allowed_ops:
            parsed = self.sqlparse.parse(sql)[0]
            stmt_type = parsed.get_type()
            if stmt_type not in allowed_ops:
                return VerificationResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    error_message=f"SQL operation '{stmt_type}' not allowed. Allowed: {allowed_ops}",
                    error_type="forbidden_operation",
                    auto_fixable=False,
                    severity=rule.severity,
                )

        # 4. Check forbidden tables
        forbidden_tables = rule_spec.get("forbidden_tables", [])
        if forbidden_tables:
            sql_upper = sql.upper()
            for table in forbidden_tables:
                if re.search(rf"\bFROM\s+{table.upper()}\b", sql_upper):
                    return VerificationResult(
                        rule_id=rule.rule_id,
                        passed=False,
                        error_message=f"Access to table '{table}' is forbidden",
                        error_type="forbidden_table",
                        auto_fixable=False,
                        severity=rule.severity,
                    )

        # 5. Check for missing semicolon (auto-fixable)
        if not sql.rstrip().endswith(";"):
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message="SQL query missing semicolon",
                error_type="missing_semicolon",
                auto_fixable=True,
                fix_strategy="add_semicolon",
                severity="warning",
            )

        # 4. Check constraints if enabled and provided
        if self.use_constraint_solver and rule_spec.get("constraints"):
            constraints = self._build_constraints(rule_spec["constraints"])
            if constraints:
                try:
                    # Extract schema if provided
                    schema = context.get("schema", {}) if context else {}
                    is_sat, violated = self.constraint_solver.verify_sql_constraints(
                        sql, schema, constraints
                    )

                    if not is_sat:
                        return VerificationResult(
                            rule_id=rule.rule_id,
                            passed=False,
                            error_message=f"SQL violates constraints: {', '.join(violated)}",
                            error_type="constraint_violation",
                            auto_fixable=False,
                            severity=rule.severity,
                            metadata={"violated_constraints": violated},
                        )
                except Exception as e:
                    # Log but don't fail on constraint checking errors
                    import logging

                    logging.warning("Constraint verification failed: %s", e)

        # All checks passed
        return VerificationResult(
            rule_id=rule.rule_id,
            passed=True,
            severity=rule.severity,
        )

    def _build_constraints(self, constraint_specs: list[dict[str, Any]]) -> list[Constraint]:
        """Build Constraint objects from constraint specifications.

        Args:
            constraint_specs: List of constraint specifications

        Returns:
            List of Constraint objects
        """
        constraints = []
        for spec in constraint_specs:
            try:
                constraint = Constraint(
                    name=spec.get("name", "unnamed"),
                    constraint_type=spec.get("type", "general"),
                    expression=spec.get("expression", ""),
                    description=spec.get("description", ""),
                )
                constraints.append(constraint)
            except Exception as e:
                import logging

                logging.warning("Failed to build constraint from spec %s: %s", spec, e)

        return constraints

    def supports_auto_fix(self, error_type: str) -> bool:
        """Check if error type can be auto-fixed."""
        return error_type in ["sql_format", "missing_semicolon"]

    async def auto_fix(
        self,
        output: Any,
        error: VerificationResult,
    ) -> tuple[Any, bool]:
        """Auto-fix SQL errors.

        Args:
            output: Original SQL query
            error: Verification error

        Returns:
            (fixed_sql, success)
        """
        if not isinstance(output, str):
            return output, False

        sql = output.strip()

        if error.error_type == "sql_format":
            try:
                formatted = self.sqlparse.format(
                    sql,
                    reindent=True,
                    keyword_case="upper",
                )
                return formatted, True
            except Exception:
                return output, False

        elif error.error_type == "missing_semicolon":
            return sql.rstrip() + ";", True

        return output, False
