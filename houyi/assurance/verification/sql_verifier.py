"""SQL verification implementation."""

import re
from typing import Any

from houyi.assurance.verification.constraint_solver import Constraint, SQLConstraintSolver
from houyi.assurance.verification.verifier import VerificationResult, VerificationRule, Verifier


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
        self.constraint_solver: SQLConstraintSolver | None
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
        sql, invalid_result = self._normalize_sql_input(output, rule)
        if invalid_result is not None:
            return invalid_result

        rule_spec = rule.rule_spec

        syntax_error = self._check_syntax(sql, rule, rule_spec)
        if syntax_error is not None:
            return syntax_error

        injection_error = self._check_injection(sql, rule, rule_spec)
        if injection_error is not None:
            return injection_error

        operation_error = self._check_allowed_operations(sql, rule, rule_spec)
        if operation_error is not None:
            return operation_error

        table_error = self._check_forbidden_tables(sql, rule, rule_spec)
        if table_error is not None:
            return table_error

        semicolon_error = self._check_missing_semicolon(sql, rule)
        if semicolon_error is not None:
            return semicolon_error

        constraint_error = self._verify_constraints(sql, rule, rule_spec, context)
        if constraint_error is not None:
            return constraint_error

        return VerificationResult(rule_id=rule.rule_id, passed=True, severity=rule.severity)

    def _normalize_sql_input(
        self,
        output: Any,
        rule: VerificationRule,
    ) -> tuple[str, VerificationResult | None]:
        if not isinstance(output, str):
            return "", VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message="Output is not a string",
                error_type="invalid_type",
                severity=rule.severity,
            )
        sql = output.strip()
        if sql:
            return sql, None
        return "", VerificationResult(
            rule_id=rule.rule_id,
            passed=False,
            error_message="SQL query is empty",
            error_type="empty_query",
            severity=rule.severity,
        )

    def _check_syntax(
        self,
        sql: str,
        rule: VerificationRule,
        rule_spec: dict[str, Any],
    ) -> VerificationResult | None:
        if not rule_spec.get("check_syntax", True):
            return None
        try:
            parsed = self.sqlparse.parse(sql)
            if parsed:
                return None
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message="Failed to parse SQL query",
                error_type="sql_syntax",
                auto_fixable=False,
                severity=rule.severity,
            )
        except Exception as exc:
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"SQL syntax error: {exc!s}",
                error_type="sql_syntax",
                auto_fixable=False,
                severity=rule.severity,
            )

    def _check_injection(
        self,
        sql: str,
        rule: VerificationRule,
        rule_spec: dict[str, Any],
    ) -> VerificationResult | None:
        if not rule_spec.get("check_injection", True):
            return None
        sql_upper = sql.upper()
        for pattern in self.UNSAFE_PATTERNS:
            if not re.search(pattern, sql_upper, re.IGNORECASE):
                continue
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"Potential SQL injection detected: pattern '{pattern}'",
                error_type="sql_injection",
                auto_fixable=False,
                severity="error",
                fix_suggestion="Use parameterized queries instead of string concatenation",
            )
        return None

    def _check_allowed_operations(
        self,
        sql: str,
        rule: VerificationRule,
        rule_spec: dict[str, Any],
    ) -> VerificationResult | None:
        allowed_ops = rule_spec.get("allowed_operations")
        if not allowed_ops:
            return None
        stmt_type = self.sqlparse.parse(sql)[0].get_type()
        if stmt_type in allowed_ops:
            return None
        return VerificationResult(
            rule_id=rule.rule_id,
            passed=False,
            error_message=f"SQL operation '{stmt_type}' not allowed. Allowed: {allowed_ops}",
            error_type="forbidden_operation",
            auto_fixable=False,
            severity=rule.severity,
        )

    def _check_forbidden_tables(
        self,
        sql: str,
        rule: VerificationRule,
        rule_spec: dict[str, Any],
    ) -> VerificationResult | None:
        forbidden_tables = rule_spec.get("forbidden_tables", [])
        if not forbidden_tables:
            return None
        sql_upper = sql.upper()
        for table in forbidden_tables:
            if not re.search(rf"\bFROM\s+{table.upper()}\b", sql_upper):
                continue
            return VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"Access to table '{table}' is forbidden",
                error_type="forbidden_table",
                auto_fixable=False,
                severity=rule.severity,
            )
        return None

    def _check_missing_semicolon(
        self,
        sql: str,
        rule: VerificationRule,
    ) -> VerificationResult | None:
        if sql.rstrip().endswith(";"):
            return None
        return VerificationResult(
            rule_id=rule.rule_id,
            passed=False,
            error_message="SQL query missing semicolon",
            error_type="missing_semicolon",
            auto_fixable=True,
            fix_strategy="add_semicolon",
            severity="warning",
        )

    def _verify_constraints(
        self,
        sql: str,
        rule: VerificationRule,
        rule_spec: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> VerificationResult | None:
        if not self.use_constraint_solver or not rule_spec.get("constraints"):
            return None
        solver = self.constraint_solver
        if solver is None:
            return None
        constraints = self._build_constraints(rule_spec["constraints"])
        if not constraints:
            return None
        try:
            schema = context.get("schema", {}) if context else {}
            is_sat, violated = solver.verify_sql_constraints(sql, schema, constraints)
        except Exception as exc:
            import logging

            logging.warning("Constraint verification failed: %s", exc)
            return None
        if is_sat:
            return None
        return VerificationResult(
            rule_id=rule.rule_id,
            passed=False,
            error_message=f"SQL violates constraints: {', '.join(violated)}",
            error_type="constraint_violation",
            auto_fixable=False,
            severity=rule.severity,
            metadata={"violated_constraints": violated},
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
