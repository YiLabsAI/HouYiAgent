"""Python code verification implementation."""

import ast
from typing import Any

from houyi.assurance.verification.constraint_solver import Constraint, PythonConstraintSolver
from houyi.assurance.verification.verifier import VerificationResult, VerificationRule, Verifier


class PythonVerifier(Verifier):
    """Verifies Python code for syntax, security, and best practices."""

    UNSAFE_IMPORTS = {"os", "subprocess", "sys", "eval", "exec", "compile", "__import__"}
    UNSAFE_BUILTINS = {"eval", "exec", "compile", "__import__", "open"}

    def __init__(self, use_constraint_solver: bool = True, use_cache: bool = True):
        """Initialize Python verifier.

        Args:
            use_constraint_solver: Whether to use Z3 constraint solver for formal verification
            use_cache: Enable caching of verification results
        """
        super().__init__(use_cache=use_cache)

        self.use_constraint_solver = use_constraint_solver
        self.constraint_solver: PythonConstraintSolver | None
        if use_constraint_solver:
            self.constraint_solver = PythonConstraintSolver(use_cache=use_cache)
        else:
            self.constraint_solver = None

    async def _verify_impl(
        self,
        output: Any,
        rule: VerificationRule,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Internal Python verification implementation."""
        code, invalid_result = self._validate_code_input(output, rule)
        if invalid_result is not None:
            return invalid_result

        rule_spec = rule.rule_spec

        tree, syntax_error = self._parse_code(code, rule)
        if syntax_error is not None:
            return syntax_error

        import_error = self._check_unsafe_imports(tree, rule)
        if import_error is not None:
            return import_error

        constraint_error = self._verify_constraints(code, rule, rule_spec)
        if constraint_error is not None:
            return constraint_error

        return VerificationResult(rule_id=rule.rule_id, passed=True)

    def _validate_code_input(
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
        code = output.strip()
        if code:
            return code, None
        return "", VerificationResult(
            rule_id=rule.rule_id,
            passed=False,
            error_message="Python code is empty",
            error_type="empty_code",
            severity=rule.severity,
        )

    def _parse_code(
        self,
        code: str,
        rule: VerificationRule,
    ) -> tuple[ast.AST | None, VerificationResult | None]:
        try:
            return ast.parse(code), None
        except SyntaxError as exc:
            return None, VerificationResult(
                rule_id=rule.rule_id,
                passed=False,
                error_message=f"Python syntax error: {exc.msg}",
                error_type="python_syntax",
                auto_fixable=True,
                severity=rule.severity,
            )

    def _check_unsafe_imports(
        self,
        tree: ast.AST | None,
        rule: VerificationRule,
    ) -> VerificationResult | None:
        if tree is None or not rule.rule_spec.get("check_imports", True):
            return None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import):
                continue
            for alias in node.names:
                if alias.name in self.UNSAFE_IMPORTS:
                    return VerificationResult(
                        rule_id=rule.rule_id,
                        passed=False,
                        error_message=f"Unsafe import: {alias.name}",
                        error_type="unsafe_import",
                        severity="error",
                    )
        return None

    def _verify_constraints(
        self,
        code: str,
        rule: VerificationRule,
        rule_spec: dict[str, Any],
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
            is_sat, violated = solver.verify_python_constraints(code, constraints)
        except Exception as exc:
            import logging

            logging.warning("Constraint verification failed: %s", exc)
            return None
        if is_sat:
            return None
        return VerificationResult(
            rule_id=rule.rule_id,
            passed=False,
            error_message=f"Python code violates constraints: {', '.join(violated)}",
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
        return error_type in ["python_syntax", "python_indent"]

    async def auto_fix(self, output: Any, error: VerificationResult) -> tuple[Any, bool]:
        """Auto-fix Python errors."""
        if error.error_type in ["python_syntax", "python_indent"]:
            try:
                import black

                return black.format_str(output, mode=black.Mode()), True
            except Exception:
                return output, False
        return output, False
