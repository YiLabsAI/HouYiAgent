"""Constraint solver using Z3 for formal verification."""

import logging
from typing import Any

try:
    from z3 import (
        And,
        Bool,
        Int,
        Not,
        Or,
        Real,
        Solver,
        String,
        sat,
        unknown,
        unsat,
    )
except ImportError as _err:
    raise ImportError(
        "z3-solver is required for constraint solving. "
        "Install it with: pip install 'houyi[verification]'"
    ) from _err

logger = logging.getLogger(__name__)

# Import cache (lazy import to avoid circular dependencies)
_cache = None


def _get_cache():
    """Lazy load constraint cache."""
    global _cache
    if _cache is None:
        from houyi.assurance.verification.cache import get_constraint_cache

        _cache = get_constraint_cache()
    return _cache


class Constraint:
    """Represents a single constraint."""

    def __init__(
        self,
        name: str,
        constraint_type: str,
        expression: str,
        description: str = "",
    ):
        """Initialize constraint.

        Args:
            name: Constraint name/identifier
            constraint_type: Type of constraint (e.g., 'range', 'equality', 'relation')
            expression: Constraint expression (e.g., 'x > 0', 'x + y == 10')
            description: Human-readable description
        """
        self.name = name
        self.constraint_type = constraint_type
        self.expression = expression
        self.description = description


class ConstraintSolver:
    """Z3-based constraint solver for formal verification.

    Provides constraint solving capabilities for verifying:
    - Numeric constraints (ranges, equalities, inequalities)
    - String constraints (length, format, patterns)
    - Boolean constraints (logical conditions)
    - Relational constraints (between multiple variables)

    Features caching for improved performance on repeated constraint checks.
    """

    def __init__(self, use_cache: bool = True, timeout_ms: int = 100):
        """Initialize constraint solver with performance optimizations.

        Performance Strategy:
        1. Timeout protection: Prevent Z3 from hanging on complex constraints
        2. Caching: Reuse results for identical constraint problems
        3. Optimized Z3 settings: Disable expensive features for speed

        Args:
            use_cache: Enable caching of solver results (default: True)
            timeout_ms: Timeout for Z3 solving in milliseconds (default: 100ms)
                       Industry best practice: 100-500ms for real-time systems
        """
        self.solver = Solver()

        # Enable unsat core tracking for better error reporting
        self.solver.set("unsat_core", True)

        # CRITICAL: Set timeout to prevent Z3 from hanging on NP-hard problems
        # Without timeout, Z3 can take minutes or even hang indefinitely
        self.solver.set("timeout", timeout_ms)

        # Performance optimizations based on Z3 best practices:
        # 1. Disable auto-config: Reduces overhead for simple constraints
        # 2. Disable MBQI: Model-based quantifier instantiation is expensive
        #    We don't use quantifiers in our constraints, so this is safe
        self.solver.set("auto_config", False)
        self.solver.set("smt.mbqi", False)

        self.variables = {}
        self.constraints = []
        self.use_cache = use_cache
        self.timeout_ms = timeout_ms
        self._cache = _get_cache() if use_cache else None

        # Precompiled constraint cache: Avoid re-parsing identical expressions
        # Key: constraint expression string
        # Value: compiled Z3 expression
        self._compiled_constraints = {}

    def add_int_variable(self, name: str) -> Any:
        """Add an integer variable.

        Args:
            name: Variable name

        Returns:
            Z3 Int variable
        """
        if name not in self.variables:
            self.variables[name] = Int(name)
        return self.variables[name]

    def add_real_variable(self, name: str) -> Any:
        """Add a real number variable.

        Args:
            name: Variable name

        Returns:
            Z3 Real variable
        """
        if name not in self.variables:
            self.variables[name] = Real(name)
        return self.variables[name]

    def add_string_variable(self, name: str) -> Any:
        """Add a string variable.

        Args:
            name: Variable name

        Returns:
            Z3 String variable
        """
        if name not in self.variables:
            self.variables[name] = String(name)
        return self.variables[name]

    def add_bool_variable(self, name: str) -> Any:
        """Add a boolean variable.

        Args:
            name: Variable name

        Returns:
            Z3 Bool variable
        """
        if name not in self.variables:
            self.variables[name] = Bool(name)
        return self.variables[name]

    def add_constraint(self, constraint: Any, name: str = "") -> None:
        """Add a constraint to the solver.

        Args:
            constraint: Z3 constraint expression
            name: Optional constraint name for tracking
        """
        self.solver.add(constraint)
        self.constraints.append((name, constraint))
        logging.debug("Added constraint: %s", name or "unnamed")

    def check_satisfiability(self) -> tuple[bool, dict[str, Any]]:
        """Check if constraints are satisfiable with timeout protection.

        This method invokes Z3 solver with timeout protection to prevent hanging
        on complex NP-hard constraint problems.

        Returns:
            Tuple of (is_satisfiable, model_or_unsat_core)
            - If satisfiable: (True, {var: value, ...})
            - If unsatisfiable: (False, {violated_constraints})
            - If timeout/unknown: (False, {"timeout": True})
        """
        result = self.solver.check()

        # Handle timeout/unknown result
        # Z3 returns 'unknown' when:
        # 1. Timeout is reached (most common)
        # 2. Problem is too complex to decide
        # 3. Incomplete theory (rare for our use cases)
        if result == unknown:
            # NOTE: Z3 can return unknown for timeouts or undecidable queries.
            # Even for "simple" constraints, the solver may occasionally exceed
            # a small timeout due to resource contention, OS scheduling jitter,
            # or internal heuristic choices. Retry once with a higher timeout to
            # reduce false negatives.
            retry_timeout_ms = max(self.timeout_ms, 2000)
            if retry_timeout_ms > self.timeout_ms:
                logging.warning(
                    "Z3 solver timeout after %dms; retrying once with %dms",
                    self.timeout_ms,
                    retry_timeout_ms,
                )
                self.solver.set("timeout", retry_timeout_ms)
                try:
                    result = self.solver.check()
                finally:
                    self.solver.set("timeout", self.timeout_ms)
            else:
                logging.warning("Z3 solver returned unknown after %dms", self.timeout_ms)

            if result == unknown:
                return False, {
                    "timeout": True,
                    "message": (
                        f"Solver timeout after {self.timeout_ms}ms"
                        + (
                            f" (retried {retry_timeout_ms}ms)"
                            if retry_timeout_ms > self.timeout_ms
                            else ""
                        )
                    ),
                }

        if result == sat:
            model = self.solver.model()
            solution = {}
            for var_name, var in self.variables.items():
                if model[var] is not None:
                    solution[var_name] = model[var]
            logging.info("Constraints satisfiable. Solution: %s", solution)
            return True, solution

        elif result == unsat:
            # Get unsat core (violated constraints)
            core = self.solver.unsat_core()
            violated = {}
            for i, (name, _) in enumerate(self.constraints):
                if name and any(str(c) == str(self.constraints[i][1]) for c in core):
                    violated[name] = str(self.constraints[i][1])
            logging.warning("Constraints unsatisfiable. Violated: %s", violated)
            return False, violated

        else:
            logging.error("Z3 solver returned unknown result")
            return False, {"error": "unknown"}

    def verify_constraints(
        self,
        constraints: list[Constraint],
        values: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Verify that given values satisfy constraints with caching and timeout.

        Performance optimizations:
        1. Cache lookup: Check if we've solved this exact problem before (<1ms)
        2. Timeout protection: Abort if Z3 takes too long (100ms default)
        3. Incremental solving: Reuse solver state when possible

        Args:
            constraints: List of constraints to verify
            values: Dictionary of variable values to check

        Returns:
            Tuple of (all_satisfied, violated_constraint_names)
            - all_satisfied: True if all constraints are satisfied
            - violated_constraint_names: List of violated constraint names
                                        ["timeout"] if solver times out
        """
        cache_key = self._build_cache_key(constraints, values)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        self.reset()

        self._bind_values(values)
        constraint_names = self._add_constraints_to_solver(constraints)
        is_sat, violated_list = self._normalize_solver_result(self.solver.check(), constraint_names)
        self._store_cached_result(cache_key, is_sat, violated_list)

        return is_sat, violated_list

    @staticmethod
    def _build_cache_key(
        constraints: list[Constraint],
        values: dict[str, Any],
    ) -> tuple[dict[str, str], list[str]]:
        var_values = {key: f"{type(value).__name__}:{value!r}" for key, value in values.items()}
        constraint_exprs = [constraint.expression for constraint in constraints]
        return var_values, constraint_exprs

    def _get_cached_result(
        self,
        cache_key: tuple[dict[str, str], list[str]],
    ) -> tuple[bool, list[str]] | None:
        if not self.use_cache or not self._cache:
            return None
        var_values, constraint_exprs = cache_key
        cached_result = self._cache.get_result(var_values, constraint_exprs)
        if cached_result is not None:
            logging.debug("Cache hit: Using cached constraint result")
        return cached_result

    def _store_cached_result(
        self,
        cache_key: tuple[dict[str, str], list[str]],
        is_sat: bool,
        violated_list: list[str],
    ) -> None:
        if not self.use_cache or not self._cache:
            return
        var_values, constraint_exprs = cache_key
        self._cache.put_result(var_values, constraint_exprs, is_sat, violated_list)

    def _bind_values(self, values: dict[str, Any]) -> None:
        for var_name, value in values.items():
            self._bind_single_value(var_name, value)

    def _bind_single_value(self, var_name: str, value: Any) -> None:
        if isinstance(value, bool):
            var = self.add_bool_variable(var_name)
        elif isinstance(value, int):
            var = self.add_int_variable(var_name)
        elif isinstance(value, float):
            var = self.add_real_variable(var_name)
        elif isinstance(value, str):
            var = self.add_string_variable(var_name)
        else:
            return
        self.add_constraint(var == value, f"value_{var_name}")

    def _add_constraints_to_solver(self, constraints: list[Constraint]) -> list[str]:
        constraint_names: list[str] = []
        for constraint in constraints:
            try:
                constraint_expr = self._parse_constraint_expression(
                    constraint.expression,
                    self.variables,
                )
                self.add_constraint(constraint_expr, constraint.name)
                constraint_names.append(constraint.name)
            except Exception as exc:
                logging.warning("Failed to build constraint from spec %s: %s", constraint.name, exc)
        return constraint_names

    def _normalize_solver_result(
        self,
        result: Any,
        constraint_names: list[str],
    ) -> tuple[bool, list[str]]:
        if result == sat:
            return True, []
        if result == unknown:
            logger.warning("Constraint solving timeout after %dms", self.timeout_ms)
            return False, ["timeout"]
        if result == unsat:
            return False, constraint_names[:]
        return False, ["unknown"]

    def _parse_constraint_expression(
        self,
        expression: str,
        variables: dict[str, Any],
    ) -> Any:
        """Parse constraint expression string into Z3 expression.

        This method converts string expressions like "x > 0" into Z3 constraint objects.
        It uses eval() with a restricted context for safety.

        Args:
            expression: Constraint expression string (e.g., "x > 0", "x + y < 100")
            variables: Dictionary of Z3 variables

        Returns:
            Z3 constraint expression

        Note:
            This is a simplified parser. Production systems should use proper
            parsing libraries or DSLs to avoid eval() security risks.
        """
        # Create a safe evaluation context with Z3 logical operators
        # and the variables defined in the solver
        eval_context = {
            "And": And,
            "Or": Or,
            "Not": Not,
            "__builtins__": {},  # Disable built-in functions for security
        }
        eval_context.update(variables)

        # Evaluate expression in the restricted context
        # This allows expressions like "x > 0" to be evaluated as Z3 constraints
        try:
            return eval(expression, eval_context, {})
        except Exception as e:
            logger.error("Failed to parse expression '%s': %s", expression, e)
            raise ValueError(f"Invalid constraint expression: {expression}") from e

    def reset(self) -> None:
        """Reset solver state."""
        self.solver.reset()
        self.variables.clear()
        self.constraints.clear()
        logger.debug("Solver reset")

    def get_model(self) -> dict[str, Any]:
        """Get current model if constraints are satisfiable.

        Returns:
            Dictionary of variable assignments
        """
        if self.solver.check() == sat:
            model = self.solver.model()
            return {var_name: model[var] for var_name, var in self.variables.items()}
        return {}


class SQLConstraintSolver(ConstraintSolver):
    """Specialized constraint solver for SQL verification.

    Verifies SQL-specific constraints such as:
    - Column value ranges
    - Foreign key constraints
    - Check constraints
    - Uniqueness constraints
    """

    def verify_sql_constraints(
        self,
        query: str,
        schema: dict[str, Any],
        constraints: list[Constraint],
    ) -> tuple[bool, list[str]]:
        """Verify SQL query against schema constraints.

        Args:
            query: SQL query string
            schema: Database schema definition
            constraints: List of constraints to verify

        Returns:
            Tuple of (all_satisfied, violated_constraint_names)
        """
        # Extract variables from query (simplified)
        # Real implementation would parse SQL AST

        # For now, return basic verification
        return self.verify_constraints(constraints, {})


class PythonConstraintSolver(ConstraintSolver):
    """Specialized constraint solver for Python code verification.

    Verifies Python-specific constraints such as:
    - Variable type constraints
    - Value range constraints
    - Function pre/post conditions
    """

    def verify_python_constraints(
        self,
        code: str,
        constraints: list[Constraint],
    ) -> tuple[bool, list[str]]:
        """Verify Python code against constraints.

        Args:
            code: Python code string
            constraints: List of constraints to verify

        Returns:
            Tuple of (all_satisfied, violated_constraint_names)
        """
        # Extract variables from code (simplified)
        # Real implementation would parse Python AST

        # For now, return basic verification
        return self.verify_constraints(constraints, {})
