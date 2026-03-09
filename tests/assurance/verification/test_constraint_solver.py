"""Tests for constraint solver."""

from houyi.assurance.verification.constraint_solver import (
    Constraint,
    ConstraintSolver,
    PythonConstraintSolver,
    SQLConstraintSolver,
)


class TestConstraint:
    """Tests for Constraint class."""

    def test_create_constraint(self):
        """Test creating a constraint."""
        constraint = Constraint(
            name="age_range",
            constraint_type="range",
            expression="age >= 0 and age <= 120",
            description="Age must be between 0 and 120",
        )

        assert constraint.name == "age_range"
        assert constraint.constraint_type == "range"
        assert "age >= 0" in constraint.expression


class TestConstraintSolver:
    """Tests for ConstraintSolver class."""

    def test_add_int_variable(self):
        """Test adding integer variable."""
        solver = ConstraintSolver()
        x = solver.add_int_variable("x")

        assert "x" in solver.variables
        assert solver.variables["x"] == x

    def test_add_real_variable(self):
        """Test adding real variable."""
        solver = ConstraintSolver()
        y = solver.add_real_variable("y")

        assert "y" in solver.variables
        assert solver.variables["y"] == y

    def test_add_bool_variable(self):
        """Test adding boolean variable."""
        solver = ConstraintSolver()
        b = solver.add_bool_variable("b")

        assert "b" in solver.variables
        assert solver.variables["b"] == b

    def test_add_constraint(self):
        """Test adding constraint."""
        solver = ConstraintSolver()
        x = solver.add_int_variable("x")

        solver.add_constraint(x > 0, "positive_x")

        assert len(solver.constraints) == 1
        assert solver.constraints[0][0] == "positive_x"

    def test_check_satisfiability_sat(self):
        """Test checking satisfiable constraints."""
        solver = ConstraintSolver()
        x = solver.add_int_variable("x")
        y = solver.add_int_variable("y")

        solver.add_constraint(x > 0, "x_positive")
        solver.add_constraint(y > 0, "y_positive")
        solver.add_constraint(x + y == 10, "sum_10")

        is_sat, solution = solver.check_satisfiability()

        assert is_sat is True
        assert "x" in solution
        assert "y" in solution

    def test_check_satisfiability_unsat(self):
        """Test checking unsatisfiable constraints."""
        solver = ConstraintSolver()
        x = solver.add_int_variable("x")

        solver.add_constraint(x > 10, "x_gt_10")
        solver.add_constraint(x < 5, "x_lt_5")

        is_sat, violated = solver.check_satisfiability()

        assert is_sat is False

    def test_verify_constraints_satisfied(self):
        """Test verifying satisfied constraints."""
        solver = ConstraintSolver()

        constraints = [
            Constraint(
                name="x_positive",
                constraint_type="range",
                expression="x > 0",
            ),
        ]

        values = {"x": 5}

        is_sat, violated = solver.verify_constraints(constraints, values)

        assert is_sat is True
        assert len(violated) == 0

    def test_verify_constraints_violated(self):
        """Test verifying violated constraints."""
        solver = ConstraintSolver()

        constraints = [
            Constraint(
                name="x_positive",
                constraint_type="range",
                expression="x > 0",
            ),
        ]

        values = {"x": -5}

        is_sat, violated = solver.verify_constraints(constraints, values)

        assert is_sat is False
        assert "x_positive" in violated

    def test_reset(self):
        """Test resetting solver."""
        solver = ConstraintSolver()
        x = solver.add_int_variable("x")
        solver.add_constraint(x > 0, "x_positive")

        assert len(solver.variables) == 1
        assert len(solver.constraints) == 1

        solver.reset()

        assert len(solver.variables) == 0
        assert len(solver.constraints) == 0

    def test_get_model(self):
        """Test getting model."""
        solver = ConstraintSolver()
        x = solver.add_int_variable("x")
        solver.add_constraint(x == 42, "x_is_42")

        model = solver.get_model()

        assert "x" in model

    def test_multiple_variable_types(self):
        """Test solver with multiple variable types."""
        solver = ConstraintSolver()

        x = solver.add_int_variable("x")
        y = solver.add_real_variable("y")
        b = solver.add_bool_variable("b")

        solver.add_constraint(x > 0, "x_positive")
        solver.add_constraint(y > 0.5, "y_gt_half")
        solver.add_constraint(b, "b_true")

        is_sat, solution = solver.check_satisfiability()

        assert is_sat is True


class TestSQLConstraintSolver:
    """Tests for SQLConstraintSolver."""

    def test_create_sql_solver(self):
        """Test creating SQL constraint solver."""
        solver = SQLConstraintSolver()

        assert isinstance(solver, ConstraintSolver)

    def test_verify_sql_constraints(self):
        """Test verifying SQL constraints."""
        solver = SQLConstraintSolver()

        query = "SELECT * FROM users WHERE age > 0"
        schema = {"users": {"columns": ["id", "name", "age"]}}
        constraints = []

        is_sat, violated = solver.verify_sql_constraints(query, schema, constraints)

        assert is_sat is True


class TestPythonConstraintSolver:
    """Tests for PythonConstraintSolver."""

    def test_create_python_solver(self):
        """Test creating Python constraint solver."""
        solver = PythonConstraintSolver()

        assert isinstance(solver, ConstraintSolver)

    def test_verify_python_constraints(self):
        """Test verifying Python constraints."""
        solver = PythonConstraintSolver()

        code = "x = 5"
        constraints = []

        is_sat, violated = solver.verify_python_constraints(code, constraints)

        assert is_sat is True
