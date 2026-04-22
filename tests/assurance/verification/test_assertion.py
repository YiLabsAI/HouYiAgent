"""Tests for AssertionSpec."""

from houyi.assurance.verification.assertion import AssertionSpec


class TestAssertionEvaluate:
    """Test AssertionSpec.evaluate() method."""

    def test_evaluate_with_callable_true(self):
        """Test evaluate with callable condition returning True."""

        def check_positive(context):
            return context.get("value", 0) > 0

        assertion = AssertionSpec(name="positive_check", condition=check_positive)

        assert assertion.evaluate({"value": 10}) is True

    def test_evaluate_with_callable_false(self):
        """Test evaluate with callable condition returning False."""

        def check_positive(context):
            return context.get("value", 0) > 0

        assertion = AssertionSpec(name="positive_check", condition=check_positive)

        assert assertion.evaluate({"value": -5}) is False

    def test_string_expression_true(self):
        """Test evaluate with string expression returning True."""
        assertion = AssertionSpec(name="cost_check", condition="cost < 1.0")

        assert assertion.evaluate({"cost": 0.5}) is True

    def test_string_expression_false(self):
        """Test evaluate with string expression returning False."""
        assertion = AssertionSpec(name="cost_check", condition="cost < 1.0")

        assert assertion.evaluate({"cost": 1.5}) is False

    def test_evaluate_with_safe_functions(self):
        """Test evaluate with safe built-in functions."""
        assertion = AssertionSpec(name="length_check", condition="len(text) > 10")

        assert assertion.evaluate({"text": "short"}) is False
        assert assertion.evaluate({"text": "this is a long text"}) is True

    def test_evaluate_with_int_function(self):
        """Test evaluate with int() function."""
        assertion = AssertionSpec(name="int_check", condition="int(value) > 5")

        assert assertion.evaluate({"value": "10"}) is True
        assert assertion.evaluate({"value": "3"}) is False

    def test_evaluate_with_float_function(self):
        """Test evaluate with float() function."""
        assertion = AssertionSpec(name="float_check", condition="float(value) > 5.5")

        assert assertion.evaluate({"value": "10.5"}) is True
        assert assertion.evaluate({"value": "3.5"}) is False

    def test_evaluate_with_min_max(self):
        """Test evaluate with min/max functions."""
        assertion = AssertionSpec(
            name="range_check", condition="min(values) > 0 and max(values) < 100"
        )

        assert assertion.evaluate({"values": [1, 50, 99]}) is True
        assert assertion.evaluate({"values": [0, 50, 99]}) is False

    def test_evaluate_with_abs(self):
        """Test evaluate with abs() function."""
        assertion = AssertionSpec(name="abs_check", condition="abs(value) < 10")

        assert assertion.evaluate({"value": -5}) is True
        assert assertion.evaluate({"value": 15}) is False

    def test_evaluate_with_dangerous_import(self):
        """Test evaluate rejects dangerous import."""
        assertion = AssertionSpec(name="dangerous", condition="import os")

        result = assertion.evaluate({})
        assert result is False

    def test_evaluate_with_dangerous_exec(self):
        """Test evaluate rejects dangerous exec."""
        assertion = AssertionSpec(name="dangerous", condition="exec('print(1)')")

        result = assertion.evaluate({})
        assert result is False

    def test_evaluate_with_dangerous_eval(self):
        """Test evaluate rejects dangerous eval."""
        assertion = AssertionSpec(name="dangerous", condition="eval('1+1')")

        result = assertion.evaluate({})
        assert result is False

    def test_evaluate_with_dangerous_dunder(self):
        """Test evaluate rejects dangerous dunder methods."""
        assertion = AssertionSpec(name="dangerous", condition="__import__('os')")

        result = assertion.evaluate({})
        assert result is False

    def test_evaluate_with_exception(self):
        """Test evaluate handles exceptions gracefully."""
        assertion = AssertionSpec(name="error", condition="undefined_var > 0")

        result = assertion.evaluate({})
        assert result is False

    def test_evaluate_with_invalid_syntax(self):
        """Test evaluate handles invalid syntax."""
        assertion = AssertionSpec(name="error", condition="x >")

        result = assertion.evaluate({"x": 5})
        assert result is False


class TestAssertionProperties:
    """Test AssertionSpec properties."""

    def test_assertion_name(self):
        """Test assertion name property."""
        assertion = AssertionSpec(name="test_assertion", condition="x > 0")

        assert assertion.name == "test_assertion"

    def test_assertion_condition_string(self):
        """Test assertion condition as string."""
        assertion = AssertionSpec(name="test", condition="x > 0")

        assert assertion.condition == "x > 0"

    def test_assertion_condition_callable(self):
        """Test assertion condition as callable."""

        def check(context):
            return True

        assertion = AssertionSpec(name="test", condition=check)

        assert callable(assertion.condition)

    def test_assertion_on_failure_default(self):
        """Test assertion on_failure default value."""
        assertion = AssertionSpec(name="test", condition="x > 0")

        assert assertion.on_failure == "abort"

    def test_assertion_on_failure_retry(self):
        """Test assertion on_failure set to retry."""
        assertion = AssertionSpec(name="test", condition="x > 0", on_failure="retry")

        assert assertion.on_failure == "retry"

    def test_assertion_on_failure_human(self):
        """Test assertion on_failure set to human."""
        assertion = AssertionSpec(name="test", condition="x > 0", on_failure="human")

        assert assertion.on_failure == "human"

    def test_assertion_with_metadata(self):
        """Test assertion with metadata."""
        assertion = AssertionSpec(
            name="test", condition="x > 0", metadata={"priority": "high", "category": "validation"}
        )

        assert assertion.metadata["priority"] == "high"
        assert assertion.metadata["category"] == "validation"

    def test_assertion_without_metadata(self):
        """Test assertion without metadata."""
        assertion = AssertionSpec(name="test", condition="x > 0")

        assert assertion.metadata == {}
