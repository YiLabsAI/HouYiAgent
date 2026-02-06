"""Tests for evaluation/dataset.py"""

import csv
import json
import tempfile
from pathlib import Path

from houyi.evaluation.dataset import Dataset
from houyi.evaluation.dataset import TestCase as EvalTestCase


def test_testcase_creation():
    """Test TestCase creation."""
    tc = EvalTestCase(
        input="What is 2+2?", expected_output="4", metadata={"category": "math", "id": "test1"}
    )

    assert tc.input == "What is 2+2?"
    assert tc.expected_output == "4"
    assert tc.metadata["category"] == "math"
    assert tc.metadata["id"] == "test1"


def test_testcase_without_expected():
    """Test TestCase without expected output."""
    tc = EvalTestCase(input="Tell me a joke", metadata={"id": "test2"})

    assert tc.input == "Tell me a joke"
    assert tc.expected_output is None


def test_dataset_creation():
    """Test Dataset creation."""
    cases = [
        EvalTestCase(input="test1", metadata={"id": "1"}),
        EvalTestCase(input="test2", metadata={"id": "2"}),
    ]

    dataset = Dataset(name="Test Dataset", test_cases=cases, metadata={"version": "1.0"})

    assert dataset.name == "Test Dataset"
    assert len(dataset.test_cases) == 2
    assert dataset.metadata["version"] == "1.0"


def test_dataset_from_json():
    """Test loading dataset from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "test.json"

        data = {
            "name": "JSON Dataset",
            "test_cases": [
                {"input": "test1", "expected_output": "output1", "metadata": {"id": "1"}},
                {"input": "test2", "metadata": {"id": "2"}},
            ],
        }

        with open(json_file, "w") as f:
            json.dump(data, f)

        dataset = Dataset.from_file(str(json_file))

        assert dataset.name == "JSON Dataset"
        assert len(dataset.test_cases) == 2
        assert dataset.test_cases[0].input == "test1"
        assert dataset.test_cases[0].expected_output == "output1"


def test_dataset_from_csv():
    """Test loading dataset from CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "expected_output"])
            writer.writerow(["test1", "output1"])
            writer.writerow(["test2", ""])

        dataset = Dataset.from_file(str(csv_file))

        assert len(dataset.test_cases) == 2
        assert dataset.test_cases[0].input == "test1"
        assert dataset.test_cases[0].expected_output == "output1"


def test_dataset_to_json():
    """Test saving dataset to JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "output.json"

        dataset = Dataset(
            name="Output Dataset",
            test_cases=[
                EvalTestCase(input="test1", expected_output="output1"),
                EvalTestCase(input="test2"),
            ],
        )

        dataset.to_file(str(json_file))

        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)

        assert data["name"] == "Output Dataset"
        assert len(data["test_cases"]) == 2


def test_dataset_to_csv():
    """Test saving dataset to CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "output.csv"

        dataset = Dataset(
            name="CSV Dataset",
            test_cases=[
                EvalTestCase(input="test1", expected_output="output1"),
                EvalTestCase(input="test2"),
            ],
        )

        dataset.to_file(str(csv_file))

        assert csv_file.exists()

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["input"] == "test1"


def test_dataset_iteration():
    """Test iterating over dataset."""
    dataset = Dataset(
        name="Iter Dataset",
        test_cases=[
            EvalTestCase(input="test1"),
            EvalTestCase(input="test2"),
            EvalTestCase(input="test3"),
        ],
    )

    count = 0
    for case in dataset:
        count += 1
        assert isinstance(case, EvalTestCase)

    assert count == 3


def test_dataset_length():
    """Test dataset length."""
    dataset = Dataset(
        name="Length Dataset",
        test_cases=[EvalTestCase(input="test1"), EvalTestCase(input="test2")],
    )

    assert len(dataset) == 2


def test_dataset_getitem():
    """Test dataset indexing."""
    dataset = Dataset(
        name="Index Dataset",
        test_cases=[
            EvalTestCase(input="test1", expected_output="output1", metadata={"id": "1"}),
            EvalTestCase(input="test2", metadata={"id": "2"}),
        ],
    )

    assert dataset[0].input == "test1"
    assert dataset[1].input == "test2"
    assert dataset[-1].input == "test2"


def test_dataset_empty():
    """Test empty dataset."""
    dataset = Dataset(name="Empty Dataset", test_cases=[])

    assert len(dataset) == 0
    assert list(dataset) == []


def test_testcase_with_context():
    """Test testcase with context."""
    testcase = EvalTestCase(
        input={"query": "test"}, expected={"answer": "result"}, context={"source": "test_source"}
    )

    assert testcase.context == {"source": "test_source"}


def test_dataset_add_testcase():
    """Test adding testcase to dataset."""
    dataset = Dataset(name="test", test_cases=[])
    testcase = EvalTestCase(input="What is 1+1?", expected_output="2")

    # Manually add to test_cases list
    dataset.test_cases.append(testcase)

    assert len(dataset) == 1
    assert dataset[0] == testcase


def test_dataset_slice():
    """Test slicing dataset."""
    dataset = Dataset(
        name="test",
        test_cases=[
            EvalTestCase(input="Question 1", expected_output="Answer 1"),
            EvalTestCase(input="Question 2", expected_output="Answer 2"),
            EvalTestCase(input="Question 3", expected_output="Answer 3"),
        ],
    )

    # Test that we can iterate and access by index
    assert len(dataset) == 3
    assert dataset[0].input == "Question 1"
    assert dataset[2].input == "Question 3"

    # Test iteration
    inputs = [tc.input for tc in dataset]
    assert len(inputs) == 3
    assert inputs[0] == "Question 1"


def test_dataset_filter():
    """Test filtering dataset."""
    dataset = Dataset(
        name="test",
        test_cases=[
            EvalTestCase(input="Q1", metadata={"category": "math"}),
            EvalTestCase(input="Q2", metadata={"category": "science"}),
            EvalTestCase(input="Q3", metadata={"category": "math"}),
        ],
    )

    math_cases = [tc for tc in dataset if tc.metadata.get("category") == "math"]
    assert len(math_cases) == 2


def test_dataset_from_list():
    """Test creating dataset from list of dicts."""
    data = [
        {"input": "Q1", "expected_output": "A1"},
        {"input": "Q2", "expected_output": "A2"},
    ]

    cases = [EvalTestCase(**item) for item in data]
    dataset = Dataset(name="test", test_cases=cases)

    assert len(dataset) == 2
    assert dataset[0].expected_output == "A1"


def test_testcase_repr():
    """Test TestCase string representation."""
    tc = EvalTestCase(input="test input", expected_output="test output")

    repr_str = repr(tc)
    assert "test input" in repr_str or "TestCase" in repr_str


def test_dataset_filter():
    """Test filtering dataset."""
    dataset = Dataset(
        name="test",
        test_cases=[
            EvalTestCase(input="Q1", metadata={"category": "math"}),
            EvalTestCase(input="Q2", metadata={"category": "science"}),
            EvalTestCase(input="Q3", metadata={"category": "math"}),
        ],
    )

    math_cases = [tc for tc in dataset if tc.metadata.get("category") == "math"]
    assert len(math_cases) == 2


def test_dataset_from_list():
    """Test creating dataset from list of dicts."""
    data = [
        {"input": "Q1", "expected_output": "A1"},
        {"input": "Q2", "expected_output": "A2"},
    ]

    cases = [EvalTestCase(**item) for item in data]
    dataset = Dataset(name="test", test_cases=cases)

    assert len(dataset) == 2
    assert dataset[0].expected_output == "A1"


def test_testcase_repr():
    """Test TestCase string representation."""
    tc = EvalTestCase(input="test input", expected_output="test output")

    repr_str = repr(tc)
    assert "test input" in repr_str or "TestCase" in repr_str


def test_testcase_with_context():
    """Test TestCase with context."""
    tc = EvalTestCase(
        input="Question",
        expected_output="Answer",
        context="Background information",
        metadata={"has_context": True, "id": "ctx1"},
    )

    assert tc.context == "Background information"
    assert tc.metadata["has_context"] is True
    assert tc.metadata["id"] == "ctx1"
