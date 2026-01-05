"""Dataset support for batch evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """Single test case for evaluation."""

    input: str = Field(..., description="Input text")
    expected_output: str | None = Field(None, description="Expected output")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Optional fields for specific evaluators
    expected_skills: list[str] = Field(
        default_factory=list, description="Expected skills to be used"
    )
    context: str | None = Field(None, description="Context for RAG scenarios")
    context_chunks: list[str] = Field(default_factory=list, description="Context chunks for RAG")


class Dataset(BaseModel):
    """Dataset for batch evaluation.

    Supports loading from JSON, CSV, and YAML files.
    """

    name: str = Field(..., description="Dataset name")
    description: str | None = Field(None, description="Dataset description")
    test_cases: list[TestCase] = Field(default_factory=list, description="List of test cases")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")

    @classmethod
    def from_file(cls, path: str) -> Dataset:
        """Load dataset from file.

        Supports JSON, CSV, and YAML formats.

        Args:
            path: Path to dataset file

        Returns:
            Dataset instance

        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return cls._from_json(file_path)
        elif suffix == ".csv":
            return cls._from_csv(file_path)
        elif suffix in [".yaml", ".yml"]:
            return cls._from_yaml(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported: .json, .csv, .yaml, .yml"
            )

    @classmethod
    def _from_json(cls, path: Path) -> Dataset:
        """Load dataset from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array of test cases and full dataset format
        if isinstance(data, list):
            # Simple array of test cases
            test_cases = [TestCase(**case) for case in data]
            return cls(name=path.stem, test_cases=test_cases)
        else:
            # Full dataset format
            test_cases = [TestCase(**case) for case in data.get("test_cases", [])]
            return cls(
                name=data.get("name", path.stem),
                description=data.get("description"),
                test_cases=test_cases,
                metadata=data.get("metadata", {}),
            )

    @classmethod
    def _from_csv(cls, path: Path) -> Dataset:
        """Load dataset from CSV file.

        Expected columns: input, expected_output, metadata (optional JSON)
        """
        test_cases = []

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse metadata if present
                metadata = {}
                if "metadata" in row and row["metadata"]:
                    try:
                        metadata = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        pass

                test_case = TestCase(
                    input=row["input"],
                    expected_output=row.get("expected_output"),
                    metadata=metadata,
                )
                test_cases.append(test_case)

        return cls(name=path.stem, test_cases=test_cases)

    @classmethod
    def _from_yaml(cls, path: Path) -> Dataset:
        """Load dataset from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML files. Install with: pip install pyyaml"
            ) from e

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Handle both array and full format
        if isinstance(data, list):
            test_cases = [TestCase(**case) for case in data]
            return cls(name=path.stem, test_cases=test_cases)
        else:
            test_cases = [TestCase(**case) for case in data.get("test_cases", [])]
            return cls(
                name=data.get("name", path.stem),
                description=data.get("description"),
                test_cases=test_cases,
                metadata=data.get("metadata", {}),
            )

    def to_file(self, path: str, format: str | None = None) -> None:
        """Save dataset to file.

        Args:
            path: Path to save dataset
            format: File format (json, csv, yaml). If None, inferred from path extension
        """
        file_path = Path(path)

        if format is None:
            format = file_path.suffix.lower().lstrip(".")

        if format == "json":
            self._to_json(file_path)
        elif format == "csv":
            self._to_csv(file_path)
        elif format in ["yaml", "yml"]:
            self._to_yaml(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _to_json(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "test_cases": [case.model_dump() for case in self.test_cases],
            "metadata": self.metadata,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _to_csv(self, path: Path) -> None:
        """Save dataset to CSV file."""
        if not self.test_cases:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["input", "expected_output", "metadata"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for case in self.test_cases:
                writer.writerow(
                    {
                        "input": case.input,
                        "expected_output": case.expected_output or "",
                        "metadata": json.dumps(case.metadata) if case.metadata else "",
                    }
                )

    def _to_yaml(self, path: Path) -> None:
        """Save dataset to YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to save YAML files. Install with: pip install pyyaml"
            ) from e

        data = {
            "name": self.name,
            "description": self.description,
            "test_cases": [case.model_dump() for case in self.test_cases],
            "metadata": self.metadata,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def __len__(self) -> int:
        """Return number of test cases."""
        return len(self.test_cases)

    def __getitem__(self, index: int) -> TestCase:
        """Get test case by index."""
        return self.test_cases[index]

    def __iter__(self):
        """Iterate over test cases."""
        return iter(self.test_cases)
