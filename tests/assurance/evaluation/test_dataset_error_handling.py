"""Error-path tests for evaluation/dataset.py."""

import tempfile
from pathlib import Path

import pytest

from houyi.assurance.evaluation.dataset import Dataset


class TestDatasetErrorHandling:
    def test_dataset_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Dataset.from_file("nonexistent.json")

    def test_dataset_from_file_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "test.txt"
            invalid_file.write_text("invalid content")

            with pytest.raises(ValueError, match="Unsupported file format"):
                Dataset.from_file(str(invalid_file))
