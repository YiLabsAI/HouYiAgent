"""RAG integration test configuration.

Ensures the local fastembed ONNX model is cached before running tests
that fall back to local embedding. Without this, huggingface_hub will
attempt an online revision check (and possibly a ~66 MB download) which
blocks CI/sandbox environments.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _resolve_fastembed_cache() -> Path:
    """Resolve the fastembed cache directory, matching fastembed's own logic.

    Priority: $FASTEMBED_CACHE_PATH > <home>/.cache/fastembed (all platforms).
    Uses Path.home() which is cross-platform (Windows: C:\\Users\\<user>).
    """
    env = os.getenv("FASTEMBED_CACHE_PATH")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "fastembed"


_FASTEMBED_CACHE = _resolve_fastembed_cache()
_MODEL_DIR = _FASTEMBED_CACHE / "models--qdrant--bge-small-en-v1.5-onnx-q"
_MODEL_BLOB_MIN_SIZE = 60_000_000  # ~60 MB expected for the ONNX model


def _model_is_cached() -> bool:
    """Return True when the ONNX model files exist locally."""
    if not _MODEL_DIR.is_dir():
        return False
    snapshots = _MODEL_DIR / "snapshots"
    if not snapshots.is_dir():
        return False
    for rev_dir in snapshots.iterdir():
        onnx = rev_dir / "model_optimized.onnx"
        if onnx.is_file() and onnx.stat().st_size >= _MODEL_BLOB_MIN_SIZE:
            return True
    return False


@pytest.fixture(autouse=True)
def _force_offline_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent huggingface_hub from making network requests during tests.

    The local model must already be cached (via make install-all or a
    prior manual run).  Setting HF_HUB_OFFLINE=1 forces the library to
    use only cached files, avoiding hangs when the network is slow or
    sandboxed.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip RAG tests that need fastembed when the ONNX model is not cached."""
    if _model_is_cached():
        return

    skip = pytest.mark.skip(
        reason=(
            f"fastembed ONNX model not cached at {_MODEL_DIR}. "
            "Run `python -c \"from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')\"` "
            "to download once, then re-run tests."
        ),
    )
    for item in items:
        if "rag" in str(item.fspath):
            item.add_marker(skip)
