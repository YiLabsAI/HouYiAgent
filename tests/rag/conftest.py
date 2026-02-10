"""Pytest fixtures for RAG tests."""

from __future__ import annotations

import gc

import pytest


@pytest.fixture(autouse=True)
def cleanup_gc():
    """Clean up garbage after each test to close SQLite connections.

    bm25s uses Snowball Stemmer which has internal SQLite connections.
    Force GC after each test to ensure connections are closed.
    """
    yield
    gc.collect()
