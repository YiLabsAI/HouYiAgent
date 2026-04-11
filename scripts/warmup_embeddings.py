#!/usr/bin/env python3
"""Warm up local embedding model and print cache diagnostics.

This script is intended to run before starting the backend service.
It does not change server core logic; it only preloads local fastembed models
and prints cache hit/miss hints.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("embedding_warmup")


def _cache_candidates() -> list[Path]:
    explicit = os.getenv("FASTEMBED_CACHE_PATH")
    if explicit:
        return [Path(explicit).expanduser()]

    home = Path.home()
    xdg_cache_home = Path(os.getenv("XDG_CACHE_HOME", home / ".cache")).expanduser()
    hf_home = Path(os.getenv("HF_HOME", xdg_cache_home / "huggingface")).expanduser()
    hf_hub_cache = Path(os.getenv("HUGGINGFACE_HUB_CACHE", hf_home / "hub")).expanduser()

    return [
        xdg_cache_home / "fastembed",
        home / ".fastembed_cache",
        hf_hub_cache,
    ]


def _dir_stats(path: Path) -> tuple[bool, int, int]:
    if not path.exists() or not path.is_dir():
        return False, 0, 0

    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                continue
    return True, file_count, total_bytes


def _log_cache_snapshot(label: str, paths: list[Path]) -> tuple[int, int]:
    logger.info("Cache snapshot (%s)", label)
    total_files = 0
    total_bytes = 0
    for path in paths:
        exists, file_count, byte_count = _dir_stats(path)
        total_files += file_count
        total_bytes += byte_count
        logger.info(
            "  - %s | exists=%s files=%s size_bytes=%s",
            path,
            exists,
            file_count,
            byte_count,
        )
    return total_files, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Warm up local embedding model and print cache diagnostics"
    )
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        from houyi_studio.server.rag.embedding_config import resolve_embedding_config
    except Exception as exc:  # pragma: no cover - script-level guard
        logger.error("Failed to import embedding resolver: %s", exc)
        return 1

    try:
        config, provider = resolve_embedding_config(strict_explicit=True)
    except Exception as exc:
        logger.error("Embedding config resolution failed: %s", exc)
        return 1

    if not config:
        logger.info("No embedding provider resolved (%s); skipping warmup.", provider)
        return 0

    logger.info(
        "Resolved embedding provider=%s model=%s dimension=%s",
        config.provider,
        config.model,
        config.dimension,
    )

    if config.provider != "local":
        logger.info("Provider '%s' is non-local; skipping local model warmup.", config.provider)
        return 0

    cache_paths = _cache_candidates()
    logger.info(
        "Warmup target: local fastembed model '%s'",
        config.model,
    )
    logger.info("FASTEMBED_CACHE_PATH=%s", os.getenv("FASTEMBED_CACHE_PATH", "<unset>"))

    before_files, before_bytes = _log_cache_snapshot("before", cache_paths)

    try:
        from fastembed import TextEmbedding

        start = time.perf_counter()
        model = TextEmbedding(model_name=config.model)
        # Trigger an actual embedding pass so model assets are loaded eagerly.
        _ = list(model.embed(["embedding warmup"]))
        elapsed_ms = int((time.perf_counter() - start) * 1000)
    except Exception as exc:
        logger.error("Local embedding warmup failed for model '%s': %s", config.model, exc)
        return 1

    after_files, after_bytes = _log_cache_snapshot("after", cache_paths)

    if after_files > before_files or after_bytes > before_bytes:
        logger.info(
            "Warmup result: cache growth detected (likely first-time download). duration_ms=%s",
            elapsed_ms,
        )
    else:
        logger.info(
            "Warmup result: no cache growth detected (likely cache hit). duration_ms=%s",
            elapsed_ms,
        )

    logger.info("Embedding warmup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
