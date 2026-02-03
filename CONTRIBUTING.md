# Contributing to HouYi

Thank you for your interest in contributing to HouYi.

This document is intentionally minimal. The single source of truth for engineering rules is:
- [agent.md](agent.md)

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/houyi.git
cd houyi

uv python install 3.11
uv sync --extra dev

make install-dev
make setup-hooks
```

## Contribution Loop

1. Create a branch.
2. Implement changes + tests.
3. Run local checks:
   ```bash
   make check
   ```
4. Open a PR against `main`.

## Quality Gates (Required)

Follow the exact requirements in [agent.md](agent.md). In particular:
- Coverage gate is ≥80%.
- Pylint target score is 10.00/10.

## Useful Makefile Commands

```bash
make help             # Show all available commands
make quick-check      # Fast checks during development
make check            # Full checks before commit
make format           # Auto-format code
make lint             # Run all linters
make test             # Run all tests
make test-cov         # Run tests with coverage
make clean            # Clean cache files
```

## Questions

Open an issue for questions or discussions.
