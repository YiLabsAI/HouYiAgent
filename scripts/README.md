# Development Scripts

This directory contains scripts to help maintain code quality.

It also includes convenience scripts for starting the HouYi Studio backend and UI during development.

The canonical developer entrypoints are the `make` targets in the repo root. The scripts in this directory are the concrete wrappers those targets call.

## Script Conventions

- Shell scripts use `#!/usr/bin/env bash`.
- Python scripts use `#!/usr/bin/env python3`.

This is the current repo convention for portability across environments where the interpreter location may differ.

## Quick Reference

```bash
# Fast checks during development (recommended for frequent use)
./scripts/quick_check.sh

# Full checks before committing
./scripts/check_code.sh

# Local integration gate for env-backed tests
./scripts/check_integration.sh

# Or use Makefile (easier)
make quick-check
make check
make check-integration
```

## Pre-Commit Verification

Before committing, make sure the quick checks are green:

```bash
make quick-check
```

For a more comprehensive local check, run:

```bash
make check
```

For env-backed integration coverage after the main gate is green, run:

```bash
make check-integration
```

These commands are the canonical entrypoints. The shell scripts below are convenience wrappers.

## Startup Scripts (Development)

### All-in-one: `dev.sh`

Starts the backend and the UI in a single tmux session.

```bash
./scripts/dev.sh
```

Notes:
- This script expects `tmux` to be installed.
- The backend is started via `python -m houyi_studio.server`.
- Before backend startup, it also runs embedding warmup (`scripts/warmup_embeddings.py`).

### Split: `restart-backend.sh`

Restarts only the backend (useful when iterating on Python code).

```bash
./scripts/restart-backend.sh
```

Notes:
- This script runs embedding warmup before launching the backend.
- Default embedding cache path is `${ROOT_DIR}/.cache/fastembed`.
- You can override it via environment variable:

```bash
FASTEMBED_CACHE_PATH=/custom/cache/path ./scripts/restart-backend.sh
```

### Warmup Helper: `warmup_embeddings.py`

You can run warmup directly:

```bash
uv run python scripts/warmup_embeddings.py
```

Or run it as an executable script:

```bash
chmod +x scripts/warmup_embeddings.py
./scripts/warmup_embeddings.py
```

It logs:
- resolved embedding provider/model
- cache directory snapshots (before/after)
- warmup duration and likely cache-hit / first-download hint

### Split: `restart-frontend.sh`

Restarts only the UI (useful when iterating on frontend code).

```bash
./scripts/restart-frontend.sh
```

## Scripts

### `quick_check.sh`

**Purpose**: Fast feedback during development

**What it does**:
- ✅ Ruff auto-fix (formatting + basic linting)
- ✅ Quick test run (fail fast)

**When to use**: Run frequently during development for quick feedback

**Time**: ~5-10 seconds

### `check_code.sh`

**Purpose**: Comprehensive checks before committing

**What it does**:
- ✅ Ruff formatting and linting (source code)
- ✅ Ruff basic checks (tests)
- ✅ Mypy type checking
- ✅ Changed-file complexity and class-size gates
- ✅ SDK unit tests
- ✅ Studio server tests
- ✅ SDK coverage check (must be ≥85%)

**When to use**: Before committing code, before opening PR

**Time**: ~1-3 minutes (depends on test runtime)

### `check_integration.sh`

**Purpose**: Local-only integration gate for env-backed tests

**What it does**:
- ✅ Verifies local integration dependencies
- ✅ Runs `tests/integration/`
- ✅ Exercises real env / `.env` backed integrations, including provider-backed paths when configured

**When to use**: After `make check` is green and you need to validate integration paths that depend on local credentials, provider SDKs, or other env-backed setup

**Time**: Variable; depends on enabled integrations and network/provider latency

### `check_class_size.py`

**Purpose**: Report or gate oversized SDK classes

**What it does**:
- ✅ Evaluates SDK class size against warning/error thresholds
- ✅ Supports changed-file-only gating from `check_code.sh`

**When to use**: Usually indirectly via `make check`; run directly only when investigating class-size warnings or tuning thresholds

### `dev.sh`

**Purpose**: Start backend and frontend together in tmux

### `restart-backend.sh`

**Purpose**: Restart only the backend during Python iteration

### `restart-frontend.sh`

**Purpose**: Restart only the frontend during UI iteration

### `warmup_embeddings.py`

**Purpose**: Warm embedding runtime and cache before backend startup or provider troubleshooting

## README Maintenance Rule

- Add or update this README whenever a script becomes a recommended developer entrypoint, a quality gate wrapper, or a common troubleshooting command.
- Internal helper scripts that are only called by another script do not need full end-user documentation here.
- If a script changes the canonical local workflow, update both this README and any corresponding Makefile help text in the same change.

## Exit Codes

- `0`: All checks passed ✅
- `1`: Some checks failed ❌

## Integration with Git

### Option 1: Pre-commit Hooks (Recommended)

Automatically run checks before every commit:

```bash
make setup-hooks
```

### Option 2: Manual Checks

Run before committing:

```bash
make check
make check-integration  # when env-backed integrations are part of your change
git add .
git commit -m "your message"
```

### Option 3: Git Alias

Add to your `~/.gitconfig`:

```ini
[alias]
    check = !cd $(git rev-parse --show-toplevel) && make check
    qcheck = !cd $(git rev-parse --show-toplevel) && make quick-check
```

Then use:
```bash
git qcheck  # Quick check
git check   # Full check
```

## Troubleshooting

### "Permission denied"

Make scripts executable:
```bash
chmod +x scripts/*.sh scripts/warmup_embeddings.py
```

### "Command not found"

Ensure you have `uv` installed and `.venv` created:
```bash
uv sync --extra dev
make install-dev
```

### Checks failing

1. Read the error messages carefully
2. Fix the issues
3. Run `make quick-check` to verify
4. Run `make check` before committing
5. If your change touches env-backed or provider-backed paths, run `make check-integration`

## CI/CD

These same checks run in GitHub Actions on every push/PR. Ensure they pass locally before pushing to avoid CI failures.
