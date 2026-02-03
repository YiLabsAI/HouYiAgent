# Development Scripts

This directory contains scripts to help maintain code quality.

It also includes convenience scripts for starting the HouYi Studio backend and UI during development.

## Quick Reference

```bash
# Fast checks during development (recommended for frequent use)
./scripts/quick_check.sh

# Full checks before committing
./scripts/check_code.sh

# Or use Makefile (easier)
make quick-check
make check
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

These commands are the canonical entrypoints. The shell scripts below are convenience wrappers.

## Startup Scripts (Development)

### All-in-one: `dev.sh`

Starts the backend and the UI in a single tmux session.

```bash
./scripts/dev.sh
```

Notes:
- This script expects `tmux` to be installed.
- The backend is started via `python -m houyi_studio.server.app`.

### Split: `restart-backend.sh`

Restarts only the backend (useful when iterating on Python code).

```bash
./scripts/restart-backend.sh
```

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
- ✅ Pylint deep analysis (must score 10.00/10)
- ✅ All unit tests
- ✅ Coverage check (must be ≥80%)

**When to use**: Before committing code, before opening PR

**Time**: ~20-30 seconds

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
chmod +x scripts/*.sh
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

## CI/CD

These same checks run in GitHub Actions on every push/PR. Ensure they pass locally before pushing to avoid CI failures.
