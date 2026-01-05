# Development Scripts

This directory contains scripts to help maintain code quality.

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
- ✅ Coverage check (must be ≥75%)

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

Ensure you're in the conda environment:
```bash
conda activate houyi
make install-dev
```

### Checks failing

1. Read the error messages carefully
2. Fix the issues
3. Run `make quick-check` to verify
4. Run `make check` before committing

## CI/CD

These same checks run in GitHub Actions on every push/PR. Ensure they pass locally before pushing to avoid CI failures.
