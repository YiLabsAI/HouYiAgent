# Development Scripts

This directory contains scripts to help maintain code quality.

It also includes convenience scripts for starting the HouYi Studio backend and UI during development.

The canonical developer entrypoints are the `make` targets in the repo root. The scripts in this directory are the concrete wrappers those targets call.

## Script Conventions

- Shell scripts use `#!/usr/bin/env bash`.
- Python scripts use `#!/usr/bin/env python3`.
- Every top-level script in this directory should support `-h` / `--help` as a self-describing entrypoint.
- Keep this README as a lightweight index. Detailed flags and examples belong to each script's own help output.

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

## Script Index

Use `-h` or `--help` on the script itself for detailed flags and examples.

| Script | Category | Purpose |
|---|---|---|
| `quick_check.sh` | Quality gate | Fast local development checks |
| `check_code.sh` | Quality gate | Main local code-quality gate behind `make check` |
| `check_integration.sh` | Quality gate | Local integration gate for env-backed coverage |
| `check_class_size.py` | Quality gate | Report or gate oversized SDK classes |
| `run_make_check.py` | Quality gate | Wrapper that enforces a global wall-clock budget for `make check` |
| `dev.sh` | Dev startup | Start backend and frontend together in tmux |
| `restart-backend.sh` | Dev startup | Restart the local backend service |
| `restart-frontend.sh` | Dev startup | Restart the local frontend dev server |
| `warmup_embeddings.py` | Dev startup | Warm local embedding runtime and print cache diagnostics |
| `run_benchmark.py` | Benchmark | Legacy HouYi benchmark runner for internal/raw benchmark generation |
| `run_benchmark_v2.py` | Benchmark | Bench II-aligned runner: generate HouYi articles, then invoke the official-style sidecar scoring flow |

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
make check-integration  # when your change touches local integration coverage
make benchmark BENCH_TARGET=memory
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
5. If your change touches local integration behavior, run `make check-integration`
6. If you are working on performance-sensitive code, run `make benchmark BENCH_TARGET=<memory|rag|runtime|verification|observability|all>` or `make benchmark BENCH_KIND=arena`

## CI/CD

These same checks run in GitHub Actions on every push/PR. Ensure they pass locally before pushing to avoid CI failures.
