# Development Scripts

This directory contains scripts to help maintain code quality.

It also includes convenience scripts for starting the HouYi Studio backend and UI during development.

The canonical developer entrypoints are the `make` targets in the repo root. The scripts in this directory are the concrete wrappers those targets call.

## Script Conventions

- Shell scripts use `#!/usr/bin/env bash`.
- Python scripts use `#!/usr/bin/env python3`.
- Every top-level script in this directory MUST support `-h` / `--help` as a self-describing entrypoint. Run a script with `-h` for its full flag list and examples — this README is a lightweight index and intentionally does not duplicate per-script detail. Enforced by `scripts/check_style_rules.py` (rule 4), which runs in `make check` and `make quick-check`.

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

See the `-h` / `--help` convention in [Script Conventions](#script-conventions) above.

| Script | Category | Purpose |
|---|---|---|
| `quick_check.sh` | Quality gate | Fast local development checks |
| `check_code.sh` | Quality gate | Main local code-quality gate behind `make check` |
| `check_integration.sh` | Quality gate | Local integration gate for env-backed coverage |
| `check_class_size.py` | Quality gate | Report or gate oversized SDK classes |
| `check_style_rules.py` | Quality gate | Enforce HouYi style conventions: reject raw CJK in source, over-long test names, and scripts/ entrypoints that drop `-h`/`--help`; warn on dense `\uXXXX` escapes without an ASCII pinyin/English gloss |
| `run_make_check.py` | Quality gate | Wrapper that enforces a global wall-clock budget for `make check` |
| `dev.sh` | Dev startup | Start backend and frontend together in tmux |
| `restart-backend.sh` | Dev startup | Restart the local backend service |
| `restart-frontend.sh` | Dev startup | Restart the local frontend dev server |
| `warmup_embeddings.py` | Dev startup | Warm local embedding runtime and print cache diagnostics |
| `run_benchmark.py` | Benchmark | Legacy HouYi benchmark runner for internal/raw benchmark generation |
| `run_benchmark_v2.py` | Benchmark | Bench II-aligned runner: generate HouYi articles, then invoke the official-style sidecar scoring flow |

## Bench II Probe Quick Guide

Use `scripts/run_benchmark_v2.py` when you want a reproducible DeepResearch Bench II probe without wiring the official shell scripts by hand.

This section is written for **user-facing reproducibility** rather than internal smoke checks.

### Recommended Commands

```bash
# 1) Reproducible adapted_env run with fresh generation
uv run python scripts/run_benchmark_v2.py \
  --limit 100 \
  --depth deep \
  --mode delegate \
  --target-model houyi \
  --bench-runtime adapted_env

# 2) Re-score an existing adapted_env raw_data export
uv run python scripts/run_benchmark_v2.py \
  --skip-generate \
  --limit 100 \
  --target-model houyi \
  --bench-runtime adapted_env \
  --raw-data-path benchmark/output/<run>/raw_data/houyi.jsonl

# 3) Official-runtime re-score (only when upstream keys are available)
uv run python scripts/run_benchmark_v2.py \
  --skip-generate \
  --limit 100 \
  --target-model houyi \
  --bench-runtime official \
  --raw-data-path benchmark/output/<run>/raw_data/houyi.jsonl
```

### Runtime Expectations

- `--limit 100` is a realistic reproducibility setting for users who want a meaningful Bench II readout without committing to a full run.
- FACT stages can take a long time, especially `fact_scrape` and `fact_validate`; plan for a materially longer wall-clock time than unit or integration checks.
- If you need the strongest confidence, run the full set instead of a capped sample.

### What To Read After A Probe

| Artifact | Why it matters |
|---|---|
| `<output-root>/houyi.summary.json` | Machine-readable run summary: arguments, steps, RACE/FACT aggregate scores, artifact paths |
| `<output-root>/houyi.bench2.log` | End-to-end stage log with timing and failure context |
| `<output-root>/fact/houyi/fact_result.txt` | Final FACT aggregate numbers |
| `<output-root>/race/houyi/race_result.txt` | Final RACE aggregate numbers |

### Interpretation Notes

- `adapted_env` keeps the official stage topology but uses HouYi's local compatibility layer; it is useful for fast local regression and probe comparison.
- `official` is closer to the public benchmark dependency path, but requires the upstream evaluator keys and should be treated as the stronger leaderboard-facing check.
- Use `--limit 100` when you want a practical user-facing reproduction command that still controls cost.
- Use the full set when you need the strongest comparison signal and can afford the additional runtime.

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
