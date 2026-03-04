# HouYi Development Guidelines

This document defines architectural principles, development standards, and best practices for the HouYi project.

## Table of Contents

1. [Architectural Principles](#architectural-principles)
2. [Repository Layout](#repository-layout)
3. [Development Environment](#development-environment)
4. [Development Workflow](#development-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Collaboration](#collaboration)
8. [Version Management](#version-management)
9. [Best Practices](#best-practices)

---

## Architectural Principles

These are non-negotiable design principles for HouYi. Violating them requires explicit approval and migration plan.

### 1. Plan First

Industrial-grade distributed system design: always plan before implementation.

- **Design docs required** for features touching multiple modules
- **Non-functional requirements** must have measurable targets:

| Metric | Target |
|--------|--------|
| API Latency (P99) | < 500ms |
| Test Coverage (Core) | ≥ 85% |
| Test Coverage (Overall) | ≥ 80% |
| Ruff | Zero violations |

- **Deterministic execution**: LLM produces plans or formal artifacts (code/SQL), not execute business logic directly

### 2. Schema First

All public boundaries must be explicitly modeled and validated.

- Inputs/outputs defined with Pydantic v2 schemas
- No untyped `dict`/`Any` across module boundaries
- API contracts versioned and backward-compatible

### 3. Data-Driven & Self-Evolving

Data is the primary driver of system evolution.

- **Metrics-driven decisions**: features backed by observability data
- **Evaluation-first**: new capabilities require measurable evaluation criteria
- **Feedback loops**: runtime metrics inform optimization priorities

### 4. Immutable State

Prefer immutable state snapshots over in-place mutation.

- State transitions produce new versions, not modify existing
- Enables time-travel debugging and reliable replay
- Critical for distributed consistency and checkpoint/restore

### 5. Full-Stack Observability

Every critical path must be traceable end-to-end.

- **Tracing**: OpenTelemetry-compatible spans for all operations
- **Logging**: Structured logs with correlation IDs
- **Metrics**: Latency, throughput, error rates exported
- **No silent failures**: tool execution errors must surface with context

### 6. Security by Default

- No `eval()` / `exec()` / `pickle` on user-controlled input
- Explicit filesystem/network side effects
- Resource cleanup guaranteed (files, sockets, subprocesses)
- Permission boundaries enforced for skill/tool execution

### 7. Documentation as Code

Public APIs must be self-documenting for external developers.

- **Docstrings required** for all public classes, methods, and functions
- **Type hints** are mandatory (enforced by mypy)
- **`__init__.py` exports** define the public surface; document each export
- **Examples** in docstrings for complex APIs
- **CHANGELOG** updated for any public API changes

### 8. Architecture Artifacts

Major features must include design documentation with visual artifacts:

| Artifact | When Required |
|----------|---------------|
| **Class Diagram** | New module or significant refactor |
| **Sequence Diagram** | Cross-module workflows, async flows |
| **Component Diagram** | System integration, service boundaries |
| **ER Diagram** | Database schema changes |
| **Dependency Graph** | New external dependencies |

Location: `docs/design/<feature>_design.md` (Mermaid preferred for version control)

### 9. Dependency Governance

Treat dependencies as critical infrastructure to avoid "dependency hell".

- **Minimal dependencies**: prefer stdlib over external packages
- **Version pinning**: use `>=x.y,<x+1` ranges, never `*`
- **Review required**: all `pyproject.toml` changes require explicit approval
- **Justification**: new dependencies must document:
  - Why needed (no stdlib alternative?)
  - License compatibility (MIT/Apache preferred)
  - Maintenance status (active, security updates?)
  - Transitive dependency impact
- **Optional extras**: heavy dependencies (ML frameworks, DB drivers) as optional extras
- **Audit**: periodic `uv tree` review for transitive bloat

### 10. Bug-Driven Testing

While full TDD is not mandated, bug fixes follow the red-green-refactor cycle:

1. **Run tests** → Identify failure (red)
2. **Fix the bug** → Tests pass (green)
3. **Refactor** → Clean up with confidence

When a bug is found without test coverage:
- **Analyze**: Should this case have been covered?
- **Add test**: Write a failing test that reproduces the bug first
- **Then fix**: Proceed with the fix

Tests enable safe refactoring; prioritize coverage for code that changes frequently.

---

## Repository Layout

HouYi is a monorepo with multiple subsystems.

- **Python core**: `houyi/`
- **Console UI**: `houyi-studio/ui/`
- **Automation / CI**: `scripts/`, `.github/`

**Policy**:
- Changes MUST respect subsystem boundaries
- Cross-subsystem changes SHOULD be split into small, reviewable commits with regression coverage

---

## Development Environment

### Virtual Environment (Required)

HouYi uses `uv` + a local virtual environment at `.venv/` for isolated and reproducible development.

**Policy**:
- Development MUST run inside `.venv`
- Dependency resolution MUST use `pyproject.toml` + `uv.lock` as the single source of truth

```bash
# Install uv (see https://docs.astral.sh/uv/)

# Use Python 3.11 by default
uv python install 3.11

# Create/sync the virtualenv and install dev dependencies
uv sync --extra dev
```

### Development Tools

| Tool | Purpose |
|------|---------|
| Python | 3.11+ |
| Environment | `.venv` (required) |
| Package manager | `uv` |
| Dependencies | `pyproject.toml` + `uv.lock` |
| Lint/format | `ruff` |
| Type checking | `mypy` |
| Tests | `pytest` + `pytest-asyncio` + `pytest-cov` |

### Frontend Tools (Console UI)

Console UI lives in `houyi-studio/ui/`.

**Policy**:
- Package manager MUST be `pnpm@9` (via `corepack`)
- MUST NOT use `npm` in scripts, docs, or CI

```bash
# In houyi-studio/ui/
corepack enable
corepack prepare pnpm@9 --activate
pnpm install
```

---

## Development Workflow

All commands MUST be run inside `.venv` via `uv run` (or through Makefile targets).

### Dependency Profiles

HouYi has multiple dependency profiles for different scenarios. Using the wrong
profile causes `ModuleNotFoundError` at runtime — the most common source of
"it worked on main but broke after merge" issues.

| Scenario | What to install | One-liner |
|----------|----------------|-----------|
| **SDK development** | Core SDK + dev tools | `make install-dev` |
| **Console development** | SDK + RAG + Studio server + UI | `make install-all` (single `uv sync --extra dev --extra rag`) |
| **Run backend only** | SDK + RAG + Studio server | `uv sync --extra dev --extra rag && uv pip install -e houyi-studio/server` |
| **Run frontend only** | UI node_modules | `cd houyi-studio/ui && pnpm install` |
| **Unit tests** | SDK + dev tools | `make install-dev && make test` |
| **Integration tests** | SDK + Studio server | `make install-dev && make test-integration` |
| **E2E tests** | Full stack + Playwright | `make install-all && pnpm run e2e:install-browsers && make test-e2e` |
| **PyPI release** | Production deps only | `uv sync && uv build` |

**Key invariant**: `uv sync` manages the root `pyproject.toml` deps but does NOT
install `houyi-studio/server` (it's a separate package). After every `uv sync`,
you MUST re-run `uv pip install -e houyi-studio/server` if you need the backend.
The scripts (`dev.sh`, `restart-backend.sh`) handle this automatically.

### HouYi Studio Server

**Policy**:
- MUST install the Studio server as an installed package into `.venv`.
- MUST NOT start the server by injecting `PYTHONPATH` (this bypasses dependency governance
  and can hide missing dependencies).

**Install (one-time)**:

```bash
# From repo root — full setup (recommended)
make install-all

# Or manually:
uv sync --extra dev --extra rag
uv pip install -e houyi-studio/server
cd houyi-studio/ui && pnpm install --frozen-lockfile
```

**Start server (standard)**:

```bash
# Recommended: auto-installs deps if missing
./scripts/restart-backend.sh

# Or via Makefile (starts backend + frontend via tmux)
make dev

# Or manually:
uv run python -m houyi_studio.server
```

### Makefile Commands (Recommended)

```bash
# Setup
make install-dev      # Install SDK + dev dependencies
make install-studio   # Install Studio server + UI deps
make install-all      # Full setup (SDK + RAG + Studio + UI)
make setup-hooks      # Setup pre-commit hooks

# Development
make dev              # Start backend + frontend (tmux)

# Code Quality (use before committing!)
make quick-check      # Fast checks (ruff + quick tests)
make check            # Full checks (ruff + tests + coverage)
make format           # Auto-format code
make lint             # Run all linters
make lint-fix         # Run linters with auto-fix

# Testing
make test             # Run SDK unit tests
make test-server      # Run Studio server tests
make test-cov         # Run tests with coverage report
make test-fast        # Run tests (fail fast)
make test-integration # Run integration tests (auto-installs studio deps)
make test-e2e         # Run Playwright e2e tests

# Cleanup
make clean            # Remove cache and build files
make help             # Show all available commands
```

### Post-Merge Checklist

After rebasing or merging branches, MUST run:

```bash
# 1. Re-sync all deps (merge may have changed pyproject.toml)
make install-all

# 2. Run full regression
make test

# 3. Verify backend starts
./scripts/restart-backend.sh
```

### Manual Commands

```bash
# Lint (Ruff)
uv run ruff check houyi/
uv run ruff check houyi/ --fix

# Type check
uv run mypy houyi/

# Run tests with coverage
uv run pytest tests/ -v --cov=houyi

# Run specific test
uv run pytest tests/test_core.py -v

# Run example
uv run python examples/quickstart.py
```

### Frontend Commands

All UI commands MUST be run from `houyi-studio/ui/`.

```bash
# Install deps
pnpm install

# Install Playwright browsers (one-time)
pnpm run e2e:install-browsers

# Lint / typecheck / unit tests
pnpm lint
pnpm type-check
pnpm test

# E2E tests
pnpm test:e2e

# If browser download blocked, use system Chrome:
HOUYI_USE_SYSTEM_CHROME=1 pnpm test:e2e
```

### Pre-commit Hooks

**One-time setup:**

```bash
make setup-hooks
```

Hooks automatically run on `git commit`:
- Ruff formatting and linting
- Trailing whitespace removal
- File encoding checks
- YAML/JSON validation
- Large file detection

**Skip hooks (emergency only):**

```bash
git commit --no-verify -m "emergency fix"
```

---

## Coding Standards

### Code Quality Tools

HouYi uses a two-tier approach:

#### Ruff (Linting & Formatting)

- **Purpose**: Fast code formatting and comprehensive error detection
- **Usage**: `ruff check houyi/` or `ruff check houyi/ --fix`
- **Checks**: PEP 8, import sorting, unused variables, anti-patterns, complexity thresholds
- **Speed**: Milliseconds

**Run before committing:**

```bash
ruff check houyi/ --fix
```

### API Stability

- **Preserve public interfaces**: No signature changes without migration guidance
- **New parameters**: Use keyword-only: `def f(a: int, *, new_param: str = "default")`
- **Public surface**: Items exported from `__init__.py` and documented APIs

### Error Handling

- Validate early at boundaries
- Raise structured, actionable errors
- No bare `except:`
- Rethrow tool execution failures with context

---

## Testing

### Test Categories

| Category | Description |
|----------|-------------|
| **Unit** | Fast, deterministic, no network |
| **Integration** | May include network/external services |
| **E2E (Playwright)** | Full UI workflow testing |

### Test Location

- **All tests in `tests/` directory** (never in project root)
- Mirror source layout: `houyi/core/agent.py` → `tests/core/test_agent.py`

### Naming Convention

✅ **Correct**: `test_<class_or_module>.py`
- `tests/evaluation/test_evaluators.py`
- `tests/execution/test_skill_executor.py`
- `tests/observability/test_observability.py`

❌ **Incorrect**:
- `test_phase3_evaluators.py` (phase-based)
- `test_advanced_features.py` (vague)
- `test_v0_2_0.py` (version-based)

### Test Structure

```python
class TestAccuracyEvaluator:
    def test_exact_match(self):
        ...

    def test_similar_match(self):
        ...
```

- Use fixtures/mocks for external dependencies
- Use `pytest.mark` for categorization
- Group tests by class

### Coverage Requirements

| Scope | Target |
|-------|--------|
| Overall project | ≥ 80% |
| Core business logic | ≥ 85% |

Core modules: `core/`, `evaluation/`, `execution/`, `orchestration/`, `observability/`

### E2E Testing (Playwright)

**Policy**:
- Tests categorized as `smoke` or `full`
- PRs run `smoke` by default
- `full` triggered by: PR label `e2e-full`, nightly schedule, or manual dispatch

**Smoke scope**: UI startup, home page render, `position_test` workflow

---

## Collaboration

### PR Checklist

- [ ] Schemas/types updated; validations added
- [ ] No breaking public API changes (or migration notes provided)
- [ ] `make check` passes (ruff, pytest)
- [ ] New critical paths include tracing/logging
- [ ] Tool execution respects sandbox/permission boundaries

### GitHub Rules

- **Issues**: Use GitHub Issue Forms for reproducible bug reports
- **PRs**: Follow repository PR template
- **Stale management**: Automated workflow labels/closes inactive issues

### Translator Workflow

- Automated translator normalizes issue titles to English
- Includes anti-abuse measures (idempotency, rate limiting)
- Ignores bot activity

---

## Version Management

HouYi follows [Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`

### When to Increment

| Type | When |
|------|------|
| **MAJOR** | Breaking changes to public APIs |
| **MINOR** | New features (backward-compatible) |
| **PATCH** | Bug fixes, no API changes |

### Release Requirements

Before any release:

1. **Coverage**: Overall ≥ 80%, Core ≥ 85%
2. **Tests**: 100% pass rate
3. **Quality**: `ruff` and `mypy` pass (zero violations)
4. **Docs**: CHANGELOG.md updated, README.md current

### Release Process

1. Update `CHANGELOG.md`
2. Update `pyproject.toml` version
3. Run full test suite: `make test-cov`
4. Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Update `README.md` if major/minor release

---

## Best Practices

### Linear Commit Workflow

Maintain linear commit history in multi-contributor environments.

**Standard Workflow:**

```bash
# 1. Stash local changes (tracked files only)
git stash push -m "WIP: description"

# 2. Pull with rebase (keeps linear history)
git pull --rebase origin main

# 3. Restore local changes
git stash pop

# 4. Squash local commits into one (if needed)
git reset --soft origin/main

# 5. Stage and commit
git add .
git commit -m "feat: description"

# 6. Push
git push origin main
```

**Handling Untracked Files:**

```bash
# Include untracked files (use with caution)
git stash push -u -m "WIP: including untracked"
```

**Caution with `-u` flag:**
- Build artifacts may be accidentally stashed
- Large binaries could bloat stash
- Files intended to stay untracked may be removed

**Best Practice for Untracked Files:**
1. **Preferred**: Add to `.gitignore` or explicitly stage before stash
2. **If using `-u`**: Review `git status` first
3. **Alternative**: Commit WIP to a temporary branch instead

**When Conflicts Occur:**

```bash
git stash pop
# Resolve conflicts manually
git add <resolved-files>
```

### Key Development Principles

- **Always `make check` before commit**: Never commit code that fails local checks
- **Confirm before push**: Get approval in collaborative workflows
- **One logical change per commit**: Keep commits atomic
- **Meaningful commit messages**: Follow conventional commits (feat/fix/docs/refactor/test)

### Design Alignment

If you add or rename major abstractions, update this guide and the README accordingly.
