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
| **Unit tests** | SDK + dev tools | `make install-dev && make test-unit` |
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
make test-sdk-unit    # Run SDK unit tests
make test-server-unit # Run Studio server unit tests
make test-unit        # Run all unit tests
make test-cov         # Run tests with coverage report
make test-fast        # Run tests (fail fast)
make test-sdk-integration # Run SDK integration tests (excludes live)
make test-server-integration # Run Studio server integration tests
make test-integration # Run all local integration tests
make test-sdk-integration-live # Run all SDK live integration tests
make test-sdk-integration-live-ddg # Run the DDG live integration variant
make test-sdk-integration-live-searxng # Run the SearxNG live integration variant
make test-sdk-integration-live-tavily # Run the Tavily live integration variant
make test-sdk-integration-live-serper # Run the Serper live integration variant
make test-server-integration-live # Run Studio server live integration tests
make test-e2e-smoke   # Run Playwright smoke e2e tests
make test-e2e         # Run full Playwright e2e tests
make check-unit       # Static checks + all unit tests
make check-integration # Local integration gate (SDK + server, excludes live)
make check-e2e-smoke  # Smoke browser gate
make check            # Aggregate pre-commit gate

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

| Category | Description | Recommended Frameworks / Tools |
|----------|-------------|-------------------------------|
| **Unit** | Fast, deterministic tests for a single module or class. External calls such as database, network, filesystem side effects, and upper-layer to lower-layer dependencies SHOULD be mocked or stubbed at the boundary. | `pytest`, `pytest-mock`, `unittest.mock`, `pytest-asyncio` |
| **Integration** | Local integration tests for real collaboration across modules, adapters, persistence, websocket, and backend boundaries. These tests use isolated local services and MUST remain suitable for default gates. | `pytest`, `pytest-asyncio`, fixture factories, local test services |
| **E2E Smoke (Playwright)** | Minimal browser validation for core product paths. Smoke tests MUST stay fast, deterministic, and suitable for frequent local or CI execution. | `Playwright`, `pnpm`, backend startup scripts |
| **E2E (Playwright)** | Full user workflow validation through the actual UI and backend stack. | `Playwright`, `pnpm`, backend startup scripts |
| **Live** | Explicit opt-in tests that hit real external providers or real network-backed services. Live tests MUST NOT run in default local or CI gates. | `pytest`, `Playwright`, provider credentials, explicit environment variables |

### Test Layout Policy

- **All tests in `tests/` directory** (never in project root)

> **Principles**
>
> - Test directories MUST stay aligned with the source module/package structure.
> - Test files MUST stay aligned with the module file they validate.
> - Test classes are for grouping inside a file only; they MUST NOT determine directory layout.
>
> **Unified Standard**
>
> - **[Default]** Mirror the source ownership directory under `tests/`.
> - **[Default]** `foo.py` → `test_foo.py`.
> - **[Prohibited]** Deep source tests flattened into a shallow root test directory.
> - **[Allowed exceptions]** Only `integration`, `e2e`, and `export-surface` tests may intentionally break source mirroring.
>
> **Detailed Rules**
>
> 1. **Single-module unit test rule**
>    - **[Rule]** One source module SHOULD map to one peer test file.
>    - **[Naming]** `foo.py` → `test_foo.py`.
>    - **[Path]** The test path SHOULD mirror the source ownership path under `tests/`.
>    - **[Example]** `houyi/rag/indexed/retrieval_execution.py` → `tests/rag/indexed/test_retrieval_execution.py`
>
> 2. **Class tests are not a directory layout unit**
>    - **[Rule]** Organize test directories by module/package, not by class name.
>    - `TestXxx` classes are only for in-file grouping and readability.
>    - **[Example]** Do not create a directory such as `tests/rag/indexed_mode/` just because the main class is `IndexedMode`. Keep the test file under the module ownership path, for example `houyi/rag/indexed/mode.py` → `tests/rag/indexed/test_mode.py`.
>
> 3. **No “deep source + shallow test” layout**
>    - **[Rule]** If source code lives two or more levels deep, the test MUST descend with it.
>    - Do not flatten these tests into directories such as `tests/rag/` or `tests/core/` just for convenience.
>    - **[Example]** `houyi/rag/indexed/search_backend.py` → `tests/rag/indexed/test_search_backend.py`, not `tests/rag/test_search_backend.py`.
>
> 4. **Separate facade tests from collaborator tests**
>    - **[Rule]** Collaborator unit tests belong in the mirrored directory of the collaborator module.
>    - **[Rule]** Facade behavior tests remain in the facade test file.
>    - **[Example]** `houyi/rag/indexed/result_processing.py` → `tests/rag/indexed/test_result_processing.py`, while `houyi/rag/indexed/mode.py` keeps facade behavior coverage in `tests/rag/indexed/test_mode.py`.
>
> 5. **Exceptions must be explicit**
>    - **[Rule]** Only `integration`, `e2e`, and `export-surface` tests may intentionally avoid source mirroring.
>    - All other tests default to mirrored placement.
>    - **[Example]** `tests/integration/...` may validate cross-module workflows without mapping to a single source file.

### Execution Policy

- **Default gate**
  - `make check` is the aggregate pre-commit gate: static checks + unit + local integration + e2e smoke.
  - `make check-unit` runs static checks plus SDK/server unit tests.
  - `make check-integration` runs only local integration tests.
  - `make check-e2e-smoke` runs only the smoke browser gate.
  - Neither command may depend on real external providers.
- **Live tests**
  - Tests that require real providers or billable network calls MUST live under a `live/` directory.
  - Live tests MUST be opt-in and MUST NOT execute only because credentials happen to exist in the shell.
  - The default live opt-in for the SDK live tool-call scenario is `HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1`.
  - Use `make test-sdk-integration-live` for the aggregate SDK live integration run.
  - Use `make test-server-integration-live` for Studio server live integration tests (chat, memory).
  - Prefer provider-specific commands such as `make test-sdk-integration-live-ddg` and `make test-sdk-integration-live-serper` when validating a single provider.
- **UI E2E policy**
  - `smoke/` is the fast browser gate and should cover only a minimal stable path.
  - Full E2E remains broader product coverage and may be slower.
  - Live UI E2E, if added later, must live under `houyi-studio/ui/tests/e2e/live/` and stay outside default gates.

### Naming Convention

✅ **Correct**: `test_<module>.py`
- `tests/application/tool_calling/test_tool_call_runner.py`
- `tests/adapters/rag/test_rag_exports.py`
- `tests/rag/indexed/test_retrieval_execution.py`

❌ **Incorrect**:
- `test_phase3_evaluators.py` (phase-based)
- `test_advanced_features.py` (vague)
- `test_v0_2_0.py` (version-based)
- `test_smoke_login.py` (execution scope in filename)
- `class TestSmokeWorkflow:` (execution scope in class name)

### Test Function Naming

- Test function names SHOULD follow `test_<subject>_<behavior>[_<condition>]`.
- Prefer concise subject + behavior names over implementation-heavy sentences.
- Use `_when_<condition>` only when the condition is essential to understand the scenario.
- Use `_with_<input_variant>` or `_without_<dependency>` only for clear setup variants.
- Keep test names short. Prefer **2-3 semantic segments after `test_`**. Treat **more than 3 underscore-separated semantic segments after `test_`** as a smell that should usually be refactored into the test class, fixture, or docstring.
- Keep test names readable. Prefer **about 35 characters or fewer**, and treat **45 characters** as a hard limit unless there is a strong reason.
- Enforced by `scripts/check_style_rules.py` (wired into `make check` and `make quick-check`): tests whose name has **more than 4 underscore segments after `test_`** fail the gate. The same script blocks raw CJK characters in `.py` sources and warns when a file has 5+ `\uXXXX` CJK escapes without an ASCII comment (add a short pinyin or English gloss).
- Avoid repeating file or module context that is already clear from the test file path or test class.
- Prefer contract-visible behavior over implementation-detail phrasing.
- Do not encode execution tier into function names; use `pytest.mark` or CI/workflow selection instead.

✅ **Correct**:
- `test_write_file_creates_parents()`
- `test_find_files_uses_contains()`
- `test_vertex_uses_httpx()`
- `test_gemini_emits_reasoning()`

❌ **Incorrect**:
- `test_write_file_executor_creates_parents()` (repeats implementation context)
- `test_find_files_executor_iterative_subdirs_exact_match()` (too implementation-heavy)
- `test_create_vertex_adapter_falls_back_when_google_sdk_missing()` (too long and over-specifies implementation)
- `test_vertex_gemini_build_generate_config_ignores_parallel_tool_calls()` (too long and carries too many underscore segments)
- `test_smoke_find_files()` (execution tier in function name)

### Test Structure

```python
class TestAccuracyEvaluator:
    def test_exact_match(self):
        ...

    def test_similar_match(self):
        ...
```

- Group tests by module behavior or subject class
- Use fixtures and factories to keep setup reusable and readable
- Use `pytest.mark` for categorization and runtime selection
- Mock external calls at architectural boundaries for unit tests
- In layered architecture, upper-layer tests SHOULD mock lower-layer collaborators unless the purpose is explicit integration coverage

### High-Quality Test Design

Use the following checklist when designing or generating tests, especially with AI assistance:

| Dimension | What to Cover | Typical Questions |
|----------|---------------|-------------------|
| **Happy path** | Expected valid behavior | Does the primary use case succeed with representative inputs? |
| **Boundary conditions** | Min/max/empty/one-off thresholds | What happens at `0`, `1`, empty string, empty list, max size, timeout edge, exact threshold? |
| **Equivalence classes** | Representative valid/invalid input groups | Which inputs are logically the same, so one case can stand for many? |
| **Error paths** | Exceptions, validation failures, partial failure | Does the code fail with actionable errors and preserve invariants? |
| **State transitions** | Before/after mutation or persistence | Does state move correctly across create/update/delete/retry flows? |
| **Interaction contracts** | Calls to collaborators, adapters, or gateways | Are downstream dependencies called with the right arguments, order, and retry behavior? |
| **Idempotency / replay** | Repeated execution safety | Does running the same operation twice remain safe where required? |
| **Fallback / degradation** | Timeout, unavailable dependency, optional component off | Does the system degrade predictably and surface metadata or warnings correctly? |
| **Distributed workflow invariants** | Cross-step consistency in async or distributed flows | Are ordering, deduplication, state convergence, compensation, and observability invariants preserved across retries, partial failures, and concurrent execution? |

**Policy**:
- Boundary-condition tests SHOULD exist for code with thresholds, slicing, pagination, retries, scoring, timeout, and size-based branching.
- Each non-trivial module SHOULD cover at least: one happy path, one boundary case, one invalid/error path, and one dependency-interaction assertion where applicable.
- Do not enforce a single minimum test-count rule across the codebase. For distributed systems, test adequacy SHOULD be driven by risk, failure modes, and contract surface rather than a fixed number of cases.
- Higher-risk paths SHOULD add targeted coverage for timeout, retry, partial failure, fallback, ordering, idempotency, concurrency, and observability metadata where applicable.
- Distributed workflows SHOULD verify invariants explicitly, such as exactly-once vs at-least-once behavior, ordering guarantees, deduplication, compensation behavior, eventual consistency, and trace or event correlation integrity where applicable.
- AI-generated tests MUST be reviewed for missing edge cases, over-mocking, and assertions that only restate implementation details.
- Prefer assertions on observable behavior, returned data, emitted metadata, and state transitions over assertions on private implementation structure.

### Coverage Requirements

| Scope | Target |
|-------|--------|
| Overall project | ≥ 85% |
| Business-critical modules | ≥ 85% |

Business-critical modules are the actively owned product and platform layers under `houyi/`, especially `domain/`, `application/`, `adapters/`, `rag/`, `infrastructure/`, and `interface/`. Coverage review SHOULD prioritize modules on critical request, execution, persistence, and retrieval paths rather than relying on a legacy static directory list.

### Integration Test Execution Strategy

**Policy**:
- Integration suites MUST be stratified by runtime cost and change frequency so the default developer and PR path remains fast.
- Scope selection belongs to `pytest` markers, CI workflow configuration, or dedicated Make targets — not file names or test class names.
- File and test names describe the feature or contract under test, not execution tier.

**Recommended tiers**:
- **[Default PR integration suite]** Fast, high-signal integration coverage for critical contracts and common workflows.
- **[Extended integration suite]** Broader cross-module and adapter coverage triggered by CI policy, nightly runs, or manual dispatch.

**Examples**:
- `tests/integration/test_skill_loading.py` remains feature-named whether it runs in the default or extended integration tier.
- Use markers such as `@pytest.mark.integration` plus an additional tier marker selected by CI, instead of names like `test_smoke_*` or `test_full_*`.

### E2E Testing (Playwright)

**Policy**:
- E2E scope selection belongs to test metadata, CI workflow configuration, or runtime tags — not file names or test class names.
- Do not encode execution scope into names such as `test_smoke_*`, `test_full_*`, `TestSmoke*`, or `TestFull*`.
- The default PR E2E suite SHOULD stay small and high-signal to control feedback time.
- The extended E2E suite is triggered by CI policy such as PR label, nightly schedule, or manual dispatch.

**Recommended default PR E2E scope**: UI startup, home page render, and one critical end-user workflow.

**Recommended extended E2E scope**: broader multi-workflow journeys, failure recovery, and lower-frequency scenarios that are too expensive for every PR run.

**Recommended naming**:
- File names describe the feature or workflow: `test_home_page.py`, `test_position_workflow.py`
- Test names describe the user-visible behavior: `test_home_page_renders()`, `test_position_workflow_completes_successfully()`
- Use markers, tags, or CI selection rules to distinguish default PR vs extended suites

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
