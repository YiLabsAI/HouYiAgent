# HouYi Development Guidelines

This document defines development standards, coding conventions, version management, and best practices for the HouYi project.

## Development Environment

### Conda Environment (Recommended)

HouYi uses Conda for environment management to ensure isolated and reproducible development environments.

```bash
# Create conda environment
conda create -n houyi python=3.11 -y

# Activate environment
conda activate houyi

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Development Tools

- **Python**: 3.10+ (3.11 recommended)
- **Environment**: Conda for isolation
- **Package manager**: `pip` with `requirements.txt`
- **Dependency management**: `pyproject.toml` for dev dependencies
- **Lint/format**: `ruff` for code formatting
- **Code quality**: `pylint` for comprehensive code quality checks
- **Type checking**: `mypy`
- **Tests**: `pytest` with `pytest-asyncio` and `pytest-cov`

## Common Commands

All commands should be run within the activated conda environment:

```bash
# Activate conda environment first
conda activate houyi

# Install production dependencies
pip install -r requirements.txt

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Using Makefile (Recommended)

The project includes a Makefile for common development tasks:

```bash
# Setup
make install-dev      # Install all development dependencies
make setup-hooks      # Setup pre-commit hooks

# Code Quality (use these before committing!)
make quick-check      # Fast checks (ruff + quick tests) - use frequently
make check            # Full checks (ruff + pylint + tests + coverage) - before commit
make format           # Auto-format code
make lint             # Run all linters
make lint-fix         # Run linters with auto-fix

# Testing
make test             # Run all tests
make test-cov         # Run tests with coverage report
make test-fast        # Run tests (fail fast)

# Cleanup
make clean            # Remove cache and build files

# Help
make help             # Show all available commands
```

### Manual Commands

If you prefer to run commands manually:

```bash
# Lint / format
ruff check houyi/
ruff check houyi/ --fix

# Code quality check
pylint houyi/ --rcfile=.pylintrc

# Type check
mypy houyi/

# Run tests with coverage
pytest tests/ -v --cov=houyi

# Run specific test
pytest tests/test_core.py -v

# Run example
python examples/quickstart.py
```

## Pre-commit Workflow

### Recommended Development Flow

**Before committing code, always run:**

```bash
# Option 1: Quick check during development (fast)
make quick-check

# Option 2: Full check before commit (comprehensive)
make check
```

### Automated Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before every commit:

```bash
# One-time setup
make setup-hooks

# Or manually
pip install pre-commit
pre-commit install
```

Once installed, hooks will automatically run on `git commit`:
- ✅ Ruff formatting and linting
- ✅ Trailing whitespace removal
- ✅ File encoding checks
- ✅ YAML/JSON validation
- ✅ Large file detection

**Skip hooks (emergency only)**:
```bash
git commit --no-verify -m "emergency fix"
```

### Manual Pre-commit Checklist

If not using automated hooks, run these before committing:

1. **Format code**: `make format`
2. **Run linters**: `make lint`
3. **Run tests**: `make test-fast`
4. **Check coverage**: `make test-cov` (ensure ≥75%)

## Coding Standards

### Code Quality Tools

HouYi uses a two-tier approach for code quality:

#### 1. Ruff (Fast Linting & Formatting)
- **Purpose**: Fast code formatting and common error detection
- **Usage**: `ruff check houyi/` or `ruff check houyi/ --fix`
- **Checks**:
  - Code formatting (PEP 8)
  - Import sorting
  - Unused variables
  - Common anti-patterns
- **Speed**: Very fast, runs in milliseconds

#### 2. Pylint (Comprehensive Code Quality)
- **Purpose**: Deep code quality analysis and best practices enforcement
- **Usage**: `pylint houyi/ --rcfile=.pylintrc`
- **Configuration**: `.pylintrc` in project root
- **Target Score**: **10.00/10** (current score)
- **Checks**:
  - Code complexity and maintainability
  - Naming conventions
  - Design issues
  - Potential bugs
  - Documentation completeness
- **CI/CD**: Both ruff and pylint run in GitHub Actions

**Best Practice**: Run both tools before committing:
```bash
# Quick check with ruff (auto-fix)
ruff check houyi/ --fix

# Comprehensive check with pylint
pylint houyi/ --rcfile=.pylintrc
```

### API Stability (Critical)

- **Preserve public interfaces**.
  - Do not change function signatures, argument order, or parameter names for public APIs without explicit migration guidance.
  - When adding new parameters, prefer keyword-only parameters: `def f(a: int, *, new_param: str = "default") -> ...`.
- **Determine what is public**.
  - Treat items exported from `__init__.py` as public.
  - Also treat anything documented as part of the framework API as public.

### Typing and Schemas

- Use type hints everywhere for non-trivial code paths.
- Keep boundaries **schema-first**:
  - Inputs/outputs should be explicitly modeled and validated (Pydantic v2).
  - Avoid passing around untyped `dict`/`Any` across module boundaries.

### Error Handling

- Validate early at boundaries.
- Raise structured, actionable errors.
- Avoid bare `except:`.
- Do not swallow tool execution failures; rethrow with context.

### Security Guidelines

- No `eval()` / `exec()` / `pickle` on user-controlled input.
- Be explicit about any filesystem/network side effects.
- Ensure proper resource cleanup (files, sockets, subprocesses, threads).

## Testing Requirements

Every new feature or bugfix must be covered by tests.

- **Framework**: `pytest`
- **Test categories**:
  - **Unit tests**: fast, deterministic, no network calls.
  - **Integration tests**: can include network calls and external services (when explicitly configured).
- **Test location**:
  - **All tests must be in `tests/` directory**
  - **Never place test files in project root**
  - Mirror the source layout: `houyi/core/agent.py` → `tests/test_core.py`

### Test Naming Convention (CRITICAL)

**Follow the `test_<class_or_module_name>.py` pattern:**

✅ **Correct naming**:
- `tests/test_evaluators.py` - tests all Evaluator classes
- `tests/test_skill_executor.py` - tests SkillExecutor class
- `tests/test_observability.py` - tests TraceManager, Span, Exporters
- `tests/test_dataset.py` - tests Dataset class
- `tests/test_llm.py` - tests LLM adapters (OpenAI, Anthropic)
- `tests/test_orchestration.py` - tests orchestration classes (Plan, Planner, State)

❌ **Incorrect naming** (DO NOT USE):
- `tests/test_phase3_evaluators.py` - phase-based naming
- `tests/test_advanced_features.py` - vague feature-based naming
- `tests/test_v0_2_0.py` - version-based naming
- `tests/test_new_stuff.py` - non-descriptive naming

**Rationale**:
- Test files should be named after **what they test**, not when they were created
- Makes it easy to find tests for a specific class/module
- Avoids confusion when phases/versions change
- Maintains long-term clarity and maintainability

### Test Structure

- **Prefer fixtures/mocks** for external dependencies
- **Use `pytest.mark`** for categorization (e.g., `@pytest.mark.integration`)
- **Group tests by class**: Use `class Test<ClassName>:` for organizing related tests
  ```python
  class TestAccuracyEvaluator:
      def test_exact_match(self):
          ...

      def test_similar_match(self):
          ...
  ```

### Test Quality Checklist

- Tests fail when the new logic is broken
- Happy path is covered
- Edge cases and error conditions are covered
- Tests are deterministic (no flakes)
- **Test coverage target: 80%+**

### Release Requirements (CRITICAL)

**Before any version release (x.y.z), the following requirements MUST be met:**

1. **Test Coverage Requirements**
   - **Overall project coverage: ≥ 80%**
   - **Core business logic modules: ≥ 85%**
   - Run: `pytest tests/ -v --cov=houyi --cov-report=term`
   - Core modules include: core/, evaluation/, execution/, orchestration/, observability/

2. **All Tests Passing**
   - 100% test pass rate required
   - No skipped tests without explicit justification

3. **Code Quality**
   - `ruff check houyi/` passes with no errors
   - `mypy houyi/` passes with no type errors

4. **Documentation**
   - CHANGELOG.md updated with all changes
   - README.md reflects current features
   - API changes documented

## Architectural Invariants (Do Not Break)

- **Schema-first**: public boundaries should be defined with explicit schemas.
- **Deterministic execution**: LLM should produce plans or formal artifacts (e.g., code/SQL), not execute business logic directly.
- **Immutable state**: prefer immutable state snapshots (versioned) over in-place mutation.
- **Observability**: critical steps should emit trace/log/metric events (OpenTelemetry-compatible).

## PR Checklist

- **Correctness**: schemas/types updated; validations added where needed.
- **Compatibility**: no breaking public API changes without migration notes.
- **Quality**: `ruff`, `mypy`, and `pytest` pass.
- **Observability**: new critical paths include tracing/logging.
- **Security**: tool execution respects sandbox/permission boundaries.

## Version Management

HouYi follows [Semantic Versioning 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH`

### When to Increment

- **MAJOR**: Breaking changes to public APIs
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes, no API changes

### Release Process

1. Update `CHANGELOG.md` with changes (Added/Changed/Fixed/Removed)
2. Update `pyproject.toml` version
3. Run full test suite: `pytest tests/ -v --cov=houyi`
4. Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Update `README.md` if major/minor release

See [CHANGELOG.md](https://github.com/your-repo/CHANGELOG.md) for version history and release notes.

## Design Alignment

If you add or rename major abstractions, update this guide and the README accordingly.
