# Contributing to HouYi

Thank you for your interest in contributing to HouYi! This document provides guidelines and best practices for contributing.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/houyi.git
cd houyi
```

### 2. Create Conda Environment

```bash
conda create -n houyi python=3.11 -y
conda activate houyi
```

### 3. Install Dependencies

```bash
make install-dev
make setup-hooks
```

## Development Workflow

### Before You Start

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make sure all checks pass:
   ```bash
   make check
   ```

### During Development

1. **Write code** following our [coding standards](agent.md#coding-standards)

2. **Run quick checks frequently**:
   ```bash
   make quick-check
   ```

3. **Write tests** for new features:
   - Place tests in `tests/` directory
   - Follow naming convention: `test_<module_name>.py`
   - **REQUIRED**: Maintain ≥80% coverage

4. **Check test coverage** during development:
   ```bash
   make test-cov
   ```
   This generates a coverage report showing which lines are not covered.

### Before Committing

**CRITICAL**: Always run full checks before committing:

```bash
make check
```

This will run:
- ✅ Ruff formatting and linting
- ✅ Pylint code quality checks (must score ≥9.80/10)
- ✅ All unit tests (must pass 100%)
- ✅ **Coverage check (≥80% REQUIRED)**

**Coverage is enforced**: Commits will be rejected if coverage drops below 80%.

If you have pre-commit hooks installed (recommended), they will run automatically on `git commit`.

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(evaluation): add ContextRecall evaluator

Implement ContextRecall evaluator for RAG evaluation.
Includes unit tests and documentation.

Closes #123
```

```
fix(llm): handle missing API key gracefully

Add proper error handling when OpenAI API key is not provided.
```

## Code Quality Standards

### Linting

- **Ruff**: Fast linting and formatting (must pass)
- **Pylint**: Deep code quality analysis (must score 10.00/10)

### Testing

- **Coverage**: Minimum 75% required
- **Test types**: Unit tests (fast, no external dependencies)
- **Framework**: pytest with pytest-asyncio

### Type Hints

- Use type hints for all public APIs
- Use Pydantic models for data validation

## Pull Request Process

### 1. Ensure Quality

Before opening a PR:

```bash
# Run all checks
make check

# Ensure tests pass
make test

# Check coverage
make test-cov
```

### 2. Update Documentation

- Update README.md if adding new features
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)
- Add docstrings to new functions/classes

### 3. Create Pull Request

1. Push your branch to your fork
2. Open a PR against `main` branch
3. Fill out the PR template
4. Link related issues

### 4. Code Review

- Address review comments promptly
- Keep commits clean and logical
- Squash commits if requested

### 5. CI/CD Checks

All PRs must pass:
- ✅ Ruff linting
- ✅ Pylint (10.00/10)
- ✅ All tests
- ✅ Coverage ≥75%

## Development Tools

### Makefile Commands

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

### Pre-commit Hooks

Automatically run checks before each commit:

```bash
make setup-hooks
```

## Questions?

- Check [agent.md](agent.md) for detailed development guidelines
- Open an issue for questions or discussions
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
