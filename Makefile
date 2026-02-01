.PHONY: help install install-dev test test-cov lint lint-fix quick-check check clean format

# Default target
help:
	@echo "HouYi Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make setup-hooks      Setup pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make quick-check      Quick checks (ruff + fast tests)"
	@echo "  make check            Full checks (ruff + pylint + tests + coverage)"
	@echo "  make format           Auto-format code with ruff"
	@echo "  make lint             Run all linters (ruff + pylint)"
	@echo "  make lint-fix         Run linters with auto-fix"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-fast        Run tests (fail fast)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove cache and build files"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pre-commit pylint

setup-hooks:
	pre-commit install
	@echo "✓ Pre-commit hooks installed"

# Quick checks (for frequent use during development)
quick-check:
	@./scripts/quick_check.sh

# Full checks (run before committing)
check:
	@./scripts/check_code.sh

# Formatting
format:
	ruff format houyi/ tests/
	ruff check houyi/ tests/ --fix

# Linting
lint:
	ruff check .
	pylint houyi/ --rcfile=.pylintrc

lint-fix:
	ruff check . --fix

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=houyi --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -x --tb=short

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ dist/ build/
	@echo "✓ Cleaned up cache and build files"
