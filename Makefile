.PHONY: help install install-dev install-studio install-all dev test test-server test-cov test-fast test-integration test-e2e lint lint-fix lint-imports quick-check check clean format typecheck

# Default target
help:
	@echo "HouYi Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make install-studio   Install Studio server + UI deps into .venv"
	@echo "  make install-all      Full setup (dev + rag + studio + UI)"
	@echo "  make setup-hooks      Setup pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start backend + frontend (tmux)"
	@echo "  make quick-check      Quick checks (ruff + fast tests)"
	@echo "  make check            Full checks (ruff + mypy + tests + coverage)"
	@echo "  make typecheck        Run mypy type checking"
	@echo "  make format           Auto-format code with ruff"
	@echo "  make lint             Run all linters (ruff)"
	@echo "  make lint-fix         Run linters with auto-fix"
	@echo "  make lint-imports     Check import layer boundaries"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run SDK unit tests"
	@echo "  make test-server      Run Studio server tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-fast        Run tests (fail fast)"
	@echo "  make test-integration Run integration tests (requires studio server deps)"
	@echo "  make test-e2e         Run Playwright e2e tests (requires running backend)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove cache and build files"
	@echo ""

# Installation
install:
	uv sync

install-dev:
	uv sync --extra dev

# Install Studio server (backend) + UI deps into .venv
install-studio:
	uv pip install -e houyi-studio/server --quiet
	@echo "✓ Studio server installed"
	@cd houyi-studio/ui && pnpm install --frozen-lockfile
	@echo "✓ UI dependencies installed"

# Full setup: SDK dev + RAG extras + Studio server + UI
install-all:
	uv sync --extra dev --extra rag
	uv pip install -e houyi-studio/server --quiet
	@cd houyi-studio/ui && pnpm install --frozen-lockfile
	@echo "✓ All dependencies installed (SDK + RAG + Studio + UI)"

setup-hooks:
	uv run pre-commit install
	@echo "✓ Pre-commit hooks installed"

# Quick checks (for frequent use during development)
quick-check:
	@./scripts/quick_check.sh

# Full checks (run before committing)
check:
	@./scripts/check_code.sh

# Formatting
format:
	uv run ruff format houyi/ tests/
	uv run ruff check houyi/ tests/ --fix

# Linting
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

# Type checking
typecheck:
	uv run mypy houyi/

lint-imports:
	uv run lint-imports

# Start dev environment (backend + frontend via tmux)
dev:
	@./scripts/dev.sh

# Testing
test:
	uv run pytest tests/ -v -n auto

test-cov:
	uv run pytest tests/ --cov=houyi --cov-report=term-missing --cov-report=html

test-fast:
	uv run pytest tests/ -x --tb=short

# Studio server tests
test-server:
	@uv run python -c "import houyi_studio" 2>/dev/null || (echo '📦 Installing studio server...' && uv pip install -e houyi-studio/server --quiet)
	uv run pytest houyi-studio/server/tests/ -v

# Integration tests (requires studio server deps)
test-integration:
	@uv run python -c "import houyi_studio" 2>/dev/null || (echo '📦 Installing studio server...' && uv pip install -e houyi-studio/server --quiet)
	uv run pytest tests/integration/ -v

# E2E tests (requires backend running + Playwright browsers)
test-e2e:
	@cd houyi-studio/ui && pnpm install --frozen-lockfile && pnpm test:e2e

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
