.PHONY: help install install-dev install-studio install-all lock lock-check dev \
	test test-server test-cov test-fast \
	test-sdk-unit test-server-unit test-unit \
	test-sdk-integration test-server-integration test-integration \
	test-sdk-integration-live test-sdk-integration-live-ddg test-sdk-integration-live-searxng test-sdk-integration-live-tavily test-sdk-integration-live-serper \
	test-server-integration-live \
	test-e2e test-e2e-smoke \
	check check-unit check-integration check-e2e-smoke \
	lint lint-fix lint-imports quick-check clean format typecheck \
	benchmark

BENCH_KIND ?= pytest
BENCH_TARGET ?= all
BENCH_DEPTH ?= deep
BENCH_MODE ?= direct
BENCH_CONCURRENCY ?= 3

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
	@echo "  make lock             Refresh uv.lock from official PyPI"
	@echo "  make lock-check       Fail if uv.lock contains non-PyPI registries"
	@echo "  make setup-hooks      Setup pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start backend + frontend (tmux)"
	@echo "  make quick-check      Quick checks (ruff + fast tests)"
	@echo "  make check            Aggregate pre-commit gate (static + unit + integration + e2e smoke)"
	@echo "  make check-unit       Static checks + SDK unit tests + server unit tests"
	@echo "  make check-integration Local-only integration gate (SDK + server, excludes live)"
	@echo "  make check-e2e-smoke  Smoke browser gate"
	@echo "  make typecheck        Run mypy type checking"
	@echo "  make format           Auto-format code with ruff"
	@echo "  make lint             Run all linters (ruff)"
	@echo "  make lint-fix         Run linters with auto-fix"
	@echo "  make lint-imports     Check import layer boundaries"
	@echo "  make benchmark        Run benchmarks (default: pytest -m benchmark, use BENCH_TARGET=memory|rag|runtime|verification|observability|all or BENCH_PATH=...; use BENCH_KIND=arena for DeepResearch-Bench)"
	@echo ""
	@echo "Testing:"
	@echo "  make test-sdk-unit    Run SDK unit tests"
	@echo "  make test-server-unit Run Studio server unit tests"
	@echo "  make test-unit        Run all unit tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make test-fast        Run tests (fail fast)"
	@echo "  make test-sdk-integration Run SDK integration tests (excludes live)"
	@echo "  make test-server-integration Run Studio server integration tests"
	@echo "  make test-integration Run all local integration tests"
	@echo "  make test-sdk-integration-live Run all SDK live integration tests"
	@echo "  make test-sdk-integration-live-ddg Run the DDG live integration variant"
	@echo "  make test-sdk-integration-live-searxng Run the SearxNG live integration variant"
	@echo "  make test-sdk-integration-live-tavily Run the Tavily live integration variant"
	@echo "  make test-sdk-integration-live-serper Run the Serper live integration variant"
	@echo "  make test-server-integration-live Run Studio server live integration tests"
	@echo "  make test-e2e-smoke   Run Playwright smoke e2e tests"
	@echo "  make test-e2e         Run full Playwright e2e tests"
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
	uv sync --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability
	uv pip install -e houyi-studio/server --quiet
	@cd houyi-studio/ui && pnpm install --frozen-lockfile
	@echo "✓ All dependencies installed (SDK + RAG + Studio + UI)"

lock:
	UV_NO_CONFIG=1 UV_INDEX_URL=https://pypi.org/simple uv lock --refresh

lock-check:
	@python3 -c "from pathlib import Path; import re, sys; registries=set(re.findall(r'registry = \"([^\"]+)\"', Path('uv.lock').read_text(encoding='utf-8'))); bad=sorted(r for r in registries if r != 'https://pypi.org/simple'); print('Unexpected lock registries: ' + ', '.join(bad)) if bad else print('uv.lock registry sources are pinned to official PyPI.'); sys.exit(1 if bad else 0)"

setup-hooks:
	uv run pre-commit install
	@echo "✓ Pre-commit hooks installed"

# Quick checks (for frequent use during development)
quick-check:
	@./scripts/quick_check.sh

# Full checks (run before committing)
check:
	@python3 scripts/run_make_check.py --budget 60

check-unit:
	@./scripts/check_code.sh

check-integration:
	@./scripts/check_integration.sh

check-e2e-smoke:
	@$(MAKE) test-e2e-smoke

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
test-sdk-unit:
	uv run pytest tests/ --ignore=tests/integration -v -n auto

test-server-unit:
	@uv run python -c "import houyi_studio" 2>/dev/null || (echo '📦 Installing studio server...' && uv pip install -e houyi-studio/server --quiet)
	uv run pytest houyi-studio/server/tests/ --ignore=houyi-studio/server/tests/integration -v -n auto

test-unit:
	@$(MAKE) test-sdk-unit
	@$(MAKE) test-server-unit

test:
	@$(MAKE) test-sdk-unit

test-cov:
	uv run pytest tests/ --cov=houyi --cov-report=term-missing --cov-report=html

test-fast:
	uv run pytest tests/ --ignore=tests/integration -x --tb=short

# Studio server tests
test-server:
	@$(MAKE) test-server-unit

# Integration tests (requires studio server deps)
test-sdk-integration:
	@uv run python -c "import houyi_studio" 2>/dev/null || (echo '📦 Installing studio server...' && uv pip install -e houyi-studio/server --quiet)
	uv run pytest tests/integration/ --ignore=tests/integration/live -v

test-server-integration:
	@uv run python -c "import houyi_studio" 2>/dev/null || (echo '📦 Installing studio server...' && uv pip install -e houyi-studio/server --quiet)
	uv run pytest houyi-studio/server/tests/integration/ -v


test-integration:
	@$(MAKE) test-sdk-integration
	@$(MAKE) test-server-integration

# Live tests (explicitly executed, never part of default gates)
test-sdk-integration-live:
	HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 uv run pytest tests/integration/live/ -v

test-sdk-integration-live-ddg:
	HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 DDG_INTEGRATION_TEST=1 uv run pytest tests/integration/live/ -k 'ddg' -v

test-sdk-integration-live-searxng:
	@if [ -z "$$SEARXNG_BASE_URL" ]; then echo 'SEARXNG_BASE_URL is required'; exit 1; fi
	HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 SEARXNG_BASE_URL="$$SEARXNG_BASE_URL" uv run pytest tests/integration/live/ -k 'searxng' -v

test-sdk-integration-live-tavily:
	@if [ -z "$$TAVILY_API_KEY" ]; then echo 'TAVILY_API_KEY is required'; exit 1; fi
	HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 TAVILY_API_KEY="$$TAVILY_API_KEY" uv run pytest tests/integration/live/ -k 'tavily' -v

test-sdk-integration-live-serper:
	@if [ -z "$$SERPER_API_KEY" ]; then echo 'SERPER_API_KEY is required'; exit 1; fi
	HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 SERPER_API_KEY="$$SERPER_API_KEY" uv run pytest tests/integration/live/ -k 'serper' -v

# Studio server live tests (real LLM, explicitly executed)
test-server-integration-live:
	@uv run python -c "import houyi_studio" 2>/dev/null || (echo '📦 Installing studio server...' && uv pip install -e houyi-studio/server --quiet)
	HOUYI_RUN_LIVE_LLM_TOOL_SCENARIO_TESTS=1 uv run pytest houyi-studio/server/tests/integration/live/ -v

# E2E tests (requires backend running + Playwright browsers)
test-e2e-smoke:
	@cd houyi-studio/ui && HOUYI_E2E_BACKEND_PORT=$${HOUYI_E2E_BACKEND_PORT:-19000} HOUYI_E2E_UI_PORT=$${HOUYI_E2E_UI_PORT:-13100} pnpm install --frozen-lockfile && HOUYI_E2E_BACKEND_PORT=$${HOUYI_E2E_BACKEND_PORT:-19000} HOUYI_E2E_UI_PORT=$${HOUYI_E2E_UI_PORT:-13100} pnpm exec playwright test tests/e2e/smoke

test-e2e:
	@cd houyi-studio/ui && pnpm install --frozen-lockfile && pnpm test:e2e

benchmark:
	@if [ "$(BENCH_KIND)" = "arena" ]; then \
		BENCH_OUTPUT_PATH="$${BENCH_OUTPUT:-benchmark/output/houyi.jsonl}"; \
		set -- uv run python scripts/run_benchmark.py --queries "$${BENCH_QUERIES:-benchmark/data/query.jsonl}" --output "$$BENCH_OUTPUT_PATH" --depth "$(BENCH_DEPTH)" --mode "$(BENCH_MODE)" --concurrency "$(BENCH_CONCURRENCY)"; \
		if [ -n "$${BENCH_LIMIT:-}" ]; then set -- "$${@}" --limit "$${BENCH_LIMIT}"; fi; \
		if [ -n "$${BENCH_TIMEOUT:-}" ]; then set -- "$${@}" --timeout "$${BENCH_TIMEOUT}"; fi; \
		if [ "$${BENCH_NO_RESUME:-0}" = "1" ]; then set -- "$${@}" --no-resume; fi; \
		echo "Running DeepResearch-Bench via scripts/run_benchmark.py"; \
		echo "  output=$$BENCH_OUTPUT_PATH depth=$(BENCH_DEPTH) mode=$(BENCH_MODE) concurrency=$(BENCH_CONCURRENCY)"; \
		"$$@"; \
	else \
		case "$(BENCH_TARGET)" in \
			all) bench_path="$${BENCH_PATH:-tests/}" ;; \
			memory) bench_path="tests/integration/benchmark/test_memory.py" ;; \
			rag) bench_path="tests/rag/benchmark" ;; \
			runtime) bench_path="tests/application/runtime/test_runtime_benchmark.py" ;; \
			verification) bench_path="tests/integration/verification/test_performance.py" ;; \
			observability) bench_path="tests/infrastructure/observability/test_golden_metrics.py" ;; \
			*) bench_path="$${BENCH_PATH:-$(BENCH_TARGET)}" ;; \
		esac; \
		echo "Running pytest benchmarks from $$bench_path"; \
		uv run pytest "$$bench_path" -m benchmark -v -s; \
	fi

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
