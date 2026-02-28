#!/usr/bin/env bash
# Code quality check script — fail-fast mode
# Run this before committing code.
# Stops at the FIRST failure so the error output stays visible.

set -euo pipefail

echo "🔍 Running code quality checks..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}✗ uv is not installed or not on PATH.${NC}"
    echo "Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ .venv not found.${NC}"
    echo "Run: uv sync --extra dev"
    exit 1
fi

# ── Ensure dependencies ──────────────────────────────────────────────
ensure_deps() {
    echo -e "${YELLOW}▶ Verifying dependencies...${NC}"

    local need_sync=0

    if ! uv run python -c "import pytest" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import pytest_cov" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import xdist" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import mypy" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import importlinter" 2>/dev/null; then need_sync=1; fi

    if [ $need_sync -eq 1 ]; then
        echo -e "${YELLOW}  Installing missing dev dependencies...${NC}"
        uv sync --extra dev --extra websearch-ddg --extra websearch-tavily --extra websearch-readability --quiet
    fi

    if ! uv run python -c "import houyi_studio" 2>/dev/null; then
        echo -e "${YELLOW}  Installing studio server...${NC}"
        uv pip install -e houyi-studio/server --quiet
    fi

    echo -e "${GREEN}✓ Dependencies verified${NC}"
    echo ""
}

ensure_deps

# ── Fail-fast runner ─────────────────────────────────────────────────
# Runs the command; on failure prints a clear banner and exits immediately.
run_check() {
    local name=$1
    shift
    echo -e "${YELLOW}▶ Running $name...${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}✗ $name FAILED — stopping here.${NC}"
        echo -e "${RED}  Fix the errors above, then re-run: make check${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 1
    fi
}

# ── 1-2. Ruff lint + format (changed files only) ────────────────────
CHANGED_PY_FILES=$(
    {
        git diff --name-only --cached
        git diff --name-only
        git ls-files --others --exclude-standard
    } 2>/dev/null | awk '($0 ~ /\.pyi?$/) && ($0 !~ /^skills\//) { print }' | sort -u | while read -r f; do
        [ -f "$f" ] && echo "$f"
    done
)

if [ -n "$CHANGED_PY_FILES" ]; then
    CHANGED_PY_FILES_ONELINE=$(echo "$CHANGED_PY_FILES" | tr '\n' ' ')
    run_check "Ruff (lint)" uv run ruff check --fix $CHANGED_PY_FILES_ONELINE
    run_check "Ruff (format)" uv run ruff format $CHANGED_PY_FILES_ONELINE
else
    echo -e "${YELLOW}▶ Ruff (lint/format) skipped (no changed Python files)${NC}"
    echo ""
fi

# ── 3. Type check ───────────────────────────────────────────────────
run_check "Type Check (mypy)" uv run mypy houyi/

# ── 4. SDK unit tests (with coverage, single pass) ──────────────────
run_check "SDK Tests + Coverage" uv run pytest tests/ -x -n auto \
    --cov=houyi --cov-report=term-missing --cov-fail-under=80

# ── 5. Server tests ─────────────────────────────────────────────────
run_check "Server Tests" uv run pytest houyi-studio/server/tests/ -x

# ── Done ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ All checks passed! Ready to commit.${NC}"
