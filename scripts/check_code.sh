#!/bin/bash
# Code quality check script
# Run this before committing code

set -e  # Exit on error

echo "🔍 Running code quality checks..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if any check fails
FAILED=0

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

# Check dev dependencies are installed
check_dev_deps() {
    local missing=()

    echo -e "${YELLOW}▶ Verifying dev dependencies...${NC}"

    # Check critical dev tools via uv run
    if ! uv run python -c "import pylint" 2>/dev/null; then
        missing+=("pylint")
    fi
    if ! uv run python -c "import pytest" 2>/dev/null; then
        missing+=("pytest")
    fi
    if ! uv run python -c "import pytest_cov" 2>/dev/null; then
        missing+=("pytest-cov")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}✗ Missing dev dependencies: ${missing[*]}${NC}"
        echo -e "${YELLOW}Run: uv sync --extra dev${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dev dependencies verified${NC}"
    echo ""
}

check_dev_deps

# Function to run a check
run_check() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}▶ Running $name...${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        FAILED=1
    fi
}

# Determine changed Python files (staged + unstaged + untracked)
get_changed_python_files() {
    {
        git diff --name-only --cached
        git diff --name-only
        git ls-files --others --exclude-standard
    } 2>/dev/null | awk '
        ($0 ~ /\.pyi?$/) { print $0 }
    ' | sort -u | while read -r file; do
        [ -f "$file" ] && echo "$file"
    done
}

CHANGED_PY_FILES=$(get_changed_python_files)

# 1-2. Ruff - Lint/format only changed Python files
if [ -n "$CHANGED_PY_FILES" ]; then
    CHANGED_PY_FILES_ONELINE=$(echo "$CHANGED_PY_FILES" | tr '\n' ' ')
    run_check "Ruff (lint)" "uv run ruff check --fix $CHANGED_PY_FILES_ONELINE"
    run_check "Ruff (format)" "uv run ruff format $CHANGED_PY_FILES_ONELINE"
else
    echo -e "${YELLOW}▶ Ruff (lint/format) skipped (no changed Python files)${NC}"
    echo ""
fi

# 3. Pylint - Deep code quality check
run_check "Pylint (source code)" "uv run pylint houyi/ --rcfile=.pylintrc"

# 4. Type checking with MyPy (optional, can be slow)
# run_check "MyPy" "mypy houyi/ --ignore-missing-imports"

# 5. Run tests
run_check "Unit Tests" "uv run pytest tests/ -v --tb=short -x"

# 6. Check test coverage
echo -e "${YELLOW}▶ Checking test coverage...${NC}"
uv run pytest tests/ --cov=houyi --cov-report=term-missing --cov-fail-under=80 || {
    echo -e "${RED}✗ Coverage below 80%${NC}"
    FAILED=1
}
echo ""

# Final result
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready to commit.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix before committing.${NC}"
    exit 1
fi
