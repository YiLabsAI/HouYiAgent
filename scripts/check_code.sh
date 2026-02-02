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

# 1. Ruff - Format and lint all code
run_check "Ruff" "uv run ruff check . --fix"

# 3. Pylint - Deep code quality check
run_check "Pylint (source code)" "uv run pylint houyi/ --rcfile=.pylintrc"

# 4. Type checking with MyPy (optional, can be slow)
# run_check "MyPy" "mypy houyi/ --ignore-missing-imports"

# 5. Run tests
run_check "Unit Tests" "uv run pytest tests/ -v --tb=short -x"

# 6. Check test coverage
echo -e "${YELLOW}▶ Checking test coverage...${NC}"
uv run pytest tests/ --cov=houyi --cov-report=term-missing --cov-fail-under=75 || {
    echo -e "${RED}✗ Coverage below 75%${NC}"
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
