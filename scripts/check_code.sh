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

PYTHON_BIN=${PYTHON_BIN:-python3}
CONDA_ENV=${CONDA_ENV:-houyi}

resolve_cmd() {
    local bin_name=$1
    local py_module=$2

    if command -v "$bin_name" >/dev/null 2>&1; then
        echo "$bin_name"
        return 0
    fi

    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        if $PYTHON_BIN -c "import $py_module" >/dev/null 2>&1; then
            echo "$PYTHON_BIN -m $py_module"
            return 0
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
            if conda run -n "$CONDA_ENV" python -c "import $py_module" >/dev/null 2>&1; then
                echo "conda run -n $CONDA_ENV python -m $py_module"
                return 0
            fi
        fi
    fi

    return 1
}

RUFF_CMD="$(resolve_cmd ruff ruff)" || RUFF_CMD=""
PYLINT_CMD="$(resolve_cmd pylint pylint)" || PYLINT_CMD=""
PYTEST_CMD="$(resolve_cmd pytest pytest)" || PYTEST_CMD=""

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
run_check "Ruff" "[ -n '$RUFF_CMD' ] && $RUFF_CMD check . --fix"

# 3. Pylint - Deep code quality check
run_check "Pylint (source code)" "[ -n '$PYLINT_CMD' ] && $PYLINT_CMD houyi/ --rcfile=.pylintrc"

# 4. Type checking with MyPy (optional, can be slow)
# run_check "MyPy" "mypy houyi/ --ignore-missing-imports"

# 5. Run tests
run_check "Unit Tests" "[ -n '$PYTEST_CMD' ] && $PYTEST_CMD tests/ -v --tb=short -x"

# 6. Check test coverage
echo -e "${YELLOW}▶ Checking test coverage...${NC}"
[ -n "$PYTEST_CMD" ] && $PYTEST_CMD tests/ --cov=houyi --cov-report=term-missing --cov-fail-under=75 || {
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
