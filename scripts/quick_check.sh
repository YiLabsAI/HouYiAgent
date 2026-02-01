#!/bin/bash
# Quick code quality check (faster, for frequent use)
# Use this during development for quick feedback

set -e

echo "⚡ Running quick checks..."
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
PYTEST_CMD="$(resolve_cmd pytest pytest)" || PYTEST_CMD=""

# 1. Ruff auto-fix
echo -e "${YELLOW}▶ Ruff auto-fix...${NC}"
if [ -n "$RUFF_CMD" ] && $RUFF_CMD check . --fix; then
    echo -e "${GREEN}✓ Ruff passed${NC}"
else
    echo -e "${RED}✗ Ruff failed${NC}"
    FAILED=1
fi
echo ""

# 2. Quick test run (fail fast)
echo -e "${YELLOW}▶ Quick test run...${NC}"
if [ -n "$PYTEST_CMD" ] && $PYTEST_CMD tests/ -x --tb=short -q; then
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    FAILED=1
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Quick checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Quick checks failed.${NC}"
    exit 1
fi
