#!/usr/bin/env bash
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

# 1. Ruff auto-fix
echo -e "${YELLOW}▶ Ruff auto-fix...${NC}"
if uv run ruff check . --fix --exclude skills/; then
    echo -e "${GREEN}✓ Ruff passed${NC}"
else
    echo -e "${RED}✗ Ruff failed${NC}"
    FAILED=1
fi
echo ""

# 2. Quick test run (fail fast)
echo -e "${YELLOW}▶ Quick test run...${NC}"
if uv run pytest tests/ -m "not benchmark" -x --tb=short -q -n auto; then
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
