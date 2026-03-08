#!/usr/bin/env bash
# Local-only integration gate.
# Run this manually after `make check` when you need to validate env-backed
# integrations under tests/integration/.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}✗ uv is not installed or not on PATH.${NC}"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ .venv not found.${NC}"
    echo "Run: uv sync --extra dev --extra rag-full --extra vertex-ai"
    exit 1
fi

ensure_integration_deps() {
    echo -e "${YELLOW}▶ Verifying local integration dependencies...${NC}"

    local need_sync=0

    if ! uv run python -c "import pytest" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import dotenv" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "from google import genai" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import google.auth" 2>/dev/null; then need_sync=1; fi

    if [ $need_sync -eq 1 ]; then
        echo -e "${YELLOW}  Installing missing local integration dependencies...${NC}"
        uv sync --extra dev --extra rag-full --extra vertex-ai --quiet
    fi

    echo -e "${GREEN}✓ Local integration dependencies verified${NC}"
    echo ""
}

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
        echo -e "${RED}  Fix the errors above, then re-run: make check-integration${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 1
    fi
}

ensure_integration_deps

run_check "Integration Tests" \
    uv run python -m pytest tests/integration/ houyi-studio/server/tests/integration/ -v -s

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Local integration gate passed.${NC}"
