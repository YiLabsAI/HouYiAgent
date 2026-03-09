#!/usr/bin/env bash
# Local-only integration gate.
# Run this manually after `make check` when you need to validate env-backed
# integrations under tests/integration/.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOUYI_INTEGRATION_BACKEND_PORT=${HOUYI_INTEGRATION_BACKEND_PORT:-18000}
INTEGRATION_BACKEND_PID=""

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}✗ uv is not installed or not on PATH.${NC}"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ .venv not found.${NC}"
    echo "Run: uv sync --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai"
    exit 1
fi

ensure_integration_deps() {
    echo -e "${YELLOW}▶ Verifying local integration dependencies...${NC}"

    local need_sync=0

    if ! uv run python -c "import pytest" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import dotenv" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "from google import genai" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import google.auth" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import houyi_studio" 2>/dev/null; then need_sync=1; fi

    if [ $need_sync -eq 1 ]; then
        echo -e "${YELLOW}  Installing missing local integration dependencies...${NC}"
        uv sync --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --quiet
    fi

    if ! uv run python -c "import houyi_studio" 2>/dev/null; then
        echo -e "${YELLOW}  Linking local houyi-studio-server package...${NC}"
        uv pip install --no-deps -e houyi-studio/server --quiet
    fi

    echo -e "${GREEN}✓ Local integration dependencies verified${NC}"
    echo ""
}

cleanup_backend() {
    if [ -n "${INTEGRATION_BACKEND_PID}" ] && kill -0 "${INTEGRATION_BACKEND_PID}" 2>/dev/null; then
        kill "${INTEGRATION_BACKEND_PID}" 2>/dev/null || true
        wait "${INTEGRATION_BACKEND_PID}" 2>/dev/null || true
    fi
}

start_isolated_backend() {
    echo -e "${YELLOW}▶ Starting isolated integration backend on port ${HOUYI_INTEGRATION_BACKEND_PORT}...${NC}"
    trap cleanup_backend EXIT

    local chat_data_dir
    local settings_dir
    local settings_path
    local knowledge_storage
    chat_data_dir=$(mktemp -d "${TMPDIR:-/tmp}/houyi-integration-chat.XXXXXX")
    settings_dir=$(mktemp -d "${TMPDIR:-/tmp}/houyi-integration-settings.XXXXXX")
    settings_path="${settings_dir}/settings.json"
    knowledge_storage=$(mktemp -d "${TMPDIR:-/tmp}/houyi-integration-knowledge.XXXXXX")

    env \
        HOUYI_PORT=${HOUYI_INTEGRATION_BACKEND_PORT} \
        HOUYI_CHAT_DATA_DIR=${chat_data_dir} \
        HOUYI_CHAT_SETTINGS_PATH=${settings_path} \
        HOUYI_KNOWLEDGE_STORAGE=${knowledge_storage} \
        HOUYI_EMBEDDING_STARTUP_TIMEOUT_SECONDS=${HOUYI_EMBEDDING_STARTUP_TIMEOUT_SECONDS:-1} \
        uv run python -m houyi_studio.server >/tmp/houyi-check-integration-backend.log 2>&1 &
    INTEGRATION_BACKEND_PID=$!

    for _ in $(seq 1 40); do
        if lsof -nP -iTCP:${HOUYI_INTEGRATION_BACKEND_PORT} -sTCP:LISTEN >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Isolated integration backend is ready${NC}"
            echo ""
            return
        fi
        sleep 0.5
    done

    echo -e "${RED}✗ Isolated integration backend did not start in time.${NC}"
    echo -e "${RED}  See /tmp/houyi-check-integration-backend.log for details.${NC}"
    exit 1
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
start_isolated_backend

run_check "Integration Tests" \
    env HOUYI_PORT=${HOUYI_INTEGRATION_BACKEND_PORT} HOUYI_INTEGRATION_BACKEND_PORT=${HOUYI_INTEGRATION_BACKEND_PORT} uv run python -m pytest tests/integration/ houyi-studio/server/tests/integration/ -v -s

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Local integration gate passed.${NC}"
