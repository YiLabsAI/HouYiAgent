#!/usr/bin/env bash
# Local-only integration gate.
# Run this manually after `make check` when you need to validate env-backed
# integrations under tests/integration/.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./scripts/check_integration.sh

Run the local integration gate for env-backed integration coverage.

Options:
  -h, --help    Show this help message and exit
EOF
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "Unknown option: $1" >&2
    echo "Run ./scripts/check_integration.sh -h for usage." >&2
    exit 2
fi

set -euo pipefail

# Prevent uv from rewriting uv.lock with local mirror URLs.
# UV_NO_CONFIG ignores ~/.config/uv/uv.toml (may contain extra-index-url).
# UV_FROZEN prevents lock file updates; lock changes go through `make lock`.
export UV_NO_CONFIG=1
export UV_INDEX_URL=https://pypi.org/simple
export UV_FROZEN=1

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOUYI_INTEGRATION_BACKEND_PORT=${HOUYI_INTEGRATION_BACKEND_PORT:-18000}
FASTEMBED_SEED_CACHE_PATH=${FASTEMBED_SEED_CACHE_PATH:-${HOME}/.cache/fastembed}
HOUYI_INTEGRATION_FASTEMBED_CACHE=${HOUYI_INTEGRATION_FASTEMBED_CACHE:-${ROOT_DIR}/.cache/integration/fastembed}
FASTEMBED_CACHE_PATH=${HOUYI_INTEGRATION_FASTEMBED_CACHE}
INTEGRATION_BACKEND_PID=""

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}✗ uv is not installed or not on PATH.${NC}"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ .venv not found.${NC}"
    echo "Run: uv sync --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability"
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
        uv sync --frozen --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability --quiet
    fi

    if ! uv run python -c "import houyi_studio" 2>/dev/null; then
        echo -e "${YELLOW}  Linking local houyi-studio-server package...${NC}"
        uv pip install --no-deps -e houyi-studio/server --quiet
    fi

    echo -e "${GREEN}✓ Local integration dependencies verified${NC}"
    echo ""
}

fastembed_cache_ready() {
    local cache_path=$1
    env FASTEMBED_CACHE_PATH=${cache_path} uv run python - <<'PY'
from pathlib import Path
import os

cache_root = Path(os.environ["FASTEMBED_CACHE_PATH"])
model_dir = cache_root / "models--qdrant--bge-small-en-v1.5-onnx-q" / "snapshots"
if not model_dir.is_dir():
    raise SystemExit(1)
for rev_dir in model_dir.iterdir():
    onnx = rev_dir / "model_optimized.onnx"
    if onnx.is_file() and onnx.stat().st_size >= 60_000_000:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

seed_fastembed_cache() {
    env FASTEMBED_SEED_CACHE_PATH=${FASTEMBED_SEED_CACHE_PATH} FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} uv run python - <<'PY'
from pathlib import Path
import os
import shutil

source_root = Path(os.environ["FASTEMBED_SEED_CACHE_PATH"])
target_root = Path(os.environ["FASTEMBED_CACHE_PATH"])
model_name = "models--qdrant--bge-small-en-v1.5-onnx-q"
source_model_dir = source_root / model_name
target_model_dir = target_root / model_name

if not source_model_dir.exists():
    raise SystemExit(1)

target_root.mkdir(parents=True, exist_ok=True)
if target_model_dir.exists():
    shutil.rmtree(target_model_dir)
shutil.copytree(source_model_dir, target_model_dir)
raise SystemExit(0)
PY
}

ensure_fastembed_test_cache() {
    mkdir -p "${FASTEMBED_CACHE_PATH}"

    if fastembed_cache_ready "${FASTEMBED_CACHE_PATH}"; then
        echo -e "${GREEN}✓ Integration fastembed cache ready at ${FASTEMBED_CACHE_PATH}${NC}"
        echo ""
        return
    fi

    if [ "${FASTEMBED_SEED_CACHE_PATH}" != "${FASTEMBED_CACHE_PATH}" ] && fastembed_cache_ready "${FASTEMBED_SEED_CACHE_PATH}"; then
        echo -e "${YELLOW}▶ Seeding integration fastembed cache from ${FASTEMBED_SEED_CACHE_PATH}...${NC}"
        if seed_fastembed_cache; then
            if fastembed_cache_ready "${FASTEMBED_CACHE_PATH}"; then
                echo -e "${GREEN}✓ Seeded integration fastembed cache at ${FASTEMBED_CACHE_PATH}${NC}"
                echo ""
                return
            fi
        fi
    fi

    echo -e "${YELLOW}▶ Warming integration fastembed cache at ${FASTEMBED_CACHE_PATH}...${NC}"
    env \
        FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} \
        EMBEDDING_PROVIDER=local \
        EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
        HF_HUB_OFFLINE=0 \
        uv run python scripts/warmup_embeddings.py

    if ! fastembed_cache_ready "${FASTEMBED_CACHE_PATH}"; then
        echo -e "${RED}✗ Integration fastembed cache warmup failed at ${FASTEMBED_CACHE_PATH}.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Warmed integration fastembed cache at ${FASTEMBED_CACHE_PATH}${NC}"
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
        FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} \
        HOUYI_PORT=${HOUYI_INTEGRATION_BACKEND_PORT} \
        HOUYI_CHAT_DATA_DIR=${chat_data_dir} \
        HOUYI_CHAT_SETTINGS_PATH=${settings_path} \
        HOUYI_KNOWLEDGE_STORAGE=${knowledge_storage} \
        OPENAI_API_KEY= \
        SILICONFLOW_API_KEY= \
        GOOGLE_API_KEY= \
        GOOGLE_CLOUD_PROJECT= \
        GOOGLE_APPLICATION_CREDENTIALS= \
        GEMINI_API_KEY= \
        TAVILY_API_KEY= \
        SERPER_API_KEY= \
        BOCHA_API_KEY= \
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
ensure_fastembed_test_cache
start_isolated_backend

# Parallelize integration pytest (same isolated backend; HTTP is concurrent-safe).
# -v/--tb=short: per-worker PASSED lines (easier to spot failures than -q progress dots).
run_check "Integration Tests" \
    env FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} HOUYI_PORT=${HOUYI_INTEGRATION_BACKEND_PORT} HOUYI_INTEGRATION_BACKEND_PORT=${HOUYI_INTEGRATION_BACKEND_PORT} OPENAI_API_KEY= SILICONFLOW_API_KEY= GOOGLE_API_KEY= GOOGLE_CLOUD_PROJECT= GOOGLE_APPLICATION_CREDENTIALS= GEMINI_API_KEY= TAVILY_API_KEY= SERPER_API_KEY= BOCHA_API_KEY= uv run python -m pytest tests/integration/ --ignore=tests/integration/live -m "not benchmark" houyi-studio/server/tests/integration/ -v --tb=short -n auto

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Local integration gate passed.${NC}"
