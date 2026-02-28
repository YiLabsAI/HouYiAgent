#!/usr/bin/env bash
# Restart backend service and stream logs

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -d "${ROOT_DIR}/.venv" ]; then
    echo "❌ .venv not found. Run: uv sync --extra dev --extra rag-full"
    exit 1
fi

echo "🔄 Stopping existing backend process..."
# Stop existing backend process(es) (including those started outside tmux)
pkill -f "houyi_studio\.server" 2>/dev/null || true
PIDS=$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null || true)
if [ -n "${PIDS}" ]; then
    echo "🔪 Killing process(es) listening on port 8000: ${PIDS}"
    kill -TERM ${PIDS} 2>/dev/null || true
    sleep 1
    kill -KILL ${PIDS} 2>/dev/null || true
fi
sleep 1

echo "✅ Backend process stopped"
echo ""

cd "$ROOT_DIR"

# Ensure SDK + RAG deps are synced
echo "🔄 Syncing SDK dependencies..."
uv sync --extra dev --extra rag-full --extra websearch-ddg --extra websearch-tavily --extra websearch-readability --quiet

# Ensure houyi-studio-server is installed (uv sync may uninstall it)
if ! uv run python -c "import houyi_studio" 2>/dev/null; then
    echo "📦 Installing houyi-studio-server..."
    uv pip install -e houyi-studio/server --quiet
fi

FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH:-${ROOT_DIR}/.cache/fastembed}

echo "🚀 Warming up embedding models..."
env FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} uv run python "${ROOT_DIR}/scripts/warmup_embeddings.py"
echo "✅ Embedding warmup complete"

echo "🚀 Starting backend service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Web search cache defaults (can be overridden by exporting env vars before running this script)
WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED:-1}
WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL:-600}
WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE:-256}
WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS:-1}
WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER:-ddg}

env FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED} WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL} WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE} WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS} WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER} uv run python -m houyi_studio.server
