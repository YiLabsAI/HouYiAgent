#!/usr/bin/env bash
# Restart backend service and stream logs

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./scripts/restart-backend.sh

Restart the local backend service and stream logs.

Options:
  -h, --help    Show this help message and exit
EOF
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "Unknown option: $1" >&2
    echo "Run ./scripts/restart-backend.sh -h for usage." >&2
    exit 2
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -d "${ROOT_DIR}/.venv" ]; then
    echo "❌ .venv not found. Run: uv sync --extra dev --extra studio-server --extra rag-full --extra memory --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability"
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

# Sync backend runtime dependencies from the root project extras
echo "🔄 Syncing backend dependencies..."
uv sync --extra dev --extra studio-server --extra rag-full --extra memory --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability --quiet

# Ensure the local houyi_studio package is linked into the venv after sync
if ! uv run python -c "import houyi_studio" 2>/dev/null; then
    echo "� Linking local houyi-studio-server package..."
    uv pip install --no-deps -e houyi-studio/server --quiet
    uv run python -c "import houyi_studio"
fi

FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH:-${HOME}/.cache/fastembed}
EMBED_WARMUP_TIMEOUT_SECONDS=${EMBED_WARMUP_TIMEOUT_SECONDS:-30}

echo "🚀 Warming up embedding models..."
(env FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} EMBED_WARMUP_TIMEOUT_SECONDS=${EMBED_WARMUP_TIMEOUT_SECONDS} uv run python -c "import os, subprocess, sys; timeout=int(os.environ.get('EMBED_WARMUP_TIMEOUT_SECONDS', '30')); env=os.environ.copy(); cmd=['python', '${ROOT_DIR}/scripts/warmup_embeddings.py']; result=subprocess.run(cmd, env=env, timeout=timeout, check=False); sys.exit(result.returncode)" && echo "✅ Embedding warmup complete") || echo "⚠️ Embedding warmup failed or timed out; continuing backend startup without warm cache"

echo "🚀 Starting backend service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Web search cache defaults (can be overridden by exporting env vars before running this script)
WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED:-1}
WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL:-600}
WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE:-256}
WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS:-1}
# WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER:-ddg}

env FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED} WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL} WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE} WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS} uv run python -m houyi_studio.server
