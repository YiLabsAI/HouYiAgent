#!/usr/bin/env bash
# Dev startup script - uses tmux to run backend + frontend and show logs

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./scripts/dev.sh

Start the local backend and frontend together in a tmux session.

Options:
  -h, --help    Show this help message and exit
EOF
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "Unknown option: $1" >&2
    echo "Run ./scripts/dev.sh -h for usage." >&2
    exit 2
fi

SESSION_NAME="houyi-dev"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed. Please install it first:"
    echo "   brew install tmux"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -d "${ROOT_DIR}/.venv" ]; then
    echo "❌ .venv not found. Run: uv sync --extra dev --extra studio-server --extra rag-full --extra memory --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability"
    exit 1
fi

# Stop existing frontend process(es)
echo "🔄 Stopping existing frontend process(es)..."
pkill -f "node.*vite" 2>/dev/null

# Stop existing backend process(es) (including those started outside tmux)
echo "🔄 Stopping existing backend process(es)..."
pkill -f "houyi_studio\.server\.app" 2>/dev/null

# If the tmux session already exists, kill it
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
    echo "🔄 Stopping existing tmux session..."
    tmux kill-session -t $SESSION_NAME
fi

echo "🚀 Starting dev environment..."
echo ""

# Web search cache defaults (can be overridden by exporting env vars before running this script)
WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED:-1}
WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL:-600}
WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE:-256}
WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS:-1}
# WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER:-ddg}
FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH:-${HOME}/.cache/fastembed}
EMBED_WARMUP_TIMEOUT_SECONDS=${EMBED_WARMUP_TIMEOUT_SECONDS:-30}

# Port configuration (override with env vars to avoid conflicts)
HOUYI_PORT=${HOUYI_PORT:-8000}
HOUYI_UI_PORT=${HOUYI_UI_PORT:-3000}
VITE_WS_HOST=${VITE_WS_HOST:-localhost:${HOUYI_PORT}}

# Create a new session and run backend
tmux new-session -d -s $SESSION_NAME -n "backend" "cd ${ROOT_DIR} && env FASTEMBED_CACHE_PATH=${FASTEMBED_CACHE_PATH} EMBED_WARMUP_TIMEOUT_SECONDS=${EMBED_WARMUP_TIMEOUT_SECONDS} WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED} WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL} WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE} WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS} HOUYI_PORT=${HOUYI_PORT} ./scripts/restart-backend.sh"

# Create a new window for frontend
tmux new-window -t $SESSION_NAME -n "frontend" "cd ${ROOT_DIR} && env VITE_WS_HOST=${VITE_WS_HOST} HOUYI_UI_PORT=${HOUYI_UI_PORT} ./scripts/restart-frontend.sh"

# Select backend window
tmux select-window -t $SESSION_NAME:0

echo "✅ Dev environment started!"
echo ""
echo "📋 Usage:"
echo "   - Attach to session: tmux attach -t $SESSION_NAME"
echo "   - Switch windows: Ctrl+B then 0 (backend) or 1 (frontend)"
echo "   - Detach (keep running): Ctrl+B then D"
echo "   - Stop all services: tmux kill-session -t $SESSION_NAME"
echo ""
echo "Attaching to session..."
sleep 2
tmux attach -t $SESSION_NAME
