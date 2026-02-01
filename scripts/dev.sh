#!/bin/bash
# Dev startup script - uses tmux to run backend + frontend and show logs

SESSION_NAME="houyi-dev"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux is not installed. Please install it first:"
    echo "   brew install tmux"
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
WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER:-ddg}

# Create a new session and run backend
# NOTE: We intentionally avoid `conda run` here. In some environments, `conda run` can cause
# stdout/stderr capture issues under tmux/PTY (especially combined with uvicorn reload), which makes
# backend logs appear blank even though the server is running. Using `bash -lc` + `conda activate`
# keeps IO behavior consistent with an interactive shell.
tmux new-session -d -s $SESSION_NAME -n "backend" "cd ${ROOT_DIR} && bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate houyi && env WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED} WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL} WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE} WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS} WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER} python -m houyi_studio.server.app'"

# Create a new window for frontend
tmux new-window -t $SESSION_NAME -n "frontend" "cd ${ROOT_DIR}/houyi-studio/ui && npm run dev"

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
