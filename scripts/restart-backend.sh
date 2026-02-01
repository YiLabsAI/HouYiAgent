#!/bin/bash
# Restart backend service and stream logs

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "🔄 Stopping existing backend process..."
# Stop existing backend process(es) (including those started outside tmux)
pkill -f "houyi_studio\.server\.app" 2>/dev/null || true
sleep 1

echo "✅ Backend process stopped"
echo ""
echo "🚀 Starting backend service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Web search cache defaults (can be overridden by exporting env vars before running this script)
WEB_SEARCH_CACHE_ENABLED=${WEB_SEARCH_CACHE_ENABLED:-1}
WEB_SEARCH_CACHE_TTL=${WEB_SEARCH_CACHE_TTL:-600}
WEB_SEARCH_CACHE_MAX_SIZE=${WEB_SEARCH_CACHE_MAX_SIZE:-256}
WEB_SEARCH_CACHE_LOG_HITS=${WEB_SEARCH_CACHE_LOG_HITS:-1}
WEB_SEARCH_PROVIDER=${WEB_SEARCH_PROVIDER:-ddg}

cd "$ROOT_DIR"

bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate houyi && env WEB_SEARCH_CACHE_ENABLED='"${WEB_SEARCH_CACHE_ENABLED}"' WEB_SEARCH_CACHE_TTL='"${WEB_SEARCH_CACHE_TTL}"' WEB_SEARCH_CACHE_MAX_SIZE='"${WEB_SEARCH_CACHE_MAX_SIZE}"' WEB_SEARCH_CACHE_LOG_HITS='"${WEB_SEARCH_CACHE_LOG_HITS}"' WEB_SEARCH_PROVIDER='"${WEB_SEARCH_PROVIDER}"' python -m houyi_studio.server.app'
