#!/bin/bash
# Restart frontend service and stream logs

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UI_DIR="${ROOT_DIR}/houyi-studio/ui"

echo "🔄 Stopping existing frontend process..."
pkill -f "node.*vite" 2>/dev/null || true
sleep 1

echo "✅ Frontend process stopped"
echo ""

# Ensure pnpm dependencies are installed
if [ ! -d "${UI_DIR}/node_modules" ]; then
    echo "📦 Installing UI dependencies..."
    cd "${UI_DIR}" && pnpm install --frozen-lockfile
fi

echo "🚀 Starting frontend service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "${UI_DIR}" && pnpm run dev:strict-3000
