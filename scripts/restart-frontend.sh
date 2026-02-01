#!/bin/bash
# Restart frontend service and stream logs

echo "🔄 Stopping existing frontend process..."
pkill -f "node.*vite" 2>/dev/null
sleep 1

echo "✅ Frontend process stopped"
echo ""
echo "🚀 Starting frontend service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd "$(dirname "$0")/../houyi-studio/ui" && npm run dev
