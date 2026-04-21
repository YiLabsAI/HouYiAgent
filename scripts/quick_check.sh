#!/usr/bin/env bash
# Quick code quality check (faster, for frequent use)
# Use this during development for quick feedback

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./scripts/quick_check.sh

Run the fast local development checks.

Options:
  -h, --help    Show this help message and exit
EOF
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "Unknown option: $1" >&2
    echo "Run ./scripts/quick_check.sh -h for usage." >&2
    exit 2
fi

set -e

echo "⚡ Running quick checks..."
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILED=0

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}✗ uv is not installed or not on PATH.${NC}"
    echo "Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ .venv not found.${NC}"
    echo "Run: uv sync --extra dev"
    exit 1
fi

# 1. Ruff auto-fix
echo -e "${YELLOW}▶ Ruff auto-fix...${NC}"
if uv run ruff check . --fix --exclude skills/; then
    echo -e "${GREEN}✓ Ruff passed${NC}"
else
    echo -e "${RED}✗ Ruff failed${NC}"
    FAILED=1
fi
echo ""

# 2. Style rules (no raw CJK, short test names, -h/--help on scripts/ entrypoints)
# Rules 1-3 run against changed Python files; rule 4 is always enforced across
# every scripts/ entrypoint so a new script cannot silently drop -h support.
echo -e "${YELLOW}▶ Style rules (CJK / test names / help)...${NC}"
CHANGED=$(
    {
        git diff --name-only --cached
        git diff --name-only
        git ls-files --others --exclude-standard
    } 2>/dev/null | awk '($0 ~ /\.pyi?$/) && ($0 !~ /^skills\//) { print }' | sort -u | tr '\n' ' '
)
# shellcheck disable=SC2086
if uv run python scripts/check_style_rules.py $CHANGED scripts/*.sh scripts/*.py; then
    echo -e "${GREEN}✓ Style rules passed${NC}"
else
    echo -e "${RED}✗ Style rules failed${NC}"
    FAILED=1
fi
echo ""

# 3. Quick test run (fail fast)
echo -e "${YELLOW}▶ Quick test run...${NC}"
if uv run pytest tests/ -m "not benchmark" -x --tb=short -q -n auto; then
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    FAILED=1
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Quick checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Quick checks failed.${NC}"
    exit 1
fi
