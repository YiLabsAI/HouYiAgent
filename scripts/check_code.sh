#!/usr/bin/env bash
# Code quality check script — fail-fast mode
# Run this before committing code.
# Stops at the FIRST failure so the error output stays visible.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./scripts/check_code.sh

Run the main local code-quality gate.

Options:
  -h, --help    Show this help message and exit
EOF
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "Unknown option: $1" >&2
    echo "Run ./scripts/check_code.sh -h for usage." >&2
    exit 2
fi

set -euo pipefail

# Prevent uv from rewriting uv.lock with local mirror URLs.
# UV_NO_CONFIG ignores ~/.config/uv/uv.toml (may contain extra-index-url).
# UV_FROZEN prevents lock file updates; lock changes go through `make lock`.
export UV_NO_CONFIG=1
export UV_INDEX_URL=https://pypi.org/simple
export UV_FROZEN=1

COVERAGE_MIN=${HOUYI_SDK_COVERAGE_MIN:-90}
CHECK_STARTED_AT=$(python3 -c 'import time; print(time.perf_counter())')
STEP_TIMINGS=()
CHECK_FAILED=0
FAILED_STEP=""
TIMING_FILE="${HOUYI_CHECK_TIMING_FILE:-}"
SUPPRESS_SUMMARY="${HOUYI_CHECK_SUPPRESS_SUMMARY:-0}"

echo "🔍 Running code quality checks..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v uv >/dev/null 2>&1; then
    echo -e "${RED}✗ uv is not installed or not on PATH.${NC}"
    echo "Install uv first: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ .venv not found.${NC}"
    echo "Run: uv sync --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability"
    exit 1
fi

# ── Ensure dependencies ──────────────────────────────────────────────
ensure_deps() {
    echo -e "${YELLOW}▶ Verifying dependencies...${NC}"

    local need_sync=0

    if ! uv run python -c "import pytest" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import pytest_cov" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import xdist" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import mypy" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import importlinter" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "from google import genai" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import google.auth" 2>/dev/null; then need_sync=1; fi
    if ! uv run python -c "import bm25s" 2>/dev/null; then need_sync=1; fi

    if [ $need_sync -eq 1 ]; then
        echo -e "${YELLOW}  Installing missing dev dependencies...${NC}"
        uv sync --frozen --extra dev --extra studio-server --extra rag-full --extra model-adapters --extra vertex-ai --extra websearch-ddg --extra websearch-tavily --extra websearch-readability --quiet
    fi

    if ! uv run python -c "import houyi_studio" 2>/dev/null; then
        echo -e "${YELLOW}  Installing studio server...${NC}"
        uv pip install -e houyi-studio/server --quiet
    fi

    echo -e "${GREEN}✓ Dependencies verified${NC}"
    echo ""
}

ensure_deps

COMPLEXITY_RULES="C901,PLR0912,PLR0915"
CLASS_WARN_LINES="${HOUYI_CLASS_WARN_LINES:-500}"
CLASS_ERROR_LINES="${HOUYI_CLASS_ERROR_LINES:-800}"

# ── Fail-fast runner ─────────────────────────────────────────────────
# Runs the command; on failure prints a clear banner and exits immediately.
run_check() {
    local name=$1
    shift
    echo -e "${YELLOW}▶ Running $name...${NC}"
    local started_at
    local elapsed_s
    started_at=$(python3 -c 'import time; print(time.perf_counter())')
    if "$@"; then
        elapsed_s=$(python3 - "$started_at" <<'PY'
import sys, time
print(f"{time.perf_counter() - float(sys.argv[1]):.2f}")
PY
)
        STEP_TIMINGS+=("$name:$elapsed_s")
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        elapsed_s=$(python3 - "$started_at" <<'PY'
import sys, time
print(f"{time.perf_counter() - float(sys.argv[1]):.2f}")
PY
)
        STEP_TIMINGS+=("$name:$elapsed_s")
        CHECK_FAILED=1
        FAILED_STEP="$name"
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}✗ $name FAILED — stopping here.${NC}"
        echo -e "${RED}  Fix the errors above, then re-run: make check${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        exit 1
    fi
}

run_parallel_test_checks() {
    echo -e "${YELLOW}▶ Running SDK Tests + Coverage and Server Tests in parallel...${NC}"

    local tmp_dir
    local sdk_cov_dir
    local server_cov_dir
    local sdk_elapsed_file
    local server_elapsed_file
    local sdk_pid
    local server_pid
    local sdk_rc=0
    local server_rc=0
    local sdk_elapsed_s="0.00"
    local server_elapsed_s="0.00"

    tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/houyi-check-code.XXXXXX")
    sdk_cov_dir="${tmp_dir}/sdk"
    server_cov_dir="${tmp_dir}/server"
    sdk_elapsed_file="${tmp_dir}/sdk.elapsed"
    server_elapsed_file="${tmp_dir}/server.elapsed"
    mkdir -p "${sdk_cov_dir}" "${server_cov_dir}"

    (
        started_at=$(python3 -c 'import time; print(time.perf_counter())')
        COVERAGE_FILE="${sdk_cov_dir}/.coverage" uv run pytest tests/ --ignore=tests/integration -m "not benchmark" -x -n auto \
            --cov=houyi --cov-report=
        rc=$?
        elapsed_s=$(python3 - "$started_at" <<'PY'
import sys, time
print(f"{time.perf_counter() - float(sys.argv[1]):.2f}")
PY
)
        printf '%s\n' "$elapsed_s" > "$sdk_elapsed_file"
        exit "$rc"
    ) &
    sdk_pid=$!

    (
        started_at=$(python3 -c 'import time; print(time.perf_counter())')
        COVERAGE_FILE="${server_cov_dir}/.coverage" uv run pytest houyi-studio/server/tests/ -m "not benchmark" \
            --ignore=houyi-studio/server/tests/integration -x -n 2 --cov=houyi --cov-report=
        rc=$?
        elapsed_s=$(python3 - "$started_at" <<'PY'
import sys, time
print(f"{time.perf_counter() - float(sys.argv[1]):.2f}")
PY
)
        printf '%s\n' "$elapsed_s" > "$server_elapsed_file"
        exit "$rc"
    ) &
    server_pid=$!

    if wait "$sdk_pid"; then
        sdk_rc=0
    else
        sdk_rc=$?
    fi

    if wait "$server_pid"; then
        server_rc=0
    else
        server_rc=$?
    fi

    if [ -f "$sdk_elapsed_file" ]; then
        sdk_elapsed_s=$(cat "$sdk_elapsed_file")
    fi
    if [ -f "$server_elapsed_file" ]; then
        server_elapsed_s=$(cat "$server_elapsed_file")
    fi

    STEP_TIMINGS+=("SDK Tests + Coverage:$sdk_elapsed_s")
    STEP_TIMINGS+=("Server Tests:$server_elapsed_s")

    if [ "$sdk_rc" -ne 0 ] || [ "$server_rc" -ne 0 ]; then
        CHECK_FAILED=1
        if [ "$sdk_rc" -ne 0 ]; then
            FAILED_STEP="SDK Tests + Coverage"
        else
            FAILED_STEP="Server Tests"
        fi
        echo ""
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}✗ ${FAILED_STEP} FAILED — stopping here.${NC}"
        echo -e "${RED}  Fix the errors above, then re-run: make check${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        rm -rf "$tmp_dir"
        exit 1
    fi

    echo -e "${GREEN}✓ SDK Tests + Coverage passed${NC}"
    echo -e "${GREEN}✓ Server Tests passed${NC}"
    echo ""

    run_check "SDK Native Coverage Gate" \
        env COVERAGE_FILE="${sdk_cov_dir}/.coverage" uv run coverage report --fail-under="$COVERAGE_MIN"
    run_check "Coverage Data Combine" \
        env COVERAGE_FILE="${tmp_dir}/.coverage" uv run coverage combine "${sdk_cov_dir}/.coverage" "${server_cov_dir}/.coverage"
    run_check "SDK Coverage Gate" \
        env COVERAGE_FILE="${tmp_dir}/.coverage" uv run coverage report --fail-under="$COVERAGE_MIN"

    rm -rf "$tmp_dir"
}

print_timing_summary() {
    local total_seconds
    total_seconds=$(python3 - "$CHECK_STARTED_AT" <<'PY'
import sys, time
print(f"{time.perf_counter() - float(sys.argv[1]):.2f}")
PY
)

    if [ -n "$TIMING_FILE" ]; then
        : > "$TIMING_FILE"
        for step in "${STEP_TIMINGS[@]}"; do
            printf '%s\n' "$step" >> "$TIMING_FILE"
        done
        printf 'total:%s\n' "$total_seconds" >> "$TIMING_FILE"
        if [ "$CHECK_FAILED" -eq 1 ]; then
            printf 'status:failed:%s\n' "$FAILED_STEP" >> "$TIMING_FILE"
        else
            printf 'status:passed\n' >> "$TIMING_FILE"
        fi
    fi

    if [ "$SUPPRESS_SUMMARY" = "1" ]; then
        return
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "check-unit timing summary"
    for step in "${STEP_TIMINGS[@]}"; do
        echo "  - ${step} s"
    done
    echo "  - total:${total_seconds} s"
    if [ "$CHECK_FAILED" -eq 1 ]; then
        echo "  - status: failed at ${FAILED_STEP}"
    else
        echo "  - status: passed"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

trap print_timing_summary EXIT

# ── 1. Ruff lint + format (changed files only) ────────────────────
CHANGED_PY_FILES=$(
    {
        git diff --name-only --cached
        git diff --name-only
        git ls-files --others --exclude-standard
    } 2>/dev/null | awk '($0 ~ /\.pyi?$/) && ($0 !~ /^skills\//) { print }' | sort -u | while read -r f; do
        if [ -f "$f" ]; then
            echo "$f"
        fi
    done
)

if [ -n "$CHANGED_PY_FILES" ]; then
    CHANGED_PY_FILES_ONELINE=$(echo "$CHANGED_PY_FILES" | tr '\n' ' ')
    run_check "Ruff (lint)" uv run ruff check --fix $CHANGED_PY_FILES_ONELINE
    run_check "Ruff (format)" uv run ruff format $CHANGED_PY_FILES_ONELINE
else
    echo -e "${YELLOW}▶ Ruff (lint/format) skipped (no changed Python files)${NC}"
    echo ""
fi

# ── 2. SDK complexity housekeeping (report + changed-file gate) ─────
echo -e "${YELLOW}▶ Running SDK Complexity Report (full houyi/, non-blocking)...${NC}"
COMPLEXITY_REPORT=$(uv run ruff check houyi --select "$COMPLEXITY_RULES" --output-format concise 2>/dev/null || true)
if [ -n "$COMPLEXITY_REPORT" ]; then
    COMPLEXITY_COUNT=$(echo "$COMPLEXITY_REPORT" | wc -l | tr -d ' ')
    echo -e "${YELLOW}  Found ${COMPLEXITY_COUNT} complexity findings in houyi/ (report-only).${NC}"
    echo "$COMPLEXITY_REPORT" | head -n 20
    if [ "$COMPLEXITY_COUNT" -gt 20 ]; then
        echo "  ... (${COMPLEXITY_COUNT} total findings)"
    fi
else
    echo -e "${GREEN}  No complexity findings in houyi/.${NC}"
fi
echo ""

CHANGED_SDK_FILES=$(echo "$CHANGED_PY_FILES" | awk '/^houyi\// { print }')
if [ -n "$CHANGED_SDK_FILES" ]; then
    CHANGED_SDK_FILES_ONELINE=$(echo "$CHANGED_SDK_FILES" | tr '\n' ' ')
    run_check "SDK Complexity Gate (changed files)" \
        uv run ruff check --select "$COMPLEXITY_RULES" $CHANGED_SDK_FILES_ONELINE
else
    echo -e "${YELLOW}▶ SDK Complexity Gate skipped (no changed files under houyi/)${NC}"
    echo ""
fi

if [ -n "$CHANGED_SDK_FILES" ]; then
    run_check "Class Size Gate (changed SDK files)" \
        uv run python scripts/check_class_size.py \
            --warn-lines "$CLASS_WARN_LINES" \
            --error-lines "$CLASS_ERROR_LINES" \
            $CHANGED_SDK_FILES_ONELINE
else
    echo -e "${YELLOW}▶ Class Size Gate skipped (no changed files under houyi/)${NC}"
    echo ""
fi

# ── 3. Type check ───────────────────────────────────────────────────
run_check "Type Check (mypy)" uv run mypy houyi/

# ── 4. SDK + server tests (parallel) with combined SDK coverage ─────
run_parallel_test_checks

# ── Done ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ All checks passed! Ready to commit.${NC}"
