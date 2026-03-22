#!/usr/bin/env bash
# Pre-flight test suite for autoresearch.
# Run this before any deployment or after environment changes.
# Usage: bash scripts/test.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC}: $1"; }
fail() { echo -e "${RED}FAIL${NC}: $1"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== autoresearch pre-flight tests ==="
echo ""

# 1. Python version
echo -n "Python version... "
PYVER=$(uv run python --version 2>&1)
if [[ "$PYVER" == *"3.12"* ]] || [[ "$PYVER" == *"3.13"* ]]; then
    pass "$PYVER"
else
    fail "Expected Python 3.12+, got: $PYVER"
fi

# 2. Key dependencies
echo -n "PyTorch import... "
uv run python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')" 2>/dev/null && pass "OK" || fail "torch import failed"

echo -n "Anthropic SDK... "
uv run python -c "import anthropic; print(f'v{anthropic.__version__}')" 2>/dev/null && pass "OK" || fail "anthropic import failed"

# 3. GPU check
echo -n "GPU available... "
GPU_OK=$(uv run python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)
if [ "$GPU_OK" = "yes" ]; then
    GPU_NAME=$(uv run python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    pass "$GPU_NAME"
else
    fail "No GPU detected"
fi

# 4. train.py matches .train_clean.py
echo -n "train.py == .train_clean.py... "
if diff -q train.py .train_clean.py > /dev/null 2>&1; then
    pass "files match"
else
    fail "train.py and .train_clean.py differ! Run: cp train.py .train_clean.py"
fi

# 5. API key
echo -n "ANTHROPIC_API_KEY set... "
if [ -n "$ANTHROPIC_API_KEY" ]; then
    pass "...${ANTHROPIC_API_KEY: -4}"
else
    fail "not set"
fi

# 6. Sonnet API call
echo -n "Sonnet 4.6 API... "
SONNET_OK=$(uv run python -c "
import anthropic
c = anthropic.Anthropic(timeout=15.0)
r = c.messages.create(model='claude-sonnet-4-6', max_tokens=5, messages=[{'role':'user','content':'hi'}])
print('ok')
" 2>/dev/null)
if [ "$SONNET_OK" = "ok" ]; then
    pass "claude-sonnet-4-6 responds"
else
    fail "Sonnet API call failed"
fi

# 7. Opus API call (optional -- only if --opus flag passed)
if [[ "$*" == *"--opus"* ]]; then
    echo -n "Opus 4.6 API... "
    OPUS_OK=$(uv run python -c "
import anthropic
c = anthropic.Anthropic(timeout=30.0)
r = c.messages.create(model='claude-opus-4-6', max_tokens=36000,
    thinking={'type':'enabled','budget_tokens':32000},
    messages=[{'role':'user','content':'hi'}])
for b in r.content:
    if b.type == 'text':
        print('ok')
        break
" 2>/dev/null)
    if [ "$OPUS_OK" = "ok" ]; then
        pass "claude-opus-4-6 with thinking responds"
    else
        fail "Opus API call failed"
    fi
fi

# 8. Training smoke test
# torch.compile on Blackwell can hit intermittent inductor bugs on first compile.
# run_forever.sh handles this with auto-restart. We retry up to 3 times here.
echo -n "Training smoke test... "
export AUTORESEARCH_DEPTH=8
export AUTORESEARCH_BATCH_SIZE=16
export AUTORESEARCH_DATASET=${AUTORESEARCH_DATASET:-pubmed}

TRAIN_OK=0
for attempt in 1 2 3; do
    TRAIN_OUT=$(timeout 120 uv run python train.py 2>&1 | tr '\r' '\n')

    # Check for fatal non-inductor errors
    if echo "$TRAIN_OUT" | grep -qi "Traceback" && ! echo "$TRAIN_OUT" | grep -qi "InductorError\|BackendCompilerFailed"; then
        echo ""
        echo "$TRAIN_OUT" | tail -5
        fail "training crashed (non-inductor error)"
    fi

    STEPS=$(echo "$TRAIN_OUT" | grep -c "^step " || true)
    if [ "$STEPS" -ge 10 ]; then
        TRAIN_OK=1
        break
    fi

    # Inductor failure -- retry (this is expected on Blackwell)
    echo -n "(compile retry $attempt) "
done

if [ "$TRAIN_OK" -eq 1 ]; then
    pass "$STEPS steps in attempt $attempt"
else
    fail "training failed after 3 attempts"
fi

echo ""
echo -e "${GREEN}=== All tests passed ===${NC}"
echo ""
