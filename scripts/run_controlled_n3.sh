#!/usr/bin/env bash
# run_controlled_n3.sh -- Run n=3 controlled experiments with integrity checks
#
# Usage:
#   AZURE_OPENAI_ENDPOINT="https://bmd-openai.services.ai.azure.com" \
#     bash scripts/run_controlled_n3.sh --model gpt-oss-120b --tag gptoss120b
#
# Runs 3 independent sessions of 100 experiments each from the frozen
# baseline (commit 3fb6704). Between each run:
#   - Validates exactly 100 clean rows in the TSV
#   - Archives results to data/controlled/
#   - Resets train.py to baseline
#   - Clears leftover files
#
# Prerequisites:
#   - On branch controlled/baseline
#   - .train_clean.py matches commit 3fb6704
#   - AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY set
#   - venv at .venv/bin/python

set -euo pipefail

# ---------- Args ----------
MODEL=""
TAG=""
PROVIDER="azure"  # azure, anthropic, opus
MAX_RUNS=100
DEPTH=""   # empty = auto-detect from GPU
BATCH=""   # empty = auto-detect from GPU

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)    MODEL="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --max)      MAX_RUNS="$2"; shift 2 ;;
    --depth)    DEPTH="$2"; shift 2 ;;
    --batch)    BATCH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$TAG" ]]; then
  echo "Usage: bash scripts/run_controlled_n3.sh --model MODEL --tag TAG [--provider PROVIDER]"
  echo "  --model    Model name or Azure deployment (e.g. gpt-oss-120b, sonnet46, Kimi-K2.5)"
  echo "  --tag      Short tag for branches/files (e.g. gptoss120b, sonnet46-5090)"
  echo "  --provider azure|anthropic|opus (default: azure)"
  echo "  --depth    Force DEPTH (default: auto-detect from GPU VRAM)"
  echo "  --batch    Force DEVICE_BATCH_SIZE (default: auto-detect from GPU VRAM)"
  exit 1
fi

CD="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CD"

PYTHON=".venv/bin/python"
AGENT="scripts/agent.py"
TSV="agent_results_pubmed.tsv"
BASELINE_SHA="3fb6704"
CLEAN_PY=".train_clean.py"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] FATAL:${NC} $*"; exit 1; }

# ---------- Preflight checks ----------
preflight() {
  log "=== PREFLIGHT CHECKS ==="

  # Branch
  local branch
  branch=$(git branch --show-current)
  if [[ "$branch" != "controlled/baseline" ]]; then
    fail "Must be on controlled/baseline (currently on $branch)"
  fi
  log "Branch: $branch"

  # Baseline match
  if ! diff -q train.py <(git show "${BASELINE_SHA}:train.py") > /dev/null 2>&1; then
    warn "train.py differs from $BASELINE_SHA -- restoring"
    cp "$CLEAN_PY" train.py
  fi
  if ! diff -q train.py "$CLEAN_PY" > /dev/null 2>&1; then
    fail ".train_clean.py does not match train.py after restore"
  fi
  log "train.py: matches frozen baseline $BASELINE_SHA"

  # Clean state
  if [[ -f "$TSV" ]]; then
    warn "Removing leftover $TSV"
    rm -f "$TSV"
  fi
  rm -f .train_results.json .suite.pid
  log "Working directory: clean"

  # Python/venv
  if [[ ! -x "$PYTHON" ]]; then
    fail "Python not found at $PYTHON"
  fi
  log "Python: $($PYTHON --version 2>&1)"

  # Source env file for API keys
  if [[ -f /etc/profile.d/autoresearch.sh ]]; then
    source /etc/profile.d/autoresearch.sh
  fi

  # Provider-specific checks and API test
  local test_result
  if [[ "$PROVIDER" == "azure" ]]; then
    if [[ -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
      fail "AZURE_OPENAI_ENDPOINT not set"
    fi
    log "Azure endpoint: ${AZURE_OPENAI_ENDPOINT}"
    test_result=$($PYTHON -c "
from scripts.agent import call_azure
r = call_azure('Say OK', deployment='$MODEL')
print('OK' if r and 'OK' in r.upper() else 'FAIL')
" 2>/dev/null | tail -1)
  elif [[ "$PROVIDER" == "anthropic" ]]; then
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      fail "ANTHROPIC_API_KEY not set"
    fi
    log "Anthropic API: key set"
    test_result=$($PYTHON -c "
from scripts.agent import call_claude
r = call_claude('Say OK')
print('OK' if r and 'OK' in r.upper() else 'FAIL')
" 2>/dev/null | tail -1)
  elif [[ "$PROVIDER" == "opus" ]]; then
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      fail "ANTHROPIC_API_KEY not set"
    fi
    log "Anthropic API (Opus): key set"
    test_result=$($PYTHON -c "
from scripts.agent import call_claude_opus
r = call_claude_opus('Say OK')
print('OK' if r and 'OK' in r.upper() else 'FAIL')
" 2>/dev/null | tail -1)
  else
    fail "Unknown provider: $PROVIDER (use azure, anthropic, or opus)"
  fi

  if [[ "$test_result" != "OK" ]]; then
    fail "API test failed for $PROVIDER / $MODEL"
  fi
  log "API test: $PROVIDER / $MODEL responds"

  log "=== PREFLIGHT PASSED ==="
  echo
}

# ---------- Validate results ----------
validate_run() {
  local run_num=$1
  local tsv_path=$2

  log "Validating R${run_num}..."

  if [[ ! -f "$tsv_path" ]]; then
    fail "Results file missing: $tsv_path"
  fi

  # Count clean rows (5+ fields, known status, parseable val_bpb)
  local clean_count
  clean_count=$($PYTHON -c "
import csv
count = 0
with open('$tsv_path') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        status = (r.get('status') or '').strip().lower()
        if status in ('keep','discard','crash','fail','skip'):
            try:
                float(r.get('val_bpb', 'x'))
                count += 1
            except ValueError:
                pass
print(count)
")

  if [[ "$clean_count" -ne "$MAX_RUNS" ]]; then
    fail "R${run_num}: expected $MAX_RUNS clean rows, got $clean_count"
  fi

  # Summary stats
  $PYTHON -c "
import csv
rows = list(csv.DictReader(open('$tsv_path'), delimiter='\t'))
valid = [r for r in rows if (r.get('status') or '').strip().lower() in ('keep','discard','crash','fail','skip')]
keeps = [r for r in valid if r['status']=='keep']
crashes = [r for r in valid if r['status']=='crash']
baseline = float(keeps[0]['val_bpb']) if keeps else 0
best = min(float(r['val_bpb']) for r in valid if float(r['val_bpb'])>0)
print(f'  R${run_num}: {len(valid)} exp | {len(keeps)} kept ({100*len(keeps)/len(valid):.1f}%) | {len(crashes)} crash ({100*len(crashes)/len(valid):.1f}%) | baseline={baseline:.4f} | best={best:.4f}')
"

  log "R${run_num}: VALID ($clean_count rows)"
}

# ---------- Archive results ----------
archive_run() {
  local run_num=$1

  local dest="data/controlled/${TAG}_r${run_num}.tsv"
  mkdir -p data/controlled

  # Clean corrupt rows before archiving
  $PYTHON -c "
import csv
with open('$TSV') as f:
    lines = f.readlines()
header = lines[0]
clean = [header]
for line in lines[1:]:
    parts = line.strip().split('\t')
    if len(parts) >= 5 and parts[3].strip().lower() in ('keep','discard','crash','fail','skip'):
        try:
            float(parts[1])
            clean.append(line)
        except ValueError:
            pass
with open('$dest', 'w') as f:
    f.writelines(clean)
print(f'Archived {len(clean)-1} clean rows to $dest')
"

  log "Archived: $dest"
}

# ---------- Reset between runs ----------
reset_for_next() {
  log "Resetting for next run..."
  cp "$CLEAN_PY" train.py
  rm -f "$TSV" .train_results.json

  # Verify
  if ! diff -q train.py "$CLEAN_PY" > /dev/null 2>&1; then
    fail "train.py restore failed"
  fi
  log "Reset complete"
  echo
}

# ---------- Single run ----------
run_one() {
  local run_num=$1
  local run_tag="controlled-${TAG}-r${run_num}"

  log "=========================================="
  log "  STARTING R${run_num}: $MODEL"
  log "  Tag: $run_tag"
  log "  Max: $MAX_RUNS experiments"
  log "=========================================="

  # Build agent command based on provider
  local agent_cmd="$PYTHON $AGENT --max-runs $MAX_RUNS --tag $run_tag --no-dashboard"
  if [[ "$PROVIDER" == "azure" ]]; then
    agent_cmd="$agent_cmd --azure $MODEL"
  elif [[ "$PROVIDER" == "opus" ]]; then
    agent_cmd="$agent_cmd --opus"
  fi
  # anthropic (Sonnet 4.6) is the default, no flag needed

  # Set env vars for DEPTH/BATCH only if explicitly provided
  local env_prefix=""
  if [[ -n "$DEPTH" ]]; then
    env_prefix="AUTORESEARCH_DEPTH=$DEPTH "
  fi
  if [[ -n "$BATCH" ]]; then
    env_prefix="${env_prefix}AUTORESEARCH_BATCH_SIZE=$BATCH "
  fi

  eval "${env_prefix}${agent_cmd}"

  log "R${run_num} agent finished"
}

# ---------- Main ----------
main() {
  echo
  echo "====================================================="
  echo "  CONTROLLED N=3 RUNNER"
  echo "  Model: $MODEL"
  echo "  Tag: $TAG"
  echo "  Max per run: $MAX_RUNS"
  echo "  DEPTH=$DEPTH  BATCH=$BATCH"
  echo "====================================================="
  echo

  preflight

  for run_num in 1 2 3; do
    run_one "$run_num"
    validate_run "$run_num" "$TSV"
    archive_run "$run_num"

    if [[ "$run_num" -lt 3 ]]; then
      reset_for_next
    fi
  done

  echo
  log "====================================================="
  log "  ALL 3 RUNS COMPLETE"
  log "====================================================="

  # Final summary
  for run_num in 1 2 3; do
    validate_run "$run_num" "data/controlled/${TAG}_r${run_num}.tsv"
  done

  log "Data saved to data/controlled/${TAG}_r{1,2,3}.tsv"
  log "Ready to convert and push to repos"
}

main
