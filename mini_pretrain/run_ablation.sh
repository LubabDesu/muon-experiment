#!/usr/bin/env bash
# Mini pretrain ablation: adamw, muon_global, muon_bank
#
# Prereqs (from muon/):
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r mini_pretrain/requirements.txt
#   python -m mini_pretrain.data --download fineweb --chunks 10
#
# Usage:
#   bash mini_pretrain/run_ablation.sh smoke
#   bash mini_pretrain/run_ablation.sh mini
#   SEED=1 bash mini_pretrain/run_ablation.sh mini
#
# After all runs: prints comparison table and appends to mini_pretrain/RESULTS.md

set -euo pipefail

MODE="${1:-smoke}"
SEED="${SEED:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

RESULTS_DIR="results/mini_pretrain"
SESSION_FILE="${RESULTS_DIR}/session_${MODE}_seed${SEED}.txt"
mkdir -p "$RESULTS_DIR"
: > "$SESSION_FILE"

run_one() {
  local run_mode="$1"
  export MINI_PRETRAIN_PRESET="$MODE"
  export RUN_MODE="$run_mode"
  export SEED="$SEED"
  export RUN_ID="${run_mode}-${MODE}-seed${SEED}-$(date +%s)"
  echo "$RUN_ID" >> "$SESSION_FILE"
  echo "========================================"
  echo "RUN: mode=$run_mode preset=$MODE seed=$SEED"
  echo "RUN_ID=$RUN_ID"
  echo "========================================"
  python -m mini_pretrain.train --preset "$MODE" --run-mode "$run_mode"
}

case "$MODE" in
  smoke)
    CHECKPOINTS="100,300"
    run_one adamw
    run_one muon_global
    run_one muon_bank
    ;;
  mini)
    CHECKPOINTS="500,1000,3000"
    run_one adamw
    run_one muon_global
    run_one muon_bank
    ;;
  bank_sweep)
    # global once, then bank at each delta (set BETA_BANK_DELTA per run)
    CHECKPOINTS="500,1000,3000"
    export RUN_MODE=muon_global
    export MINI_PRETRAIN_PRESET=mini
    export SEED="$SEED"
    export RUN_ID="muon_global-${MODE}-seed${SEED}-$(date +%s)"
    echo "$RUN_ID" >> "$SESSION_FILE"
    python -m mini_pretrain.train --preset mini --run-mode muon_global
    for delta in 0.01 0.02 0.03; do
      export BETA_BANK_DELTA="$delta"
      run_one muon_bank
      unset BETA_BANK_DELTA
    done
    ;;
  quick)
    echo "quick mode removed — use smoke on GPU VM only (never local CPU)."
    exit 1
    ;;
  *)
    echo "Unknown mode: $MODE (use smoke | mini | bank_sweep)"
    exit 1
    ;;
esac

echo ""
echo "========== Run comparison =========="
python -m mini_pretrain.summarize_results \
  --session "$SESSION_FILE" \
  --checkpoints "${CHECKPOINTS:-500,3000}" \
  --append-results-md

echo "Done. JSONL in ${RESULTS_DIR}/ · summary appended to mini_pretrain/RESULTS.md"
