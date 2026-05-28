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

set -euo pipefail

MODE="${1:-smoke}"
SEED="${SEED:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

run_one() {
  local run_mode="$1"
  export MINI_PRETRAIN_PRESET="$MODE"
  export RUN_MODE="$run_mode"
  export SEED="$SEED"
  export RUN_ID="${run_mode}-${MODE}-seed${SEED}-$(date +%s)"
  echo "========================================"
  echo "RUN: mode=$run_mode preset=$MODE seed=$SEED"
  echo "========================================"
  python -m mini_pretrain.train --preset "$MODE" --run-mode "$run_mode"
}

case "$MODE" in
  smoke)
    run_one adamw
    run_one muon_global
    run_one muon_bank
    ;;
  mini)
    run_one adamw
    run_one muon_global
    run_one muon_bank
    ;;
  quick)
    echo "quick mode removed — use smoke on GPU VM only (never local CPU)."
    exit 1
    ;;
  *)
    echo "Unknown mode: $MODE (use smoke | mini | quick)"
    exit 1
    ;;
esac

echo "Done. Results in results/mini_pretrain/"
