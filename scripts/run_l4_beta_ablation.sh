#!/usr/bin/env bash
# L4 pilot: global vs bank beta only. No grid search.
#
# Prereqs on VM:
#   cd muon/modded-nanogpt
#   pip install -r requirements.txt
#   python data/cached_fineweb10B.py 20
#
# Usage:
#   bash ../scripts/run_l4_beta_ablation.sh smoke   # 300-step debug x2
#   bash ../scripts/run_l4_beta_ablation.sh mini    # 3k-step A/B
#   SEED=1 bash ../scripts/run_l4_beta_ablation.sh mini

set -euo pipefail

MODE="${1:-smoke}"
SEED="${SEED:-0}"
NGPU="${NGPU:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/modded-nanogpt"

export OPTIMIZER_MODE=muon
export DISABLE_FP8=1
export DATA_PATH="${DATA_PATH:-.}"
# L4 = sm_89; H100 record stack defaults to sm_90 in triton_kernels CE kernel
export CUDA_COMPUTE_CAPABILITY="${CUDA_COMPUTE_CAPABILITY:-89}"

run_one() {
  local policy="$1"
  local steps="$2"
  local val_every="$3"
  export BETA_POLICY="$policy"
  export NUM_SCHEDULED_ITERATIONS="$steps"
  export NUM_EXTENSION_ITERATIONS=0
  export VAL_LOSS_EVERY="$val_every"
  export RUN_ID="${policy}-steps${steps}-seed${SEED}-$(date +%s)"
  echo "========================================"
  echo "RUN: policy=$policy steps=$steps seed=$SEED"
  echo "========================================"
  if [[ "$NGPU" -eq 1 ]]; then
    torchrun --standalone --nproc_per_node=1 train_gpt.py
  else
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py
  fi
}

case "$MODE" in
  smoke)
    # 1–2: debug (NaN, compile, dataloader)
    run_one global 300 100
    run_one bank 300 100
    ;;
  mini)
    # 3–4: real mini-ablation
    run_one global 3000 500
    run_one bank 3000 500
    ;;
  *)
    echo "Unknown mode: $MODE (use smoke | mini)"
    exit 1
    ;;
esac

echo "Done. Compare val_loss in logs/ — search BETA_POLICY in step-0 print."
