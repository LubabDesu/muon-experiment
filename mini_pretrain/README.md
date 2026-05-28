# mini_pretrain

Small tied-weight GPT on FineWeb (or synthetic data) for **global vs bank** Muon β ablation. No `modded-nanogpt` dependency.

## Setup

```bash
cd muon
python -m venv .venv && source .venv/bin/activate
pip install -r mini_pretrain/requirements.txt
python -m mini_pretrain.data --download fineweb --chunks 10
```

## Run (GPU VM only — not on your laptop)

**Do not run training locally.** Use your L4 VM (`l4-muon`). Local CPU/MPS runs can freeze the machine.

```bash
# 300-step smoke (adamw + global + bank)
bash mini_pretrain/run_ablation.sh smoke

# 3000-step mini pilot
bash mini_pretrain/run_ablation.sh mini
```

Single run:

```bash
export PYTHONPATH=.
python -m mini_pretrain.train --preset mini --run-mode muon_bank
```

## Presets

| preset | layers | d_model | seq | steps | ~params (tied) |
|--------|--------|---------|-----|-------|----------------|
| smoke | 6 | 512 | 512 | 300 | ~44M |
| mini | 8 | 768 | 1024 | 3000 | ~95M |

## Outputs

- `results/mini_pretrain/{run_id}.jsonl` — train/val metrics
- `results/mini_pretrain/{run_id}_meta.json` — full β assignment table

## Bank β offsets

```bash
# symmetric: qk = -delta, mlp = +delta, vo = 0
export BETA_BANK_DELTA=0.02

# or per-bank
export BETA_OFFSET_QK=-0.03
export BETA_OFFSET_MLP=0.03
```

Track runs in [`RESULTS.md`](RESULTS.md).

## OOM on L4

```bash
export BATCH_TOKENS=4096    # mini on L4
```
