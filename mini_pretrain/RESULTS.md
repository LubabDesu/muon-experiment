# Mini pretrain — experiment log

Living doc for β ablation runs. Update after each batch of jobs.

**Setup (default mini):** 96M tied GPT · 8L×768 · seq 1024 · FineWeb · seed 0 unless noted.

**Random-guess CE:** `log(50257) ≈ 10.825`

---

## Hyperparameters (fill per batch)

| Field | smoke (done) | mini (done) | notes |
|-------|--------------|-------------|-------|
| `BATCH_TOKENS` | 4096 | 4096 | |
| `LR_ADAM` | 1e-4 | 1e-4 | |
| `LR_MUON` | 0.005 | 0.005 | |
| `WEIGHT_DECAY_ADAM` | 0.01 | 0.1 | Adam late diverge likely WD |
| `WEIGHT_DECAY_MUON` | 0.0 | 0.005 | |
| `BASE_BETA` | 0.95 | 0.95 | |
| train shards | 2 | 10 | |
| steps | 300 | 3000 | |

---

## Bank β offsets (configurable)

Offsets add to `BASE_BETA` for `muon_bank` only:

| bank | params | default offset | default β |
|------|--------|----------------|-----------|
| qk | `q_proj`, `k_proj` | −0.01 | 0.94 |
| vo | `v_proj`, `o_proj` | 0 | 0.95 |
| mlp | `c_fc`, `c_proj` | +0.01 | 0.96 |

**Env (symmetric sweep):**

```bash
export BETA_BANK_DELTA=0.02   # qk=-0.02, mlp=+0.02, vo=0
```

**Env (per-bank):**

```bash
export BETA_OFFSET_QK=-0.03
export BETA_OFFSET_VO=0
export BETA_OFFSET_MLP=0.03
```

---

## Results — smoke (300 steps, 51M, seed 0)

| run | val @ 100 | val @ 200 | val @ 300 | notes |
|-----|-----------|-----------|-----------|-------|
| `muon_global` | 7.839 | 7.239 | **7.157** | |
| `muon_bank` (Δ=0.01) | 7.840 | 7.243 | 7.159 | ≈ global (null) |

**Conclusion:** smoke null; β offsets ±0.01 too small to matter.

---

## Results — mini (3000 steps, 96M, seed 0)

### Validation loss

| step | AdamW | Muon global | Muon bank (Δ=0.01) | best |
|------|-------|-------------|-------------------|------|
| 0 | 10.947 | 10.947 | 10.947 | — |
| 500 | 7.174 | **6.921** | 6.923 | Muon |
| 1000 | 7.635 | 7.299 | **7.296** | Muon |
| 1500 | 8.252 | 7.852 | **7.845** | Muon |
| 2000 | 8.567 | **8.063** | 8.070 | Muon |
| 2500 | 8.627 | **8.047** | 8.063 | Muon global |
| 3000 | 9.278 | 8.605 | **8.600** | Muon (not Adam) |

### vs global @ 3000

| run | val @ 3000 | Δ vs global |
|-----|------------|-------------|
| muon_global | 8.605 | — |
| muon_bank | 8.600 | −0.005 (noise) |

**Conclusion:** bank Δ=0.01 still null. Muon hybrid ≈ global ≈ bank. AdamW diverged late (check `WEIGHT_DECAY_ADAM=0.01` rerun).

**Best checkpoint (all runs):** ~step **500** (val ~6.92 Muon, ~7.17 Adam).

---

## Sweep log — bank delta (fill as you run)

Copy row template. `delta` = `BETA_BANK_DELTA` (symmetric).

| date | preset | delta | seed | val@500 | val@1000 | val@3000 | vs global@3000 | notes |
|------|--------|-------|------|---------|----------|----------|----------------|-------|
| | mini | 0.01 | 0 | 6.923 | 7.296 | 8.600 | −0.005 | baseline bank |
| | mini | 0.02 | 0 | | | | | |
| | mini | 0.03 | 0 | | | | | |
| | mini | 0.02 | 1 | | | | | optional confirm |

---

## Stability pass — cosine profile (seed 0, Δ=0.03)

Profile: `mini_pretrain/hparams/stable_mini.env` (`LR_MUON=0.002`, `MUON_NS_STEPS=3`, warmup 200, cosine decay, early stop).

| run | delta | best val | val@500 | val@1000 | val@1500 | val@2000 | stop |
|-----|-------|----------|---------|----------|----------|----------|------|
| `muon_bank` | 0.03 | **7.0169 @ 500** | 7.0169 | 7.3400 | 7.7951 | 7.8694 | early stop: `val_loss > best + 0.800` |

**Issue found:** LR schedule worked, but validation still degraded after step 500. Likely bug/config mismatch: AdamW weight decay was applied to embeddings and LayerNorm params. In Muon mode, Adam side is mostly embeddings/norms, so this can destabilize the hybrid. Fixed in `mini_pretrain/optim.py` by splitting AdamW params into decay / no-decay groups.

**Command template:**

```bash
BATCH_TOKENS=4096 LR_MUON=0.005 LR_ADAM=1e-4 BETA_BANK_DELTA=0.02 \
  python -m mini_pretrain.train --preset mini --run-mode muon_bank
```

Compare to global (once per preset):

```bash
BATCH_TOKENS=4096 LR_MUON=0.005 LR_ADAM=1e-4 \
  python -m mini_pretrain.train --preset mini --run-mode muon_global
```

---

## TODO / next runs

- [ ] AdamW mini rerun: `WEIGHT_DECAY_ADAM=0.01`
- [ ] Bank sweep: `BETA_BANK_DELTA` = 0.02, 0.03
- [ ] If any delta wins: seed 1 confirm
- [ ] Optional: LR schedule / early stop @ 500

---

## Claims (draft)

- **Safe now:** At mini scale, per-bank β with ±0.01 did not beat global β.
- **Safe now:** Hybrid Muon+Adam more stable than AdamW under this config (Adam val 9.28 @ 3k).
- **Not yet:** Optimal bank offset magnitude or fair Adam baseline.

---

## Latest comparison (auto)

<!-- AUTO_SUMMARY_START -->
<!-- AUTO_SUMMARY_END -->

Re-generated after `run_ablation.sh` finishes, or manually:

```bash
python -m mini_pretrain.summarize_results --session results/mini_pretrain/session_mini_seed0.txt --checkpoints 500,1000,3000 --append-results-md
```

---

## Raw logs

JSONL: `results/mini_pretrain/{run_id}.jsonl`  
Meta: `results/mini_pretrain/{run_id}_meta.json`  
Session IDs: `results/mini_pretrain/session_{preset}_seed{N}.txt`
