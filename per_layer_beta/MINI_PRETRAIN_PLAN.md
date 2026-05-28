# Mini Pretrain β-Ablation — Implementation Plan

> **For agentic workers:** Implement task-by-task. Check off boxes as you go.

**Goal:** Answer whether **bank-specific Muon β** (qk / vo / mlp) beats **global β** on a small GPT pretrain pilot that fits **1× L4**, without `modded-nanogpt` speedrun stack.

**Architecture:** New `mini_pretrain/` package: small GPT + standard CE + dual optimizers (AdamW on embed/head/norms; Muon on 2D matrices). Reuse `per_layer_beta/muon_beta.py` for Newton–Schulz update; add thin `beta_policy` extensions for `global` vs `bank`. No Triton CE, no FP8, no NorMuon banks, no multi-GPU NCCL.

**Tech stack:** PyTorch 2.x, CUDA L4, optional FineWeb `.bin` shards (or WikiText-103 for first smoke).

**Out of scope (v1):** Reproducing modded-nanogpt leaderboard; `shape` β grid; grad-norm adaptive β; 8-GPU.

---

## Why this instead of full modded-nanogpt

| full stack | mini pilot |
|------------|------------|
| H100-tuned Triton kernels (`sm_90`) | `F.cross_entropy` |
| NorMuon param banks + sharding | per-layer `nn.Linear` / standard GPT blocks |
| NCCL + 8-GPU batch schedule | single GPU, fixed batch |
| ~7 min compile warmup | `torch.compile` optional/off |

Science question unchanged: **same model/data/seed, only β assignment differs.**

---

## Reuse map (do not rewrite)

| asset | path | use |
|-------|------|-----|
| Muon step + NS | `per_layer_beta/muon_beta.py` | core update |
| eligibility helpers | `per_layer_beta/beta_policy.py` | matrix vs AdamW params |
| tests | `per_layer_beta/tests/` | extend for `bank` policy |
| momentum schedule (optional v2) | port `get_muon_momentum` from `modded-nanogpt/train_gpt.py` | v1: fixed β=0.95 ok |

**Do not depend on** `modded-nanogpt/` at runtime for v1.

---

## Target file layout

```
muon/
  mini_pretrain/
    __init__.py
    config.py          # dataclass: model/data/train; env overrides
    model_gpt.py       # ~6–12L, 512–768 dim, RoPE, no exotic extras
    data.py            # .bin shard loader OR HF dataset
    optim.py           # AdamW + PerLayerBetaMuon wiring
    beta_assign.py     # global | bank (qk-0.01, vo, mlp+0.01)
    train.py           # main loop, val, logging
    run_ablation.sh    # smoke + mini presets
  per_layer_beta/
    beta_policy.py     # + bank_for_name()
    tests/             # + test_bank_beta.py
  results/mini_pretrain/   # jsonl + plots (gitignored)
```

---

## Model & budget (L4-safe defaults)

| knob | smoke | mini (real pilot) |
|------|-------|-------------------|
| layers | 6 | 8 |
| `d_model` | 512 | 768 |
| heads | 8 | 12 |
| embed/head | tied | tied |
| seq len | 512 | 1024 |
| batch tokens/step | 32k–65k | tune to VRAM |
| train steps | 300 | 3000–5000 |
| val every | 100 | 500 |
| seeds | 0 | 0, then 1 if signal |
| data | 1–2 FineWeb shards | 10–20 shards |

**OOM rule:** halve `batch_tokens` before touching β logic.

---

## Optimizer split (match modded spirit, simplified)

| params | optimizer | notes |
|--------|-----------|-------|
| `embed`, `lm_head` | AdamW | fixed β not applied |
| `norm`, biases, scalars | AdamW | |
| `q_proj,k_proj,v_proj,o_proj, mlp` 2D weights | Muon | β from policy |
| gates / routers | AdamW | skip for v1 |

### β policies (v1 ablation only)

- **`global`:** all Muon matrices → `β = 0.95` (or schedule base `β_t`)
- **`bank`:** name-based offsets (from your modded patch):
  - q/k: `β_t - 0.01`
  - v/o: `β_t`
  - mlp (fc/proj): `β_t + 0.01`

Optional baseline: **`adamw`** full model (same LR budget).

---

## Run matrix

| run_id | optimizer | β |
|--------|-----------|---|
| `adamw` | AdamW all | — |
| `muon_global` | Muon + AdamW split | global 0.95 |
| `muon_bank` | Muon + AdamW split | bank offsets |

Same: seed, data order, steps, LR, WD, model arch.

---

## Success criteria (pilot, not SOTA)

1. **Smoke:** 300 steps, no NaN, val loss finite, logs show distinct β tables for global vs bank.
2. **Mini:** val loss curve; if `muon_bank` best at 3k steps by ≥ small margin (you pick ε, e.g. 0.02 val loss) → run seed 1.
3. **Claim wording:** “on L4 mini-GPT pilot, bank β showed [signal / no signal] vs global β” — not “Muon universally better.”

---

## Implementation phases

### Phase 0 — Preconditions (VM, ~30 min)

- [ ] L4 VM + driver OK (`nvidia-smi`)
- [ ] venv in `muon/` (not committed): `pip install torch numpy tqdm datasets tiktoken`
- [ ] Data: `python mini_pretrain/data.py --download fineweb --chunks 10` (or document WikiText fallback)
- [ ] Confirm **no** import of `modded-nanogpt` in `mini_pretrain`

### Phase 1 — Core modules (~2–3 hrs)

- [ ] **`config.py`:** CLI + env (`SEED`, `BETA_POLICY`, `STEPS`, `RUN_ID`)
- [ ] **`model_gpt.py`:** minimal causal LM; param names include `q_proj`, `k_proj`, `mlp.c_fc`, etc. for bank routing
- [ ] **`beta_assign.py`:** `beta_for_name(name, shape, policy, base_beta) -> float | None`
- [ ] **`optim.py`:** build param groups; wrap `PerLayerBetaMuon` from `muon_beta.py`
- [ ] **Tests:** `test_bank_beta.py` — q/k/mlp get different β; embed gets `None`

### Phase 2 — Train loop (~2 hrs)

- [ ] **`data.py`:** iterator yielding `(input, target)` token batches; pinned memory optional
- [ ] **`train.py`:** train/val loop, `results/mini_pretrain/{run_id}.jsonl`
- [ ] Log at step 0: full `name → β → optim` table (reproducibility)
- [ ] Optional: `get_muon_momentum(step)` port for v2 (v1 can use constant 0.95)

### Phase 3 — Smoke on L4 (~30 min wall + compile if enabled)

- [ ] `run_ablation.sh smoke` → 300 steps × (`muon_global`, `muon_bank`)
- [ ] Fix OOM / dataloader / dtype issues
- [ ] Inspect logs under `results/mini_pretrain/`

### Phase 4 — Mini ablation (~few hours GPU)

- [ ] 3000 steps × global, bank (seed 0)
- [ ] Plot val loss vs step (`analysis/plot_mini_pretrain.py` or notebook)
- [ ] Optional seed 1 if curves separate

### Phase 5 — Repo hygiene

- [ ] `.gitignore`: `results/`, `.venv`, large data
- [ ] Short README section in `per_layer_beta/README.md` pointing to `mini_pretrain/`
- [ ] Commit **code only** to `muon-experiment` (no data, no venv)

---

## Testing strategy

| level | what |
|-------|------|
| unit | `beta_assign`, `muon_beta` direction changes when β changes (`test_muon_beta.py` exists) |
| cpu smoke | 10 steps, tiny random data, `BETA_POLICY=bank` |
| gpu smoke | 300 steps L4 |
| regression | global vs bank produce different momentum buffers at step 1 (same seed, diff β) |

No requirement for 80% coverage on v1; require **deterministic β table** + **no crash**.

---

## Risks & mitigations

| risk | mitigation |
|------|------------|
| L4 OOM | reduce `seq_len` or batch tokens; bf16 autocast |
| FineWeb download slow | WikiText smoke first; FineWeb for mini |
| β effect too small at 3k | extend steps or slightly widen bank offsets (document as v2) |
| Adam undertuned | one LR sweep on `adamw` only (2–3 values max) |
| Diverges from modded NorMuon | document as “standard GPT + Muon”, not NorMuon reproduction |

---

## Decision gate (after Phase 3)

| outcome | next |
|---------|------|
| smoke fails | stop; fix infra, do not scale |
| smoke OK, mini flat | report null result; optional shape-β v2 |
| mini favors bank | seed 1 + short writeup; **do not** port back to full modded until justified |

---

## What to extract from modded-nanogpt (reference only)

Read, don’t import:

1. `get_muon_momentum(step)` — schedule shape  
2. `get_bank_beta(label, base_beta)` — your offsets  
3. Which labels are Muon vs Adam in `ParamConfig` table — mirror naming in `model_gpt.py`

Ignore: Triton CE, FP8, banks, YaRN, MTP, sparse gates.

---

## Single command target (end state)

```bash
cd muon
source .venv/bin/activate
bash mini_pretrain/run_ablation.sh smoke    # 300 × global, bank
bash mini_pretrain/run_ablation.sh mini     # 3000 × global, bank
```

---

## Locked choices (v1)

1. **Data:** FineWeb `.bin` — 10 chunks smoke / 20 mini  
2. **Baseline:** `adamw` + `muon_global` + `muon_bank`  
3. **β schedule:** constant `0.95`  
4. **`torch.compile`:** off  
5. **Embed / LM head:** **tied** (shared `vocab × d` matrix; AdamW on that tensor, Muon on attn/MLP only)

## Open choices (defer)

- Untied embed/head (v2, modded parity)  
- Ported `get_muon_momentum` schedule (v2)
