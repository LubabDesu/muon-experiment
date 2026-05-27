# Muon Optimizer: Phase 1 Quick Signals

## Quick Start

### Run a quick experiment (10 min, 1000 steps):

```bash
# Muon with oracle tracking (slow but informative)
python train_quick_signals.py --optimizer muon --steps 1000 --compute_oracle

# Just baseline comparison (fast)
python train_quick_signals.py --optimizer muon --steps 1000
python train_quick_signals.py --optimizer sgd --steps 1000
python train_quick_signals.py --optimizer adam --steps 1000

# Muon ablation: without orthogonalization
python train_quick_signals.py --optimizer muon_no_ortho --steps 1000
```

### Analyze results:

```bash
python analyze_quick_signals.py
```

---

## What You Get

### Files Created
- `metrics_tracker.py` — Track metrics during training (loss, lr, cosine_sim)
- `optimizers.py` — Muon + baselines (SGD, Adam, Muon without orthogonalization)
- `quick_oracle.py` — Lightweight oracle estimator (3 CG iterations)
- `train_quick_signals.py` — Main training loop with multiple optimizers
- `analyze_quick_signals.py` — Simple analysis script

### Results Saved
Results go to `results/quick_signals_{optimizer}_s{seed}.json` with:
- **step**: Training step
- **loss**: Current training loss
- **lr**: Learning rate at this step
- **cosine_sim** (if --compute_oracle): Cosine similarity between Muon and oracle updates
- **elapsed_s**: Wall-clock time

---

## Interpreting Results

### Convergence Comparison
Look at `Final Loss` across optimizers. If Muon ≈ SGD/Adam, great! If Muon is much worse, investigate.

### Muon-Oracle Alignment (`--compute_oracle`)
- **Cosine Sim = 1.0**: Muon perfectly matches oracle (best case)
- **Cosine Sim = 0.5-0.8**: Reasonable approximation
- **Cosine Sim < 0.3**: Muon diverges significantly from second-order optimal

**Key insight:** If cosine_sim stays high throughout training, the layer-wise orthogonalization approximates full cross-layer coupling well. If it drops early, you have a signal that cross-layer effects matter.

---

## Next Steps (Phase 2)

Once you see initial signals:
- Full Hessian analysis (condition numbers, off-diagonal energy)
- Validation loop (perplexity)
- Learning rate sweep for fair comparison
- Per-layer analysis (which layers couple most?)

See `plan.md` for full Phase 2 roadmap.

---

## Troubleshooting

**Out of memory with --compute_oracle?**
- Reduce batch size in `muon.py` training_config
- Or run without `--compute_oracle` first

**Slow convergence?**
- These are short runs (1000 steps) on a 10M transformer
- Loss won't drop dramatically; look at trends, not absolute values

**Oracle computation errors?**
- Make sure model is on same device as data
- Check that gradients flow properly
