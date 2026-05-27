# Phase 1 Refactoring Summary

## What Changed

### New Files Created
```
muon/
├── metrics_tracker.py          [NEW] 50 lines - Lightweight metrics logging
├── optimizers.py               [NEW] 110 lines - Muon + baselines (SGD, Adam, Muon-no-ortho)
├── quick_oracle.py             [NEW] 140 lines - Fast oracle estimator (3 CG iters)
├── train_quick_signals.py      [NEW] 170 lines - Main training loop with multiple opts
├── analyze_quick_signals.py    [NEW] 100 lines - Analysis/comparison script
├── run_experiment.py           [NEW] 80 lines - Batch experiment runner
├── PHASE1_README.md            [NEW] Usage guide
└── results/                    [NEW] Directory for results
```

### Existing Files (Unchanged)
- `model.py` — No changes (Transformer architecture)
- `muon.py` — No changes (original Muon still available for reference)
- `train.py` — No changes (original training loop untouched)

### Files You Can Keep/Remove
- `analysis_utils.py` — Kept for reference, not used in Phase 1 (will integrate in Phase 2)

## Key Design Decisions

### 1. Minimal Invasiveness
- All new code in new files
- Original `train.py` and `muon.py` untouched
- Old code still works exactly as before

### 2. Quick Iteration Path
- `train_quick_signals.py` is simpler than original `train.py`
- Focused on comparison, not production training
- Easy to debug individual optimizers

### 3. Oracle Integration
- Lightweight oracle in `quick_oracle.py` (3 CG iterations, not full solve)
- Computed every 200 steps (not every step) to balance signal vs. cost
- Optional flag `--compute_oracle` to turn off

### 4. Modular Optimizers
- All in `optimizers.py` for easy comparison
- Includes ablation: `MuonWithoutOrthogonalization` (momentum only, no orthogonalization)
- Fair comparison: same momentum, LR, weight decay across variants

## How to Use

### Option 1: Quick Signals (5 min)
```bash
python train_quick_signals.py --optimizer muon --steps 500
python train_quick_signals.py --optimizer sgd --steps 500
python train_quick_signals.py --optimizer adam --steps 500
python analyze_quick_signals.py
```

### Option 2: With Oracle (30 min, informative)
```bash
python train_quick_signals.py --optimizer muon --steps 1000 --compute_oracle
python analyze_quick_signals.py
```

### Option 3: Batch Run
```bash
python run_experiment.py --steps 1000 --quick     # 3 optimizers, no oracle
python run_experiment.py --steps 1000 --oracle    # With oracle tracking
python run_experiment.py --steps 1000 --ablation  # Include Muon ablation
```

## What Happens When You Run It

1. **train_quick_signals.py** starts training with specified optimizer
2. Every 100 steps: prints current loss and lr
3. Every 200 steps: if `--compute_oracle`, estimates oracle direction and computes cosine similarity
4. After training: saves JSON to `results/quick_signals_{optimizer}_s{seed}.json`
5. **analyze_quick_signals.py** reads all JSONs and prints comparison table

### Example Output
```
================================================================================
QUICK SIGNALS ANALYSIS
================================================================================

📊 CONVERGENCE COMPARISON

Optimizer            Final Loss      Min Loss        Time/Step (ms)   
----
muon                 1.234567        1.123456        15.34            
sgd                  1.245678        1.134567        14.23            
adam                 1.223456        1.112345        16.78            

🎯 MUON-ORACLE ALIGNMENT

  Cosine Sim (Muon): 0.7234 (range: 0.6891-0.8123)
```

## Interpretation Guide

### Convergence Comparison
- **Final Loss**: Lower is better
- **Min Loss**: Best loss reached
- **Time/Step**: Speed (Muon should be similar to SGD)

### Oracle Alignment (`--compute_oracle`)
- **0.8-1.0**: Excellent - Muon matches second-order optimal
- **0.5-0.8**: Good - Reasonable approximation despite missing cross-layer info
- **0.3-0.5**: Poor - Cross-layer effects likely matter
- **<0.3**: Bad - Layer-wise approximation breaks down

## Next Steps After Phase 1

If signals look good (cosine_sim > 0.7, convergence comparable):
- Move to Phase 2: rigorous framework
- Compute full Hessian blocks
- Run validation loop
- Sweep hyperparameters

If signals suggest cross-layer coupling matters (cosine_sim < 0.5):
- Investigate which layers couple most
- Test hybrid approaches (Muon + periodic full CG)
- Analyze Hessian block structure

## Files You'll Modify in Phase 2

New files in Phase 2:
- `experiment.py` — orchestrator for full experimental runs
- `hessian_analysis.py` — detailed Hessian computation
- `visualization.py` — comprehensive plotting
- `config.py` — centralized configuration

Phase 1 stays untouched; Phase 2 builds on top.

---

**Total new code: ~600 lines (well-organized, reusable)**  
**Time to first results: 5-30 minutes depending on run length**
