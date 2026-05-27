# Muon Optimizer Research: File Index

## 📚 Documentation Files (Read These First)

| File | Purpose | Read Time |
|------|---------|-----------|
| **QUICK_START.txt** | Fast overview + 3 command options | 5 min |
| **PHASE1_README.md** | Detailed usage guide for all files | 10 min |
| **REFACTORING_SUMMARY.md** | Design decisions + interpretation guide | 10 min |
| **../plan.md** | Full Phase 1 + Phase 2 roadmap | 5 min |

**Recommendation:** Start with `QUICK_START.txt`, then pick one experiment option.

---

## 🔧 Phase 1 Code Files

| File | Lines | Purpose |
|------|-------|---------|
| **metrics_tracker.py** | 50 | Track loss/lr/cosine_sim during training |
| **optimizers.py** | 110 | Muon + SGD/Adam + Muon ablation |
| **quick_oracle.py** | 140 | Fast oracle direction estimator (CG-based) |
| **train_quick_signals.py** | 170 | Main training loop (multiple optimizers) |
| **run_experiment.py** | 80 | Batch experiment runner |
| **analyze_quick_signals.py** | 100 | Parse results, print comparison table |

**Total New Code:** ~650 lines (well-organized, reusable)

---

## 🏗️ Reference/Unchanged Files

| File | Purpose | Note |
|------|---------|------|
| **model.py** | Transformer architecture | Unchanged |
| **muon.py** | Original Muon implementation | Reference only |
| **train.py** | Original training script | Reference only |
| **analysis_utils.py** | Hessian analysis tools | Will integrate in Phase 2 |

---

## 🎯 Quick Navigation by Task

### "I want to run experiments right now"
→ `QUICK_START.txt` (2 min read) → Pick one command → Run it

### "I want to understand the code structure"
→ `REFACTORING_SUMMARY.md` → Browse optimizers.py and train_quick_signals.py

### "I want to run multiple experiments"
→ `run_experiment.py` with `--quick`, `--oracle`, or `--ablation` flags

### "I want to analyze results"
→ Run `python analyze_quick_signals.py` after experiments complete

### "I want to extend this to Phase 2"
→ `../plan.md` + `REFACTORING_SUMMARY.md` (explains extensibility)

### "I want to understand the research question"
→ `PHASE1_README.md` (first section) + Your question in intro

---

## 📊 Expected Outputs

After running experiments, you'll have:

```
results/
├── quick_signals_muon_s42.json       # Metrics from Muon run
├── quick_signals_sgd_s42.json        # Metrics from SGD run
├── quick_signals_adam_s42.json       # Metrics from Adam run
└── quick_signals_muon_no_ortho_s42.json  # (Optional) ablation
```

Run `python analyze_quick_signals.py` to see comparison table:
- Final Loss per optimizer
- Min Loss achieved
- Time per step
- Cosine Similarity (if --compute_oracle was used)

---

## 🔑 Key Metrics to Watch

### Convergence Comparison
- **Final Loss**: Lower is better
- **Min Loss**: Best performance
- **Time/Step**: Should be ≈ same for SGD/Muon (Adam may differ)

### Oracle Alignment (with --compute_oracle)
- **> 0.8**: Excellent — Muon matches second-order optimal
- **0.5-0.8**: Good — Reasonable despite layer-wise approximation
- **0.3-0.5**: Moderate — Cross-layer effects visible
- **< 0.3**: Poor — Layer-wise breaks down

---

## 🚀 Command Reference

```bash
# Single run (5 min)
python train_quick_signals.py --optimizer muon --steps 500

# All baselines (15 min)
python run_experiment.py --steps 1000 --quick

# With oracle tracking (30 min)
python run_experiment.py --steps 1000 --oracle

# With ablation (20 min)
python run_experiment.py --steps 1000 --ablation

# Analyze all results
python analyze_quick_signals.py
```

---

## 💡 Next Steps

1. **Read** `QUICK_START.txt` (2 min)
2. **Run** one experiment option (5-30 min depending on choice)
3. **Check** results with `python analyze_quick_signals.py`
4. **Interpret** using `PHASE1_README.md` (Interpretation Guide section)
5. **Decide** Phase 2 direction based on cosine_sim value

---

## 📞 Troubleshooting

**Code doesn't run?**
→ Check Python/PyTorch environment is set up
→ See `PHASE1_README.md` Troubleshooting section

**Oracle computation slow/OOM?**
→ Either skip --compute_oracle for now
→ Or reduce batch size in muon.py training_config

**Want to modify experiments?**
→ Each file is self-contained and well-commented
→ Easiest to start with `train_quick_signals.py`

---

**Ready to start?** → `QUICK_START.txt`
