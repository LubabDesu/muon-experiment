"""
Quick analysis of training results.

Usage:
  python analyze_quick_signals.py
"""

import json
import glob
from pathlib import Path

def analyze_run(filepath):
    """Extract key metrics from a run."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if not data:
        return None
    
    final = data[-1]
    
    # Find min loss
    losses = [m['loss'] for m in data]
    min_loss = min(losses)
    min_loss_step = data[[m['loss'] for m in data].index(min_loss)]['step']
    
    results = {
        'file': Path(filepath).name,
        'final_loss': final['loss'],
        'min_loss': min_loss,
        'min_loss_step': min_loss_step,
        'steps': final['step'],
        'total_time_s': final['elapsed_s'],
        'time_per_step_ms': final['elapsed_s'] / (final['step'] + 1) * 1000,
    }
    
    # Average cosine similarity if available
    cosine_sims = [m.get('cosine_sim') for m in data if 'cosine_sim' in m]
    if cosine_sims:
        results['avg_cosine_sim'] = sum(cosine_sims) / len(cosine_sims)
        results['min_cosine_sim'] = min(cosine_sims)
        results['max_cosine_sim'] = max(cosine_sims)
    
    return results

def main():
    result_files = sorted(glob.glob("results/quick_signals_*.json"))
    
    if not result_files:
        print("No result files found in results/")
        return
    
    print("\n" + "="*80)
    print("QUICK SIGNALS ANALYSIS")
    print("="*80)
    
    analyses = []
    for f in result_files:
        a = analyze_run(f)
        if a:
            analyses.append(a)
    
    # Group by optimizer
    by_optimizer = {}
    for a in analyses:
        name = a['file'].split('_')[2]  # Extract optimizer name
        if name not in by_optimizer:
            by_optimizer[name] = []
        by_optimizer[name].append(a)
    
    # Print comparison table
    print("\n📊 CONVERGENCE COMPARISON\n")
    print(f"{'Optimizer':<20} {'Final Loss':<15} {'Min Loss':<15} {'Time/Step (ms)':<15}")
    print("-" * 65)
    
    for opt_name in sorted(by_optimizer.keys()):
        runs = by_optimizer[opt_name]
        avg_final = sum(r['final_loss'] for r in runs) / len(runs)
        avg_min = sum(r['min_loss'] for r in runs) / len(runs)
        avg_time = sum(r['time_per_step_ms'] for r in runs) / len(runs)
        print(f"{opt_name:<20} {avg_final:<15.6f} {avg_min:<15.6f} {avg_time:<15.2f}")
    
    # Oracle alignment (if computed)
    print("\n🎯 MUON-ORACLE ALIGNMENT\n")
    muon_runs = by_optimizer.get('muon', [])
    if muon_runs and 'avg_cosine_sim' in muon_runs[0]:
        for a in muon_runs:
            print(f"  Cosine Sim (Muon): {a.get('avg_cosine_sim', 'N/A'):.4f} "
                  f"(range: {a.get('min_cosine_sim', 0):.4f}-{a.get('max_cosine_sim', 1):.4f})")
    else:
        print("  (Run with --compute_oracle to see alignment)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
