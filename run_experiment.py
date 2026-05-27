#!/usr/bin/env python3
"""
Batch runner for quick signals experiments.

Usage:
  python run_experiment.py --steps 1000 --quick
  python run_experiment.py --steps 5000 --oracle  # Slower, includes oracle
"""

import subprocess
import argparse
import sys
from pathlib import Path


def run_experiment(optimizer, steps, compute_oracle=False, seed=42):
    """Run a single experiment."""
    cmd = [
        'python', 'train_quick_signals.py',
        '--optimizer', optimizer,
        '--steps', str(steps),
        '--seed', str(seed),
    ]
    if compute_oracle:
        cmd.append('--compute_oracle')
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch runner for Muon experiments")
    parser.add_argument('--steps', type=int, default=1000, help='Steps per run')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 3 optimizers, no oracle')
    parser.add_argument('--oracle', action='store_true', help='Run with oracle tracking (slow)')
    parser.add_argument('--ablation', action='store_true', help='Include ablation (muon_no_ortho)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    optimizers = ['muon']
    
    if args.quick:
        optimizers = ['muon', 'sgd', 'adam']
        compute_oracle = False
    elif args.oracle:
        optimizers = ['muon', 'sgd', 'adam']
        compute_oracle = True
    elif args.ablation:
        optimizers = ['muon', 'muon_no_ortho', 'sgd', 'adam']
        compute_oracle = False
    else:
        compute_oracle = False
    
    print(f"\n🚀 Muon Quick Signals Experiment")
    print(f"   Steps: {args.steps}")
    print(f"   Optimizers: {', '.join(optimizers)}")
    print(f"   Oracle tracking: {compute_oracle}")
    print(f"   Seed: {args.seed}\n")
    
    failed = []
    for opt in optimizers:
        success = run_experiment(opt, args.steps, compute_oracle, args.seed)
        if not success:
            failed.append(opt)
    
    print(f"\n{'='*70}")
    if failed:
        print(f"❌ Failed optimizers: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅ All runs completed successfully!")
        print("\nAnalyze results:")
        print("  python analyze_quick_signals.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
