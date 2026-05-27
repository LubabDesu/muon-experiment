"""
Ablation sweep: depth ∈ {2,4,6} × skip ∈ {False,True} → 6 configs total.

Usage:
  python scripts/run_ablations.py --steps 1000 --samples 10
  python scripts/run_ablations.py --steps 200 --samples 5 --quick
  python scripts/run_ablations.py --steps 1000 --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

FULL_CONFIGS = [(d, s) for d in [2, 4, 6] for s in [False, True]]
QUICK_CONFIGS = [(3, False), (3, True)]


def run_config(depth: int, skip: bool, args: argparse.Namespace) -> tuple[bool, float]:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train_coupling.py"),
        "--depth", str(depth),
        "--steps", str(args.steps),
        "--samples", str(args.samples),
        "--cg-iters", str(args.cg_iters),
        "--damping", str(args.damping),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    if skip:
        cmd.append("--skip")

    label = f"depth={depth} skip={skip}"
    print(f"\n{'='*50}")
    print(f"Running: {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*50}")

    if args.dry_run:
        print("  [dry-run, skipping]")
        return True, 0.0

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    success = result.returncode == 0
    status = "OK" if success else f"FAILED (exit {result.returncode})"
    print(f"  {label}: {status} in {elapsed:.1f}s")
    return success, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--cg-iters", type=int, default=20)
    p.add_argument("--damping", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Depth=3 only (2 configs) for fast iteration")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    args = p.parse_args()

    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS
    print(f"Ablation sweep: {len(configs)} configs, {args.steps} steps each")
    if args.quick:
        print("  (--quick: depth=3 only)")

    results = []
    total_t0 = time.time()
    for depth, skip in configs:
        ok, elapsed = run_config(depth, skip, args)
        results.append((depth, skip, ok, elapsed))

    total = time.time() - total_t0
    print(f"\n{'='*50}")
    print(f"Summary ({len(configs)} configs, {total:.1f}s total):")
    print(f"{'depth':>6}  {'skip':>5}  {'status':>8}  {'time':>8}")
    print(f"{'-'*36}")
    for depth, skip, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"{depth:>6}  {str(skip):>5}  {status:>8}  {elapsed:>7.1f}s")

    n_failed = sum(1 for _, _, ok, _ in results if not ok)
    if n_failed:
        print(f"\n{n_failed}/{len(configs)} configs FAILED")
        sys.exit(1)
    else:
        print(f"\nAll {len(configs)} configs succeeded.")


if __name__ == "__main__":
    main()
