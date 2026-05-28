#!/usr/bin/env python3
"""Plot val_loss curves from results/mini_pretrain/*.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_val_points(path: Path) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "val_loss" in rec:
            steps.append(rec["step"])
            losses.append(rec["val_loss"])
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/mini_pretrain")
    parser.add_argument("--out", default="results/mini_pretrain/val_loss.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in sorted(results_dir.glob("*.jsonl")):
        steps, losses = load_val_points(path)
        if steps:
            ax.plot(steps, losses, label=path.stem)
    ax.set_xlabel("step")
    ax.set_ylabel("val_loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
