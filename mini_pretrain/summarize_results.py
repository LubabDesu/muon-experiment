#!/usr/bin/env python3
"""Build comparison table from mini_pretrain JSONL logs (val loss + wall time)."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunSummary:
    run_id: str
    run_mode: str = ""
    preset: str = ""
    seed: int = 0
    steps: int = 0
    bank_delta: str = "—"
    val_by_step: dict[int, float] = field(default_factory=dict)
    final_train_loss: float | None = None
    wall_s: float = 0.0

    @property
    def final_step(self) -> int:
        return max(self.val_by_step) if self.val_by_step else 0

    @property
    def final_val(self) -> float | None:
        if not self.val_by_step:
            return None
        return self.val_by_step[self.final_step]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_meta(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def summarize_jsonl(jsonl_path: Path) -> RunSummary:
    meta = load_meta(jsonl_path.with_name(jsonl_path.stem + "_meta.json"))
    rows = load_jsonl(jsonl_path)
    run_id = jsonl_path.stem
    s = RunSummary(
        run_id=run_id,
        run_mode=meta.get("run_mode", ""),
        preset=meta.get("preset", ""),
        seed=int(meta.get("seed", 0)),
        steps=int(meta.get("steps", 0)),
    )
    offsets = meta.get("bank_offsets") or {}
    if offsets:
        qk = offsets.get("qk", 0)
        mlp = offsets.get("mlp", 0)
        s.bank_delta = f"qk{qk:+.2f} mlp{mlp:+.2f}"
    for r in rows:
        if "val_loss" in r:
            step = int(r["step"])
            s.val_by_step[step] = float(r["val_loss"])
            s.wall_s = max(s.wall_s, float(r.get("elapsed_s", 0)))
        if "train_loss" in r:
            s.final_train_loss = float(r["train_loss"])
    if not s.run_mode:
        for mode in ("adamw", "muon_global", "muon_bank"):
            if run_id.startswith(mode):
                s.run_mode = mode
                break
    return s


def read_session_run_ids(session_file: Path) -> list[str]:
    return [ln.strip() for ln in session_file.read_text().splitlines() if ln.strip()]


def latest_run_per_mode(
    results_dir: Path, preset: str, seed: int, modes: list[str]
) -> list[Path]:
    paths: list[Path] = []
    for mode in modes:
        pattern = f"{mode}-{preset}-seed{seed}-*.jsonl"
        matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if matches:
            paths.append(matches[-1])
    return paths


def format_val(v: float | None, digits: int = 4) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def build_markdown_table(
    summaries: list[RunSummary],
    checkpoint_steps: list[int],
    title: str,
) -> str:
    headers = ["run", "bank Δ", *[f"val@{s}" for s in checkpoint_steps], "val@final", "train@last", "wall"]
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for s in summaries:
        row = [
            s.run_mode or s.run_id,
            s.bank_delta,
        ]
        for step in checkpoint_steps:
            row.append(format_val(s.val_by_step.get(step)))
        row.append(format_val(s.final_val))
        row.append(format_val(s.final_train_loss))
        row.append(format_time(s.wall_s))
        lines.append("| " + " | ".join(row) + " |")

    if len(summaries) >= 2:
        best = min((x for x in summaries if x.final_val is not None), key=lambda x: x.final_val)
        lines.extend(
            [
                "",
                f"**Best final val:** `{best.run_mode}` ({best.final_val:.4f}) · "
                f"**Fastest:** `{min(summaries, key=lambda x: x.wall_s).run_mode}` "
                f"({format_time(min(s.wall_s for s in summaries))})",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def append_to_results_md(results_md: Path, block: str) -> None:
    start = "<!-- AUTO_SUMMARY_START -->"
    end = "<!-- AUTO_SUMMARY_END -->"
    if results_md.exists():
        text = results_md.read_text()
    else:
        text = "# Mini pretrain — experiment log\n\n"
    if start in text and end in text:
        before = text.split(start)[0]
        after = text.split(end)[1]
        text = before + start + "\n" + block + end + after
    else:
        text = text.rstrip() + f"\n\n## Latest comparison (auto)\n\n{start}\n{block}{end}\n"
    results_md.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize mini_pretrain JSONL runs")
    parser.add_argument("--results-dir", default="results/mini_pretrain")
    parser.add_argument("--session", type=str, default=None, help="File with one run_id per line")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--modes",
        type=str,
        default="adamw,muon_global,muon_bank",
        help="Comma-separated run modes for --latest",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="500,1000,3000",
        help="Val steps to show as columns (missing steps shown as —)",
    )
    parser.add_argument("--append-results-md", action="store_true")
    parser.add_argument("--results-md", default="mini_pretrain/RESULTS.md")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir

    jsonl_paths: list[Path] = []
    if args.session:
        session = Path(args.session)
        if not session.is_absolute():
            session = repo_root / session
        for run_id in read_session_run_ids(session):
            p = results_dir / f"{run_id}.jsonl"
            if p.exists():
                jsonl_paths.append(p)
            else:
                print(f"warning: missing {p}")
    elif args.preset is not None and args.seed is not None:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        jsonl_paths = latest_run_per_mode(results_dir, args.preset, args.seed, modes)
    else:
        jsonl_paths = sorted(results_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)

    if not jsonl_paths:
        print("No runs found.")
        return

    summaries = [summarize_jsonl(p) for p in jsonl_paths]
    order = {"adamw": 0, "muon_global": 1, "muon_bank": 2}
    summaries.sort(key=lambda s: (order.get(s.run_mode, 99), s.run_id))

    checkpoint_steps = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    preset = summaries[0].preset or args.preset or "?"
    seed = summaries[0].seed if args.seed is None else args.seed
    title = (
        f"Comparison · preset={preset} seed={seed} · "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    block = build_markdown_table(summaries, checkpoint_steps, title)
    print(block)

    if args.append_results_md:
        md_path = Path(args.results_md)
        if not md_path.is_absolute():
            md_path = Path(__file__).resolve().parents[1] / md_path
        append_to_results_md(md_path, block)
        print(f"Appended to {md_path}")


if __name__ == "__main__":
    main()
