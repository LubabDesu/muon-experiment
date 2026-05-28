#!/usr/bin/env python3
"""Mini GPT pretrain for global vs bank Muon beta ablation."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_pretrain.config import TrainConfig, load_config
from mini_pretrain.data import create_train_iterator, create_val_iterator
from mini_pretrain.model_gpt import build_model
from mini_pretrain.optim import build_optimizers, optimizer_step


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def print_assignment(rows: list[dict], beta_policy: str, run_mode: str) -> None:
    print(f"run_mode={run_mode} beta_policy={beta_policy}")
    muon_rows = [r for r in rows if r["optimizer"] == "muon"]
    unique_betas = sorted({r["beta"] for r in muon_rows})
    print(f"muon_params={len(muon_rows)} unique_betas={unique_betas}")
    for row in muon_rows[:12]:
        print(f"  {row['name']}: beta={row['beta']}")
    if len(muon_rows) > 12:
        print(f"  ... {len(muon_rows) - 12} more muon tensors")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_iter,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(max_batches):
        x, y = val_iter.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    model = build_model(cfg.model).to(device)
    n_params = model.count_parameters()
    print(f"model_params={n_params/1e6:.2f}M preset={cfg.preset} tie_weights={cfg.model.tie_weights}")

    batch_size = max(1, cfg.batch_tokens // cfg.model.max_seq_len)
    random_ce = math.log(cfg.model.vocab_size)
    print(
        f"device={device} batch_tokens={cfg.batch_tokens} seq_len={cfg.model.max_seq_len} "
        f"batch_size={batch_size} lr_adam={cfg.lr_adam} lr_muon={cfg.lr_muon} "
        f"wd_adam={cfg.weight_decay_adam} wd_muon={cfg.weight_decay_muon} "
        f"use_synthetic={cfg.data.use_synthetic} random_guess_ce={random_ce:.4f}"
    )
    if cfg.data.use_synthetic:
        print("WARNING: USE_SYNTHETIC=1 — random tokens only.")
    else:
        from mini_pretrain.data import resolve_data_dir

        print(f"data_dir_resolved={resolve_data_dir(cfg.data.data_dir)}")
    train_iter = create_train_iterator(
        cfg.data.data_dir,
        cfg.data.train_glob,
        cfg.data.num_train_shards,
        batch_size,
        cfg.model.max_seq_len,
        cfg.seed,
        cfg.data.use_synthetic,
        cfg.data.synthetic_tokens,
        cfg.model.vocab_size,
    )
    val_iter = create_val_iterator(
        cfg.data.data_dir,
        cfg.data.val_file,
        batch_size,
        cfg.model.max_seq_len,
        cfg.seed,
        cfg.data.use_synthetic,
        cfg.data.synthetic_tokens,
        cfg.model.vocab_size,
        num_shards=cfg.data.num_train_shards,
        train_glob=cfg.data.train_glob,
    )

    optimizers, assign_rows = build_optimizers(
        model,
        cfg.run_mode,
        cfg.beta_policy,
        cfg.base_beta,
        cfg.lr_adam,
        cfg.lr_muon,
        cfg.weight_decay_adam,
        cfg.weight_decay_muon,
        cfg.muon_ns_steps,
    )
    print_assignment(assign_rows, cfg.beta_policy, cfg.run_mode)

    results_path = Path(cfg.results_dir) / f"{cfg.run_id}.jsonl"
    meta_path = Path(cfg.results_dir) / f"{cfg.run_id}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "run_mode": cfg.run_mode,
                "beta_policy": cfg.beta_policy,
                "preset": cfg.preset,
                "seed": cfg.seed,
                "steps": cfg.steps,
                "n_params": n_params,
                "model": model.config_dict(),
                "assignment": assign_rows,
            },
            indent=2,
        )
    )

    model.train()
    t0 = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % cfg.val_every == 0 or step == cfg.steps:
            val_loss = evaluate(model, val_iter, device)
            elapsed = time.perf_counter() - t0
            rec = {
                "step": step,
                "val_loss": val_loss,
                "elapsed_s": elapsed,
                "run_mode": cfg.run_mode,
                "beta_policy": cfg.beta_policy,
            }
            log_jsonl(results_path, rec)
            print(f"step {step}/{cfg.steps} val_loss={val_loss:.4f} elapsed={elapsed:.1f}s")

        if step == cfg.steps:
            break

        x, y = train_iter.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
            _, loss = model(x, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer_step(optimizers)

        if step % cfg.log_every == 0:
            log_jsonl(
                results_path,
                {"step": step, "train_loss": loss.item(), "run_mode": cfg.run_mode},
            )
            print(f"step {step}/{cfg.steps} train_loss={loss.item():.4f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "mini"], default=None)
    parser.add_argument("--run-mode", choices=["adamw", "muon_global", "muon_bank"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.preset)
    if args.run_mode:
        cfg.run_mode = args.run_mode
        if cfg.run_mode == "muon_global":
            cfg.beta_policy = "global"
        elif cfg.run_mode == "muon_bank":
            cfg.beta_policy = "bank"

    train(cfg)


if __name__ == "__main__":
    main()
