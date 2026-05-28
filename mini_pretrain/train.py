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


def apply_hparams_file(path: str) -> None:
    """Load KEY=VALUE lines into environment for config parsing."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = _ROOT / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"hparams file not found: {file_path}")
    for raw in file_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid hparams line (expected KEY=VALUE): {raw}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid hparams key in line: {raw}")
        # Keep ad-hoc shell exports higher priority than file defaults.
        os.environ.setdefault(key, value)


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


def _lr_scale(step: int, cfg: TrainConfig) -> float:
    if cfg.steps <= 0:
        return 1.0
    warmup = min(cfg.lr_warmup_steps, cfg.steps)
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if cfg.lr_schedule == "constant":
        return 1.0
    # cosine decay from 1.0 to cfg.min_lr_scale
    start = warmup
    total = max(1, cfg.steps - start)
    t = min(max(0.0, (step - start) / total), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine


def _set_optimizer_lrs(optimizers: list[torch.optim.Optimizer], cfg: TrainConfig, step: int) -> tuple[float, float]:
    scale = _lr_scale(step, cfg)
    lr_adam_now = cfg.lr_adam * scale
    lr_muon_now = cfg.lr_muon * scale
    if cfg.run_mode == "adamw":
        for group in optimizers[0].param_groups:
            group["lr"] = lr_adam_now
        return lr_adam_now, 0.0
    for group in optimizers[0].param_groups:
        group["lr"] = lr_adam_now
    for group in optimizers[1].param_groups:
        group["lr"] = lr_muon_now
    return lr_adam_now, lr_muon_now


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
        f"batch_size={batch_size} grad_accum={cfg.grad_accum_steps} "
        f"effective_batch={batch_size * cfg.grad_accum_steps} "
        f"lr_adam={cfg.lr_adam} lr_muon={cfg.lr_muon} "
        f"lr_schedule={cfg.lr_schedule} warmup={cfg.lr_warmup_steps} min_lr_scale={cfg.min_lr_scale} "
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
        bank_offsets=cfg.bank_offsets,
    )
    if cfg.run_mode == "muon_bank":
        o = cfg.bank_offsets
        print(
            f"bank_offsets: qk={o.qk:+.4f} vo={o.vo:+.4f} mlp={o.mlp:+.4f} "
            f"(betas @ base={cfg.base_beta}: "
            f"qk={cfg.base_beta + o.qk:.3f} vo={cfg.base_beta + o.vo:.3f} mlp={cfg.base_beta + o.mlp:.3f})"
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
                "base_beta": cfg.base_beta,
                "bank_offsets": {
                    "qk": cfg.bank_offsets.qk,
                    "vo": cfg.bank_offsets.vo,
                    "mlp": cfg.bank_offsets.mlp,
                },
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
    log_jsonl(
        results_path,
        {
            "event": "run_start",
            "run_id": cfg.run_id,
            "run_mode": cfg.run_mode,
            "beta_policy": cfg.beta_policy,
            "preset": cfg.preset,
            "seed": cfg.seed,
            "steps": cfg.steps,
            "batch_tokens": cfg.batch_tokens,
            "lr_adam": cfg.lr_adam,
            "lr_muon": cfg.lr_muon,
            "lr_schedule": cfg.lr_schedule,
            "lr_warmup_steps": cfg.lr_warmup_steps,
            "min_lr_scale": cfg.min_lr_scale,
            "early_stop_patience_evals": cfg.early_stop_patience_evals,
            "early_stop_min_delta": cfg.early_stop_min_delta,
            "max_val_increase_from_best": cfg.max_val_increase_from_best,
            "weight_decay_adam": cfg.weight_decay_adam,
            "weight_decay_muon": cfg.weight_decay_muon,
            "muon_ns_steps": cfg.muon_ns_steps,
        },
    )

    model.train()
    t0 = time.perf_counter()
    best_val = float("inf")
    best_step = 0
    stale_evals = 0
    stop_reason: str | None = None
    final_step = 0
    for step in range(cfg.steps + 1):
        final_step = step
        lr_adam_now, lr_muon_now = _set_optimizer_lrs(optimizers, cfg, step)
        if step % cfg.val_every == 0 or step == cfg.steps:
            val_loss = evaluate(model, val_iter, device)
            elapsed = time.perf_counter() - t0
            rec = {
                "step": step,
                "val_loss": val_loss,
                "elapsed_s": elapsed,
                "run_mode": cfg.run_mode,
                "beta_policy": cfg.beta_policy,
                "lr_adam": lr_adam_now,
                "lr_muon": lr_muon_now,
            }
            log_jsonl(results_path, rec)
            if val_loss + cfg.early_stop_min_delta < best_val:
                best_val = val_loss
                best_step = step
                stale_evals = 0
            else:
                stale_evals += 1
            print(
                f"step {step}/{cfg.steps} val_loss={val_loss:.4f} best={best_val:.4f}@{best_step} "
                f"lr_adam={lr_adam_now:.6g} lr_muon={lr_muon_now:.6g} elapsed={elapsed:.1f}s"
            )
            if (
                cfg.max_val_increase_from_best > 0
                and step >= cfg.val_every
                and val_loss > best_val + cfg.max_val_increase_from_best
            ):
                stop_reason = (
                    f"val_diverged: val_loss={val_loss:.4f} > best+{cfg.max_val_increase_from_best:.3f}"
                )
                print(f"early_stop step={step}: {stop_reason}")
                break
            if cfg.early_stop_patience_evals > 0 and stale_evals >= cfg.early_stop_patience_evals:
                stop_reason = f"no_improvement_for_{stale_evals}_evals"
                print(f"early_stop step={step}: {stop_reason}")
                break

        if step == cfg.steps:
            break

        accum_loss = 0.0
        for micro in range(cfg.grad_accum_steps):
            x, y = train_iter.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                stop_reason = f"non_finite_loss at step {step} micro {micro}: {loss.item()}"
                print(f"early_stop step={step}: {stop_reason}")
                break
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item()
        if stop_reason is not None:
            break
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer_step(optimizers)
        loss_for_log = accum_loss / cfg.grad_accum_steps

        if step % cfg.log_every == 0:
            log_jsonl(
                results_path,
                {
                    "step": step,
                    "train_loss": loss_for_log,
                    "run_mode": cfg.run_mode,
                    "lr_adam": lr_adam_now,
                    "lr_muon": lr_muon_now,
                },
            )
            print(
                f"step {step}/{cfg.steps} train_loss={loss_for_log:.4f} "
                f"lr_adam={lr_adam_now:.6g} lr_muon={lr_muon_now:.6g}"
            )
    elapsed = time.perf_counter() - t0
    log_jsonl(
        results_path,
        {
            "event": "run_end",
            "step": final_step,
            "elapsed_s": elapsed,
            "best_val": best_val,
            "best_step": best_step,
            "stopped_early": stop_reason is not None,
            "stop_reason": stop_reason,
        },
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "mini"], default=None)
    parser.add_argument("--run-mode", choices=["adamw", "muon_global", "muon_bank"], default=None)
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="Path to KEY=VALUE file with env-style training overrides",
    )
    args = parser.parse_args()

    if args.hparams:
        apply_hparams_file(args.hparams)
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
