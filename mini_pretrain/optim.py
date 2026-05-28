"""Build AdamW / Muon optimizers for mini_pretrain."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_pretrain.beta_assign import assignment_table, beta_for_name, should_use_muon
from per_layer_beta.muon_beta import PerLayerBetaMuon


def build_optimizers(
    model: nn.Module,
    run_mode: str,
    beta_policy: str,
    base_beta: float,
    lr_adam: float,
    lr_muon: float,
    weight_decay_adam: float,
    weight_decay_muon: float,
    muon_ns_steps: int,
) -> tuple[list[torch.optim.Optimizer], list[dict[str, object]]]:
    """Return optimizers list (step all) and assignment metadata rows."""
    if run_mode == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=weight_decay_adam)
        rows = [
            {"name": n, "shape": tuple(p.shape), "optimizer": "adamw", "beta": None}
            for n, p in model.named_parameters()
        ]
        return [opt], rows

    adam_params: list[nn.Parameter] = []
    muon_named: list[tuple[str, nn.Parameter]] = []

    for name, param in model.named_parameters():
        if should_use_muon(name, tuple(param.shape)):
            muon_named.append((name, param))
        else:
            adam_params.append(param)

    def beta_fn(name: str, shape: tuple[int, ...]) -> float:
        beta = beta_for_name(name, shape, beta_policy, base_beta)
        if beta is None:
            raise RuntimeError(f"Muon optimizer received non-Muon param: {name}")
        return beta

    adam_opt = torch.optim.AdamW(adam_params, lr=lr_adam, weight_decay=weight_decay_adam)
    muon_opt = PerLayerBetaMuon(
        muon_named,
        beta_for_name=beta_fn,
        lr=lr_muon,
        weight_decay=weight_decay_muon,
        ns_steps=muon_ns_steps,
    )
    rows = assignment_table(model.named_parameters(), beta_policy, base_beta)
    return [adam_opt, muon_opt], rows


def optimizer_step(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
