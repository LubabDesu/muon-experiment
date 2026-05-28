"""Build AdamW / Muon optimizers for mini_pretrain."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_pretrain.beta_assign import (
    BankBetaOffsets,
    DEFAULT_BANK_OFFSETS,
    assignment_table,
    beta_for_name,
    should_use_muon,
)
from per_layer_beta.muon_beta import PerLayerBetaMuon


def _uses_adam_weight_decay(name: str, param: nn.Parameter) -> bool:
    """Decay dense weights, not embeddings or normalization params."""
    lowered = name.lower()
    if param.ndim < 2:
        return False
    if "wte" in lowered or "wpe" in lowered or "embed" in lowered or "lm_head" in lowered:
        return False
    if "norm" in lowered or "ln_" in lowered:
        return False
    return True


def _adamw_param_groups(
    named_params: list[tuple[str, nn.Parameter]],
    weight_decay: float,
) -> list[dict[str, object]]:
    decay = [p for name, p in named_params if _uses_adam_weight_decay(name, p)]
    no_decay = [p for name, p in named_params if not _uses_adam_weight_decay(name, p)]
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


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
    bank_offsets: BankBetaOffsets = DEFAULT_BANK_OFFSETS,
) -> tuple[list[torch.optim.Optimizer], list[dict[str, object]]]:
    """Return optimizers list (step all) and assignment metadata rows."""
    if run_mode == "adamw":
        named_params = list(model.named_parameters())
        opt = torch.optim.AdamW(
            _adamw_param_groups(named_params, weight_decay_adam),
            lr=lr_adam,
        )
        rows = [
            {"name": n, "shape": tuple(p.shape), "optimizer": "adamw", "beta": None}
            for n, p in named_params
        ]
        return [opt], rows

    adam_named: list[tuple[str, nn.Parameter]] = []
    muon_named: list[tuple[str, nn.Parameter]] = []

    for name, param in model.named_parameters():
        if should_use_muon(name, tuple(param.shape)):
            muon_named.append((name, param))
        else:
            adam_named.append((name, param))

    def beta_fn(name: str, shape: tuple[int, ...]) -> float:
        beta = beta_for_name(name, shape, beta_policy, base_beta, bank_offsets)
        if beta is None:
            raise RuntimeError(f"Muon optimizer received non-Muon param: {name}")
        return beta

    adam_opt = torch.optim.AdamW(
        _adamw_param_groups(adam_named, weight_decay_adam),
        lr=lr_adam,
    )
    muon_opt = PerLayerBetaMuon(
        muon_named,
        beta_for_name=beta_fn,
        lr=lr_muon,
        weight_decay=weight_decay_muon,
        ns_steps=muon_ns_steps,
    )
    rows = assignment_table(model.named_parameters(), beta_policy, base_beta, bank_offsets)
    return [adam_opt, muon_opt], rows


def optimizer_step(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.step()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
