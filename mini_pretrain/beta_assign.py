"""Beta assignment for mini_pretrain (global vs bank)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from per_layer_beta.beta_policy import is_muon_eligible


@dataclass(frozen=True)
class BankBetaOffsets:
    """Per-bank offsets added to base_beta (modded-style: qk low, vo mid, mlp high)."""

    qk: float = -0.01
    vo: float = 0.0
    mlp: float = 0.01


DEFAULT_BANK_OFFSETS = BankBetaOffsets()


def _clamp_beta(beta: float, low: float = 0.5, high: float = 0.999) -> float:
    return max(low, min(high, beta))


def bank_offset(name: str, offsets: BankBetaOffsets = DEFAULT_BANK_OFFSETS) -> float:
    lowered = name.lower()
    if "q_proj" in lowered or "k_proj" in lowered:
        return offsets.qk
    if "v_proj" in lowered or "o_proj" in lowered:
        return offsets.vo
    if "c_fc" in lowered or "c_proj" in lowered:
        return offsets.mlp
    return 0.0


def should_use_muon(name: str, shape: tuple[int, ...]) -> bool:
    if not is_muon_eligible(shape):
        return False
    lowered = name.lower()
    if "wte" in lowered or "wpe" in lowered or "embed" in lowered or "lm_head" in lowered:
        return False
    if "norm" in lowered or "ln_" in lowered:
        return False
    if "bias" in lowered:
        return False
    return True


def beta_for_name(
    name: str,
    shape: tuple[int, ...],
    policy: str,
    base_beta: float = 0.95,
    bank_offsets: BankBetaOffsets = DEFAULT_BANK_OFFSETS,
) -> float | None:
    """Return Muon beta for a parameter, or None if AdamW should own it."""
    if not should_use_muon(name, shape):
        return None
    if policy == "global":
        return _clamp_beta(base_beta)
    if policy == "bank":
        return _clamp_beta(base_beta + bank_offset(name, bank_offsets))
    raise ValueError(f"Unknown beta policy: {policy}")


def assignment_table(
    named_parameters,
    policy: str,
    base_beta: float = 0.95,
    bank_offsets: BankBetaOffsets = DEFAULT_BANK_OFFSETS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, param in named_parameters:
        beta = beta_for_name(name, tuple(param.shape), policy, base_beta, bank_offsets)
        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "optimizer": "muon" if beta is not None else "adamw",
                "beta": beta,
            }
        )
    return rows
