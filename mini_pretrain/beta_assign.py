"""Beta assignment for mini_pretrain (global vs bank)."""

from __future__ import annotations

import sys
from pathlib import Path

# per_layer_beta on path when running as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from per_layer_beta.beta_policy import is_muon_eligible


def _clamp_beta(beta: float, low: float = 0.5, high: float = 0.999) -> float:
    return max(low, min(high, beta))


def bank_offset(name: str) -> float:
    """Offsets relative to base beta (modded-nanogpt bank policy)."""
    lowered = name.lower()
    if "q_proj" in lowered or "k_proj" in lowered:
        return -0.01
    if "v_proj" in lowered or "o_proj" in lowered:
        return 0.0
    if "c_fc" in lowered or "c_proj" in lowered:
        return 0.01
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
) -> float | None:
    """Return Muon beta for a parameter, or None if AdamW should own it."""
    if not should_use_muon(name, shape):
        return None
    if policy == "global":
        return _clamp_beta(base_beta)
    if policy == "bank":
        return _clamp_beta(base_beta + bank_offset(name))
    raise ValueError(f"Unknown beta policy: {policy}")


def assignment_table(
    named_parameters,
    policy: str,
    base_beta: float = 0.95,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, param in named_parameters:
        beta = beta_for_name(name, tuple(param.shape), policy, base_beta)
        rows.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "optimizer": "muon" if beta is not None else "adamw",
                "beta": beta,
            }
        )
    return rows
