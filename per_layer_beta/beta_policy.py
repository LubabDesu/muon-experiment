"""Per-parameter beta policies for Muon experiments.

These helpers are intentionally independent of a specific training loop so they
can be ported into modded-nanoGPT's optimizer setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class BetaAssignment:
    name: str
    shape: tuple[int, ...]
    beta: float | None
    reason: str


def is_muon_eligible(shape: tuple[int, ...]) -> bool:
    """Muon orthogonalization is only meaningful for matrix parameters."""
    return len(shape) == 2


def shape_based_beta(info: ParamInfo) -> BetaAssignment:
    """Assign the MVP static beta from name and matrix shape.

    Returns beta=None for parameters that should not be handled by Muon.
    """
    if not is_muon_eligible(info.shape):
        return BetaAssignment(info.name, info.shape, None, "not_matrix")

    lowered = info.name.lower()
    rows, cols = info.shape
    min_dim = min(rows, cols)
    aspect = max(rows, cols) / max(1, min_dim)

    if "embed" in lowered or "wte" in lowered or "lm_head" in lowered:
        return BetaAssignment(info.name, info.shape, None, "adamw_embedding_or_head")
    if "gate" in lowered or "router" in lowered:
        return BetaAssignment(info.name, info.shape, None, "adamw_gate")
    if min_dim < 64:
        return BetaAssignment(info.name, info.shape, 0.90, "small_matrix")
    if aspect >= 4.0:
        return BetaAssignment(info.name, info.shape, 0.97, "high_aspect_matrix")
    return BetaAssignment(info.name, info.shape, 0.95, "default_matrix")


def assign_shape_betas(params: Iterable[ParamInfo]) -> list[BetaAssignment]:
    return [shape_based_beta(param) for param in params]


def beta_summary(assignments: Iterable[BetaAssignment]) -> dict[str, object]:
    assignments = list(assignments)
    active = [item for item in assignments if item.beta is not None]
    by_reason: dict[str, int] = {}
    for item in assignments:
        by_reason[item.reason] = by_reason.get(item.reason, 0) + 1

    betas = [item.beta for item in active if item.beta is not None]
    return {
        "n_params": len(assignments),
        "n_muon_params": len(active),
        "unique_betas": sorted(set(betas)),
        "by_reason": by_reason,
    }


def named_parameters_to_infos(named_parameters) -> list[ParamInfo]:
    """Convert a PyTorch named_parameters() iterator without importing torch."""
    return [ParamInfo(name=name, shape=tuple(param.shape)) for name, param in named_parameters]
