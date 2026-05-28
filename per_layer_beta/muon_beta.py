"""Reference Muon update with per-parameter beta.

This is a compact implementation meant to clarify the mechanical change needed
in modded-nanoGPT: beta is selected per parameter before updating the momentum
buffer, then Newton-Schulz is applied to the resulting direction.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    x = x / (x.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + b * gram @ x + c * gram @ gram @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    ns_steps: int = 5,
) -> torch.Tensor:
    """Return the Muon update direction for one parameter."""
    momentum_buffer.mul_(beta).add_(grad)
    direction = grad.add(momentum_buffer, alpha=beta)
    if direction.ndim == 2:
        return newton_schulz(direction, steps=ns_steps)
    return direction


class PerLayerBetaMuon(torch.optim.Optimizer):
    """Minimal optimizer reference for static per-parameter beta policies."""

    def __init__(
        self,
        named_params,
        beta_for_name: Callable[[str, tuple[int, ...]], float],
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ):
        named_params = list(named_params)
        params = [param for _, param in named_params]
        defaults = dict(lr=lr, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self._names = {param: name for name, param in named_params}
        self._betas = {
            param: beta_for_name(name, tuple(param.shape))
            for name, param in named_params
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)

                if group["weight_decay"] != 0:
                    param.add_(param, alpha=-group["lr"] * group["weight_decay"])

                beta = self._betas[param]
                update = muon_update(
                    param.grad,
                    state["momentum_buffer"],
                    beta=beta,
                    ns_steps=group["ns_steps"],
                )
                scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5 if param.ndim == 2 else 1.0
                param.add_(update, alpha=-group["lr"] * scale)

        return loss

    def beta_by_name(self) -> dict[str, float]:
        return {self._names[param]: beta for param, beta in self._betas.items()}
