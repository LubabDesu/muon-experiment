import torch
import torch.nn as nn

# ── Muon Optimizer ──────────────────────────────────────────────────────────

def newton_schulz(G, steps=5, eps=1e-7):
    """Orthogonalize G via Newton-Schulz (same as Muon)"""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = (a * X + b * A @ X + c * A @ A @ X)
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer: Momentum + Newton-Schulz orthogonalization.
    Applies orthogonalization only to 2D weight matrices (attention, dense layers).
    """
    def __init__(self, params, lr=3e-4, momentum=0.9, n_steps=5, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, n_steps=n_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                momentum = group['momentum']
                buf.mul_(momentum).add_(grad)
                
                # Nesterov momentum update
                nesterov_grad = grad.add(buf, alpha=momentum)

                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

                # Muon update: orthogonalize if 2D, else standard SGD-like
                if p.ndim == 2:
                    update = newton_schulz(nesterov_grad, steps=group['n_steps'])
                    # Scale update by sqrt(rows/cols) as per Muon paper/impl
                    scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                    p.add_(update, alpha=-group['lr'] * scale)
                else:
                    p.add_(nesterov_grad, alpha=-group['lr'])

                state['step'] += 1

        return loss


class MuonWithoutOrthogonalization(torch.optim.Optimizer):
    """Muon without orthogonalization (just momentum) for ablation."""
    def __init__(self, params, lr=3e-4, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                momentum = group['momentum']
                buf.mul_(momentum).add_(grad)
                nesterov_grad = grad.add(buf, alpha=momentum)

                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

                p.add_(nesterov_grad, alpha=-group['lr'])
                state['step'] += 1

        return loss
