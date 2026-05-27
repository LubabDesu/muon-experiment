import torch
import torch.nn as nn
from muon import newton_schulz

# ── 1. HUTCHINSON CROSS-LAYER BLOCK ESTIMATOR ─────────────────────────────────
def hessian_block_frob_sq(model, loss_fn, x, y, layer_i, layer_j, n_samples=50):
    """
    Estimates ||d²L / dWi dWj||_F^2 using Hutchinson.
    """
    Wi = layer_i.weight
    Wj = layer_j.weight
    estimates = []

    for _ in range(n_samples):
        v = torch.randint(0, 2, Wj.shape, device=Wj.device).float() * 2 - 1
        v = v / (v.norm() + 1e-8)

        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        grad_j = torch.autograd.grad(loss, Wj, create_graph=True)[0]
        scalar = (grad_j * v).sum()
        grad2 = torch.autograd.grad(scalar, Wi, retain_graph=False)[0]
        estimates.append((grad2 ** 2).sum().item())

    return sum(estimates) / len(estimates)


def diagonal_block_frob_sq(model, loss_fn, x, y, layer_i, n_samples=50):
    """Same but for diagonal block H_ii"""
    Wi = layer_i.weight
    estimates = []
    for _ in range(n_samples):
        v = torch.randint(0, 2, Wi.shape, device=Wi.device).float() * 2 - 1
        v = v / (v.norm() + 1e-8)
        model.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        grad_i = torch.autograd.grad(loss, Wi, create_graph=True)[0]
        scalar = (grad_i * v).sum()
        grad2 = torch.autograd.grad(scalar, Wi, retain_graph=False)[0]
        estimates.append((grad2 ** 2).sum().item())
    return sum(estimates) / len(estimates)

# ── 2. ORACLE UPDATE DIRECTION (cross-layer corrected) ───────────────────────
    
def oracle_update(model, loss_fn, x, y, params, lr=0.01, cg_iters=5, damping=1e-3):
    """
    Approximate 'true' update for layer_i accounting for cross-layer coupling.
    """
    model.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    muon_dirs = {id(p): newton_schulz(g.detach()) for p, g in zip(params, grads)}
    b = -torch.nn.utils.parameters_to_vector(grads).detach()
    
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    offsets = [0] + torch.cumsum(torch.tensor(numels), 0).tolist()
    
    def global_hvp(v_flat):
        v_tensors = []
        for i, p in enumerate(params):
            v_tensors.append(v_flat[offsets[i]:offsets[i+1]].view_as(p))
        grad_v_dot = sum((g * v).sum() for g, v in zip(grads, v_tensors))
        hvp = torch.autograd.grad(grad_v_dot, params, retain_graph=True)
        return torch.nn.utils.parameters_to_vector(hvp)
    
    x_flat = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    r_dot_old = torch.dot(r, r)
    
    for _ in range(cg_iters):
        Hp = global_hvp(p) + damping * p
        pHp = torch.dot(p, Hp)
        if pHp <= 0:
            break
        alpha = r_dot_old / pHp
        x_flat += alpha * p
        r -= alpha * Hp
        r_dot_new = torch.dot(r, r)
        if r_dot_new < 1e-10 * b.norm()**2:
            break
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    
    true_dirs = {}
    for i, p in enumerate(params):
        true_dirs[id(p)] = lr * x_flat[offsets[i]:offsets[i+1]].view_as(p)
    
    return muon_dirs, true_dirs

def cosine_similarity_updates(muon_dirs, true_dirs, layer_id):
    m = muon_dirs[layer_id].flatten()
    t = true_dirs[layer_id].flatten()
    return (m @ t / (m.norm() * t.norm() + 1e-8)).item()
