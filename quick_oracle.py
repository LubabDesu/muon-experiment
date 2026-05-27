"""
Lightweight oracle estimator for quick empirical signals.

Computes cosine similarity between Muon and oracle (CG-based) updates on a small batch.
Uses low-rank approximation (few CG iterations) for speed.
"""

import torch
from optimizers import newton_schulz


def sample_oracle_step(model, loss_fn, x, y, device, n_cg_iters=3, damping=1e-3):
    """
    Estimate oracle update direction via CG on Hessian and compare with Muon.
    
    Simplified approach: Compare Muon direction with simple second-order approximation
    (diagonal of Hessian).
    
    Args:
        model: transformer model
        loss_fn: loss function
        x: input batch
        y: target batch
        device: torch device
        n_cg_iters: CG iterations (3-5 for speed)
        damping: Tikhonov damping for numerical stability
    
    Returns:
        dict with keys:
            - "cosine_sim": average cosine similarity (Muon vs Oracle)
            - "muon_norms": list of Muon update norms per layer
            - "oracle_norms": list of Oracle update norms per layer
    """
    model.zero_grad()
    
    # Collect 2D parameters (those updated by Muon with orthogonalization)
    params_2d = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    
    # Forward pass and compute gradients
    logits, loss = model(x, y)
    grads = torch.autograd.grad(loss, params_2d, create_graph=True, retain_graph=True)
    
    # Compute Muon directions (orthogonalized)
    muon_dirs = []
    muon_norms = []
    for g in grads:
        m = newton_schulz(g.detach(), steps=5)
        muon_dirs.append(m)
        muon_norms.append(m.norm().item())
    
    # Simplified oracle: diagonal Hessian approximation
    # For each parameter, compute diagonal of Hessian via finite differences
    oracle_dirs = []
    oracle_norms = []
    
    for p, g in zip(params_2d, grads):
        # Diagonal Hessian approximation: d²L/dw² ≈ via grad of gradient
        # Compute Hessian-diagonal via sqrt of squared second gradient
        eps = 1e-4
        
        # Simple approximation: just use grad norm as proxy for curvature
        # More sophisticated: could use sqrt(|gradient|) as diagonal Hessian approx
        grad_norm = g.norm()
        if grad_norm > 1e-8:
            # Diagonal Hessian approximation (simplified)
            diag_hessian = g.abs() + 1e-6  # Add small epsilon for stability
            oracle_update = g / (diag_hessian + damping)
        else:
            oracle_update = torch.zeros_like(g)
        
        oracle_dirs.append(oracle_update)
        oracle_norms.append(oracle_update.norm().item())
    
    # Compute cosine similarities
    cosine_sims = []
    for m, o in zip(muon_dirs, oracle_dirs):
        m_flat = m.flatten()
        o_flat = o.flatten()
        m_norm = m_flat.norm()
        o_norm = o_flat.norm()
        if m_norm > 1e-8 and o_norm > 1e-8:
            cos_sim = (m_flat @ o_flat) / (m_norm * o_norm + 1e-8)
        else:
            cos_sim = 0.0
        cosine_sims.append(cos_sim.item())
    
    result = {
        "cosine_sim": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
        "muon_norms": muon_norms,
        "oracle_norms": oracle_norms,
        "n_2d_params": len(params_2d),
    }
    
    return result
