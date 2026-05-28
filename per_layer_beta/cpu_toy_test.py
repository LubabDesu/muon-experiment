import sys
import os
import torch
import torch.nn as nn

# 1. Add modded-nanogpt to sys.path
sys.path.append("/Users/lucasyan/Desktop/UCSD/Spring 26/CSE151B/experiments/muon/modded-nanogpt")

# Set dummy env vars for import time checks and configuration
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["DISABLE_FP8"] = "1"

# Mock torch.distributed to prevent process group errors on single CPU
import torch.distributed as dist
dist.get_world_size = lambda *args, **kwargs: 1
dist.get_rank = lambda *args, **kwargs: 0
dist.is_initialized = lambda *args, **kwargs: False

# Mock torch.compile before importing train_gpt to prevent Inductor C++ compile space-in-path bug
torch.compile = lambda fn, *args, **kwargs: fn

# Import train_gpt (which does not crash on CPU anymore!)
import train_gpt

# 2. Implement CPU fallbacks for Triton / CUDA functions

def cpu_XTX(A, out):
    if A.ndim == 3:
        torch.bmm(A.transpose(-1, -2), A, out=out)
    else:
        torch.mm(A.t(), A, out=out)
    return out

def cpu_XXT(A, out):
    if A.ndim == 3:
        torch.bmm(A, A.transpose(-1, -2), out=out)
    else:
        torch.mm(A, A.t(), out=out)
    return out

def cpu_ba_plus_cAA(A, alpha, beta, out):
    if A.ndim == 3:
        torch.baddbmm(beta * A, A, A, alpha=alpha, beta=1.0, out=out)
    else:
        torch.addmm(beta * A, A, A, alpha=alpha, beta=1.0, out=out)
    return out

def cpu_transpose_copy(src, dst):
    dst.copy_(src.t())

def cpu_transpose_add(src, dst):
    dst.add_(src.t())

class CPUFusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        x_flat = x.view(-1, x.shape[-1])
        pre = x_flat @ W1.t()
        post = torch.relu(pre) ** 2
        x3 = post @ W2
        ctx.save_for_backward(x_flat, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, W1, W2, pre, post = ctx.saved_tensors
        grad_flat = grad_output.view(-1, grad_output.shape[-1])
        dW2 = post.t() @ grad_flat
        dpost = grad_flat @ W2.t()
        dpre = 2 * dpost * torch.relu(pre)
        dW1 = dpre.t() @ x_flat
        dx = dpre @ W1
        return dx.view(grad_output.shape), dW1, dW2

class CPUFusedSoftcappedCrossEntropy:
    @staticmethod
    def apply(x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, A=23.0, B=5.0, C=7.5):
        x_f = x.float()
        w_f = lm_head_weight.float()
        logits = x_f @ w_f
        
        softcapped_logits = A * torch.sigmoid((logits + B) / C)
        lse = torch.logsumexp(softcapped_logits, dim=-1)
        
        N, vocab_size = logits.shape
        if mtp_weights is None:
            mtp_weights = torch.tensor([1.0], dtype=torch.float32, device=x.device)
        n_predict = mtp_weights.shape[0]
        
        losses = torch.zeros(N, dtype=x.dtype, device=x.device)
        
        for k in range(n_predict):
            target_shifted = torch.full((N,), -1, dtype=torch.long, device=x.device)
            target_shifted[:N-k] = targets[k:]
            
            mask = (target_shifted >= 0) & (target_shifted < vocab_size)
            
            z_target = torch.zeros(N, dtype=torch.float32, device=x.device)
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) > 0:
                valid_targets = target_shifted[valid_indices]
                z_target[valid_indices] = softcapped_logits[valid_indices, valid_targets]
            
            losses += mtp_weights[k] * (lse - z_target).to(x.dtype)
            
        return losses

def cpu_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True, softmax_scale=None, window_size=None):
    T, num_heads, head_dim = q.shape
    out = torch.zeros_like(q)
    seqlens = cu_seqlens_q.tolist()
    
    for idx in range(len(seqlens) - 1):
        start, end = seqlens[idx], seqlens[idx+1]
        if start >= end:
            continue
        
        q_s = q[start:end].transpose(0, 1)
        k_s = k[start:end].transpose(0, 1)
        v_s = v[start:end].transpose(0, 1)
        
        seq_len = end - start
        
        scale = softmax_scale if softmax_scale is not None else (1.0 / (head_dim ** 0.5))
        scores = (q_s @ k_s.transpose(-1, -2)) * scale
        
        if causal:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
            scores = scores + causal_mask[None, :, :]
            
        if window_size is not None and window_size[0] is not None:
            w_size = window_size[0]
            row_idx = torch.arange(seq_len, device=q.device)[:, None]
            col_idx = torch.arange(seq_len, device=q.device)[None, :]
            window_mask = (row_idx - col_idx >= w_size)
            scores = scores.masked_fill(window_mask[None, :, :], float('-inf'))
            
        attn_probs = torch.softmax(scores, dim=-1).to(q.dtype)
        context = attn_probs @ v_s
        
        out[start:end] = context.transpose(0, 1)
        
    return out

# Mock/Monkeypatch train_gpt functions
train_gpt.XTX = cpu_XTX
train_gpt.XXT = cpu_XXT
train_gpt.ba_plus_cAA = cpu_ba_plus_cAA
train_gpt.transpose_copy = cpu_transpose_copy
train_gpt.transpose_add = cpu_transpose_add
train_gpt.ReLUSqrdMLP = CPUFusedLinearReLUSquareFunction.apply
train_gpt.FusedSoftcappedCrossEntropy = CPUFusedSoftcappedCrossEntropy

def cpu_get_bigram_hash(x):
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = train_gpt.args.bigram_vocab_size-1
    x = x.to(torch.int32)
    out = torch.empty_like(x)  # Disable pin_memory to avoid PyTorch MPS bug on CPU
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(rand_int_1 * out[1:], rand_int_2 * out[:-1]) % mod
    return out

train_gpt.get_bigram_hash = cpu_get_bigram_hash

class MockFlashAttnInterface:
    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        return cpu_flash_attn_varlen_func(*args, **kwargs)

train_gpt.flash_attn_interface = MockFlashAttnInterface

# Disable compile for simplicity
train_gpt.torch.compile = lambda model, *args, **kwargs: model

def run_cpu_toy_test(optimizer_mode, beta_policy):
    print(f"\n=========================================")
    print(f"RUNNING TOY TEST on CPU:")
    print(f"  OPTIMIZER_MODE: {optimizer_mode}")
    print(f"  BETA_POLICY   : {beta_policy}")
    print(f"=========================================")
    
    # 3. Create a tiny model
    vocab_size = 1000
    num_layers = 11
    num_heads = 6
    head_dim = 16
    model_dim = 96
    max_seq_len = 64
    
    # Override global config values
    train_gpt.OPTIMIZER_MODE = optimizer_mode
    train_gpt.BETA_POLICY = beta_policy
    
    # Create the model on CPU
    model = train_gpt.GPT(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        model_dim=model_dim,
        max_seq_len=max_seq_len
    )
    
    # Convert weights to bfloat16 like train_gpt.py
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.qk_bank.data = model.qk_bank.data.bfloat16()
    model.vo_bank.data = model.vo_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()
    
    # Initialize TrainingManager
    training_manager = train_gpt.TrainingManager(model)
    
    # 4. Generate synthetic toy input batch
    seq_len = 64
    inputs = torch.randint(0, vocab_size - 1, (seq_len,), dtype=torch.int32)
    targets = torch.randint(0, vocab_size - 1, (seq_len,), dtype=torch.long)
    
    # Single document of length seq_len
    num_tokens_local = seq_len
    max_num_docs = max(128, train_gpt.TRAIN_MAX_NUM_DOCS.get(num_tokens_local, train_gpt.next_multiple_of_n(num_tokens_local // 300, n=128)))
    cum_seqlens = torch.full((max_num_docs,), seq_len, dtype=torch.int32)
    cum_seqlens[0] = 0
    cum_seqlens[1] = seq_len
    
    bigram_inputs = train_gpt.get_bigram_hash(inputs)
    
    # Set step training parameters
    step = 0
    training_manager.advance_schedule(step)
    
    # Run sparse index update
    bigram_cpu = bigram_inputs.numpy()
    training_manager.sparse_index_update(step, bigram_cpu)
    
    # Forward pass
    forward_args = training_manager.get_forward_args()
    loss = model(inputs, targets, cum_seqlens, bigram_inputs, forward_args).sum()
    
    print(f"-> Forward succeeded! Loss: {loss.item():.4f}")
    
    # Backward pass
    training_manager.sparse_index_share(step)
    loss.backward()
    print("-> Backward succeeded!")
    
    # Optimizer step
    # Save initial value of qk_bank to check updates
    param_to_check = model.qk_bank
    initial_val = param_to_check.clone().detach()
    
    training_manager.step_optimizers(step)
    print("-> Optimizer step succeeded!")
    
    # Verify parameter updated
    is_updated = not torch.allclose(initial_val, param_to_check)
    print(f"-> Weight update check: {'PASSED' if is_updated else 'FAILED'}")
    assert is_updated, "Weights did not update!"

if __name__ == '__main__':
    run_cpu_toy_test("muon", "global")
    run_cpu_toy_test("muon", "shape")
    run_cpu_toy_test("adamw", "global")
    print("\nAll CPU toy tests completed successfully!")
