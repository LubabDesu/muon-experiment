import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['d_model'] % config['n_heads'] == 0
        self.c_attn = nn.Linear(config['d_model'], 3 * config['d_model'])
        self.c_proj = nn.Linear(config['d_model'], config['d_model'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_heads = config['n_heads']
        self.d_model = config['d_model']
        self.register_buffer("bias", torch.tril(torch.ones(config['max_seq_len'], config['max_seq_len']))
                                        .view(1, 1, config['max_seq_len'], config['max_seq_len']))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['d_model'], config['d_ff'])
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config['d_ff'], config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['d_model'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['d_model'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['d_model']),
            wpe = nn.Embedding(config['max_seq_len'], config['d_model']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layers'])]),
            ln_f = nn.LayerNorm(config['d_model']),
        ))
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        self.apply(self._init_weights)
        self.count_parameters()

    def count_parameters(self):
        emb_params = sum(p.numel() for p in self.transformer.wte.parameters()) + \
                     sum(p.numel() for p in self.transformer.wpe.parameters())
        block_params = sum(p.numel() for p in self.transformer.h.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Total Parameters: {total_params/1e6:.2f}M")
        print(f"  - Embeddings: {emb_params/1e6:.2f}M")
        print(f"  - Blocks: {block_params/1e6:.2f}M (Non-embedding)")
        return total_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['max_seq_len'], f"Cannot forward sequence of length {t}, max_seq_len is {self.config['max_seq_len']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
