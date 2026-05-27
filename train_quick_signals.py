"""
Quick empirical signals: Compare Muon vs baselines + oracle alignment.

Run with:
  python train_quick_signals.py --steps 1000 --compute_oracle
  python train_quick_signals.py --steps 1000 --optimizer adam
"""

import torch
from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader, IterableDataset
from model import Transformer
from muon import config, training_config
from optimizers import Muon, MuonWithoutOrthogonalization
from quick_oracle import sample_oracle_step
from metrics_tracker import MetricsTracker
import time
from tqdm import tqdm
import argparse
import json


class WikitextDataset(IterableDataset):
    def __init__(self, split='train', seq_len=128):
        self.dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split, streaming=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.seq_len = seq_len

    def __iter__(self):
        tokens = []
        for example in self.dataset:
            text = example['text']
            if not text.strip(): continue
            tokens.extend(self.tokenizer.encode_ordinary(text))
            tokens.append(self.tokenizer.eot_token)
            
            while len(tokens) >= self.seq_len + 1:
                chunk = tokens[:self.seq_len + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                tokens = tokens[self.seq_len:]


def get_optimizer(optimizer_name, model_params):
    """Factory for optimizers."""
    lr = training_config['lr']
    if optimizer_name == 'muon':
        return Muon(model_params, lr=lr, weight_decay=training_config['weight_decay'])
    elif optimizer_name == 'muon_no_ortho':
        return MuonWithoutOrthogonalization(model_params, lr=lr, weight_decay=training_config['weight_decay'])
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=training_config['weight_decay'])
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=training_config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train(optimizer_name='muon', steps=1000, compute_oracle=False, seed=42):
    """Train transformer with specified optimizer."""
    
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Optimizer: {optimizer_name} | Steps: {steps} | Compute Oracle: {compute_oracle}")

    # Initialize
    model = Transformer(config).to(device)
    optimizer = get_optimizer(optimizer_name, model.parameters())
    tracker = MetricsTracker(name=f"{optimizer_name}_s{seed}")

    # Data
    train_dataset = WikitextDataset(split='train', seq_len=config['max_seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'])
    data_iter = iter(train_loader)

    # Training loop
    model.train()
    start_time = time.time()
    
    def get_lr(step):
        if step < training_config['warmup']:
            return training_config['lr'] * (step + 1) / training_config['warmup']
        return training_config['lr']

    pbar = tqdm(range(steps), desc=f"Training {optimizer_name}")
    for step in pbar:
        # Update LR
        curr_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        log_data = {
            "loss": loss.item(),
            "lr": curr_lr,
        }

        # Compute oracle alignment (expensive, optional)
        if compute_oracle and step % 200 == 0:
            oracle_result = sample_oracle_step(model, lambda x_b, y_b: model(x_b, y_b)[1], x, y, device)
            log_data["cosine_sim"] = oracle_result["cosine_sim"]
            log_data["n_2d_params"] = oracle_result["n_2d_params"]

        tracker.log(step, **log_data)

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{curr_lr:.2e}"})

    elapsed = time.time() - start_time
    print(f"\nCompleted {steps} steps in {elapsed:.2f}s ({elapsed/steps:.3f}s/step)")
    
    # Save metrics
    output_file = f"results/quick_signals_{optimizer_name}_s{seed}.json"
    tracker.save(output_file)
    
    # Print summary
    summary = tracker.summary()
    print("\nSummary:")
    for key, stats in summary.items():
        print(f"  {key}: {stats['latest']:.6f} (min: {stats['min']:.6f}, max: {stats['max']:.6f})")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='muon', choices=['muon', 'muon_no_ortho', 'sgd', 'adam'])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--compute_oracle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(
        optimizer_name=args.optimizer,
        steps=args.steps,
        compute_oracle=args.compute_oracle,
        seed=args.seed
    )
