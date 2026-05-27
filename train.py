import torch
from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader, IterableDataset
from model import Transformer
from muon import Muon, config, training_config
import time
from tqdm import tqdm

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

def train():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model
    model = Transformer(config).to(device)

    # Initialize optimizer
    optimizer = Muon(model.parameters(), 
                     lr=training_config['lr'], 
                     weight_decay=training_config['weight_decay'])

    # Data loading
    train_dataset = WikitextDataset(split='train', seq_len=config['max_seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'])

    # Training loop
    model.train()
    start_time = time.time()
    
    # Linear warmup scheduler
    def get_lr(step):
        if step < training_config['warmup']:
            return training_config['lr'] * (step + 1) / training_config['warmup']
        return training_config['lr']

    # We use iter() on DataLoader to get a continuous stream
    data_iter = iter(train_loader)
    
    pbar = tqdm(range(training_config['steps']), desc="Training")
    for step in pbar:
        # Update learning rate
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        # Logging
        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{curr_lr:.2e}"})
        
        if step % 1000 == 0 and step > 0:
            elapsed = time.time() - start_time
            print(f"\nStep {step} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

    print(f"Training completed in {time.time() - start_time:.2f}s")
    
    # Save model
    torch.save(model.state_dict(), "muon_transformer_10m.pt")
    print("Model saved to muon_transformer_10m.pt")

if __name__ == "__main__":
    train()
