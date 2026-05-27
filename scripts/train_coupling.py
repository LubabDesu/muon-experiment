"""
Train MLP on MNIST with Muon optimizer and log cross-layer coupling metrics.

Every --log-interval steps computes:
  - coupling_ratio_{i}{i+1}: ||H_ij||_F^2 / ||H_ii||_F^2 for adjacent layer pairs
  - cosine_sim_{i}: cos(Muon_dir_i, Newton_dir_i) per layer

Usage:
  python scripts/train_coupling.py --steps 1000 --depth 3
  python scripts/train_coupling.py --steps 200 --samples 5 --depth 4 --skip
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mlp_mnist import MLP
from src.hessian_utils import coupling_ratio
from src.oracle import oracle_update, layer_cosine_sim
from optimizers import Muon
from metrics_tracker import MetricsTracker


def load_mnist(batch_size: int, measurement_size: int, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_ds = torchvision.datasets.MNIST(
        root=".mnist_data", train=True, download=True, transform=transform
    )
    val_ds = torchvision.datasets.MNIST(
        root=".mnist_data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    # fixed held-out measurement batch — loaded once, never changes
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=measurement_size, shuffle=False
    )
    mx, my = next(iter(val_loader))
    return train_loader, mx.to(device), my.to(device)


def measure(model, mx, my, params, args):
    """Compute coupling ratios and cosine sims on fixed measurement batch."""
    loss_fn = lambda out, y: F.cross_entropy(out, y)
    records = {}

    # coupling ratio for each adjacent layer pair
    for i in range(len(params) - 1):
        try:
            ratios = coupling_ratio(
                model, loss_fn, mx, my,
                params[i], params[i+1],
                n_samples=args.samples,
            )
            records[f"coupling_raw_{i}{i+1}"] = ratios["raw"]
            records[f"coupling_norm_{i}{i+1}"] = ratios["normalized"]
        except Exception:
            records[f"coupling_raw_{i}{i+1}"] = None
            records[f"coupling_norm_{i}{i+1}"] = None

    # cosine sim per layer (Muon dir vs full-Newton dir)
    try:
        muon_dirs, newton_dirs = oracle_update(
            model, loss_fn, mx, my, params,
            cg_iters=args.cg_iters,
            damping=args.damping,
        )
        sims = layer_cosine_sim(muon_dirs, newton_dirs)
        for i, s in enumerate(sims):
            records[f"cosine_sim_{i}"] = s
    except Exception as e:
        for i in range(len(params)):
            records[f"cosine_sim_{i}"] = None

    return records


def train(args):
    torch.manual_seed(args.seed)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device={device}  depth={args.depth}  skip={args.skip}  steps={args.steps}")

    model = MLP(depth=args.depth, hidden=args.hidden, skip=args.skip).to(device)
    params = model.get_weight_params()
    optimizer = Muon(model.parameters(), lr=args.lr)
    tracker = MetricsTracker(name=f"d{args.depth}_skip{int(args.skip)}")

    train_loader, mx, my = load_mnist(args.batch_size, args.measurement_batch, device)
    data_iter = iter(train_loader)

    Path("results").mkdir(exist_ok=True)
    outfile = f"results/coupling_d{args.depth}_skip{int(args.skip)}_s{args.seed}.json"

    t0 = time.time()
    for step in range(args.steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        log = {"loss": loss.item()}

        if step % args.log_interval == 0:
            model.eval()
            with torch.enable_grad():
                metrics = measure(model, mx, my, params, args)
            log.update(metrics)
            model.train()

            parts = [f"step={step}", f"loss={loss.item():.4f}"]
            for k, v in metrics.items():
                if v is not None:
                    parts.append(f"{k}={v:.3f}")
            print("  ".join(parts))

        tracker.log(step, **log)

    elapsed = time.time() - t0
    print(f"\ndone: {args.steps} steps in {elapsed:.1f}s")

    with open(outfile, "w") as f:
        json.dump({"config": vars(args), "records": tracker.metrics}, f, indent=2)
    print(f"saved -> {outfile}")
    return outfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--skip", action="store_true")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--samples", type=int, default=10,
                   help="Hutchinson samples per estimate")
    p.add_argument("--cg-iters", type=int, default=20)
    p.add_argument("--damping", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--measurement-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
