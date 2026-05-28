"""FineWeb .bin shard loader with synthetic fallback for smoke tests."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch


def load_bin_shard(path: Path) -> torch.Tensor:
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert int(header[0]) == 20240520, f"magic mismatch in {path}"
    assert int(header[1]) == 1, f"unsupported version in {path}"
    num_tokens = int(header[2])
    tokens = torch.empty(num_tokens, dtype=torch.uint16)
    with path.open("rb", buffering=0) as f:
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
    assert nbytes == 2 * num_tokens
    return tokens.long()


def download_fineweb(num_chunks: int, data_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    local = data_dir

    def get(fname: str) -> None:
        if not (local / fname).exists():
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=str(local),
            )

    get("fineweb_val_000000.bin")
    for i in range(1, num_chunks + 1):
        get(f"fineweb_train_{i:06d}.bin")


class TokenBatchIterator:
    """Yield (input, target) batches of shape [B, T] from concatenated token shards."""

    def __init__(
        self,
        token_streams: list[torch.Tensor],
        batch_size: int,
        seq_len: int,
        seed: int = 0,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.streams = token_streams
        self.stream_idx = 0
        self.pos = 0
        self._active = self.streams[0] if self.streams else torch.zeros(0, dtype=torch.long)

    def _rotate_stream(self) -> None:
        if not self.streams:
            return
        self.stream_idx = (self.stream_idx + 1) % len(self.streams)
        self._active = self.streams[self.stream_idx]
        self.pos = 0

    def _sample_batch_offsets(self, need: int) -> torch.Tensor:
        tokens = self._active
        if tokens.numel() < need + 1:
            self._rotate_stream()
            tokens = self._active
        if tokens.numel() < need + 1:
            raise RuntimeError("token stream too short for batch")
        max_start = tokens.numel() - need - 1
        starts = self.rng.integers(0, max_start + 1, size=self.batch_size)
        return torch.tensor(starts, dtype=torch.long)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.seq_len
        starts = self._sample_batch_offsets(need)
        x = torch.stack([self._active[s : s + need] for s in starts], dim=0)
        y = torch.stack([self._active[s + 1 : s + need + 1] for s in starts], dim=0)
        return x, y


def build_synthetic_stream(num_tokens: int, vocab_size: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.integers(0, vocab_size, size=num_tokens), dtype=torch.long)


def create_train_iterator(
    data_dir: str,
    train_glob: str,
    num_shards: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    use_synthetic: bool,
    synthetic_tokens: int,
    vocab_size: int,
) -> TokenBatchIterator:
    if use_synthetic:
        stream = build_synthetic_stream(synthetic_tokens, vocab_size, seed)
        return TokenBatchIterator([stream], batch_size, seq_len, seed=seed)

    pattern = os.path.join(data_dir, train_glob)
    files = sorted(glob.glob(pattern))[:num_shards]
    if not files:
        print(f"No shards at {pattern}; using synthetic data.")
        stream = build_synthetic_stream(synthetic_tokens, vocab_size, seed)
        return TokenBatchIterator([stream], batch_size, seq_len, seed=seed)

    streams = [load_bin_shard(Path(f)) for f in files]
    return TokenBatchIterator(streams, batch_size, seq_len, seed=seed)


def create_val_iterator(
    data_dir: str,
    val_file: str,
    batch_size: int,
    seq_len: int,
    seed: int,
    use_synthetic: bool,
    synthetic_tokens: int,
    vocab_size: int,
) -> TokenBatchIterator:
    if use_synthetic:
        stream = build_synthetic_stream(synthetic_tokens // 10, vocab_size, seed + 1)
        return TokenBatchIterator([stream], batch_size, seq_len, seed=seed + 1)

    path = Path(data_dir) / val_file
    if not path.exists():
        stream = build_synthetic_stream(synthetic_tokens // 10, vocab_size, seed + 1)
        return TokenBatchIterator([stream], batch_size, seq_len, seed=seed + 1)
    return TokenBatchIterator([load_bin_shard(path)], batch_size, seq_len, seed=seed + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb GPT-2 token shards")
    parser.add_argument("--download", choices=["fineweb"], required=True)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="data/fineweb10B")
    args = parser.parse_args()
    if args.download == "fineweb":
        download_fineweb(args.chunks, Path(args.data_dir))
        print(f"Downloaded {args.chunks} train shards + val to {args.data_dir}")


if __name__ == "__main__":
    main()
