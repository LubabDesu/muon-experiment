"""FineWeb .bin shard loader. Real shards only unless USE_SYNTHETIC=1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# muon/ repo root (parent of mini_pretrain/)
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(data_dir: str) -> Path:
    """Resolve data_dir relative to repo root, not shell cwd."""
    p = Path(data_dir).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def find_train_shards(data_dir: Path, train_glob: str, num_shards: int) -> list[Path]:
    direct = sorted(data_dir.glob(train_glob))
    if direct:
        return direct[:num_shards]
    # HuggingFace sometimes nests files one level deeper
    nested = sorted(data_dir.rglob("fineweb_train_*.bin"))
    if nested:
        return nested[:num_shards]
    raise FileNotFoundError(
        f"No FineWeb train shards under {data_dir}\n"
        f"  tried: {data_dir / train_glob}\n"
        f"  and:   {data_dir}/**/fineweb_train_*.bin\n"
        f"Download from repo root:\n"
        f"  python -m mini_pretrain.data --download fineweb --chunks {num_shards}\n"
        f"Or set DATA_DIR to the folder that contains fineweb_train_*.bin"
    )


def find_val_shard(data_dir: Path, val_file: str) -> Path:
    direct = data_dir / val_file
    if direct.exists():
        return direct
    matches = sorted(data_dir.rglob(val_file))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing validation shard {val_file} under {data_dir}\n"
        f"Run: python -m mini_pretrain.data --download fineweb --chunks 1"
    )


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
        dest = local / fname
        if not dest.exists():
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


def describe_data_setup(
    data_dir: str,
    train_glob: str,
    num_shards: int,
    val_file: str,
    use_synthetic: bool,
) -> tuple[Path, list[Path], Path]:
    if use_synthetic:
        return Path("."), [], Path(".")
    root = resolve_data_dir(data_dir)
    train_files = find_train_shards(root, train_glob, num_shards)
    val_path = find_val_shard(root, val_file)
    return root, train_files, val_path


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
        print("WARNING: USE_SYNTHETIC=1 — random tokens, not FineWeb.")
        stream = build_synthetic_stream(synthetic_tokens, vocab_size, seed)
        return TokenBatchIterator([stream], batch_size, seq_len, seed=seed)

    root, files, _ = describe_data_setup(data_dir, train_glob, num_shards, "", False)
    print(f"FineWeb train: {len(files)} shard(s) from {root}")
    for f in files:
        print(f"  {f.name}")
    streams = [load_bin_shard(f) for f in files]
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
    num_shards: int = 1,
    train_glob: str = "fineweb_train_*.bin",
) -> TokenBatchIterator:
    if use_synthetic:
        stream = build_synthetic_stream(synthetic_tokens // 10, vocab_size, seed + 1)
        return TokenBatchIterator([stream], batch_size, seq_len, seed=seed + 1)

    root, _, val_path = describe_data_setup(data_dir, train_glob, num_shards, val_file, False)
    print(f"FineWeb val: {val_path}")
    return TokenBatchIterator([load_bin_shard(val_path)], batch_size, seq_len, seed=seed + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FineWeb GPT-2 token shards")
    parser.add_argument("--download", choices=["fineweb"], required=True)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default="data/fineweb10B")
    args = parser.parse_args()
    if args.download == "fineweb":
        out = resolve_data_dir(args.data_dir)
        download_fineweb(args.chunks, out)
        print(f"Downloaded {args.chunks} train shards + val to {out}")


if __name__ == "__main__":
    main()
