"""Training presets and CLI/env configuration for mini_pretrain."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 12
    d_model: int = 768
    d_ff: int | None = None  # default 4 * d_model
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


@dataclass
class DataConfig:
    data_dir: str = "data/fineweb10B"
    train_glob: str = "fineweb_train_*.bin"
    val_file: str = "fineweb_val_000000.bin"
    num_train_shards: int = 20
    use_synthetic: bool = False
    synthetic_tokens: int = 10_000_000


@dataclass
class TrainConfig:
    preset: str = "mini"
    run_mode: str = "muon_global"  # adamw | muon_global | muon_bank
    beta_policy: str = "global"
    base_beta: float = 0.95
    seed: int = 0
    steps: int = 3000
    val_every: int = 500
    batch_tokens: int = 65536
    lr_adam: float = 3e-4
    lr_muon: float = 0.02
    weight_decay: float = 0.1
    muon_ns_steps: int = 5
    grad_clip: float = 1.0
    use_amp: bool = True
    device: str = "cuda"
    run_id: str = ""
    results_dir: str = "results/mini_pretrain"
    log_every: int = 10

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)


PRESETS: dict[str, dict] = {
    "smoke": {
        "steps": 300,
        "val_every": 100,
        "log_every": 10,
        "batch_tokens": 32768,
        "model": {"n_layer": 6, "n_head": 8, "d_model": 512, "max_seq_len": 512},
        "data": {"num_train_shards": 2, "use_synthetic": True},
    },
    "mini": {
        "steps": 3000,
        "val_every": 500,
        "log_every": 50,
        "batch_tokens": 65536,
        "model": {"n_layer": 8, "n_head": 12, "d_model": 768, "max_seq_len": 1024},
        "data": {"num_train_shards": 20, "use_synthetic": False},
    },
}


def _apply_dict(cfg: TrainConfig, updates: dict) -> None:
    for key, value in updates.items():
        if key == "model" and isinstance(value, dict):
            for mk, mv in value.items():
                setattr(cfg.model, mk, mv)
        elif key == "data" and isinstance(value, dict):
            for dk, dv in value.items():
                setattr(cfg.data, dk, dv)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)


def load_config(preset: str | None = None) -> TrainConfig:
    cfg = TrainConfig()
    name = preset or os.environ.get("MINI_PRETRAIN_PRESET", cfg.preset)
    if name in PRESETS:
        _apply_dict(cfg, PRESETS[name])
        cfg.preset = name

    if "RUN_MODE" in os.environ:
        cfg.run_mode = os.environ["RUN_MODE"]
    if "BETA_POLICY" in os.environ:
        cfg.beta_policy = os.environ["BETA_POLICY"]
    if "SEED" in os.environ:
        cfg.seed = int(os.environ["SEED"])
    if "STEPS" in os.environ:
        cfg.steps = int(os.environ["STEPS"])
    if "VAL_EVERY" in os.environ:
        cfg.val_every = int(os.environ["VAL_EVERY"])
    if "BATCH_TOKENS" in os.environ:
        cfg.batch_tokens = int(os.environ["BATCH_TOKENS"])
    if "RUN_ID" in os.environ:
        cfg.run_id = os.environ["RUN_ID"]
    if "DATA_DIR" in os.environ:
        cfg.data.data_dir = os.environ["DATA_DIR"]
    if "USE_SYNTHETIC" in os.environ:
        cfg.data.use_synthetic = os.environ["USE_SYNTHETIC"].lower() in ("1", "true", "yes")
    if "DEVICE" in os.environ:
        cfg.device = os.environ["DEVICE"]

    if cfg.run_mode == "muon_global":
        cfg.beta_policy = "global"
    elif cfg.run_mode == "muon_bank":
        cfg.beta_policy = "bank"
    elif cfg.run_mode == "adamw":
        pass
    else:
        raise ValueError(f"Unknown run_mode: {cfg.run_mode}")

    if not cfg.run_id:
        cfg.run_id = f"{cfg.run_mode}-{cfg.preset}-seed{cfg.seed}"

    return cfg
