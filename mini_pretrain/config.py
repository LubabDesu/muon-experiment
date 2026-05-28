"""Training presets and CLI/env configuration for mini_pretrain."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mini_pretrain.beta_assign import BankBetaOffsets


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
    bank_offsets: BankBetaOffsets = field(default_factory=BankBetaOffsets)
    seed: int = 0
    steps: int = 3000
    val_every: int = 500
    batch_tokens: int = 65536
    lr_adam: float = 1e-4
    lr_muon: float = 0.003
    lr_schedule: str = "constant"  # constant | cosine
    lr_warmup_steps: int = 0
    min_lr_scale: float = 0.1
    early_stop_patience_evals: int = 0
    early_stop_min_delta: float = 0.0
    max_val_increase_from_best: float = 1.0
    weight_decay: float = 0.1  # legacy fallback if split envs unset
    weight_decay_adam: float | None = None
    weight_decay_muon: float | None = None
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
        "batch_tokens": 4096,
        "lr_adam": 1e-4,
        "lr_muon": 0.003,
        "weight_decay_adam": 0.01,
        "weight_decay_muon": 0.0,
        "model": {"n_layer": 6, "n_head": 8, "d_model": 512, "max_seq_len": 512},
        "data": {"num_train_shards": 2},
    },
    "mini": {
        "steps": 3000,
        "val_every": 500,
        "log_every": 50,
        "batch_tokens": 65536,
        "lr_muon": 0.002,
        "weight_decay_adam": 0.01,
        "weight_decay_muon": 0.001,
        "muon_ns_steps": 3,
        "lr_schedule": "cosine",
        "lr_warmup_steps": 200,
        "min_lr_scale": 0.05,
        "early_stop_patience_evals": 4,
        "early_stop_min_delta": 0.01,
        "max_val_increase_from_best": 0.8,
        "model": {"n_layer": 8, "n_head": 12, "d_model": 768, "max_seq_len": 1024},
        "data": {"num_train_shards": 20},
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
    if "LR_ADAM" in os.environ:
        cfg.lr_adam = float(os.environ["LR_ADAM"])
    if "LR_MUON" in os.environ:
        cfg.lr_muon = float(os.environ["LR_MUON"])
    if "LR_SCHEDULE" in os.environ:
        cfg.lr_schedule = os.environ["LR_SCHEDULE"].strip().lower()
    if "LR_WARMUP_STEPS" in os.environ:
        cfg.lr_warmup_steps = int(os.environ["LR_WARMUP_STEPS"])
    if "MIN_LR_SCALE" in os.environ:
        cfg.min_lr_scale = float(os.environ["MIN_LR_SCALE"])
    if "EARLY_STOP_PATIENCE_EVALS" in os.environ:
        cfg.early_stop_patience_evals = int(os.environ["EARLY_STOP_PATIENCE_EVALS"])
    if "EARLY_STOP_MIN_DELTA" in os.environ:
        cfg.early_stop_min_delta = float(os.environ["EARLY_STOP_MIN_DELTA"])
    if "MAX_VAL_INCREASE_FROM_BEST" in os.environ:
        cfg.max_val_increase_from_best = float(os.environ["MAX_VAL_INCREASE_FROM_BEST"])
    if "WEIGHT_DECAY" in os.environ:
        cfg.weight_decay = float(os.environ["WEIGHT_DECAY"])
    if "WEIGHT_DECAY_ADAM" in os.environ:
        cfg.weight_decay_adam = float(os.environ["WEIGHT_DECAY_ADAM"])
    if "WEIGHT_DECAY_MUON" in os.environ:
        cfg.weight_decay_muon = float(os.environ["WEIGHT_DECAY_MUON"])

    if cfg.weight_decay_adam is None:
        cfg.weight_decay_adam = cfg.weight_decay
    if cfg.weight_decay_muon is None:
        cfg.weight_decay_muon = 0.0 if cfg.preset == "smoke" else cfg.weight_decay * 0.05
    if "MUON_NS_STEPS" in os.environ:
        cfg.muon_ns_steps = int(os.environ["MUON_NS_STEPS"])
    if "RUN_ID" in os.environ:
        cfg.run_id = os.environ["RUN_ID"]
    if "DATA_DIR" in os.environ:
        cfg.data.data_dir = os.environ["DATA_DIR"]
    # Synthetic only when explicitly requested (never silent fallback).
    cfg.data.use_synthetic = os.environ.get("USE_SYNTHETIC", "0").lower() in ("1", "true", "yes")
    if "DEVICE" in os.environ:
        cfg.device = os.environ["DEVICE"]
    if "BASE_BETA" in os.environ:
        cfg.base_beta = float(os.environ["BASE_BETA"])

    # Bank offsets: per-bank or symmetric shortcut BETA_BANK_DELTA (= magnitude for qk-/mlp+).
    qk, vo, mlp = cfg.bank_offsets.qk, cfg.bank_offsets.vo, cfg.bank_offsets.mlp
    if "BETA_BANK_DELTA" in os.environ:
        delta = float(os.environ["BETA_BANK_DELTA"])
        qk, vo, mlp = -delta, 0.0, delta
    if "BETA_OFFSET_QK" in os.environ:
        qk = float(os.environ["BETA_OFFSET_QK"])
    if "BETA_OFFSET_VO" in os.environ:
        vo = float(os.environ["BETA_OFFSET_VO"])
    if "BETA_OFFSET_MLP" in os.environ:
        mlp = float(os.environ["BETA_OFFSET_MLP"])
    cfg.bank_offsets = BankBetaOffsets(qk=qk, vo=vo, mlp=mlp)

    if cfg.run_mode == "muon_global":
        cfg.beta_policy = "global"
    elif cfg.run_mode == "muon_bank":
        cfg.beta_policy = "bank"
    elif cfg.run_mode == "adamw":
        pass
    else:
        raise ValueError(f"Unknown run_mode: {cfg.run_mode}")
    if cfg.lr_schedule not in {"constant", "cosine"}:
        raise ValueError(f"Unknown LR_SCHEDULE: {cfg.lr_schedule}")
    if cfg.lr_warmup_steps < 0:
        raise ValueError("LR_WARMUP_STEPS must be >= 0")
    if not (0.0 <= cfg.min_lr_scale <= 1.0):
        raise ValueError("MIN_LR_SCALE must be in [0, 1]")
    if cfg.early_stop_patience_evals < 0:
        raise ValueError("EARLY_STOP_PATIENCE_EVALS must be >= 0")
    if cfg.early_stop_min_delta < 0:
        raise ValueError("EARLY_STOP_MIN_DELTA must be >= 0")
    if cfg.max_val_increase_from_best < 0:
        raise ValueError("MAX_VAL_INCREASE_FROM_BEST must be >= 0")

    if not cfg.run_id:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        cfg.run_id = f"{cfg.run_mode}-{cfg.preset}-seed{cfg.seed}"
        if cfg.run_mode == "muon_bank":
            o = cfg.bank_offsets
            cfg.run_id += f"-qk{o.qk:+.3f}-mlp{o.mlp:+.3f}".replace(".", "p").replace("+", "")
        cfg.run_id += f"-{ts}"

    return cfg
