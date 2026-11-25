"""Serializable training defaults; time fields always denote physical seconds."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "tron2-act-spatial-command-v2"


@dataclass
class Config:
    data_root: str = "/home/zmy/workplace/insert-data/transhandr_demo/o6"
    foreground_root: str | None = None
    allow_experimental_foreground: bool = False
    deduplicate_identical_timestamps: bool = False
    task: str = "transhandr_demo"
    episodes: list[int] | None = None
    calibration: str = str(ROOT / "tracker_calibrations/left_mapping_final_20261027/calibration.json")
    quaternion_format: str = "pyvut-legacy"
    time_scale: float = 0.4
    action_hz: float = 20.0
    chunk_size: int = 50
    history_steps: int = 2
    max_source_gap_s: float = 0.1
    min_future_steps: int = 1
    image_size: list[int] = field(default_factory=lambda: [480, 640])
    seed: int = 0
    hidden_dim: int = 512
    heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 7
    feedforward_dim: int = 3200
    latent_dim: int = 32
    dropout: float = 0.1
    pretrained_backbone: bool = True
    batch_size: int = 8
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    kl_weight: float = 10.0
    gradient_clip: float = 1.0
    max_steps: int = 20000
    eval_every: int = 500
    checkpoint_every: int = 1000
    workers: int = 0
    device: str = "cuda"
    normalization_floor: float = 0.01

    @property
    def state_dim(self):
        return self.history_steps * 9 + 6

    @property
    def action_dim(self):
        return 15

    def validate(self):
        if not isinstance(self.allow_experimental_foreground, bool):
            raise ValueError("allow_experimental_foreground must be a boolean")
        if not isinstance(self.deduplicate_identical_timestamps, bool):
            raise ValueError("deduplicate_identical_timestamps must be a boolean")
        if self.allow_experimental_foreground and not self.foreground_root:
            raise ValueError("Experimental foreground requires foreground_root")
        positive = ["time_scale", "action_hz", "max_source_gap_s", "learning_rate",
                    "backbone_learning_rate", "gradient_clip", "normalization_floor"]
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ["chunk_size", "hidden_dim", "heads", "encoder_layers", "decoder_layers",
                     "feedforward_dim", "latent_dim", "batch_size", "max_steps", "eval_every", "checkpoint_every"]:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.time_scale > 1 or self.history_steps < 0 or self.workers < 0:
            raise ValueError("Require time_scale<=1, history_steps>=0, workers>=0")
        if not isinstance(self.history_steps, int) or not isinstance(self.workers, int):
            raise ValueError("history_steps and workers must be integers")
        if self.hidden_dim % self.heads or self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by heads and 4")
        if len(self.image_size) != 2 or min(self.image_size) < 32:
            raise ValueError("image_size is [height,width], both >=32")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("Invalid dropout or KL weight")
        if not 1 <= self.min_future_steps <= self.chunk_size:
            raise ValueError("min_future_steps must be in 1..chunk_size")
        if self.quaternion_format not in {"pyvut-legacy", "wxyz"}:
            raise ValueError("Specify pyvut-legacy or wxyz explicitly")
        if self.episodes is not None and (not self.episodes or len(set(self.episodes)) != len(self.episodes)):
            raise ValueError("episodes must be a nonempty unique list")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path=None, **overrides):
    data = {} if path is None else yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must contain a YAML mapping")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data).validate()
