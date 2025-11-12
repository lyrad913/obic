from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class TrainingConfig:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    submission_name: str = "submission.csv"
    val_ratio: float = 0.2
    seed: int = 42
    batch_size: int = 1024
    eval_batch_size: int = 2048
    max_epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-4
    hidden_dims: Sequence[int] = field(default_factory=lambda: (256, 128, 64))
    dropout: float = 0.1
    patience: int = 10
    num_workers: int = 0
    amp: bool = True
    target_col: str = "nins"
    group_col: str = "pv_id"
    time_col: str = "time"
    model_name: str = "feedforward"
    pv_embedding_dim: int = 32
    bce_weight: float = 1.0
    regression_weight: float = 1.0
    positive_threshold: float = 0.0

    def resolve_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def is_two_head(self) -> bool:
        return self.model_name.lower() == "two_head"
