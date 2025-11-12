from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class TrainingConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
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
    use_gpu: bool = False
    target_col: str = "nins"
    group_col: str = "pv_id"
    time_col: str = "time"

    def resolve_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.artifacts_dir = Path(self.artifacts_dir)
        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = (PROJECT_ROOT / self.artifacts_dir).resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"
