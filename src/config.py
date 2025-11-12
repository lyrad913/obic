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
    bce_label_smoothing: float = 0.0
    scale_features: bool = True
    max_checkpoints: int = 3
    tree_max_iter: int = 800
    tree_learning_rate: float = 0.05
    tree_max_depth: int | None = None
    tree_l2_regularization: float = 0.0
    tree_subsample: float = 1.0
    tree_min_samples_leaf: int = 20
    tree_max_bins: int = 255
    lgbm_n_estimators: int = 800
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 128
    lgbm_feature_fraction: float = 0.8
    lgbm_bagging_fraction: float = 0.7
    lgbm_bagging_freq: int = 1
    lgbm_lambda_l2: float = 1.0
    lgbm_min_child_samples: int = 80
    lgbm_max_depth: int = -1
    lgbm_sample_fraction: float = 0.4
    lgbm_early_stopping_rounds: int = 50
    lgbm_log_evaluation_period: int = 50

    def resolve_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_root = self.artifacts_dir / "checkpoints"
        checkpoints_root.mkdir(parents=True, exist_ok=True)
        (checkpoints_root / self.model_name).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints" / self.model_name

    @property
    def is_two_head(self) -> bool:
        return self.model_name.lower() == "two_head"

    @property
    def is_hist_gbdt(self) -> bool:
        return self.model_name.lower() == "hist_gbdt"

    @property
    def is_lightgbm(self) -> bool:
        return self.model_name.lower() == "lightgbm"

    @property
    def is_tree_model(self) -> bool:
        return self.is_hist_gbdt or self.is_lightgbm

    @staticmethod
    def _float_to_tag(value: float) -> str:
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1:
            fmt = f"{value:.2f}"
        elif magnitude >= 1e-2:
            fmt = f"{value:.3f}"
        else:
            fmt = f"{value:.1e}"
        return fmt.replace(".", "p").replace("+0", "+").replace("-0", "-")

    @property
    def checkpoint_prefix(self) -> str:
        name = self.model_name.lower()
        if name == "hist_gbdt":
            depth = "none" if self.tree_max_depth is None else str(self.tree_max_depth)
            return "-".join(
                [
                    name,
                    f"iter{self.tree_max_iter}",
                    f"lr{self._float_to_tag(self.tree_learning_rate)}",
                    f"depth{depth}",
                    f"sub{self._float_to_tag(self.tree_subsample)}",
                ]
            )
        if name == "lightgbm":
            depth = "none" if self.lgbm_max_depth in (-1, None) else str(self.lgbm_max_depth)
            return "-".join(
                [
                    name,
                    f"iter{self.lgbm_n_estimators}",
                    f"lr{self._float_to_tag(self.lgbm_learning_rate)}",
                    f"leaves{self.lgbm_num_leaves}",
                    f"depth{depth}",
                    f"bag{self._float_to_tag(self.lgbm_bagging_fraction)}",
                    f"feat{self._float_to_tag(self.lgbm_feature_fraction)}",
                ]
            )
        hidden = "x".join(str(h) for h in self.hidden_dims)
        return "-".join(
            [
                name,
                f"bs{self.batch_size}",
                f"lr{self._float_to_tag(self.lr)}",
                f"hid{hidden}",
                f"dp{self._float_to_tag(self.dropout)}",
            ]
        )
