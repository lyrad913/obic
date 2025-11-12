from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class FeedForwardRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | Iterable[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class TwoHeadRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | Iterable[int],
        dropout: float,
        pv_vocab_size: int,
        pv_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.pv_embedding = nn.Embedding(pv_vocab_size, pv_embedding_dim)

        combined_dim = input_dim + pv_embedding_dim
        self.feature_norm = nn.LayerNorm(combined_dim)
        self.feature_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        layers: list[nn.Module] = []
        prev_dim = combined_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        head_input_dim = prev_dim if layers else (input_dim + pv_embedding_dim)

        head_hidden_dim = max(32, head_input_dim // 2)
        self.classifier_head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(head_hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor, pv_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.pv_embedding(pv_indices)
        features = torch.cat([inputs, emb], dim=-1)
        features = self.feature_norm(features)
        features = self.feature_dropout(features)
        representation = self.backbone(features)
        logits = self.classifier_head(representation).squeeze(-1)
        regression = self.regression_head(representation).squeeze(-1)
        return logits, regression


class TwoHeadLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        regression_weight: float = 1.0,
        positive_threshold: float = 0.0,
        label_smoothing: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.regression_weight = regression_weight
        self.threshold = positive_threshold
        self.eps = eps
        self.label_smoothing = max(0.0, min(0.499, label_smoothing))
        self._bce = nn.BCEWithLogitsLoss()
        self._reg = nn.SmoothL1Loss(reduction="none")

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits, regression = outputs
        target = target.float()
        positive_mask = (target > self.threshold).float()
        if self.label_smoothing > 0.0:
            smoothed = positive_mask * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed = positive_mask

        classification_loss = self._bce(logits, smoothed)

        reg_losses = self._reg(regression, target)
        weighted_reg_loss = (reg_losses * positive_mask).sum()
        normalizer = positive_mask.sum().clamp_min(self.eps)
        regression_loss = weighted_reg_loss / normalizer

        total_loss = self.bce_weight * classification_loss + self.regression_weight * regression_loss
        return total_loss
