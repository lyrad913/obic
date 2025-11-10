from __future__ import annotations

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
import torch


class NormalizedMAE(Metric):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__(output_transform=lambda output: output)
        self.eps = eps

    def reset(self) -> None:
        self._abs_error = 0.0
        self._abs_target = 0.0
        self._num_examples = 0

    def update(self, output) -> None:  # type: ignore[override]
        y_pred, y = output
        if y_pred.ndim > 1:
            y_pred = y_pred.squeeze()
        if y.ndim > 1:
            y = y.squeeze()
        errors = torch.abs(y_pred - y)
        self._abs_error += errors.sum().item()
        self._abs_target += torch.abs(y).sum().item()
        self._num_examples += y.shape[0]

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("NormalizedMAE must have at least one example.")
        denominator = (self._abs_target / self._num_examples) + self.eps
        return (self._abs_error / self._num_examples) / denominator * 100.0
