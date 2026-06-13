from typing import Literal

import torch.nn as nn
from torch import Tensor

from . import functional as dF

type Reduction = Literal['mean', 'sum', 'none']

__all__ = ['CrossEntropyLoss']


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: Tensor | None = None, reduction: Reduction = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return dF.cross_entropy(
            x,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )
