from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

type Reduction = Literal['mean', 'sum', 'none']

__all__ = ['CrossEntropyLoss']


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: Reduction = 'mean',
        *,
        fast: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.cross_entropy(
                x,
                target,
                weight=self.weight,
                reduction=self.reduction,
            )
        return dF.cross_entropy(
            x,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )
