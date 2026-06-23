import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

__all__ = [
    'Sigmoid',
    'Tanh',
    'ReLU',
    'GELU',
    'Softmax',
    'LogSoftmax',
]


class Sigmoid(nn.Module):
    """Apply the sigmoid function element-wise."""

    def __init__(self, *, fast: bool = False):
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.sigmoid(x)
        return dF.sigmoid(x)


class Tanh(nn.Module):
    """Apply the hyperbolic tangent function element-wise."""

    def __init__(self, *, fast: bool = False):
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.tanh(x)
        return dF.tanh(x)


class ReLU(nn.Module):
    """Apply the rectified linear unit function element-wise."""

    def __init__(self, *, fast: bool = False):
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.relu(x)
        return dF.relu(x)


class GELU(nn.Module):
    """Apply the Gaussian Error Linear Unit function element-wise."""

    def __init__(self, *, fast: bool = False):
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.gelu(x)
        return dF.gelu(x)


class Softmax(nn.Module):
    """Apply the softmax function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softmax(x, dim=self.dim)
        return dF.softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(nn.Module):
    """Apply the log-softmax function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.log_softmax(x, dim=self.dim)
        return dF.log_softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'
