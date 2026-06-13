import torch
from torch import Tensor

__all__ = ['linear']


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a linear transformation to the incoming data.

    Args:
        x (Tensor): Input tensor with shape ``(*, in_features)``.
        weight (Tensor): Weight tensor with shape ``(out_features, in_features)``.
        bias (Tensor | None, default: None): Optional bias tensor with shape ``(out_features,)``.

    Returns:
        Transformed tensor with shape ``(*, out_features)``.
    """
    if x.ndim == 2 and bias is not None:
        return torch.addmm(bias, x, weight.T)
    elif bias is not None:
        return x @ weight.T + bias
    else:
        return x @ weight.T
