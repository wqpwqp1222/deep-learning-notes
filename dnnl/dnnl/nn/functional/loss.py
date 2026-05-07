from typing import Literal

import torch
import torch.nn.functional as torch_F
from torch import Tensor

__all__ = ['vae_loss']


def vae_loss(
    x_hat: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    loss_fn: Literal['mse', 'bce'] = 'bce',
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if loss_fn == 'mse':
        recon_loss = torch_F.mse_loss(x_hat, x, reduction='sum')
    elif loss_fn == 'bce':
        recon_loss = torch_F.binary_cross_entropy(x_hat, x, reduction='sum')
    else:
        raise NotImplementedError(f'Unsupported loss function: {loss_fn},.')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    batch_size = x.size(0)
    return loss / batch_size, recon_loss / batch_size, kl_loss / batch_size
