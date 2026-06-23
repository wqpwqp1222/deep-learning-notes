import torch
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device

import dnnlpy.nn.functional as dF

__all__ = [
    'get_batch',
    'greedy_sampling',
    'top_k_sampling',
    'top_p_sampling',
    'sample_next_token',
]


def get_batch(
    token_ids: Tensor,
    block_size: int,
    batch_size: int,
    device: Device,
) -> tuple[Tensor, Tensor]:
    """Cut out a batch of inputs and targets from the token stream."""
    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(max_start + 1, (batch_size,))

    x = torch.stack([token_ids[i : i + block_size] for i in starts])
    y = torch.stack([token_ids[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def greedy_sampling(logits: Tensor, temperature: float) -> Tensor:
    """Sample the next token greedily from the logits."""
    if temperature <= 0:
        raise AssertionError('`temperature` must be positive.')
    return dF.softmax(logits / temperature, dim=-1)


def top_k_sampling(logits: Tensor, top_k: int) -> Tensor:
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    values = logits.topk(top_k, dim=-1).values
    threshold = values[..., -1]

    logits = logits.masked_fill(logits < threshold, -torch.inf)
    probs = dF.softmax(logits, dim=-1)
    return probs


def top_p_sampling(logits: Tensor, top_p: float) -> Tensor:
    if not 0 < top_p <= 1:
        raise AssertionError('`top_p` must be in (0, 1].')
    if top_p == 1.0:
        return logits

    logits, indices = logits.sort(descending=True, dim=-1)
    probs = dF.softmax(logits, dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)

    # Remove tokens whose cumulative probability is above top_p.
    mask = cumulative_probs > top_p

    # Keep the first token above the threshold as well, so the kept set reaches top_p.
    remove_mask = F.pad(mask[..., :-1], (1, 0), value=False)
    remove_mask = remove_mask.scatter(-1, indices, remove_mask)

    logits = logits.masked_fill(remove_mask, -torch.inf)
    logits = dF.softmax(logits, dim=-1)
    return logits


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> Tensor:
    """Sample next token ids from logits with temperature, top-k, and top-p."""
    if logits.ndim != 2:
        raise ValueError('`logits` must have shape (B, V).')
    if temperature <= 0:
        raise ValueError('`temperature` must be positive.')

    if greedy:
        next_token = logits.argmax(dim=-1, keepdim=True)
        return next_token

    logits = logits / temperature
    probs = torch.zeros_like(logits)

    if top_k is not None:
        probs = top_k_sampling(logits, top_k=top_k)

    if top_p is not None:
        probs = top_p_sampling(logits, top_p=top_p)

    next_token = probs.multinomial(num_samples=1)
    return next_token
