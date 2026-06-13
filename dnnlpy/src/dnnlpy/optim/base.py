from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from torch import Tensor

__all__ = ['Optimizer']


class Optimizer(ABC):
    """Abstract interface for gradient-based parameter optimizers."""

    def __init__(self, params: Iterable[Tensor], **defaults: Any):
        """Store the parameters managed by the optimizer.

        Args:
            params (Iterable[Tensor]): Parameters whose gradients drive the
                optimizer updates.
            **defaults (Any): Additional keyword arguments to store as attributes.
                These may be used by concrete optimizer implementations.
        """
        self.params = list(params)
        self.defaults = defaults

    @abstractmethod
    def step(self):
        """Update parameters in place using their current gradients."""
        pass

    def zero_grad(self, set_to_none: bool = False):
        """Clear stored parameter gradients.

        Args:
            set_to_none (bool, default: False): If ``True``, replace existing
                gradients with ``None``. Otherwise, zero gradients in place.
        """
        for p in self.params:
            if p.grad is None:
                continue

            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()
