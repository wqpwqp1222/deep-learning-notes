from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor

__all__ = ['Optimizer']


class Optimizer(ABC):
    """Abstract interface for gradient-based parameter optimizers."""

    def __init__(self, params: Sequence[Tensor]):
        """Store the parameters managed by the optimizer.

        Args:
            params (Sequence[Tensor]): Parameters whose gradients drive the
                optimizer updates.
        """
        self.params = params

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
