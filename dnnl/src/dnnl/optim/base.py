from abc import ABC, abstractmethod

__all__ = ['Optimizer']


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        pass
