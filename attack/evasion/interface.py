from abc import ABC, abstractmethod
from typing import Tuple

import torch


class EvasionAttack(ABC):
    def __init__(self, loss_fn: torch.nn.Module, epsilon: float, domain: Tuple[float, float]) -> None:
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.domain = domain

    @abstractmethod
    def generate(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        pass
