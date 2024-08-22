from abc import ABC
from typing import Tuple
import torch
from utils.types import Norm
from utils.norm import Linf


class FastGradientSignMethod(ABC):
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 epsilon: float, domain: Tuple[float, float] = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.domain = [0., 1.] if domain is None else domain

    def generate(self, x, y=None) -> torch.utils.data.Dataset:
        if y is None:
            y = self.model(x)
        return FastGradientSignMethod.fast_gradient_dual_norm_method(
            self.model, self.loss_fn, Linf, x, y, self.epsilon, self.domain)

    @staticmethod
    def fast_gradient_dual_norm_method(model: torch.nn.Module,
                                        loss_fn: torch.nn.Module,
                                        norm: Norm, x: torch.utils.data.Dataset,
                                        y: torch.utils.data.Dataset, epsilon: float,
                                        domain: Tuple[float, float]) -> torch.utils.data.Dataset:
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        y_hat = model(x_adv)
        if y_hat.shape != y.shape:
            y_hat = y_hat[:, 0]
        model.zero_grad()
        loss = loss_fn(y_hat, y.float())
        loss.backward()
        return torch.clamp(x_adv + norm.dual_dx(x_adv.grad) * epsilon, domain[0], domain[1])
