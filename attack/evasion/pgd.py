from abc import ABC
from typing import Tuple
import torch
from utils.types import Norm


class ProjectecGradientDescent(ABC):
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 norm: Norm, proj: Norm, epsilon: float, alpha: float = 0.025,
                 k: int = 100, domain: Tuple[float, float] = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.norm = norm
        self.proj = proj
        self.alpha = alpha
        self.k = k
        self.epsilon = epsilon
        self.domain = [0., 1.] if domain is None else domain

    def generate(self, x, y=None) -> torch.utils.data.Dataset:
        if y is None:
            y = self.model(x)
        return ProjectecGradientDescent.projected_gradient_descent_norm_method(
            self.model, self.loss_fn, self.norm, self.proj, x, y, self.epsilon,
            self.alpha, self.k, self.domain)

    @staticmethod
    def projected_gradient_descent_norm_method(model: torch.nn.Module,
                                               loss_fn: torch.nn.Module,
                                               norm: Norm, proj: Norm,
                                               x: torch.utils.data.Dataset,
                                               y: torch.utils.data.Dataset,
                                               epsilon: float, alpha: float, k: int,
                                               domain: Tuple[float, float]):
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        for _ in range(k):
            y_hat = model(x_adv)
            if y_hat.shape != y.shape:
                y_hat = y_hat[:, 0]
            model.zero_grad()
            loss = loss_fn(y_hat.flatten(), y.float())
            loss.backward()
            x_adv = torch.autograd.Variable(torch.clamp(proj.proj(
                x_adv + alpha * norm.dual_dx(x_adv.grad), x, epsilon),
                domain[0], domain[1]), requires_grad=True)
        return x_adv
