from abc import ABC
from typing import Tuple
import torch


class Trades(ABC):
    criterion_kl = torch.nn.KLDivLoss(size_average=False)

    def generate(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        if y is None:
            y = model(x)
        return Trades.trades(model, self.loss_fn, x, y, self.epsilon, self.alpha, self.k)

    @staticmethod
    def trades(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               x,
               y,
               epsilon: float,
               perturb_steps: float,
               beta: float,
               sigma: float,
               domain: Tuple[float, float]):
        batch_size = len(x)
        x_nat = torch.clone(x)
        x_adv = torch.clamp(x_nat + sigma * torch.randn_like(x_nat),
                            min=domain[0], max=domain[1])
        delta = x_adv - x_nat
        delta.requires_grad = True

        # Setup optimizers
        optimizer_delta = torch.optim.SGD(
            [delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_nat + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * loss_fn(torch.F.log_softmax(model(x_adv), dim=1),
                                      torch.F.softmax(model(x_nat), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(
                    delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_nat)
            delta.data.clamp_(domain[0], domain[1]).sub_(x_nat)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = x_nat + delta
        x_adv.requires_grad = False
        model.train()

        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv.requires_grad = False
        model.zero_grad()

        logits = model(x_nat)
        loss_natural = torch.F.cross_entropy(logits.T[0], y)
        loss_robust = (1.0 / batch_size) * loss_fn(torch.F.log_softmax(model(x_adv), dim=1),
                                                   torch.F.softmax(model(x_nat), dim=1))
        loss = loss_natural + beta * loss_robust
        return loss, x_adv
