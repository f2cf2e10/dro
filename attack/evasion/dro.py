from abc import ABC
from typing import Callable, Tuple
import numpy as np
import torch
import clarabel
from scipy import sparse
import torch.utils
import torch.utils.data
from attack.evasion.interface import EvasionAttack


class Dro(EvasionAttack):

    def __init__(self, loss_fn: torch.nn.Module, zeta: float, domain: Tuple[float, float],
                 max_steps: int = 10, lamb: float = 0.0) -> None:
        self.loss_fn = loss_fn
        self.zeta = zeta
        self.max_steps = max_steps
        self.domain = domain
        self.lamb = lamb

    def generate(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        if y is None:
            y = model(x)
        return Dro.dro_W1_adv_data_driven_attack_ccp_opt(
            model, self.loss_fn, x, y, self.zeta, self.max_steps,
            self.domain, lamb=self.lamb)

    @staticmethod
    def dro_W1_adv_data_driven_attack_ccp_opt(model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor,
                                              y: torch.Tensor, zeta: torch.Tensor, max_steps: int, domain: Tuple[float, float],
                                              lamb: float = 0.0, tol: float = 1E-6) -> np.array:
        n = x.shape[0]
        x_flat = x.flatten().detach().cpu().numpy()
        d = int(len(x_flat)/n)
        val = np.hstack([[1.0/n]*d*n,
                         [-1.0]*d*n,
                         [-1.0]*d*n,
                         [1.0]*d*n,
                         [-1.0]*d*n,
                         [-1.0]*d*n,
                         [1.0]*d*n,
                         [-1.0]*d*n])
        row = np.hstack([[0]*(d*n),
                         np.arange(1, d*n+1),
                         np.arange(1, d*n+1),
                         np.arange(d*n+1, 2*d*n+1),
                         np.arange(d*n+1, 2*d*n+1),
                         np.arange(2*d*n+1, 3*d*n+1),
                         np.arange(3*d*n+1, 4*d*n+1),
                         np.arange(4*d*n+1, 5*d*n+1)])
        col = np.hstack([np.arange(d*n, 2*d*n),
                         np.arange(d*n),
                         np.arange(d*n, 2*d*n),
                         np.arange(d*n),
                         np.arange(d*n, 2*d*n),
                         np.arange(d*n, 2*d*n),
                         np.arange(d*n),
                         np.arange(d*n)])
        A = sparse.csc_matrix((val, (row, col)), shape=(5*d*n+1, 2*d*n))
        b = np.hstack([zeta, -1.0 * x_flat, x_flat, np.zeros(d*n),
                       domain[1]*np.ones(d*n), -1.0 * domain[0]*np.ones(d*n)])
        P = sparse.csc_matrix(([0.0 if i < d*n else lamb/n for i in range(2*d*n)],
                               (range(2*d*n), range(2*d*n))), shape=(2*d*n, 2*d*n))
        cones = [clarabel.NonnegativeConeT(5*d*n+1)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        x_adv = torch.rand_like(x, requires_grad=True)
        x_adv.requires_grad_()
        y_hat = model(x_adv)
        loss = loss_fn(y_hat, y)
        loss_old = loss + 10*tol
        grad_flat = None
        for _ in range(max_steps):
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad_flat = x_adv.grad.flatten().detach().cpu().numpy()
            # print(np.abs(grad_flat).mean(), np.abs(grad_flat).max())
            q = np.hstack(
                [-1.0/n*grad_flat, np.zeros(d*n)])
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            sol = solver.solve()
            v = sol.x[0:n*d]
            t = sol.x[n*d:]
            x_adv = torch.Tensor(np.array(v).reshape(
                x.shape)).to(device=x.device)
            x_adv.requires_grad_()
            loss_old = loss
            y_hat = model(x_adv)
            loss = loss_fn(y_hat, y)
            if ((loss - loss_old).abs() <= tol).cpu().detach().numpy():
                break
        return x_adv


