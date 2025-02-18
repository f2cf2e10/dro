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
        n, *_, nrow, ncol = x.shape
        d = nrow * ncol
        x_tensor_flat = x.flatten()
        x_flat = x_tensor_flat.detach().cpu().numpy()
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
        settings.verbose = True
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
            q = np.hstack(
                [-1.0/n*grad_flat, np.zeros(d*n)])
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            sol = solver.solve()
            v = sol.x[0:n*d]
            t = sol.x[n*d:]
            x_adv = torch.Tensor(np.array(v)).reshape_as(x).to(device=x.device)
            x_adv.requires_grad_()
            loss_old = loss
            y_hat = model(x_adv)
            loss = loss_fn(y_hat, y)
            if ((loss - loss_old).abs() <= tol).cpu().detach().numpy():
                break
        return x_adv


class DroEntropic(EvasionAttack):

    def __init__(self, loss_fn: torch.nn.Module, zeta: float, domain: Tuple[float, float],
                 max_steps: int = 10, lamb: float = 1.0) -> None:
        self.loss_fn = loss_fn
        self.zeta = zeta
        self.max_steps = max_steps
        self.domain = domain
        self.lamb = lamb
        self.p = 1

    def generate(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        *_, nrow, ncol = x.shape
        d = nrow * ncol
        C = torch.zeros((d, d))
        for i_row in range(nrow):
            for i_col in range(ncol):
                for j_row in range(nrow):
                    for j_col in range(ncol):
                        val = (np.abs(i_row - j_row)**self.p +
                               np.abs(i_col - j_col)**self.p)**(1/self.p)
                        C[i_row * ncol + i_col, j_row * ncol + j_col] = val
        self.C = C
        if y is None:
            y = model(x)
        return DroEntropic.dro_W1_adv_data_driven_attack_ccp_entropic(
            model, self.loss_fn, x, y, self.C, self.zeta, self.max_steps, lamb=self.lamb)

    @staticmethod
    def dro_W1_adv_data_driven_attack_ccp_entropic(model: torch.nn.Module, loss_fn: torch.nn.Module, x: torch.Tensor,
                                                   y: torch.Tensor, C: torch.Tensor, zeta: torch.Tensor, max_steps: int,
                                                   lamb: float = 0.0, tol: float = 1E-6) -> np.array:
        n, *_ = x.shape
        v = torch.rand_like(x, requires_grad=True)
        v.requires_grad_()
        y_hat = model(v)
        loss = loss_fn(y_hat, y)
        loss_old = loss + 10*tol
        x_flat = x.flatten(1)
        normalization = x_flat.sum(1)
        x_flat = torch.div(x_flat.T, normalization).T
        for _ in range(max_steps):
            model.zero_grad()
            loss.backward(retain_graph=True)
            beta = -lamb * v.grad.flatten(1)
            alpha = np.log(1./n) * torch.ones_like(beta)
            gamma = 1
            counter = 0
            while counter < max_steps:
                alpha = torch.log(
                    torch.sum(
                        torch.exp(- beta.unsqueeze(1)  # Shape: (n, 1, d)
                                  - gamma * C.unsqueeze(0)  # Shape: (1, d, d)
                                  - 1),
                        dim=2)) - torch.log(x_flat)  # Shape: (n, d)
                P = torch.exp(
                    - alpha.unsqueeze(2)  # Shape: (n, d, 1)
                    - beta.unsqueeze(1)  # Shape: (n, 1, d)
                    - gamma * C.unsqueeze(0)  # Shape: (1, d, d)
                    - 1  # Scalar
                )  # Shape: (n, d, d)
                a = torch.sum(
                    C.unsqueeze(0) * P,  # Shape: (n, d, d)
                ) - n * zeta
                b = - torch.sum(
                    (C.unsqueeze(0) ** 2) * P,  # Shape: (n, d, d)
                )
                delta_gamma = a/b
                if (delta_gamma.abs() <= tol).cpu().detach().numpy():
                    break
                else:
                    t = 1.
                    while gamma - t*delta_gamma < 0:
                        t = t/2
                    gamma = gamma - t*delta_gamma
                counter += 1
            v = torch.mul(P.sum(1).T, normalization).T.reshape_as(v)
            # v = P.sum(2).reshape_as(v)
            v.requires_grad_()

            loss_old = loss
            y_hat = model(v)
            loss = loss_fn(y_hat, y)
            if ((loss - loss_old).abs() <= tol).cpu().detach().numpy():
                break
        return v


class DroMinCorrectClassifier(EvasionAttack):

    def __init__(self, loss_fn: torch.nn.Module, zeta: float, domain: Tuple[float, float],
                 eta: float, max_iter: int, tol: float = 1E-6) -> None:
        self.loss_fn = loss_fn
        self.zeta = zeta
        self.domain = domain
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def generate(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        if y is None:
            y = model(x)
        return DroMinCorrectClassifier.projected_gradient_descent_method(model, self.loss_fn, x, y,
                                                                         self.zeta, self.eta, self.max_iter,
                                                                         self.domain, self.tol)

    @staticmethod
    def projected_gradient_descent_method(model: torch.nn.Module,
                                          loss_fn: torch.nn.Module,
                                          x: torch.utils.data.Dataset,
                                          y: torch.utils.data.Dataset,
                                          epsilon: float, eta: float,
                                          max_iter: int, domain: Tuple[float, float],
                                          tol: float):
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        n, *_, nrow, ncol = x.shape
        d = nrow * ncol
        x_tensor_flat = x.flatten()
        x_flat = x_tensor_flat.detach().cpu().numpy()
        old_loss = np.inf
        for _ in range(max_iter):
            y_hat = model(x_adv)
            if y_hat.shape != y.shape:
                y_hat = y_hat[:, 0]
            model.zero_grad()
            loss = loss_fn(y_hat.flatten(), y.float())
            loss.backward()
            if np.abs(loss.item() - old_loss) <= tol:
                break
            old_loss = loss.item()
            # Gradient update
            w = x_adv - eta * x_adv.grad
            # Projection
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
            b = np.hstack([epsilon, -1.0 * x_flat, x_flat, np.zeros(d*n),
                           domain[1]*np.ones(d*n), -1.0 * domain[0]*np.ones(d*n)])
            P = sparse.csc_matrix(([1.0] * d * n, (range(d*n), range(d*n))),
                                  shape=(2*d*n, 2*d*n))
            cones = [clarabel.NonnegativeConeT(5*d*n+1)]
            settings = clarabel.DefaultSettings()
            settings.verbose = True
            q = np.hstack([-2*w.flatten().detach().cpu().numpy(),
                           np.zeros(d*n)])
            solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            sol = solver.solve()
            v = sol.x[0:n*d]
            s = sol.x[n*d:]
            x_adv = torch.Tensor(np.array(v)).reshape_as(x).to(device=x.device)
            x_adv.requires_grad_()
        return x_adv
