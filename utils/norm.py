import torch
import numpy as np
from cvxopt import matrix, solvers
from .types import Norm


class L1(Norm):
    @staticmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        return x.norm(1)

    @staticmethod
    def norm_dx(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x)

    @staticmethod
    def dual(x: torch.Tensor) -> torch.Tensor:
        return Linf.norm(x)

    @staticmethod
    def dual_dx(x: torch.Tensor) -> float:
        return Linf.norm_dx(x)

    @staticmethod
    def proj(x: torch.Tensor, x0: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        N = len(x)
        P = matrix([[matrix(np.eye(N)), matrix(np.zeros(N * N).reshape([N, N]))],
                    [matrix(np.zeros(N * N).reshape([N, N])), matrix(np.zeros(N * N).reshape([N, N]))]])
        _q = -2.0 * (x - x0).detach().numpy()
        q = matrix([matrix(_q.astype(np.double)), matrix(np.zeros(N))])
        G = matrix([[matrix(np.eye(N)), -matrix(np.eye(N))],
                    [-matrix(np.eye(N)), -matrix(np.eye(N))],
                    [matrix(np.zeros(N * N).reshape([N, N])), matrix(-np.eye(N))],
                    [matrix(np.zeros(N)), matrix(np.ones(N))]]).T
        h = matrix([0.0] * 3 * N + [np.double(constraint.detach().numpy()[0])])
        sol = solvers.qp(P, q, G, h)
        return x0 + torch.from_numpy(np.array(sol['x'][0:N].T)[0])


class L2(Norm):
    @staticmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        return x.norm(2)

    @staticmethod
    def norm_dx(x: torch.Tensor) -> torch.Tensor:
        return x / L2.norm(x)

    @staticmethod
    def dual(x: torch.Tensor) -> torch.Tensor:
        return L2.norm(x)

    @staticmethod
    def dual_dx(x: torch.Tensor) -> torch.Tensor:
        return L2.norm_dx(x)

    @staticmethod
    def proj(x: torch.Tensor, x0: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        return x0 + (x - x0) / L2.norm(x - x0) * constraint


class Linf(Norm):
    @staticmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        return x.norm(float("inf"))

    @staticmethod
    def norm_dx(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not Implemented")

    @staticmethod
    def dual(x: torch.Tensor) -> torch.Tensor:
        return L1.norm(x)

    @staticmethod
    def dual_dx(x: torch.Tensor) -> torch.Tensor:
        return L1.norm_dx(x)

    @staticmethod
    def proj(x: torch.Tensor, x0: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        return x0 + torch.clip(x - x0, -constraint, constraint)
