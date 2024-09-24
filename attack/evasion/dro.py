from abc import ABC
from typing import Tuple
import numpy as np
import torch
from cvxopt import matrix, spmatrix, solvers
import torch.utils
import torch.utils.data

# DEBUG
# import matplotlib.pyplot as plt; plt.imshow(np.transpose(nn.Unflatten(0,(3, 32, 32))(x_adv[0]), (1, 2, 0))); plt.show()
# np.abs(np.array(nn.Unflatten(0,(3, 32, 32))(x_adv[0]) - x[0]))
# import matplotlib.pyplot as plt; plt.imshow(nn.Unflatten(0,(28, 28))(x_adv[0]), cmap='gray'); plt.show()
# import matplotlib.pyplot as plt; plt.imshow(np.transpose(x[0], (1, 2, 0)), cmap='gray'); plt.show()
####


class Dro(ABC):

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, epsilon: float,
                 k: int = 10, domain: Tuple[float, float] = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.k = k
        self.domain = [0., 1.] if domain is None else domain

    def generate(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        if y is None:
            y = self.model(x)
        return Dro.dro_W1_adv_data_driven_attack_ccp_opt(
            self.model, self.loss_fn, x, y, 
            self.epsilon, self.k, self.domain)

    @staticmethod
    def dro_W1_adv_data_driven_attack_opt(a: torch.Tensor,
                                          x: torch.Tensor,
                                          y: torch.Tensor,
                                          zeta: float,
                                          domain: Tuple[float, float]) -> Tuple[np.array, np.array]:

        def _build_data_dependent_vector(a: np.array,
                                         x: np.array,
                                         y: np.array) -> Tuple[np.array, np.array, np.array]:
            c = []
            b_minus = []
            b_plus = []
            for i in range(len(y)):
                c += [-(2. * y[i] - 1) * a]
                b_minus += [domain[1] - x[i, :]]
                b_plus += [domain[0] + x[i, :]]
            c = np.array(c).astype(np.double).flatten()
            b_minus = np.array(b_minus).astype(np.double).flatten()
            b_plus = np.array(b_plus).astype(np.double).flatten()

            return c, b_minus, b_plus

        def _zero(m: int, d: int) -> matrix:
            return matrix(np.zeros([m, d]))

        def _one(m: int, d: int) -> matrix:
            return matrix(np.ones([m, d]))

        def _id(m: int) -> matrix:
            return matrix(np.eye(m))

        c, b_minus, b_plus = _build_data_dependent_vector(a.detach().cpu().numpy(),
                                                          x.detach().cpu().numpy(),
                                                          y.detach().cpu().numpy())
        d = len(a)
        n = int(len(c) / d)
        c = matrix(np.hstack([c, np.array([0.0]*n)]))
        A = spmatrix(np.hstack([[-1.0]*d*n,
                                [1.0]*d*n,
                                [1.0]*n,
                                [1.0]*d*n,
                                [-1.0]*d*n,
                                [-1.0]*n,
                                [-1.0]*d*n,
                                [-1.0]*d*n]),
                     np.hstack([np.arange(d*n),
                                np.arange(d*n, 2*d*n),
                                [2*d*n]*n,
                                np.arange(2*d*n+1, 3*d*n+1),
                                np.arange(2*d*n+1, 3*d*n+1),
                                np.arange(3*d*n+1, 4*d*n+1),
                                np.arange(3*d*n+1, 4*d*n+1),
                                np.arange(4*d*n+1, 4*d*n+n+1)]),
                     np.hstack([np.arange(d*n),
                                np.arange(d*n),
                                np.arange(d*n, d*n+n),
                                np.arange(d*n),
                                [col for col in range(d*n, d*n + n)
                                 for _ in range(d)],
                                np.arange(d*n),
                                [col for col in range(d*n, d*n + n)
                                 for _ in range(d)],
                                np.arange(d*n, d*n + n)]))
        b = matrix([matrix(b_minus), matrix(b_plus),
                    matrix([zeta*n]), _zero(2*d*n+n, 1)])
        solvers.options["show_progress"] = True
        sol = solvers.lp(c, A, b)
        s = sol["x"][0:n*d]
        # r = sol["x"][(n*d):(n*d+n)]
        x_adv = x - \
            torch.Tensor(np.array(s).reshape(x.shape)).to(device=x.device)
        return x_adv

    @staticmethod
    def dro_W1_adv_data_driven_attack_ccp_opt(model: torch.nn.Module, loss_fn: torch.nn.Module,
                                          x: torch.Tensor, y: torch.Tensor, zeta: torch.Tensor,
                                          k: int, domain: Tuple[float, float]) -> np.array:
        def _zero(m: int, d: int) -> matrix:
            return matrix(np.zeros([m, d]))

        def _one(m: int, d: int) -> matrix:
            return matrix(np.ones([m, d]))

        def _id(m: int) -> matrix:
            return matrix(np.eye(m))

        x_adv = torch.rand_like(x, requires_grad=True)
        n, d = x_adv.shape
        A = spmatrix(np.hstack([[-1.0/n]*d*n,
                                [-1.0]*d*n,
                                [-1.0]*d*n,
                                [1.0]*d*n,
                                [-1.0]*d*n,
                                [-1.0]*d*n,
                                [1.0]*d*n,
                                [-1.0]*d*n]),
                     np.hstack([[0]*(d*n),
                                np.arange(1, d*n+1),
                                np.arange(1, d*n+1),
                                np.arange(d*n+1, 2*d*n+1),
                                np.arange(d*n+1, 2*d*n+1),
                                np.arange(2*d*n+1, 3*d*n+1),
                                np.arange(3*d*n+1, 4*d*n+1),
                                np.arange(4*d*n+1, 5*d*n+1)]),
                     np.hstack([np.arange(d*n, 2*d*n),
                                np.arange(d*n),
                                np.arange(d*n, 2*d*n),
                                np.arange(d*n),
                                np.arange(d*n, 2*d*n),
                                np.arange(d*n, 2*d*n),
                                np.arange(d*n),
                                np.arange(d*n)]))
        b = matrix([matrix([zeta]), -matrix(x.flatten().detach().cpu().numpy()), 
                    matrix(x.flatten().detach().cpu().numpy()),
                   _zero(d*n, 1), domain[1]*_one(d*n, 1),
                   domain[0]*_one(d*n, 1)])
        solvers.options["show_progress"] = True
        for _ in range(k):
            y_hat = model(x_adv)
            model.zero_grad()
            loss = loss_fn(y_hat.flatten(), 1.0*y)
            loss.backward(retain_graph=True)
            grad = x_adv.grad
            c = matrix([matrix(-grad.T.flatten().detach().cpu().numpy()), _zero(d*n, 1)])
            sol = solvers.lp(c, A, b)
            v = sol["x"][0:n*d]
            x_adv_old = x_adv
            x_adv = torch.Tensor(np.array(v).reshape(
                x.shape)).to(device=x.device)
        return x_adv
