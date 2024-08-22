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
                 alpha: float = 0.025, k: int = 10, kpgd: int = 100,
                 domain: Tuple[float, float] = None, method: str = "GD") -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k
        self.kpgd = kpgd
        self.domain = [0., 1.] if domain is None else domain
        self.method = method

    def generate(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.utils.data.Dataset:
        if y is None:
            y = self.model(x)

        if self.method == "GD":
            c = torch.Tensor([10.]).to(x.device)
            return Dro.dro_W1_adv_data_driven_attack_gd(
                self.model, self.loss_fn, x, y,
                torch.Tensor([self.epsilon]).to(x.device),
                torch.Tensor([self.alpha]).to(x.device),
                self.k, self.kpgd, c, self.domain)
        else:
            return Dro.dro_W1_adv_data_driven_attack_opt(
                list(self.model.parameters())[0],
                x, y,
                self.epsilon,
                self.domain)

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
                c += [-y[i] * a]
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
        #r = sol["x"][(n*d):(n*d+n)]
        x_adv = x - torch.Tensor(np.array(s).reshape(x.shape)).to(device=x.device)
        return x_adv

    @staticmethod
    def dro_W1_adv_data_driven_attack_gd(model: torch.nn.Module,
                                         loss_fn: torch.nn.Module,
                                         x: torch.Tensor,
                                         y: torch.Tensor,
                                         zeta: torch.Tensor,
                                         alpha: torch.Tensor,
                                         k: int, k_pgd: int, c: torch.Tensor,
                                         domain: Tuple[float, float]) -> np.array:
        lamb = torch.Tensor([0.1]).to(x.device)

        def norm(x): return torch.linalg.vector_norm(
            x, dim=1, ord=float('inf'))
        x_adv = (torch.clamp(x + (torch.rand_like(x) - 0.5)
                             * zeta, domain[0], domain[1])).clone()
        x_adv.requires_grad = True
        n = x_adv.shape[0]
        sum_norm = norm(x_adv - x).sum()
        grad = None
        for _ in range(k):
            for _ in range(k_pgd):
                y_hat = model(x_adv)
                model.zero_grad()
                loss = -loss_fn(y_hat.flatten(), y) + \
                    c/2 * (sum_norm - n * zeta)**2 - \
                    lamb * (sum_norm - n * zeta)
                loss.backward(retain_graph=True)
                x_adv_old = x_adv
                grad_old = grad
                grad = x_adv.grad
                x_adv = torch.clamp(x + alpha * grad,
                                    domain[0], domain[1]).clone()
                sum_norm_old = sum_norm
                sum_norm = norm(x_adv - x).sum()
                if sum_norm > n*zeta:
                    # x_adv = (x + (x_adv - x) * n*zeta/sum_norm).clone()
                    sum_norm = norm(x_adv - x).sum()
                x_adv.requires_grad = True
            lamb = lamb + c * (sum_norm - n * zeta)
        return x_adv
