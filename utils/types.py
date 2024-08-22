from abc import ABC, abstractmethod
import torch


class Norm(ABC):
    @staticmethod
    @abstractmethod
    def norm(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def norm_dx(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def dual(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def dual_dx(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class method")

    @staticmethod
    @abstractmethod
    def proj(x: torch.Tensor, x0: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class method")
