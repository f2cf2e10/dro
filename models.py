import torch


class LinearModel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((d, 1)))
        self.b = torch.nn.Parameter(torch.randn((1)))

    def forward(self, x):
        return x @ self.a + self.b
