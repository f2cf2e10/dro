import torch


def LinearFlat(in_features: int, out_features: int, device) -> torch.nn.Module:
    linear = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features,
                        out_features=out_features,
                        device=device),
        torch.nn.Flatten(0, -1))

    return linear


def LinearMatrix(in_nrows: int, in_ncols: int, device) -> torch.nn.Module:

    class LinearMatrix(torch.nn.Module):
        def __init__(self, in_nrows: int, in_ncols: int):
            super().__init__()
            sqrt_k = 1./(in_nrows*in_ncols)**0.5
            self.A = torch.nn.Parameter(
                2*(torch.rand(in_nrows, in_ncols)-0.5)*sqrt_k)
            self.b = torch.nn.Parameter(2*(torch.rand(1)-0.5)*sqrt_k)

        def forward(self, x):
            x_ = torch.mul(self.A, x)
            n = x_.dim()
            return x_.sum(list(range(n-2,  n))) + self.b

    model = LinearMatrix(in_nrows, in_ncols)
    model.to(device)
    return model


def CNN(in_channels: int, n_classes: int, kernel_size: int, device) -> torch.nn.Module:
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=6, kernel_size=kernel_size,
                        device=device),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(in_channels=6, out_channels=16,
                        kernel_size=kernel_size, device=device),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Flatten(1, -1),
        torch.nn.Linear(in_features=16*5*5, out_features=120, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=120, out_features=84, device=device),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=84, out_features=n_classes, device=device),
    )
    return cnn
