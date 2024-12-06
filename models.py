import torch


def Linear(in_features: int, out_features: int, device) -> torch.nn.Module:
    linear = torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features,
                        out_features=out_features,
                        device=device),
        torch.nn.Flatten(0, -1))

    return linear


def CNN(in_channels: int, n_classes: int, device) -> torch.nn.Module:
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=6, kernel_size=3,
                        device=device),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(in_channels=6, out_channels=16,
                        kernel_size=3, device=device),
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
