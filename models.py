import torch


class LinearModel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((d, 1)))
        self.b = torch.nn.Parameter(torch.randn((1)))

    def forward(self, x):
        return x @ self.a + self.b


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.F.relu(self.conv1(x)))
        x = self.pool(torch.F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.F.relu(self.fc1(x))
        x = torch.F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
