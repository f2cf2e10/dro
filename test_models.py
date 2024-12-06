import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from models import Linear, LinearMatrix
from learn.train import train_and_eval

torch.set_default_dtype(torch.float64)
torch.manual_seed(171)
# Using only 2 digits for a linear classifier
domain = [0., 1.]
digits = [0, 1]
labels = [0., 1.]
loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn_adv = loss_fn


def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()


d = 28
batch_size = 64
epochs = 100

# getting and transforming data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.nn.Flatten(1, x.dim()-1)(x))])
target_transform = transforms.Lambda(lambda y: y)


def prepare_data(transform, target_transform):
    mnist_train = datasets.MNIST("./data", train=True, download=True,
                                 transform=transform, target_transform=target_transform)
    two_digits_train = list(filter(lambda x: np.isin(
        x[1], digits), mnist_train))
    two_digits_train = [(x[0][0], labels[0] if x[1] == digits[0] else labels[1])
                        for x in two_digits_train]

    mnist_test = datasets.MNIST("./data", train=False, download=True,
                                transform=transform, target_transform=target_transform)
    two_digits_test = list(filter(lambda x: np.isin(
        x[1], digits), mnist_test))
    two_digits_test = [(x[0][0], labels[0] if x[1] == digits[0] else labels[1])
                       for x in two_digits_test]

    train_data = torch.utils.data.DataLoader(
        two_digits_train, batch_size=batch_size, shuffle=False)
    test_data = torch.utils.data.DataLoader(
        two_digits_test, batch_size=batch_size, shuffle=False)
    return train_data, test_data


train_data_plain, test_data_plain = prepare_data(transform, target_transform)
train_data, test_data = prepare_data(transforms.ToTensor(), target_transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

linear = Linear(d*d, 1, device)
linearMatrix = LinearMatrix(d, d, device)
optimizer_linear = torch.optim.SGD(linear.parameters(), lr=0.001)
optimizer_linearMatrix = torch.optim.SGD(linearMatrix.parameters(), lr=0.001)

train_and_eval(train_data_plain, test_data_plain, epochs, linear,
               loss_fn, optimizer_linear, device, eval_fn, agg_fn)

train_and_eval(train_data, test_data, epochs, linearMatrix,
               loss_fn, optimizer_linearMatrix, device, eval_fn, agg_fn)
