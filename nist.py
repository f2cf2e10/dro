import numpy as np
from torchvision import datasets, transforms
import torch
import numpy as np
from attack.evasion import FastGradientSignMethod, ProjectecGradientDescent, Dro
from models import LinearModel
from utils import generate_attack_loop, train_and_eval, eval_test
from utils.norm import Linf

torch.set_default_dtype(torch.float64)

transform = transforms.ToTensor()
target_transform = transforms.Lambda(lambda y: y)

mnist_train = datasets.MNIST("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)

mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)
