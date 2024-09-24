import numpy as np
from torchvision import datasets, transforms
import torch
import numpy as np
from attack.evasion import FastGradientSignMethod, ProjectecGradientDescent, Dro
from models import LinearModel
from utils import generate_attack_loop, train_and_eval, eval_test
from utils.norm import Linf

torch.set_default_dtype(torch.float64)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.nn.Flatten(1, x.dim()-1)(x)[0])])
target_transform = transforms.Lambda(lambda y: y)

# Using only 0s and 1s
digits = [0, 1]
mnist_train = datasets.MNIST("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)
zeros_ones_train = list(filter(lambda x: np.isin(x[1], digits), mnist_train))  # zero and ones

mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)
zeros_ones_test = list(filter(lambda x: np.isin(x[1], digits), mnist_test))  # zero and ones

d = 784
batch_size = 100
train_data = torch.utils.data.DataLoader(
    zeros_ones_train, batch_size=batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(
    zeros_ones_test, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LinearModel(d).to(device)
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 10
train_and_eval(train_data, test_data, epochs,
               model, loss_fn, optimizer, device)
eval_test(test_data, model, loss_fn, device)

epsilon = 0.1
attack = FastGradientSignMethod(model, loss_fn, epsilon) 
test_data_adv = generate_attack_loop(test_data, attack, device)
eval_test(test_data_adv, model, loss_fn, device)

attack = ProjectecGradientDescent(model, loss_fn, Linf, Linf, epsilon)
test_data_adv = generate_attack_loop(test_data, attack, device)
eval_test(test_data_adv, model, loss_fn, device)

attack = Dro(model, loss_fn, epsilon)
test_data_adv = generate_attack_loop(test_data, attack, device)
eval_test(test_data_adv, model, loss_fn, device)