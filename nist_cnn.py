import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from attack.evasion.dro import Dro
from attack.evasion.fgsm import FastGradientSignMethod
from attack.utils import generate_attack_loop
from models import CNN
from learn.train import eval_test, train_and_eval

torch.set_default_dtype(torch.float64)
torch.manual_seed(171)


def class_fn(yp): return yp.argmax(dim=1)
def eval_fn(y, yp): return y == class_fn(yp)
def agg_fn(x): return x.sum().item()


domain = [0., 1.]
channels = 1
n_classes = 10

# getting and transforming data
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.Lambda(lambda y: y)])

mnist_train = datasets.MNIST("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)

mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)

batch_size = 64
train_data_plain = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=False)
test_data_plain = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn_adv = loss_fn

models = {
    'cnn': CNN(channels, n_classes, device),
}
optimizers = {'cnn': torch.optim.SGD(models['cnn'].parameters(), lr=0.001)}

epsilon = 0.1
attacks = {
    'fgsm': FastGradientSignMethod(loss_fn_adv, epsilon, domain),
}
for lamb in [0.0, 0.01, 0.1, 1.0]:
    attacks['dro_lambda_' + str(lamb)] = Dro(loss_fn=loss_fn_adv, zeta=epsilon, domain=domain,
                                             max_steps=50, lamb=lamb)

epochs = 30
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['cnn'],
               loss_fn, optimizers['cnn'], device, eval_fn, agg_fn)

train_data_adv_model_attack = {
    model: {
        attack_name: None for attack_name in attacks.keys()
    } for model in models.keys()
}
# Plain model attacks
for attack_name in attacks.keys():
    train_data_adv_model_attack['cnn'][attack_name] = generate_attack_loop(
        test_data_plain, attacks[attack_name], models['cnn'], device)

# Start instances for adv trained models with starting parameters starting at the plain trained model
for attack_name in attacks.keys():
    models[attack_name] = CNN(channels, n_classes, device)
    optimizers[attack_name] = torch.optim.SGD(
        models[attack_name].parameters(), lr=0.001)

for model_name in train_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Test data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for attack_name in train_data_adv_model_attack[model_name].keys():
        print("  = Test data {} =".format(attack_name))
        eval_test(train_data_adv_model_attack[model_name][attack_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)
