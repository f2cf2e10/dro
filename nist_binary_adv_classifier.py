import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
import numpy as np
from attack.evasion import FastGradientSignMethod, ProjectecGradientDescent, Dro
from models import LinearModel
from learn.train import adv_train_and_eval, train_and_eval, eval_test
from attack.utils import generate_attack_loop
from utils.norm import Linf
from attack.utils import eval_adversary

torch.set_default_dtype(torch.float64)

# Using only 2 digits for a linear classifier
domain = [0., 1.]
digits = [0, 1]
labels = [0, 1]
loss_fn = torch.nn.BCEWithLogitsLoss()
def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()
# labels = [-1, 1]
# loss_fn = torch.nn.SoftMarginLoss()
# def eval_fn(y, yp): return ((y * yp) > 0)
# def agg_fn(x): return x.sum().item()


# getting and transforming data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.nn.Flatten(1, x.dim()-1)(x)[0])])
target_transform = transforms.Lambda(lambda y: y)


mnist_train = datasets.MNIST("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)
two_digits_train = list(filter(lambda x: np.isin(
    x[1], digits), mnist_train))
two_digits_train = [(x[0], labels[0] if x[1] == digits[0] else labels[1])
                    for x in two_digits_train]

mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)
two_digits_test = list(filter(lambda x: np.isin(
    x[1], digits), mnist_test))
two_digits_test = [(x[0], labels[0] if x[1] == digits[0] else labels[1])
                   for x in two_digits_test]

d = 784
batch_size = 64
train_data_plain = torch.utils.data.DataLoader(
    two_digits_train, batch_size=batch_size, shuffle=False)
test_data_plain = torch.utils.data.DataLoader(
    two_digits_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epsilon = 0.05
attacks = {'fgsm': FastGradientSignMethod(loss_fn, epsilon, domain),
           'dro': Dro(loss_fn, d*epsilon, domain, max_steps=30)}
models = {'plain': LinearModel(d).to(device),
          'fgsm': LinearModel(d).to(device),
          'dro': LinearModel(d).to(device)}
optimizers = {key: torch.optim.SGD(
    model.parameters(), lr=0.001) for key, model in zip(models.keys(), models.values())}
test_data_adv_model_attack = {
    'plain': {
        'fgsm': None,
        'dro': None
    },
    'fgsm': {
        'fgsm': None,
        'dro': None,
    },
    'dro': {
        'fgsm': None,
        'dro': None,
    }
}

epochs = 100
epochs_adv = 15
torch.manual_seed(17331)
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

# Adversarial data derived from plain model
for model_name in test_data_adv_model_attack.keys():
    for attack_name in attacks.keys():
        test_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
            test_data_plain, attacks[attack_name], models[model_name], device)

# Adversarial model training
for attack_name in attacks.keys():
    adv_train_and_eval(train_data_plain, test_data_plain, epochs_adv, models[attack_name],
                       attacks[attack_name], loss_fn, optimizers[attack_name], device, eval_fn, agg_fn)

# Evaluating models accuracies
for model_name in models.keys():
    print("=== Training procedure {} ===".format(model_name))
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)

# Evaluating
for model_name in test_data_adv_model_attack.keys():
    print("=== Attacking {} model ===".format(model_name))
    print("  = Plain test data =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for adv_data_name in test_data_adv_model_attack[model_name].keys():
        print("  = {} test data =".format(adv_data_name))
        eval_test(test_data_adv_model_attack[model_name][adv_data_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)

# Plotting model parameters
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(torch.nn.Unflatten(0, (28, 28))(
    list(models['plain'].parameters())[0]).detach().cpu().numpy(), cmap='gray')
ax1.set_title('Plain') 
ax2.imshow(torch.nn.Unflatten(0, (28, 28))(
    list(models['fgsm'].parameters())[0]).detach().cpu().numpy(), cmap='gray')
ax2.set_title('FGSM') 
ax3.imshow(torch.nn.Unflatten(0, (28, 28))(
    list(models['dro'].parameters())[0]).detach().cpu().numpy(), cmap='gray')
ax3.set_title('DRO') 


i_fgsm, x_fgsm, y_fgsm, x_adv_fgsm, y_adv_fgsm = eval_adversary(
    test_data_plain, test_data_adv_model_attack['plain']['fgsm'], models['plain'], device, eval_fn)
i_dro, x_dro, y_dro, x_adv_dro, y_adv_dro = eval_adversary(
    test_data_plain, test_data_adv_model_attack['plain']['dro'], models['plain'], device, eval_fn)

# DEBUG
i = 0
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_plain.dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax1.set_title(
    'Original: ' + str(digits[test_data_plain.dataset[i][1]]))
ax2.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_adv_model_attack['plain']['fgsm'].dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax2.set_title('FGSM classified as ' +
              str(digits[1*(models['plain'](test_data_adv_model_attack['plain']['fgsm'].dataset[i][0]) > 0)]))
ax3.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_adv_model_attack['plain']['dro'].dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax3.set_title('DRO classified as ' +
              str(digits[1*(models['plain'](test_data_adv_model_attack['plain']['dro'].dataset[i][0]) > 0)]))

import dill
filename = 'nist_binary_adv_classifier.pkl'
dill.dump_session(filename)

# and to load the session again:
dill.load_session(filename)
