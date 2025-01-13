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
domain = [-1., 1.]
digits = [0, 1]
labels = [0, 1]
loss_fn = torch.nn.BCEWithLogitsLoss()
def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()


# labels = [-1, 1]
# loss_fn = torch.nn.SoftMarginLoss()
# def eval_fn(y, yp): return ((y * yp) > 0)
# def agg_fn(x): return x.sum().item()
a_true = np.array([2.0, 1.0])
b_true = 0.0


def f_x(x): return (np.sign(x.dot(a_true) + b_true +
                            0*np.random.randn(x.shape[0])) + 1)/2


N = 1000
x = 2*np.random.rand(N, 2)-1
y = f_x(x)
two_digits_train = [(x[i, :], y[i]) for i in range(N)]


# getting and transforming data
N = 250
x = 2 * np.random.rand(N, 2) - 1
y = f_x(x)
two_digits_test = [(x[i, :], y[i]) for i in range(N)]

d = 2
batch_size = 64
train_data_plain = torch.utils.data.DataLoader(
    two_digits_train, batch_size=batch_size, shuffle=False)
test_data_plain = torch.utils.data.DataLoader(
    two_digits_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {'plain': LinearModel(d).to(device),
          'fgsm': LinearModel(d).to(device),
          'dro': LinearModel(d).to(device),
          'dro_pen': LinearModel(d).to(device)}

epsilon = 0.05
attacks = {'fgsm': FastGradientSignMethod(loss_fn, epsilon, domain),
           'dro': Dro(loss_fn, d*epsilon, domain, max_steps=30),
           'dro_pen': Dro(loss_fn, d*epsilon, domain, max_steps=30, lamb=1000000.0)}
optimizers = {key: torch.optim.SGD(
    model.parameters(), lr=0.001) for key, model in zip(models.keys(), models.values())}
train_data_adv_model_attack = {
    'plain': {
        'fgsm': None,
        'dro': None,
        'dro_pen': None,
    },
    'fgsm': {
        'fgsm': None,
        'dro': None,
        'dro_pen': None,
    },
    'dro': {
        'fgsm': None,
        'dro': None,
        'dro_pen': None,
    }
}

epochs = 1000
epochs_adv = 15
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

# Adversarial data derived from plain model
for model_name in train_data_adv_model_attack.keys():
    for attack_name in attacks.keys():
        train_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
            train_data_plain, attacks[attack_name], models[model_name], device)

# Adversarial model training
for attack_name in attacks.keys():
    adv_train_and_eval(train_data_plain, test_data_plain, epochs_adv, models[attack_name],
                       attacks[attack_name], loss_fn, optimizers[attack_name], device, eval_fn, agg_fn)

for model_name in train_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Test data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    print("  = Test data fgsm =")
    eval_test(train_data_adv_model_attack[model_name]['fgsm'], models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    print("  = Test data dro =")
    eval_test(train_data_adv_model_attack[model_name]['dro'],
              models[model_name], loss_fn, device, eval_fn, agg_fn)
    print("  = Test data dro penalized =")
    eval_test(train_data_adv_model_attack[model_name]['dro_pen'],
              models[model_name], loss_fn, device, eval_fn, agg_fn)

# Analysis full data
adv_model_name = "dro"
# plain data
# all data
x = np.vstack([x[0] for x in train_data_plain.dataset])
y = np.vstack([x[1] for x in train_data_plain.dataset])
# 1st batch
# x, y = next(iter(train_data_plain))
#x = x.detach().numpy()
#y = y.detach().numpy()
idx_1 = np.where(y == 1)[0]
idx_0 = np.where(y == 0)[0]
plt.figure()
plt.scatter(x[idx_1, 0], x[idx_1, 1], s=20, facecolors='None',
            edgecolors='b', marker="o", label='first')
# for idx in idx_1:
#    plt.annotate(str(idx), (x[idx][0], x[idx][1]+0.02))
plt.scatter(x[idx_0, 0], x[idx_0, 1], s=20, facecolors='None',
            edgecolors='r', marker="o", label='second')
# for idx in idx_0:
#    plt.annotate(str(idx), (x[idx][0], x[idx][1]+0.02))
# adv data on plain model
x = np.vstack([x[0].detach().numpy()
              for x in train_data_adv_model_attack['plain'][adv_model_name].dataset])
y = np.vstack([x[1].detach().numpy()
              for x in train_data_adv_model_attack['plain'][adv_model_name].dataset])
# 1st batch
#x, y = next(iter(train_data_adv_model_attack['plain'][adv_model_name]))
#x = x.detach().numpy()
#y = y.detach().numpy()
idx_1 = np.where(y == 1)[0]
idx_0 = np.where(y == 0)[0]
plt.scatter(x[idx_1, 0], x[idx_1, 1], s=20, c='b', marker="+", label='first')
plt.scatter(x[idx_0, 0], x[idx_0, 1], s=20, c='r', marker="+", label='second')

t = np.arange(-1, 1, .01).reshape((200, 1))
a = a_true
b = b_true
plt.plot(-(a[1]*t - b)/a[0], t, c='black', linewidth=3)

a = list(models['plain'].parameters())[0].detach().numpy()
b = list(models['plain'].parameters())[1].detach().numpy()
x0 = -(a[1]*t - b)/a[0]
x1 = t
idx = (x0 >= -1) & (x0 <= 1) & (x1 >= -1) & (x1 <= 1)
plt.plot(x0[idx], x1[idx], c='black', linestyle='dashed', linewidth=3)
plt.ylabel('x[1]')
plt.xlabel('x[0]')
plt.title(adv_model_name)
plt.show()
plt.plot()


i_fgsm, x_fgsm, y_fgsm, x_adv_fgsm, y_adv_fgsm = eval_adversary(
    test_data_plain, train_data_adv_model_attack['plain']['fgsm'], models['plain'], device, eval_fn)
i_dro, x_dro, y_dro, x_adv_dro, y_adv_dro = eval_adversary(
    test_data_plain, train_data_adv_model_attack['plain']['dro'], models['plain'], device, eval_fn)

# DEBUG
i = 0
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_plain.dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax1.set_title(
    'Original: ' + str(digits[test_data_plain.dataset[i][1]]))
ax2.imshow(torch.nn.Unflatten(0, (28, 28))(
    train_data_adv_model_attack['plain']['fgsm'].dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax2.set_title('FGSM classified as ' +
              str(digits[1*(models['plain'](train_data_adv_model_attack['plain']['fgsm'].dataset[i][0]) > 0)]))
ax3.imshow(torch.nn.Unflatten(0, (28, 28))(
    train_data_adv_model_attack['plain']['dro'].dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax3.set_title('DRO classified as ' +
              str(digits[1*(models['plain'](train_data_adv_model_attack['plain']['dro'].dataset[i][0]) > 0)]))

