import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from attack.evasion import FastGradientSignMethod, Dro, DroEntropic
from models import LinearMatrix
from learn.train import adv_train_and_eval, train_and_eval, eval_test
from attack.utils import generate_attack_loop
from utils.norm import Linf
from attack.utils import eval_adversary

torch.set_default_dtype(torch.float64)
torch.manual_seed(1771)

# Using only 2 digits for a linear classifier
domain = [0., 1.]
digits = [0, 1]
labels = [0, 1]
loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn_adv = loss_fn


def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()


# labels = [-1, 1]
# loss_fn = torch.nn.SoftMarginLoss()
# def eval_fn(y, yp): return ((y * yp) > 0)
# def agg_fn(x): return x.sum().item()
a_true = np.array([-4, -3.5])
b_true = 3.75


def f_x(x):
    y = (a_true.dot(x.T) + b_true)
    return (np.sign(y)+1)/2


N = 1000
x = np.random.rand(N, 2)
y = f_x(x)
two_digits_train = [(x[[i]], y[i]) for i in range(N)]


# getting and transforming data
N = 250
x = np.random.rand(N, 2)
y = f_x(x)
two_digits_test = [(x[[i]], y[i]) for i in range(N)]

d = 2
batch_size = 100
train_data_plain = torch.utils.data.DataLoader(
    two_digits_train, batch_size=batch_size, shuffle=True)
test_data_plain = torch.utils.data.DataLoader(
    two_digits_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epsilon = 0.1
models = {
    'plain': LinearMatrix(1, d, device),
}

models = {'plain': LinearMatrix(1, d, device)}

attacks = {'fgsm': FastGradientSignMethod(loss_fn_adv, epsilon, domain),
           'dro': Dro(loss_fn_adv, d*epsilon, domain, max_steps=50)}

for lamb in [1., 10., 100., 1000., 2500., 5000., 10000.]:
    attacks['dro_entropic_lambda_' + str(lamb)] = DroEntropic(loss_fn=loss_fn_adv, zeta=epsilon,
                                                              domain=domain, max_steps=100, lamb=lamb)
optimizers = {'plain': torch.optim.SGD(models['plain'].parameters(), lr=0.01)}

epochs = 1000
epochs_adv = 50
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

for attack_name in attacks.keys():
    models[attack_name] = LinearMatrix(1, d, device)
    optimizers[attack_name] = torch.optim.SGD(
        models[attack_name].parameters(), lr=0.001)

train_data_adv_model_attack = {
    model: {
        attack_name: None for attack_name in attacks.keys()
    } for model in models.keys()
}

# Plain model attacks
for attack_name in attacks.keys():
    print(f'===== {attack_name} =====')
    train_data_adv_model_attack['plain'][attack_name] = generate_attack_loop(
        train_data_plain, attacks[attack_name], models['plain'], device)

# Adversarial model training
for attack_name in attacks.keys():
    adv_train_and_eval(train_data_plain, test_data_plain, epochs_adv, models[attack_name],
                       attacks[attack_name], loss_fn, optimizers[attack_name], device, eval_fn, agg_fn)

# Adversarial model attacks
for model_name in train_data_adv_model_attack.keys():
    if model_name != 'plain':
        for attack_name in attacks.keys():
            train_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
                train_data_plain, attacks[attack_name], models[model_name], device)


for model_name in train_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Train data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for attack_name in train_data_adv_model_attack[model_name].keys():
        print("  = Train data {} =".format(attack_name))
        eval_test(train_data_adv_model_attack[model_name][attack_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)


# Analysis full data
adv_model_name = "fgsm"
# plain data
# all data
x = np.vstack([x[0] for x in train_data_plain.dataset])
y = np.vstack([x[1] for x in train_data_plain.dataset])
# 1st batch
# x, y = next(iter(train_data_plain))
# x = x.detach().numpy()
# y = y.detach().numpy()
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
x = np.vstack([torch.clamp(x[0], domain[0], domain[1]).detach().numpy()
              for x in train_data_adv_model_attack['plain'][adv_model_name].dataset])
y = np.vstack([x[1].detach().numpy()
              for x in train_data_adv_model_attack['plain'][adv_model_name].dataset])
# 1st batch
# x, y = next(iter(train_data_adv_model_attack['plain'][adv_model_name]))
# x = x.detach().numpy()
# y = y.detach().numpy()
idx_1 = np.where(y == 1)[0]
idx_0 = np.where(y == 0)[0]
plt.scatter(x[idx_1, 0], x[idx_1, 1], s=20, c='b', marker="+", label='first')
plt.scatter(x[idx_0, 0], x[idx_0, 1], s=20, c='r', marker="+", label='second')

t = np.arange(0, 1, .01).reshape((100, 1))
a = a_true
b = b_true
plt.plot(-(a[1]*t + b)/a[0], t, c='black', linewidth=3)

a = list(models['plain'].parameters())[0].detach().numpy()
b = list(models['plain'].parameters())[1].detach().numpy()
x0 = -(a[:, 1]*t + b)/a[:, 0]
x1 = t
idx = (x0 >= -1) & (x0 <= 1) & (x1 >= -1) & (x1 <= 1)
plt.plot(x0[idx], x1[idx], c='black', linestyle='dashed', linewidth=3)
plt.ylabel('x[1]')
plt.xlabel('x[0]')
plt.title(adv_model_name)
plt.show()
plt.plot()

# Analysis
attack_name_1 = 'fgsm'
attack_name_2 = 'dro_entropic_lambda_1000.0'

i_attack_1, x_attack_1, y_attack_1, x_adv_attack_1, y_adv_attack_1 = eval_adversary(
    test_data_plain, train_data_adv_model_attack['plain'][attack_name_1], models['plain'], device, eval_fn)
i_attack_2, x_attack_2, y_attack_2, x_adv_attack_1, y_adv_attack_2 = eval_adversary(
    test_data_plain, train_data_adv_model_attack['plain'][attack_name_2], models['plain'], device, eval_fn)

# DEBUG
i = 0
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(test_data_plain.dataset[i][0].detach().cpu().numpy(), cmap='gray')
ax1.set_title(
    'Original: ' + str(digits[int(test_data_plain.dataset[i][1])]))
ax2.imshow(train_data_adv_model_attack['plain'][attack_name_1].dataset[i][0].detach(
).cpu().numpy(), cmap='gray')
ax2.set_title(attack_name_1 + ' classified as ' +
              str(digits[1*((y_attack_1[i]) > 0)]))
ax3.imshow(train_data_adv_model_attack['plain'][attack_name_2].dataset[i][0].detach(
).cpu().numpy(), cmap='gray')
ax3.set_title(attack_name_1 + ' classified as ' +
              str(digits[1*((y_attack_2[i]) > 0)]))
