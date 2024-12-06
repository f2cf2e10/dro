import matplotlib.pyplot as plt
import numpy as np
import torch
from attack.evasion import FastGradientSignMethod, Dro
from models import Linear
from learn.train import adv_train_and_eval, train_and_eval, eval_test
from attack.utils import generate_attack_loop

torch.set_default_dtype(torch.float64)
torch.manual_seed(171)
# Using only 2 digits for a linear classifier
domain = [-1., 1.]
digits = [0, 1]
labels = [0., 1.]
loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn_adv = loss_fn


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

models = {
    'plain': Linear(d, 1, device)
}

epsilon = 0.1
attacks = {
    'fgsm': FastGradientSignMethod(loss_fn_adv, epsilon, domain),
}
for lamb in [0.0, 0.05, 0.06, 0.07, 0.075, 0.08, 0.09, 0.1]:
    attacks['dro_lambda_' + str(lamb)] = Dro(loss_fn=loss_fn_adv, zeta=d*epsilon, domain=domain,
                                             max_steps=50, lamb=lamb)
optimizers = {'plain': torch.optim.SGD(models['plain'].parameters(), lr=0.001)}

epochs = 2000
epochs_adv = 100
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

# Start instances for adv trained models with starting parameters starting at the plain trained model
for attack_name in attacks.keys():
    models[attack_name] = Linear(d, 1, device)
    optimizers[attack_name] = torch.optim.SGD(
        models[attack_name].parameters(), lr=0.001)

train_data_adv_model_attack = {
    model: {
        attack_name: None for attack_name in attacks.keys()
    } for model in models.keys()
}

# Adversary data on plain model
for attack_name in attacks.keys():
    train_data_adv_model_attack['plain'][attack_name] = generate_attack_loop(
        train_data_plain, attacks[attack_name], models['plain'], device)

# Adversarial model training
for attack_name in attacks.keys():
    adv_train_and_eval(train_data_plain, test_data_plain, epochs_adv, models[attack_name],
                       attacks[attack_name], loss_fn, optimizers[attack_name], device, eval_fn, agg_fn)

# Adversarial data derived for the other models
for model_name in train_data_adv_model_attack.keys():
    if model_name != 'plain':
        for attack_name in attacks.keys():
            train_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
                train_data_plain, attacks[attack_name], models[model_name], device)

for model_name in train_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Test data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for attack_name in train_data_adv_model_attack[model_name].keys():
        print("  = Test data {} =".format(attack_name))
        eval_test(train_data_adv_model_attack[model_name][attack_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)

# Analysis full data
adv_model_name = "dro_lambda_0.075"
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
plt.figure(figsize=[8, 8])
plt.scatter(x[idx_1, 0], x[idx_1, 1], s=20, facecolors='None',
            edgecolors='b', marker="o", label='first')
# for idx in idx_1:
#    plt.annotate(str(idx), (x[idx][0], x[idx][1]+0.02))
plt.scatter(x[idx_0, 0], x[idx_0, 1], s=20, facecolors='None',
            edgecolors='r', marker="o", label='second')
# for idx in idx_0:
#    plt.annotate(str(idx), (x[idx][0], x[idx][1]+0.02))
# adv data on plain model
x_adv = np.vstack([x[0].detach().numpy()
                   for x in train_data_adv_model_attack['plain'][adv_model_name].dataset])
y_adv = np.vstack([x[1].detach().numpy()
                   for x in train_data_adv_model_attack['plain'][adv_model_name].dataset])
# 1st batch
# x, y = next(iter(train_data_adv_model_attack['plain'][adv_model_name]))
# x = x.detach().numpy()
# y = y.detach().numpy()
idx_1 = np.where(y_adv == 1)[0]
idx_0 = np.where(y_adv == 0)[0]
plt.scatter(x_adv[idx_1, 0], x_adv[idx_1, 1],
            s=20, c='b', marker="+", label='first')
plt.scatter(x_adv[idx_0, 0], x_adv[idx_0, 1], s=20,
            c='r', marker="+", label='second')

t = np.arange(-1, 1, .01).reshape((200, 1))
a = a_true
b = b_true
plt.plot(-(a[1]*t + b)/a[0], t, c='black', linewidth=3)

a = list(models['plain'].parameters())[0].detach().numpy()
b = list(models['plain'].parameters())[1].detach().numpy()
x0 = -(a[0][1]*t + b)/a[0][0]
x1 = t
idx = (x0 >= -1) & (x0 <= 1) & (x1 >= -1) & (x1 <= 1)
plt.plot(x0[idx], x1[idx], c='black', linestyle='dashed', linewidth=3)
plt.ylabel('x[1]')
plt.xlabel('x[0]')
plt.title(adv_model_name + " - moved {} points".format(
          np.sum(np.abs(x_adv - x).sum(axis=1) > 1E-6)))
plt.show()
plt.plot()
