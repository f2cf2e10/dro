import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from attack.evasion import FastGradientSignMethod, Dro
from models import LinearFlat
from learn.train import adv_train_and_eval, train_and_eval, eval_test
from attack.utils import generate_attack_loop, eval_adversary

torch.set_default_dtype(torch.float64)
torch.manual_seed(171)
# airplane(0), cat(3) and dog(5)
domain = [0., 1.]
images = [0, 5]
labels = [0., 1.]
loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn_adv = loss_fn


def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()

# labels = [-1, 1]
# loss_fn = torch.nn.SoftMarginLoss()
# def eval_fn(y, yp): return ((y * yp) > 0)
# def agg_fn(x): return x.sum().item()


# getting and transforming data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.nn.Flatten(1, x.dim()-1)(x)),
    transforms.Lambda(lambda x: torch.nn.Flatten(0, x.dim()-1)(x))])
target_transform = transforms.Lambda(lambda y: y)


cifar_train = datasets.CIFAR10("./data", train=True, download=True,
                               transform=transform, target_transform=target_transform)
two_images_train = list(filter(lambda x: np.isin(
    x[1], images), cifar_train))
two_images_train = [(x[0], labels[0] if x[1] == images[0] else labels[1])
                    for x in two_images_train]

cifar_test = datasets.CIFAR10("./data", train=False, download=True,
                              transform=transform, target_transform=target_transform)
two_images_test = list(filter(lambda x: np.isin(
    x[1], images), cifar_test))
two_images_test = [(x[0], labels[0] if x[1] == images[0] else labels[1])
                   for x in two_images_test]

d = 3*32*32
batch_size = 64
train_data_plain = torch.utils.data.DataLoader(
    two_images_train, batch_size=batch_size, shuffle=False)
test_data_plain = torch.utils.data.DataLoader(
    two_images_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {
    'plain': LinearFlat(d, 1, device),
}

epsilon = 0.2
attacks = {
    'fgsm': FastGradientSignMethod(loss_fn_adv, epsilon, domain),
}
for lamb in [0.0, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1, 1.0]:
    attacks['dro_lambda_' + str(lamb)] = Dro(loss_fn=loss_fn_adv, zeta=d*epsilon, domain=domain,
                                             max_steps=50, lamb=lamb)
optimizers = {'plain': torch.optim.SGD(models['plain'].parameters(), lr=0.001)}

epochs = 100
epochs_adv = 15

# original_stdout = sys.stdout  # Save a reference to the original standard output

# with open('output.txt', 'w') as f:
#    sys.stdout = f  # Change the standard output to the file we created

# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

# Start instances for adv trained models with starting parameters starting at the plain trained model
for attack_name in attacks.keys():
    models[attack_name] = LinearFlat(d, 1, device)
    optimizers[attack_name] = torch.optim.SGD(
        models[attack_name].parameters(), lr=0.001)

train_data_adv_model_attack = {
    model: {
        attack_name: None for attack_name in attacks.keys()
    } for model in models.keys()
}

# Plain model attacks
for attack_name in attacks.keys():
    train_data_adv_model_attack['plain'][attack_name] = generate_attack_loop(
        test_data_plain, attacks[attack_name], models['plain'], device)

# Adversarial model training
# for attack_name in attacks.keys():
#    adv_train_and_eval(train_data_plain, test_data_plain, epochs_adv, models[attack_name],
#                       attacks[attack_name], loss_fn, optimizers[attack_name], device, eval_fn, agg_fn)

# Adversarial model attacks
# for model_name in train_data_adv_model_attack.keys():
#    if model_name != 'plain':
#        for attack_name in attacks.keys():
#            train_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
#                train_data_plain, attacks[attack_name], models[model_name], device)

for model_name in train_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Test data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for attack_name in train_data_adv_model_attack[model_name].keys():
        print("  = Test data {} =".format(attack_name))
        eval_test(train_data_adv_model_attack[model_name][attack_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)

#    sys.stdout = original_stdout


# TODO
attack_name_1 = 'fgsm'
attack_name_2 = 'dro_lambda_0.001'

i_fgsm, x_fgsm, y_fgsm, x_adv_fgsm, y_adv_fgsm = eval_adversary(
    test_data_plain, train_data_adv_model_attack['plain'][attack_name_1], models['plain'], device, eval_fn)
i_dro, x_dro, y_dro, x_adv_dro, y_adv_dro = eval_adversary(
    test_data_plain, train_data_adv_model_attack['plain'][attack_name_2], models['plain'], device, eval_fn)

# DEBUG
i = 0
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_plain.dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax1.set_title(
    'Original: ' + str(images[int(test_data_plain.dataset[i][1])]))
ax2.imshow(torch.nn.Unflatten(0, (28, 28))(
    train_data_adv_model_attack['plain'][attack_name_1].dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax2.set_title(attack_name_1 + ' classified as ' +
              str(images[1*(models['plain'](train_data_adv_model_attack['plain'][attack_name_1].dataset[i][0]) > 0)]))
ax3.imshow(torch.nn.Unflatten(0, (28, 28))(
    train_data_adv_model_attack['plain'][attack_name_2].dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax3.set_title('DRO classified as ' +
              str(images[1*(models['plain'](train_data_adv_model_attack['plain'][attack_name_2].dataset[i][0]) > 0)]))
