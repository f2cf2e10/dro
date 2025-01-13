import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
import numpy as np
from attack.evasion import FastGradientSignMethod, ProjectecGradientDescent, Dro
from models import LinearModel
from learn.train import train_and_eval, eval_test
from attack.utils import generate_attack_loop
from utils.norm import Linf
from attack.utils import eval_adversary

torch.set_default_dtype(torch.float64)

# Using only 2 digits for a linear classifier
domain = [0., 1.]
digits = [0, 1]
labels = [0, 1]
# loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = torch.nn.SoftMarginLoss()
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

attacks = {'dro_0.01': Dro(loss_fn, d*0.01, domain, max_steps=30),
           'dro_0.025': Dro(loss_fn, d*0.025, domain, max_steps=30),
           'dro_0.05': Dro(loss_fn, d*0.05, domain, max_steps=30),
           'dro_0.1': Dro(loss_fn, d*0.1, domain, max_steps=30)}
models = {'plain': LinearModel(d).to(device)}
optimizers = {key: torch.optim.SGD(
    model.parameters(), lr=0.001) for key, model in zip(models.keys(), models.values())}
test_data_adv_model_attack = {'plain': {
    'dro_0.01': None,
    'dro_0.025': None,
    'dro_0.05': None,
    'dro_0.01': None}
}

epochs = 100
epochs_adv = 15
torch.manual_seed(109823)
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

# Adversarial data derived from plain model
for model_name in test_data_adv_model_attack.keys():
    for attack_name in attacks.keys():
        test_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
            test_data_plain, attacks[attack_name], models[model_name], device)

for model_name in test_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Test data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for adv_data_name in test_data_adv_model_attack[model_name].keys():
        print("  = Test data {} =").format(adv_data_name)
        eval_test(test_data_adv_model_attack[model_name][adv_data_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)

i_fgsm, x_fgsm, y_fgsm, x_adv_fgsm, y_adv_fgsm = eval_adversary(
    test_data_plain, test_data_adv_fgsm, model, device, eval_fn)
i_pgd, x_pgd, y_pgd, x_adv_pgd, y_adv_pgd = eval_adversary(
    test_data_plain, test_data_adv_pgd, model, device, eval_fn)
i_dro, x_dro, y_dro, x_adv_dro, y_adv_dro = eval_adversary(
    test_data_plain, test_data_adv_dro, model, device, eval_fn)

# DEBUG
i = 2000
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_plain.dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax1.set_title('Original: ' + str(digits[test_data_plain.dataset[i][1]]))
ax2.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_adv_fgsm.dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax2.set_title('FGSM classified as ' +
              str(digits[1*(model(test_data_adv_fgsm.dataset[i][0]) > 0)]))
ax3.imshow(torch.nn.Unflatten(0, (28, 28))(
    test_data_adv_dro.dataset[i][0]).detach().cpu().numpy(), cmap='gray')
ax3.set_title('DRO classified as ' +
              str(digits[1*(model(test_data_adv_dro.dataset[i][0]) > 0)]))

