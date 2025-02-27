import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from attack.evasion.dro import DroEntropic
from attack.evasion.fgsm import FastGradientSignMethod
from attack.utils import eval_adversary, generate_attack_loop
from models import CNN
from learn.train import eval_test, train_and_eval

torch.set_default_dtype(torch.float64)
torch.manual_seed(171)


def class_fn(yp): return yp.argmax(dim=1) if yp.dim() > 1 else yp
def eval_fn(y, yp): return class_fn(y) == class_fn(yp)
def agg_fn(x): return x.sum().item()


domain = [0., 1.]
channels = 3
n_classes = 10

# getting and transforming data
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.Lambda(lambda y: y)])

cifar_train = datasets.CIFAR10("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)

cifar_test = datasets.CIFAR10("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)

batch_size = 100
train_data_plain = torch.utils.data.DataLoader(
    cifar_train, batch_size=batch_size, shuffle=False)
test_data_plain = torch.utils.data.DataLoader(
    cifar_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn_adv = loss_fn

models = {
    'plain': CNN(channels, n_classes, 5, device),
}

epsilon = 0.1
attacks = {
    'fgsm': FastGradientSignMethod(loss_fn_adv, epsilon, domain),
}
optimizers = {'plain': torch.optim.SGD(models['plain'].parameters(), lr=0.001)}


epochs = 100
epochs_adv = 50
# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)

for attack_name in attacks.keys():
    models[attack_name] = CNN(channels, n_classes, device)
    optimizers[attack_name] = torch.optim.SGD(
        models[attack_name].parameters(), lr=0.001)

test_data_adv_model_attack = {
    model: {
        attack_name: None for attack_name in attacks.keys()
    } for model in models.keys()
}

# Plain model attacks
for attack_name in attacks.keys():
    print(f'===== {attack_name} =====')
    test_data_adv_model_attack['plain'][attack_name] = generate_attack_loop(
        test_data_plain, attacks[attack_name], models['plain'], device)

# Adversarial model attacks
for model_name in test_data_adv_model_attack.keys():
    if model_name != 'plain':
        for attack_name in attacks.keys():
            test_data_adv_model_attack[model_name][attack_name] = generate_attack_loop(
                train_data_plain, attacks[attack_name], models[model_name], device)

for model_name in test_data_adv_model_attack.keys():
    print("=== MODEL for attack {} ===".format(model_name))
    print("  = Test data plain =")
    eval_test(test_data_plain, models[model_name],
              loss_fn, device, eval_fn, agg_fn)
    for attack_name in test_data_adv_model_attack[model_name].keys():
        print("  = Test data {} =".format(attack_name))
        eval_test(test_data_adv_model_attack[model_name][attack_name], models[model_name],
                  loss_fn, device, eval_fn, agg_fn)

# Analysis
attack_name_1 = 'fgsm'
attack_name_2 = 'dro'

i_attack_1, x_attack_1, y_attack_1, x_adv_attack_1, y_adv_attack_1 = eval_adversary(
    test_data_plain, test_data_adv_model_attack['plain'][attack_name_1], models['plain'], device, eval_fn)
i_attack_2, x_attack_2, y_attack_2, x_adv_attack_1, y_adv_attack_2 = eval_adversary(
    test_data_plain, test_data_adv_model_attack['plain'][attack_name_2], models['plain'], device, eval_fn)

# DEBUG
i = 0
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(test_data_plain.dataset[i][0]
           [0].detach().cpu().numpy(), cmap='gray')
ax1.set_title(
    'Original: ' + str(int(test_data_plain.dataset[i][1])))
ax2.imshow(test_data_adv_model_attack['plain'][attack_name_1].dataset[i][0][0].detach(
).cpu().numpy(), cmap='gray')
ax2.set_title(attack_name_1 + ' classified as ' + str(
    class_fn(models['plain'](test_data_adv_model_attack['plain'][attack_name_1].dataset[i][0].unsqueeze(0)))))
ax3.imshow(test_data_adv_model_attack['plain'][attack_name_2].dataset[i][0][0].detach(
).cpu().numpy(), cmap='gray')
ax3.set_title(attack_name_1 + ' classified as ' + str(
    class_fn(models['plain'](test_data_adv_model_attack['plain'][attack_name_2].dataset[i][0].unsqueeze(0)))))
