import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from attack.evasion import FastGradientSignMethod, DroMinCorrectClassifier
from models import CNN, LinearMatrix
from learn.train import adv_train_and_eval, train_and_eval, eval_test
from attack.utils import generate_attack_loop, eval_adversary

torch.set_default_dtype(torch.float64)
# torch.manual_seed(171)
# airplane(0), cat(3) and dog(5)
domain = [0., 1.]
images = [0, 5]
labels = [0, 1]
channels = 3
n_classes = 2
loss_fn = torch.nn.CrossEntropyLoss()


def loss_fn_adv(input, target):
    return loss_fn(-input, target)


def class_fn(yp): return yp.argmax(dim=1) if yp.dim() > 1 else yp
def eval_fn(y, yp): return class_fn(y) == class_fn(yp)
def agg_fn(x): return x.sum().item()


# getting and transforming data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

d = 32
batch_size = 64
train_data_plain = torch.utils.data.DataLoader(
    two_images_train, batch_size=batch_size, shuffle=False)
test_data_plain = torch.utils.data.DataLoader(
    two_images_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = {
    'plain': CNN(channels, n_classes, 5, device),
}

epsilon = 0.1
attacks = {
    'fgsm': FastGradientSignMethod(loss_fn, epsilon, domain),
    'dro': DroMinCorrectClassifier(loss_fn_adv, 500*epsilon, domain, 0.1, 100000)
}
optimizers = {'plain': torch.optim.SGD(models['plain'].parameters(), lr=0.001)}

epochs = 100
epochs_adv = 50

# Plain model training with plain data
train_and_eval(train_data_plain, test_data_plain, epochs, models['plain'],
               loss_fn, optimizers['plain'], device, eval_fn, agg_fn)



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

# Analysis
print("  = Test data plain =")
eval_test(test_data_plain, models['plain'],
          loss_fn, device, eval_fn, agg_fn)
N = len(test_data_plain.dataset)
for attack_name in test_data_adv_model_attack['plain'].keys():
    print("  = Test data {} =".format(attack_name))
    avg_change = sum([(test_data_adv_model_attack['plain'][attack_name].dataset[i][0] -
                       test_data_plain.dataset[i][0]).abs().sum().item() for i in range(N)])/N
    print(f"Average abs move per sample: {avg_change}")
    eval_test(test_data_adv_model_attack['plain'][attack_name], models['plain'],
              loss_fn, device, eval_fn, agg_fn)

# Start instances for adv trained models with starting parameters starting at the plain trained model
for attack_name in attacks.keys():
    models[attack_name] = LinearMatrix(d, d, device)
    optimizers[attack_name] = torch.optim.SGD(
        models[attack_name].parameters(), lr=0.001)

# Adversarial model training
for attack_name in attacks.keys():
    adv_train_and_eval(train_data_plain, test_data_plain, epochs_adv, models[attack_name],
                       attacks[attack_name], loss_fn, optimizers[attack_name], device, eval_fn, agg_fn)
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
ax1.set_title('Original: ' + str(images[int(test_data_plain.dataset[i][1])]))
ax2.imshow(test_data_adv_model_attack['plain'][attack_name_1].dataset[i][0][0].detach(
).cpu().numpy(), cmap='gray')
ax2.set_title(attack_name_1 + ' classified as ' + str(images[class_fn(
    models['plain'](test_data_adv_model_attack['plain'][attack_name_1].dataset[i][0].unsqueeze(0)))]))
ax3.imshow(test_data_adv_model_attack['plain'][attack_name_2].dataset[i][0][0].detach(
).cpu().numpy(), cmap='gray')
ax3.set_title(attack_name_1 + ' classified as ' + str(images[class_fn(
    models['plain'](test_data_adv_model_attack['plain'][attack_name_2].dataset[i][0].unsqueeze(0)))]))
