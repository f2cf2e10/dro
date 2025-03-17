import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
import torch
import numpy as np
from attack.evasion.dro import DroEntropic, DroMirroredLoss
from attack.evasion.fgsm import FastGradientSignMethod
from attack.utils import generate_attack_loop
from learn.train import eval_test

torch.set_default_dtype(torch.float64)


def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("linear_binary")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
domain = [0., 1.]
digits = [0, 1]
labels = [0., 1.]
d = 28

# Model
models = mlflow.search_runs(filter_string="params.`model_architecture` = 'Linear' AND " +
                            "params.`type` = 'Train' AND " +
                            "params.`dataset_name` = 'MNIST' AND " +
                            "params.`dataset_detail` = '" +
                            str(digits[0]) + "/" + str(digits[1]) + " digits'",
                            order_by=["attributes.`start_time` DESC"],
                            search_all_experiments=True)
if models.empty:
    print("=====> No model found.")
if models.shape[0] > 1:
    print("=====> Several models found, picking the first one.")
model = mlflow.pytorch.load_model(f'runs:/{models.run_id[0]}/model')

# Training
batch_size = 64
epochs = 100
loss_fn = torch.nn.BCEWithLogitsLoss()


def loss_fn_adv(input, target):
    return loss_fn(-input, target)


# Attacks
epsilons = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
learning_rate_adversary = 0.1
max_steps_adversary = 100000

# Data
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.Lambda(lambda y: y)])
mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)
two_digits_test = list(filter(lambda x: np.isin(
    x[1], digits), mnist_test))
two_digits_test = [(x[0][0], labels[0] if x[1] == digits[0] else labels[1])
                   for x in two_digits_test]
test_data = torch.utils.data.DataLoader(
    two_digits_test, batch_size=batch_size, shuffle=False)


def generate_attack_save_data_and_log(step, epsilon, attack, attack_name, extra_params=None):
    log = {
        "learning_rate": learning_rate_adversary,
        "batch_size": batch_size,
        "epochs": epochs,
        "model_architecture": "Linear",
        "dataset_name": "MNIST",
        "type": "Attack",
        "dataset_test_size": len(test_data.dataset),
        "dataset_detail": str(digits[0]) + "/" + str(digits[1]) + " digits",
        "attack": attack_name
    }
    mlflow.log_params(log if extra_params is None else {**log, **extra_params})
    adv_test_data = generate_attack_loop(test_data, attack, model, device)
    N = len(adv_test_data.dataset)
    abs_dev = sum([(adv_test_data.dataset[i][0] - test_data.dataset[i][0]).abs().sum().item()
                   for i in range(N)])/N
    eval_test(adv_test_data, model, loss_fn, device,
              eval_fn, agg_fn, step, mlflow=mlflow)
    mlflow.log_metric("epsilon", epsilon, step=step)
    mlflow.log_metric("abs_deviation", abs_dev, step=step)
    data_path = attack_name + "_epsilon_" + \
        str(epsilon) + "_linear_binary_mnist_" + \
        str(digits[0]) + "_" + str(digits[1]) + ".pt"
    torch.save(adv_test_data, data_path)
    mlflow.log_artifact(data_path)


with mlflow.start_run():
    for step, epsilon in enumerate(epsilons):
        attack = FastGradientSignMethod(loss_fn, epsilon, domain)
        generate_attack_save_data_and_log(step, epsilon, attack, "fgsm")

with mlflow.start_run():
    for step, epsilon in enumerate(epsilons):
        attack = DroMirroredLoss(
            loss_fn_adv, d*d*epsilon, domain, learning_rate_adversary, max_steps_adversary)
        generate_attack_save_data_and_log(
            step, epsilon, attack, "dro_mirrored")

with mlflow.start_run():
    lamb = 5000.
    for step, epsilon in enumerate(epsilons):
        attack = DroEntropic(loss_fn=loss_fn, zeta=epsilon, domain=domain,
                             max_steps=50, lamb=lamb)
        generate_attack_save_data_and_log(
            step, epsilon, attack, "dro_entropic", {"lambda": lamb})
