import pandas as pd
import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
import torch
from attack.evasion.dro import DroEntropic, DroMirroredLoss
from attack.evasion.fgsm import FastGradientSignMethod
from attack.utils import generate_attack_loop
from learn.train import eval_test

torch.set_default_dtype(torch.float64)


def class_fn(yp): return yp.argmax(dim=1) if yp.dim() > 1 else yp
def eval_fn(y, yp): return class_fn(y) == class_fn(yp)
def agg_fn(x): return x.sum().item()


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("cnn")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
domain = [0., 1.]
channels = 1
n_classes = 10
d = 28

# Model
models = mlflow.search_runs(filter_string="params.`model_architecture` = 'CNN' AND " +
                            "params.`type` = 'Train' AND " +
                            "params.`dataset_name` = 'MNIST' AND " +
                            "params.`dataset_detail` = '0-9 digits'",
                            order_by=["attributes.`start_time` DESC"],
                            search_all_experiments=True)
if models.empty:
    print("=====> No model found.")
if models.shape[0] > 1:
    print("=====> Several models found, picking the first one.")
model = mlflow.pytorch.load_model(f'runs:/{models.run_id[0]}/model')

# Training
batch_size = 100
epochs = 100
loss_fn = torch.nn.CrossEntropyLoss()


def loss_fn_adv(input, target):
    return loss_fn(-input, target)


# Attacks
epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
# Use > 0.5
learning_rate_adversary = 20
max_steps_adversary = 100000

# Data
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.Lambda(lambda y: y)])
mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)
test_data = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False)


def generate_attack_save_data_and_log(step, epsilon, attack, attack_name, extra_params=None):
    log = {
        "learning_rate": learning_rate_adversary,
        "batch_size": batch_size,
        "epochs": epochs,
        "model_architecture": "Linear",
        "dataset_name": "MNIST",
        "type": "Attack",
        "dataset_test_size": len(test_data.dataset),
        "dataset_detail": "0-9 digits",
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
    data_path = attack_name + "_epsilon_" + str(epsilon) + "_cnn_mnist_0-9.pt"
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
    lamb = 12500.
    for step, epsilon in enumerate(epsilons):
        attack = DroEntropic(loss_fn=loss_fn, zeta=epsilon, domain=domain,
                             max_steps=50, lamb=lamb)
        generate_attack_save_data_and_log(
            step, epsilon, attack, "dro_entropic", {"lambda": lamb})


# ANALYSIS
analyse = False
if analyse:
    df = None
    client = mlflow.MlflowClient()
    models = mlflow.search_runs(
        filter_string="params.`attack` = 'dro_mirrored' AND params.`dataset_name` = 'MNIST' AND params.`dataset_detail` = '0-9 digits'", search_all_experiments=True)
    for run_id in models['run_id']:
        acc = client.get_metric_history(run_id, "test_acc")
        epsilon = client.get_metric_history(run_id, "epsilon")
        abs_dev = client.get_metric_history(run_id, "abs_deviation")
        new_df = pd.DataFrame({
            "epsilon": [500*e.value for e in epsilon],
            "abs_dev": [d.value for d in abs_dev],
            "dro": [a.value for a in acc]
        })
        df = new_df if df is None else pd.concat([df, new_df])
