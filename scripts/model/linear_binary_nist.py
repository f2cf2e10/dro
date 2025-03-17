import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
import torch
import numpy as np
from models import LinearMatrix
from learn.train import train_and_eval

torch.set_default_dtype(torch.float64)


def eval_fn(y, yp): return (1*(y > 0) == 1*(yp > 0))
def agg_fn(x): return x.sum().item()


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("linear_binary")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Parameters
domain = [0., 1.]
digits = [4, 9]
labels = [0., 1.]
d = 28

# Model
model = LinearMatrix(d, d, device)

# Training
batch_size = 64
epochs = 100
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Data
transform = transforms.ToTensor()
target_transform = transforms.Lambda(lambda y: y)
mnist_train = datasets.MNIST("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)
two_digits_train = list(filter(lambda x: np.isin(
    x[1], digits), mnist_train))
two_digits_train = [(x[0][0], labels[0] if x[1] == digits[0] else labels[1])
                    for x in two_digits_train]
mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)
two_digits_test = list(filter(lambda x: np.isin(
    x[1], digits), mnist_test))
two_digits_test = [(x[0][0], labels[0] if x[1] == digits[0] else labels[1])
                   for x in two_digits_test]

train_data = torch.utils.data.DataLoader(
    two_digits_train, batch_size=batch_size, shuffle=False)
test_data = torch.utils.data.DataLoader(
    two_digits_test, batch_size=batch_size, shuffle=False)

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "model_architecture": "Linear",
        "dataset_name": "MNIST",
        "type": "Train",
        "dataset_train_size": len(train_data.dataset),
        "dataset_test_size": len(test_data.dataset),
        "dataset_detail": str(digits[0]) + "/" + str(digits[1]) + " digits"

    })
    train_and_eval(train_data, test_data, epochs, model,
                   loss_fn, optimizer, device, eval_fn, agg_fn, mlflow=mlflow)

    # Save model
    model_path = "linear_binary_mnist_" + str(digits[0]) + "_" + str(digits[1]) + ".pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")
    print("Training complete and logged to MLflow!")
