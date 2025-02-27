import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms
import torch
from models import CNN
from learn.train import train_and_eval

torch.set_default_dtype(torch.float64)


def class_fn(yp): return yp.argmax(dim=1) if yp.dim() > 1 else yp
def eval_fn(y, yp): return class_fn(y) == class_fn(yp)
def agg_fn(x): return x.sum().item()


mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("cnn_mnist")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
domain = [0., 1.]
channels = 1
n_classes = 10

# Model
model = CNN(channels, n_classes, 3, device)

# Training
batch_size = 100
epochs = 100
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Data
transform = transforms.Compose([transforms.ToTensor()])
target_transform = transforms.Compose([transforms.Lambda(lambda y: y)])

mnist_train = datasets.MNIST("./data", train=True, download=True,
                             transform=transform, target_transform=target_transform)

mnist_test = datasets.MNIST("./data", train=False, download=True,
                            transform=transform, target_transform=target_transform)

train_data = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=False)
test_data = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False)

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "model_architecture": "CNN",
        "dataset_name": "MNIST" ,
        "dataset_train_size": len(train_data),
        "dataset_test_size": len(test_data),
        "dataset_detail": "0-9 digits"

    })
    train_and_eval(train_data, test_data, epochs, model,
                   loss_fn, optimizer, device, eval_fn, agg_fn, mlflow=mlflow)
    
    # Save model
    model_path = "cnn_mnist_allclasses.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.pytorch.log_model(model, "model")
    print("Training complete and logged to MLflow!")