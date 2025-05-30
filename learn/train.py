import torch
import torch.utils

from attack.evasion.interface import EvasionAttack


def train_loop(dataloader, model, loss_fn, optimizer, device, step, mlflow=None):
    print(f"Epoch {step}\n-------------------------------")
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (x_i, y_i) in enumerate(dataloader):
        x, y = x_i.to(device), y_i.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            current = batch * dataloader.batch_size + len(x)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    if mlflow is not None:
        mlflow.log_metric("train_loss", loss.item(), step=step)


def eval_test(dataloader, model, loss_fn, device, eval_fn, agg_fn, step, mlflow=None):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x_i, y_i in dataloader:
            x, y = x_i.to(device), y_i.to(device)
            y_hat = model(x)
            test_loss += loss_fn(y_hat, y).item()
            correct += agg_fn(eval_fn(y, y_hat))
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if mlflow is not None:
        mlflow.log_metric("test_loss", test_loss, step=step)
        mlflow.log_metric("test_acc", 100*correct, step=step)


def train_and_eval(train_data, test_data, epochs, model, loss_fn, optimizer, device, eval_fn, agg_fn, mlflow=None):
    for epoch in range(epochs):
        train_loop(train_data, model, loss_fn,
                   optimizer, device, epoch+1, mlflow)
        eval_test(test_data, model, loss_fn, device,
                  eval_fn, agg_fn, epoch+1, mlflow)
    print("Finished!")


def adv_train_loop(dataloader, model, attack, loss_fn, optimizer, device, epoch, mlflow=None):
    print(f"Epoch {epoch}\n-------------------------------")
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (x_i, y_i) in enumerate(dataloader):
        x, y = x_i.to(device), y_i.to(device)
        x_adv = attack.generate(model=model, x=x, y=y)
        y_hat = model(x_adv)
        loss = loss_fn(y_hat, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Check: https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933/3

        if batch % 50 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if mlflow is not None:
        mlflow.log_metric("train_loss", loss.item(), step=epoch)


def adv_train_and_eval(train_data, test_data, epochs, model, attack, loss_fn, optimizer, device, eval_fn, agg_fn):
    for epoch in range(epochs):
        adv_train_loop(train_data, model, attack,
                       loss_fn, optimizer, device, epoch+1)
        eval_test(test_data, model, loss_fn, device, eval_fn, agg_fn)
    print("Finished!")
