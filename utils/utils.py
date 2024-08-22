import torch
import torch.utils


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (x_i, y_i) in enumerate(dataloader):
        x, y = x_i.to(device), y_i.to(device)
        yp = model(x)[:, 0]
        loss = loss_fn(yp, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def eval_test(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x_i, y_i in dataloader:
            x, y = x_i.to(device), y_i.to(device)
            yp = model(x)[:, 0]
            test_loss += loss_fn(yp, y.float()).item()
            correct += ((y * yp) > 0).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_and_eval(train_data, test_data, epochs, model, loss_fn, optimizer, device):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_data, model, loss_fn, optimizer, device)
        eval_test(test_data, model, loss_fn, device)
    print("Done!")


def generate_attack_loop(dataloader, attack, device):
    x = None
    y = None
    x_adv = None
    for x_i, y_i in dataloader:
        x = x_i.to(device) if x is None else torch.concatenate(
            [x, x_i.to(device)])
        y = y_i.to(device) if y is None else torch.concatenate(
            [y, y_i.to(device)])
        x_adv = attack.generate(x=x_i.to(device), y=y_i.to(device)) if x_adv is None else torch.concatenate(
            [x_adv, attack.generate(x=x_i.to(device), y=y_i.to(device))])
    return torch.utils.data.DataLoader([[x_adv[i], y[i]] for i in range(x_adv.shape[0])], 
                                       batch_size=dataloader.batch_size, shuffle=False)
