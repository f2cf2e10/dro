import torch
import torch.utils


def generate_attack_loop(dataloader, attack, model, device):
    x = None
    y = None
    x_adv = None
    for x_i, y_i in dataloader:
        x = x_i.to(device) if x is None else torch.concatenate(
            [x, x_i.to(device)])
        y = y_i.to(device) if y is None else torch.concatenate(
            [y, y_i.to(device)])
        x_adv_batch = attack.generate(
            model=model, x=x_i.to(device), y=y_i.to(device))
        x_adv = x_adv_batch if x_adv is None else torch.concatenate(
            [x_adv, x_adv_batch])
    return torch.utils.data.DataLoader([[x_adv[i], y[i]] for i in range(x_adv.shape[0])],
                                       batch_size=dataloader.batch_size, shuffle=False)


def eval_adversary(dataloader, dataloader_adv, model, device, eval_fn):
    model.eval()
    num_batches = len(dataloader)

    data_iter = iter(dataloader)
    adv_data_iter = iter(dataloader_adv)
    missclassified_x = None
    missclassified_x_adv = None
    missclassified_y = None
    missclassified_y_adv = None
    items = []
    batch_items = [-1]
    with torch.no_grad():
        for _ in range(num_batches):
            x_i, y_i = next(data_iter)
            x_adv_i, y_adv_i = next(adv_data_iter)
            x, y = x_i.to(device), y_i.to(device)
            x_adv, y_adv = x_adv_i.to(device), y_adv_i.to(device)
            batch_items = [batch_items[-1] + i + 1 for i in range(len(x_i))]
            if torch.any(y != y_adv):
                print("Data is not aligned!!!!")
                break
            yp = model(x)[:, 0]
            yp_adv = model(x_adv)[:, 0]
            model_correct_adversary_wrong_idx = torch.logical_and(
                eval_fn(y, yp), torch.logical_not(eval_fn(yp, yp_adv)))
            if torch.any(model_correct_adversary_wrong_idx):
                missclassified_x_adv = x_adv[model_correct_adversary_wrong_idx] if missclassified_x_adv is None else torch.vstack(
                    [missclassified_x_adv, x_adv[model_correct_adversary_wrong_idx]])
                missclassified_x = x[model_correct_adversary_wrong_idx] if missclassified_x is None else torch.vstack(
                    [missclassified_x, x[model_correct_adversary_wrong_idx]])
                missclassified_y_adv = yp_adv[model_correct_adversary_wrong_idx] if missclassified_y_adv is None else torch.hstack(
                    [missclassified_y_adv, yp_adv[model_correct_adversary_wrong_idx]])
                missclassified_y = y[model_correct_adversary_wrong_idx] if missclassified_y is None else torch.hstack(
                    [missclassified_y, y[model_correct_adversary_wrong_idx]])
                items += [batch_items[i] for i in range(
                    len(model_correct_adversary_wrong_idx)) if model_correct_adversary_wrong_idx[i]]
    return items, missclassified_x, missclassified_y, missclassified_x_adv, missclassified_y_adv
