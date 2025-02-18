import torch


def missclassification_hinge_surrogate(outputs, targets, margin=1.0):
    """
    Custom Hinge Loss for binary classification.

    Parameters:
        outputs (Tensor): Model predictions (logits, not probabilities), shape (batch_size,).
        targets (Tensor): True labels (-1 or 1), shape (batch_size,).
        margin (float): The margin parameter for hinge loss (default: 1.0).

    Returns:
        Tensor: Computed hinge loss.
    """
    targets = targets.float()  # Ensure targets are float
    # loss = torch.mean(torch.clamp(margin - outputs * 2*(targets - 0.5), min=0)
    #                  ) if zero_ones else torch.mean(torch.clamp(margin - outputs * targets, min=0))
    loss = torch.mean(torch.clamp(margin - outputs * targets, min=0))
    return loss.requires_grad_()


def missclassification_softmargin_surrogate(outputs, targets, margin=1.0):
    """
    Custom Hinge Loss for binary classification.

    Parameters:
        outputs (Tensor): Model predictions (logits, not probabilities), shape (batch_size,).
        targets (Tensor): True labels (-1 or 1), shape (batch_size,).
        margin (float): The margin parameter for hinge loss (default: 1.0).

    Returns:
        Tensor: Computed hinge loss.
    """
    targets = targets.float()  # Ensure targets are float
    # loss = torch.mean(torch.clamp(margin - outputs * 2*(targets - 0.5), min=0)
    #                  ) if zero_ones else torch.mean(torch.clamp(margin - outputs * targets, min=0))
    loss = torch.mean(torch.log(1 + torch.exp(-outputs * targets)))
    return loss.requires_grad_()


def correctclassification_hinge_surrogate(outputs, targets, margin=1.0):
    """
    Custom Hinge Loss for binary classification in PyTorch.

    Parameters:
        outputs (Tensor): Model predictions (logits, not probabilities), shape (batch_size,).
        targets (Tensor): True labels (-1 or 1), shape (batch_size,).
        margin (float): The margin parameter for hinge loss (default: 1.0).

    Returns:
        Tensor: Computed hinge loss.
    """
    targets = targets.float()  # Ensure targets are float
    # loss = torch.mean(torch.clamp(margin + outputs * 2*(targets - 0.5), min=0)
    #                  ) if zero_ones else torch.mean(torch.clamp(margin + outputs * targets, min=0))
    loss = torch.mean(torch.clamp(margin + outputs * targets, min=0))
    return loss.requires_grad_()


def correctclassification_softmargin_surrogate(outputs, targets):
    """
    Custom Hinge Loss for binary classification in PyTorch.

    Parameters:
        outputs (Tensor): Model predictions (logits, not probabilities), shape (batch_size,).
        targets (Tensor): True labels (-1 or 1), shape (batch_size,).
        margin (float): The margin parameter for hinge loss (default: 1.0).

    Returns:
        Tensor: Computed hinge loss.
    """
    targets = targets.float()  # Ensure targets are float
    # loss = torch.mean(torch.clamp(margin - outputs * 2*(targets - 0.5), min=0)
    #                  ) if zero_ones else torch.mean(torch.clamp(margin - outputs * targets, min=0))
    loss = torch.mean(torch.log(1 + torch.exp(outputs * targets)))
    return loss.requires_grad_()
