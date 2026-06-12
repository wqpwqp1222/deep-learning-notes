import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from torch import Tensor

from .configtools import get_default_device

try:
    from torchmetrics import Metric
except ImportError:
    raise ImportError(
        'torchmetrics is required to use training tools. '
        'Install it with `pip install torchmetrics`.'
    )

__all__ = [
    'train',
    'evaluate',
    'train_and_evaluate',
]


def train(
    model: nn.Module,
    dataloader: utils.DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    metric: Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Train a model for one epoch and return average loss and metric values.

    Args:
        model (Module): Model to train.
        dataloader (DataLoader): Batches of input tensors and target tensors.
        loss_fn (Module): Loss module used to optimize the model.
        optimizer (Optimizer): Optimizer that updates model parameters.
        metric (Metric): TorchMetrics metric updated from model predictions and targets.
        device (torch.device): Device where batches are moved before the forward pass.

    Returns:
        A tuple containing the average loss and computed metric value.
    """
    model.train()
    metric.reset()

    total_loss = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metric.update(logits.detach(), y)

    avg_loss = total_loss / len(dataloader)
    avg_metric = metric.compute().item()
    return avg_loss, avg_metric


def evaluate(
    model: nn.Module,
    dataloader: utils.DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    metric: Metric,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model for one epoch and return average loss and metric values.

    Args:
        model (Module): Model to evaluate.
        dataloader (DataLoader): Batches of input tensors and target tensors.
        loss_fn (Module): Loss module used to measure prediction error.
        metric (Optimizer): TorchMetrics metric updated from model predictions and targets.
        device (torch.device): Device where batches are moved before the forward pass.

    Returns:
        A tuple containing the average loss and computed metric value.
    """
    model.eval()
    metric.reset()

    total_loss = 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            metric.update(logits, y)

    avg_loss = total_loss / len(dataloader)
    avg_metric = metric.compute().item()
    return avg_loss, avg_metric


def train_and_evaluate(
    model: nn.Module,
    train_dl: utils.DataLoader[tuple[Tensor, Tensor]],
    val_dl: utils.DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train_metric: Metric,
    val_metric: Metric,
    num_epochs: int,
    device: torch.device | None = None,
) -> None:
    """Train and validate a model for multiple epochs.

    Args:
        model (Module): Model to train and evaluate.
        train_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        loss_fn (Module): Loss module used for both training and validation.
        optimizer (Optimizer): Optimizer that updates model parameters.
        train_metric (Metric): Metric used on the training split.
        val_metric (Metric): Metric used on the validation split.
        num_epochs (int): Number of epochs to run.
        device (torch.device | None, default: None): Device to use.
            Defaults to ``get_default_device()`` when omitted.
    """
    if device is None:
        device = get_default_device()

    model.to(device)
    train_metric.to(device)
    val_metric.to(device)

    for epoch in range(1, num_epochs + 1):
        loss, score = train(
            model=model,
            dataloader=train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metric=train_metric,
            device=device,
        )
        val_loss, val_score = evaluate(
            model=model,
            dataloader=val_dl,
            loss_fn=loss_fn,
            metric=val_metric,
            device=device,
        )

        w = len(str(num_epochs))
        print(
            f'Epoch [{epoch:{w}d}/{num_epochs:{w}d}] '
            f'| loss: {loss:.4f} '
            f'| metric: {score:.4f} '
            f'| val_loss: {val_loss:.4f} '
            f'| val_metric: {val_score:.4f}'
        )
