"""One training epoch and evaluation of a BRACE model."""
import torch

from .metrics import sentiment_metrics


def to_device(batch, device):
    *inputs, labels = (t.to(device) for t in batch)
    return inputs, labels


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    for batch in loader:
        inputs, labels = to_device(batch, device)
        with torch.autocast(device_type=device.type):
            loss = model.loss(inputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.item())
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        inputs, labels = to_device(batch, device)
        with torch.autocast(device_type=device.type):
            pred = model(inputs)
        preds.append(pred.float().cpu())
        trues.append(labels.cpu())
    return sentiment_metrics(torch.cat(preds).view(-1).numpy(), torch.cat(trues).view(-1).numpy())
