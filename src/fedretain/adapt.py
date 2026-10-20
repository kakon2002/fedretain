"""Test-time BatchNorm re-estimation with population statistics, prediction and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fedretain.config import Config
from fedretain.data import Loader
from fedretain.metrics import metrics_from


def autocast_ctx(device: torch.device) -> Any:
    """bfloat16 autocast on CUDA; disabled autocast on CPU (so CPU runs are bit-for-bit deterministic)."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def calibrate_bn(
    model: nn.Module, loader: Loader | None, device: torch.device, max_batches: int = 0, ema: bool = False
) -> None:
    """Reset BN running statistics and re-estimate them over `loader` (cumulative average, order-free)."""
    if loader is None:
        return
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(12345)
    saved = {}
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            saved[module] = module.momentum
            module.reset_running_stats()
            module.momentum = 0.1 if ema else None
    model.train()
    for i, (bx, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with autocast_ctx(device):
            model(bx.to(device, non_blocking=True))
    for module, momentum in saved.items():
        module.momentum = momentum
    model.eval()


@torch.no_grad()
def predict(model: nn.Module, loader: Loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Tumour probabilities and labels over a loader, in loader order."""
    model.eval()
    probs, labels = [], []
    for bx, by in loader:
        with autocast_ctx(device):
            logits = model(bx.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        labels.append(by.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def evaluate(
    model: nn.Module,
    loader: Loader,
    device: torch.device,
    bn: tuple[Loader | None, bool],
    cfg: Config,
    max_batches: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Re-estimate BN on bn=(loader, ema), then score `loader`; max_batches caps the calibration pass."""
    cap = cfg.bn_max_batches if max_batches is None else max_batches
    calibrate_bn(model, bn[0], device, cap, bn[1])
    probs, labels = predict(model, loader, device)
    return metrics_from(probs, labels, cfg.n_bins), probs, labels
